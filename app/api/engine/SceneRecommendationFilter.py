import os
import re
from typing import Any

import pandas as pd
import upstash_redis as redis
from dotenv import load_dotenv

from app.api.engine.content_based import recommend
from app.api.engine.llm_personalization import GroqPersonalizationReranker
from db.mssql import engine

load_dotenv()


class SceneRecommendationFilter:
    def __init__(self):
        self.engine = engine
        self.redis_client = redis.Redis(
            url=os.getenv("UPSTASH_URL"),
            token=os.getenv("UPSTASH_TOKEN"),
        )
        self.llm_reranker = GroqPersonalizationReranker()
        self.candidate_limit = int(os.getenv("LLM_CANDIDATE_LIMIT", "20"))
        self.interest_event_limit = int(os.getenv("USER_INTEREST_EVENT_LIMIT", "200"))

    def get_user_product_sets(self, user_id: int):
        purchased_query = """
        SELECT DISTINCT p.product_sys_id
        FROM OrderItem oi
        JOIN [Order] o ON oi.order_id = o.order_id
        JOIN Product p ON oi.product_id = p.product_id
        WHERE o.user_id = ?
        """
        cart_query = """
        SELECT DISTINCT p.product_sys_id
        FROM CartItem ci
        JOIN Cart c ON ci.cart_id = c.cart_id
        JOIN Product p ON ci.product_id = p.product_id
        WHERE c.user_id = ?
        """
        wishlist_query = """
        SELECT DISTINCT p.product_sys_id
        FROM Wishlist wl
        JOIN Product p ON wl.product_id = p.product_id
        WHERE wl.user_id = ?
        """

        purchased_df = pd.read_sql(purchased_query, self.engine, params=(user_id,))
        cart_df = pd.read_sql(cart_query, self.engine, params=(user_id,))
        wishlist_df = pd.read_sql(wishlist_query, self.engine, params=(user_id,))

        return (
            set(purchased_df["product_sys_id"].astype(str).str.strip()),
            set(cart_df["product_sys_id"].astype(str).str.strip()),
            set(wishlist_df["product_sys_id"].astype(str).str.strip()),
        )

    def get_recommendations_homepage(self, user_id: int, top_n: int = 50):
        list_key = f"user:{user_id}:latest_watched" if user_id else f"guest:{user_id}:latest_watched"
        recent_viewed_raw = self.redis_client.lrange(list_key, 0, 9)
        recent_viewed = [
            pid.decode("utf-8") if isinstance(pid, bytes) else str(pid)
            for pid in recent_viewed_raw
        ]

        if not recent_viewed:
            return []

        purchased_set, cart_set, wishlist_set = self.get_user_product_sets(user_id)
        exclude = purchased_set.union(cart_set).union(wishlist_set)

        candidate_scores: dict[str, float] = {}
        for pid in recent_viewed:
            for sim_id in recommend(pid, top_n=10):
                if sim_id in exclude:
                    continue
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0.0) + 1.0

        return self._rerank_with_llm(
            user_id=user_id,
            scene="homepage",
            source_ids=recent_viewed,
            candidate_scores=candidate_scores,
            exclude=exclude,
            top_n=top_n,
        )

    def get_recommendations_detail(self, user_id: int, product_sys_id: str, top_n: int = 11):
        recommendations = recommend(product_sys_id, top_n=top_n + 5)
        purchased_set, cart_set, wishlist_set = self.get_user_product_sets(user_id)

        exclude = purchased_set.union(cart_set).union(wishlist_set)
        exclude.add(str(product_sys_id))

        return [pid for pid in recommendations if pid not in exclude][:top_n]

    def get_recommendations_wishlist(self, user_id: int, top_n: int = 50):
        _, _, wishlist_set = self.get_user_product_sets(user_id)
        wishlist_ids = list(wishlist_set)

        if not wishlist_ids:
            return []

        purchased_set, cart_set, _ = self.get_user_product_sets(user_id)
        exclude = purchased_set.union(cart_set).union(wishlist_set)

        candidate_scores: dict[str, float] = {}
        for pid in wishlist_ids:
            for sim_id in recommend(pid, top_n=10):
                if sim_id in exclude:
                    continue
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0.0) + 1.0

        return self._rerank_with_llm(
            user_id=user_id,
            scene="wishlist",
            source_ids=wishlist_ids,
            candidate_scores=candidate_scores,
            exclude=exclude,
            top_n=top_n,
        )

    def get_recommendations_cart(self, user_id: int, top_n: int = 15):
        purchased_set, cart_set, wishlist_set = self.get_user_product_sets(user_id)
        if not cart_set:
            return []

        candidate_scores: dict[str, float] = {}
        for pid in cart_set:
            for sim_id in recommend(pid, top_n=10):
                if sim_id in purchased_set or sim_id in cart_set or sim_id in wishlist_set:
                    continue
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0.0) + 1.0

        exclude = purchased_set.union(cart_set).union(wishlist_set)
        return self._rerank_with_llm(
            user_id=user_id,
            scene="cart",
            source_ids=list(cart_set),
            candidate_scores=candidate_scores,
            exclude=exclude,
            top_n=top_n,
        )

    def _rerank_with_llm(
        self,
        user_id: int,
        scene: str,
        source_ids: list[str],
        candidate_scores: dict[str, float],
        exclude: set[str],
        top_n: int,
    ) -> list[str]:
        interest_context = self._get_user_interest_context(user_id=user_id, source_ids=source_ids)
        self._boost_candidates_from_interest(
            candidate_scores=candidate_scores,
            interest_context=interest_context,
            exclude=exclude,
        )

        affinity_candidate_ids = [
            pid
            for pid, _score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            if pid not in exclude
        ][: max(self.candidate_limit * 2, top_n)]
        affinity_profiles = self._fetch_product_profiles(affinity_candidate_ids, candidate_scores)
        self._apply_profile_affinity_scores(
            candidate_scores=candidate_scores,
            interest_context=interest_context,
            candidate_profiles=affinity_profiles,
        )

        fallback_ids = [
            pid
            for pid, _score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            if pid not in exclude
        ]
        fallback_ids = fallback_ids[:top_n]

        if not fallback_ids:
            return []

        candidate_ids = [
            pid
            for pid, _score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            if pid not in exclude
        ][: self.candidate_limit]

        source_products = self._fetch_product_profiles(source_ids)
        candidate_products = self._fetch_product_profiles(candidate_ids, candidate_scores)

        ranked_ids = self.llm_reranker.rerank(
            scene=scene,
            source_products=source_products,
            candidate_products=candidate_products,
            top_n=top_n,
        )

        return self._merge_ranked_ids(ranked_ids, fallback_ids, top_n)

    def _merge_ranked_ids(self, ranked_ids: list[str], fallback_ids: list[str], top_n: int) -> list[str]:
        result = []
        seen = set()

        for pid in ranked_ids + fallback_ids:
            if pid in seen:
                continue
            if pid not in fallback_ids and pid not in ranked_ids:
                continue
            seen.add(pid)
            result.append(pid)
            if len(result) >= top_n:
                break

        return result

    def _fetch_product_profiles(
        self,
        product_ids: list[str],
        candidate_scores: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        normalized_ids = [str(pid).strip() for pid in product_ids if str(pid).strip()]
        if not normalized_ids:
            return []

        placeholders = ",".join(["?"] * len(normalized_ids))
        query = f"""
        SELECT
            p.product_sys_id,
            p.name,
            p.description,
            p.sellPrice,
            p.stock,
            p.status,
            ct.name AS category,
            b.name AS brand
        FROM Product p
        JOIN Category ct ON p.category_id = ct.category_id
        JOIN Brand b ON p.brandId = b.BrandId
        WHERE p.product_sys_id IN ({placeholders})
        """

        df = pd.read_sql(query, self.engine, params=tuple(normalized_ids))
        if df.empty:
            return []

        profiles_by_id: dict[str, dict[str, Any]] = {}
        for _, row in df.iterrows():
            product_id = str(row["product_sys_id"]).strip()
            profiles_by_id[product_id] = {
                "product_id": product_id,
                "name": self._clean_text(row["name"], max_len=100),
                "category": self._clean_text(row["category"], max_len=60),
                "brand": self._clean_text(row["brand"], max_len=50),
                "price": float(row["sellPrice"]) if pd.notna(row["sellPrice"]) else 0.0,
                "stock": int(row["stock"]) if pd.notna(row["stock"]) else 0,
                "status": self._clean_text(row["status"], max_len=30),
                "description": self._clean_text(row["description"], max_len=240),
            }
            if candidate_scores is not None:
                profiles_by_id[product_id]["base_score"] = round(float(candidate_scores.get(product_id, 0.0)), 4)

        ordered_profiles = []
        for pid in normalized_ids:
            profile = profiles_by_id.get(pid)
            if profile:
                ordered_profiles.append(profile)
        return ordered_profiles

    def _clean_text(self, value: Any, max_len: int) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        text = str(value)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_len]

    def _get_user_interest_context(self, user_id: int, source_ids: list[str]) -> dict[str, Any]:
        query = """
        SELECT TOP (?)
            p.product_sys_id,
            ct.name AS category,
            b.name AS brand,
            upe.event_type,
            upe.weight,
            upe.created_at
        FROM UserProductEvent upe
        JOIN Product p ON upe.product_id = p.product_id
        JOIN Category ct ON p.category_id = ct.category_id
        JOIN Brand b ON p.brandId = b.BrandId
        WHERE upe.user_id = ?
        ORDER BY upe.created_at DESC, upe.id DESC
        """

        df = pd.read_sql(query, self.engine, params=(self.interest_event_limit, user_id))
        if df.empty:
            return {
                "product_scores": {},
                "category_scores": {},
                "brand_scores": {},
                "recent_interest_products": [],
            }

        source_id_set = {str(pid).strip() for pid in source_ids}
        df["product_sys_id"] = df["product_sys_id"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip()
        df["brand"] = df["brand"].astype(str).str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).astype(float)
        df["event_rank"] = range(len(df))
        df["recency_factor"] = (1.0 - (df["event_rank"] * 0.015)).clip(lower=0.35)
        df["interest_score"] = df["weight"] * df["recency_factor"]

        product_scores = df.groupby("product_sys_id")["interest_score"].sum().to_dict()
        category_scores = df.groupby("category")["interest_score"].sum().to_dict()
        brand_scores = df.groupby("brand")["interest_score"].sum().to_dict()

        recent_interest_products = [
            pid
            for pid in df["product_sys_id"].tolist()
            if pid not in source_id_set
        ]

        return {
            "product_scores": product_scores,
            "category_scores": category_scores,
            "brand_scores": brand_scores,
            "recent_interest_products": list(dict.fromkeys(recent_interest_products)),
        }

    def _boost_candidates_from_interest(
        self,
        candidate_scores: dict[str, float],
        interest_context: dict[str, Any],
        exclude: set[str],
    ) -> None:
        product_scores: dict[str, float] = interest_context["product_scores"]
        recent_interest_products: list[str] = interest_context["recent_interest_products"]

        for pid in recent_interest_products[:8]:
            base_interest_score = float(product_scores.get(pid, 0.0))
            if pid not in exclude and base_interest_score > 0:
                candidate_scores[pid] = candidate_scores.get(pid, 0.0) + (base_interest_score * 0.2)

            if base_interest_score <= 0:
                continue

            for sim_id in recommend(pid, top_n=6):
                if sim_id in exclude:
                    continue
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0.0) + (base_interest_score * 0.35)

    def _apply_profile_affinity_scores(
        self,
        candidate_scores: dict[str, float],
        interest_context: dict[str, Any],
        candidate_profiles: list[dict[str, Any]],
    ) -> None:
        category_scores: dict[str, float] = interest_context["category_scores"]
        brand_scores: dict[str, float] = interest_context["brand_scores"]

        for profile in candidate_profiles:
            product_id = str(profile["product_id"]).strip()
            category = str(profile.get("category", "")).strip()
            brand = str(profile.get("brand", "")).strip()

            affinity_boost = (
                category_scores.get(category, 0.0) * 0.12
                + brand_scores.get(brand, 0.0) * 0.08
            )
            candidate_scores[product_id] = candidate_scores.get(product_id, 0.0) + affinity_boost
