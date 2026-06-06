import os
import re
import json
from pathlib import Path
from typing import Any

import pandas as pd
import upstash_redis as redis
from dotenv import load_dotenv
from sqlalchemy import text

from app.api.engine.content_based import get_product_profiles, recommend
from app.api.engine.index_store import ACCESSORY_RULES_PATH, OFFLINE_RERANK_SCORES_PATH
from db.mssql import engine

load_dotenv()


class SceneRecommendationFilter:
    def __init__(self):
        self.engine = engine
        self.redis_client = redis.Redis(
            url=os.getenv("UPSTASH_URL"),
            token=os.getenv("UPSTASH_TOKEN"),
        )
        self.candidate_limit = int(os.getenv("LLM_CANDIDATE_LIMIT", "20"))
        self.interest_event_limit = int(os.getenv("USER_INTEREST_EVENT_LIMIT", "100"))
        self.last_llm_latency_ms = 0.0
        self.last_llm_status = "idle"
        self.last_source_ids: list[str] = []
        self.last_recommendation_items: list[dict[str, Any]] = []
        self.accessory_rules = self._load_json_file(ACCESSORY_RULES_PATH)
        self.offline_rerank_scores = self._load_json_file(OFFLINE_RERANK_SCORES_PATH)
        self.scene_similarity_weights = {
            "homepage": 0.9,
            "wishlist": 1.25,
            "cart": 1.55,
        }
        self.scene_rule_weights = {
            "homepage": {"exact": 1.1, "partial": 0.65},
            "wishlist": {"exact": 1.55, "partial": 0.95},
            "cart": {"exact": 1.95, "partial": 1.2},
        }

    def get_user_product_sets(self, user_id: int):
        purchased_query = text("""
        SELECT DISTINCT p.product_sys_id
        FROM OrderItem oi
        JOIN [Order] o ON oi.order_id = o.order_id
        JOIN Product p ON oi.product_id = p.product_id
        WHERE o.user_id = :user_id
        """)
        cart_query = text("""
        SELECT DISTINCT p.product_sys_id
        FROM CartItem ci
        JOIN Cart c ON ci.cart_id = c.cart_id
        JOIN Product p ON ci.product_id = p.product_id
        WHERE c.user_id = :user_id
        """)
        wishlist_query = text("""
        SELECT DISTINCT p.product_sys_id
        FROM Wishlist wl
        JOIN Product p ON wl.product_id = p.product_id
        WHERE wl.user_id = :user_id
        """)

        params = {"user_id": int(user_id)}
        purchased_df = pd.read_sql(purchased_query, self.engine, params=params)
        cart_df = pd.read_sql(cart_query, self.engine, params=params)
        wishlist_df = pd.read_sql(wishlist_query, self.engine, params=params)

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
        recent_count = max(len(recent_viewed), 1)
        for idx, pid in enumerate(recent_viewed):
            source_weight = max(0.45, 1.0 - (idx * (0.55 / recent_count)))
            for sim_id in recommend(pid, top_n=10):
                if sim_id in exclude:
                    continue
                candidate_scores[sim_id] = (
                    candidate_scores.get(sim_id, 0.0)
                    + (self.scene_similarity_weights["homepage"] * source_weight)
                )

        return self._rank_candidates(
            user_id=user_id,
            scene="homepage",
            source_ids=recent_viewed,
            candidate_scores=candidate_scores,
            exclude=exclude,
            top_n=top_n,
        )

    def get_recommendations_detail(self, user_id: int, product_sys_id: str, top_n: int = 11):
        self.last_llm_latency_ms = 0.0
        self.last_llm_status = "not_used"
        self.last_source_ids = [str(product_sys_id).strip()]
        self.last_recommendation_items = []
        recommendations = recommend(product_sys_id, top_n=top_n + 5)
        purchased_set, cart_set, wishlist_set = self.get_user_product_sets(user_id)

        exclude = purchased_set.union(cart_set).union(wishlist_set)
        exclude.add(str(product_sys_id))

        final_ids = [pid for pid in recommendations if pid not in exclude][:top_n]
        self.last_recommendation_items = self._build_ranked_items(
            ranked_ids=final_ids,
            candidate_scores={pid: float(max(top_n - idx, 1)) for idx, pid in enumerate(final_ids)},
            reason_code_by_id={pid: "content_similarity" for pid in final_ids},
        )
        return final_ids

    def get_recommendations_wishlist(self, user_id: int, top_n: int = 50):
        purchased_set, cart_set, wishlist_set = self.get_user_product_sets(user_id)
        wishlist_ids = list(wishlist_set)

        if not wishlist_ids:
            return []

        exclude = purchased_set.union(cart_set).union(wishlist_set)

        candidate_scores: dict[str, float] = {}
        for pid in wishlist_ids:
            for sim_id in recommend(pid, top_n=10):
                if sim_id in exclude:
                    continue
                candidate_scores[sim_id] = (
                    candidate_scores.get(sim_id, 0.0)
                    + self.scene_similarity_weights["wishlist"]
                )

        return self._rank_candidates(
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
                candidate_scores[sim_id] = (
                    candidate_scores.get(sim_id, 0.0)
                    + self.scene_similarity_weights["cart"]
                )

        exclude = purchased_set.union(cart_set).union(wishlist_set)
        return self._rank_candidates(
            user_id=user_id,
            scene="cart",
            source_ids=list(cart_set),
            candidate_scores=candidate_scores,
            exclude=exclude,
            top_n=top_n,
        )

    def _rank_candidates(
        self,
        user_id: int,
        scene: str,
        source_ids: list[str],
        candidate_scores: dict[str, float],
        exclude: set[str],
        top_n: int,
    ) -> list[str]:
        self.last_source_ids = [str(pid).strip() for pid in source_ids if str(pid).strip()]
        self.last_llm_latency_ms = 0.0
        self.last_llm_status = "not_used"
        self.last_recommendation_items = []
        interest_context = self._get_user_interest_context(user_id=user_id, source_ids=source_ids)
        reason_code_by_id: dict[str, str] = {}
        self._boost_candidates_from_interest(
            candidate_scores=candidate_scores,
            interest_context=interest_context,
            exclude=exclude,
            reason_code_by_id=reason_code_by_id,
        )
        source_profiles = self._fetch_product_profiles(source_ids)
        self._apply_scene_rule_scores(
            scene=scene,
            source_profiles=source_profiles,
            candidate_scores=candidate_scores,
            exclude=exclude,
            reason_code_by_id=reason_code_by_id,
        )

        affinity_candidate_ids = [
            pid
            for pid, _score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            if pid not in exclude
        ][: max(self.candidate_limit * 2, top_n)]
        affinity_profiles = self._fetch_product_profiles(affinity_candidate_ids, candidate_scores)
        self._apply_profile_affinity_scores(
            scene=scene,
            source_profiles=source_profiles,
            candidate_scores=candidate_scores,
            interest_context=interest_context,
            candidate_profiles=affinity_profiles,
            reason_code_by_id=reason_code_by_id,
        )
        self._apply_offline_bonus_scores(
            scene=scene,
            source_ids=source_ids,
            candidate_scores=candidate_scores,
            reason_code_by_id=reason_code_by_id,
        )

        sorted_candidate_ids = self._sorted_candidate_ids(candidate_scores=candidate_scores, exclude=exclude)
        fallback_ids = sorted_candidate_ids[:top_n]

        if not fallback_ids:
            return []

        self.last_llm_latency_ms = 0.0
        self.last_llm_status = "not_used_offline_only"
        self.last_recommendation_items = self._build_ranked_items(
            ranked_ids=fallback_ids[:top_n],
            candidate_scores=candidate_scores,
            reason_code_by_id=reason_code_by_id,
        )
        return fallback_ids[:top_n]

    def _fetch_product_profiles(
        self,
        product_ids: list[str],
        candidate_scores: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        return get_product_profiles(product_ids=product_ids, candidate_scores=candidate_scores)

    def _load_json_file(self, path: Path) -> dict[str, Any]:
        try:
            if not path.exists():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _sorted_candidate_ids(self, candidate_scores: dict[str, float], exclude: set[str]) -> list[str]:
        return [
            pid
            for pid, _score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            if pid not in exclude
        ]

    def _slugify(self, value: str) -> str:
        text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
        return text.strip("_")

    def _apply_scene_rule_scores(
        self,
        scene: str,
        source_profiles: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        exclude: set[str],
        reason_code_by_id: dict[str, str],
    ) -> None:
        if scene not in {"cart", "wishlist", "homepage"} or not self.accessory_rules:
            return

        candidate_profiles = self._fetch_product_profiles(list(candidate_scores.keys()), candidate_scores)
        candidate_category_map = {
            str(profile["product_id"]).strip(): self._slugify(str(profile.get("category", "")))
            for profile in candidate_profiles
        }

        target_categories: list[str] = []
        for profile in source_profiles:
            category_key = self._slugify(str(profile.get("category", "")))
            if not category_key:
                continue
            target_categories.extend(self.accessory_rules.get(category_key, []))

        if not target_categories:
            return

        scene_weight = self.scene_rule_weights.get(scene, {"exact": 1.0, "partial": 0.6})
        normalized_targets = [self._slugify(value) for value in target_categories if self._slugify(value)]
        for product_id, category_key in candidate_category_map.items():
            if product_id in exclude:
                continue
            rule_boost = 0.0
            for target_key in normalized_targets:
                if category_key == target_key:
                    rule_boost = max(rule_boost, float(scene_weight["exact"]))
                elif target_key and target_key in category_key:
                    rule_boost = max(rule_boost, float(scene_weight["partial"]))
            if rule_boost > 0:
                candidate_scores[product_id] = candidate_scores.get(product_id, 0.0) + rule_boost
                reason_code_by_id.setdefault(product_id, "accessory_rule")

    def _apply_offline_bonus_scores(
        self,
        scene: str,
        source_ids: list[str],
        candidate_scores: dict[str, float],
        reason_code_by_id: dict[str, str],
    ) -> None:
        if not self.offline_rerank_scores:
            return

        scene_payload = self.offline_rerank_scores.get(scene, {})
        if not isinstance(scene_payload, dict):
            return

        for source_id in source_ids:
            bonus_map = scene_payload.get(str(source_id).strip(), {})
            if not isinstance(bonus_map, dict):
                continue

            for candidate_id, score_payload in bonus_map.items():
                normalized_candidate_id = str(candidate_id).strip()
                if normalized_candidate_id not in candidate_scores:
                    continue
                if not isinstance(score_payload, dict):
                    continue
                try:
                    bonus_score = float(score_payload.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                candidate_scores[normalized_candidate_id] = (
                    candidate_scores.get(normalized_candidate_id, 0.0) + bonus_score
                )
                reason_code = str(score_payload.get("reason_code", "")).strip()
                if reason_code:
                    reason_code_by_id[normalized_candidate_id] = reason_code

    def _get_user_interest_context(self, user_id: int, source_ids: list[str]) -> dict[str, Any]:
        limit = max(int(self.interest_event_limit), 1)
        query = text(f"""
        SELECT TOP ({limit})
            p.product_sys_id,
            ct.name AS category,
            b.name AS brand,
            upe.view_count,
            upe.add_to_cart_count,
            upe.wishlist_count,
            upe.purchase_count,
            upe.interaction_score,
            upe.last_interacted_at
        FROM UserProductEvent upe
        JOIN Product p ON upe.product_id = p.product_id
        JOIN Category ct ON p.category_id = ct.category_id
        JOIN Brand b ON p.brandId = b.BrandId
        WHERE upe.user_id = :user_id
        ORDER BY
            upe.interaction_score DESC,
            upe.last_interacted_at DESC,
            upe.id DESC
        """)

        try:
            df = pd.read_sql(query, self.engine, params={"user_id": int(user_id)})
        except Exception:
            return {
                "product_scores": {},
                "category_scores": {},
                "brand_scores": {},
                "recent_interest_products": [],
            }
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
        df["view_count"] = pd.to_numeric(df["view_count"], errors="coerce").fillna(0).astype(float)
        df["add_to_cart_count"] = pd.to_numeric(df["add_to_cart_count"], errors="coerce").fillna(0).astype(float)
        df["wishlist_count"] = pd.to_numeric(df["wishlist_count"], errors="coerce").fillna(0).astype(float)
        df["purchase_count"] = pd.to_numeric(df["purchase_count"], errors="coerce").fillna(0).astype(float)
        df["interaction_score"] = pd.to_numeric(df["interaction_score"], errors="coerce").fillna(0.0).astype(float)
        df["last_interacted_at"] = pd.to_datetime(df["last_interacted_at"], errors="coerce")
        latest_interacted_at = df["last_interacted_at"].max()
        if pd.isna(latest_interacted_at):
            df["recency_factor"] = 1.0
        else:
            age_days = (latest_interacted_at - df["last_interacted_at"]).dt.total_seconds().div(86400.0).fillna(0.0)
            df["recency_factor"] = (1.0 - (age_days * 0.05)).clip(lower=0.35, upper=1.0)

        aggregate_signal = (
            (df["interaction_score"] * 1.25)
            + (df["purchase_count"] * 6.0)
            + (df["add_to_cart_count"] * 3.5)
            + (df["wishlist_count"] * 2.2)
            + (df["view_count"] * 0.5)
        )
        df["interest_score"] = aggregate_signal * df["recency_factor"]

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
        reason_code_by_id: dict[str, str],
    ) -> None:
        product_scores: dict[str, float] = interest_context["product_scores"]
        recent_interest_products: list[str] = interest_context["recent_interest_products"]

        for pid in recent_interest_products[:8]:
            base_interest_score = float(product_scores.get(pid, 0.0))
            if pid not in exclude and base_interest_score > 0:
                candidate_scores[pid] = candidate_scores.get(pid, 0.0) + (base_interest_score * 0.28)
                reason_code_by_id.setdefault(pid, "interest_match")

            if base_interest_score <= 0:
                continue

            for sim_id in recommend(pid, top_n=6):
                if sim_id in exclude:
                    continue
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0.0) + (base_interest_score * 0.12)
                reason_code_by_id.setdefault(sim_id, "interest_match")

    def _apply_profile_affinity_scores(
        self,
        scene: str,
        source_profiles: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        interest_context: dict[str, Any],
        candidate_profiles: list[dict[str, Any]],
        reason_code_by_id: dict[str, str],
    ) -> None:
        category_scores: dict[str, float] = interest_context["category_scores"]
        brand_scores: dict[str, float] = interest_context["brand_scores"]
        source_category_counts: dict[str, int] = {}
        source_brand_counts: dict[str, int] = {}

        for profile in source_profiles:
            category = str(profile.get("category", "")).strip()
            brand = str(profile.get("brand", "")).strip()
            if category:
                source_category_counts[category] = source_category_counts.get(category, 0) + 1
            if brand:
                source_brand_counts[brand] = source_brand_counts.get(brand, 0) + 1

        for profile in candidate_profiles:
            product_id = str(profile["product_id"]).strip()
            category = str(profile.get("category", "")).strip()
            brand = str(profile.get("brand", "")).strip()

            affinity_boost = (
                category_scores.get(category, 0.0) * 0.06
                + brand_scores.get(brand, 0.0) * 0.035
            )
            source_affinity_boost = 0.0
            if category in source_category_counts:
                if scene == "cart":
                    source_affinity_boost += 0.15 * source_category_counts[category]
                else:
                    source_affinity_boost += 0.22 * source_category_counts[category]
            if brand in source_brand_counts:
                source_affinity_boost += 0.08 * source_brand_counts[brand]
            affinity_boost += source_affinity_boost
            candidate_scores[product_id] = candidate_scores.get(product_id, 0.0) + affinity_boost
            if affinity_boost > 0 and product_id not in reason_code_by_id:
                if source_affinity_boost > 0 and brand in source_brand_counts and category in source_category_counts:
                    reason_code_by_id[product_id] = "brand_category_affinity"
                elif source_affinity_boost > 0 and category in source_category_counts:
                    reason_code_by_id[product_id] = "category_affinity"
                elif source_affinity_boost > 0 and brand in source_brand_counts:
                    reason_code_by_id[product_id] = "brand_affinity"
                else:
                    reason_code_by_id[product_id] = "category_affinity"

    def _build_ranked_items(
        self,
        ranked_ids: list[str],
        candidate_scores: dict[str, float],
        reason_code_by_id: dict[str, str],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for product_id in ranked_ids:
            items.append(
                {
                    "product_sys_id": str(product_id).strip(),
                    "score": round(float(candidate_scores.get(product_id, 0.0)), 4),
                    "reason_code": reason_code_by_id.get(str(product_id).strip(), "ranking_score"),
                }
            )
        return items
