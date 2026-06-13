import os
import re
import json
import hashlib
import math
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.api.engine.content_based import get_product_profiles, recommend
from app.api.engine.index_store import (
    ACCESSORY_RULES_PATH,
    COMPATIBILITY_RULES_PATH,
    OFFLINE_RERANK_SCORES_PATH,
    PRODUCTS_PATH,
)
from app.api.engine.recommendation_cache import recommendation_cache
from app.api.engine.recommendation_weights import load_recommendation_weight_config
from db.mssql import engine

load_dotenv()


class SceneRecommendationFilter:
    def __init__(self):
        self.engine = engine
        self.connection: Connection | None = None
        self.candidate_limit = int(os.getenv("LLM_CANDIDATE_LIMIT", "20"))
        self.interest_event_limit = int(os.getenv("USER_INTEREST_EVENT_LIMIT", "100"))
        self.user_state_cache_ttl_seconds = int(os.getenv("USER_STATE_CACHE_TTL_SECONDS", "15"))
        self.interest_context_cache_ttl_seconds = int(os.getenv("INTEREST_CONTEXT_CACHE_TTL_SECONDS", "30"))
        self.max_per_category = int(os.getenv("RECOMMEND_MAX_PER_CATEGORY", "4"))
        self.max_per_brand = int(os.getenv("RECOMMEND_MAX_PER_BRAND", "3"))
        self.last_llm_latency_ms = 0.0
        self.last_llm_status = "idle"
        self.last_source_ids: list[str] = []
        self.last_recommendation_items: list[dict[str, Any]] = []
        self._user_product_sets_cache: dict[int, tuple[set[str], set[str], set[str]]] = {}
        self._interest_context_cache: dict[int, dict[str, Any]] = {}
        self.accessory_rules = self._load_json_file(ACCESSORY_RULES_PATH)
        self.compatibility_rules = self._load_compatibility_rules()
        self.offline_rerank_scores = self._load_json_file(OFFLINE_RERANK_SCORES_PATH)
        self.product_catalog = self._load_product_catalog()
        self.weight_config = load_recommendation_weight_config()
        self.implicit_weights = self.weight_config.implicit
        self.decay_lambda = self.weight_config.decay_lambda
        self.scene_similarity_weights = self.weight_config.scene_similarity.as_dict()
        self.scene_rule_weights = self.weight_config.scene_rule_as_dict()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _get_connection(self) -> Connection:
        if self.connection is None:
            self.connection = self.engine.connect()
        return self.connection

    def get_scene_cache_token(self, scene: str, user_id: int) -> str | None:
        normalized_scene = str(scene).strip().lower()
        if normalized_scene not in {"detail", "wishlist", "homepage", "cart"}:
            return None

        purchased_set, cart_set, wishlist_set = self.get_user_product_sets(user_id)
        state_parts = [
            normalized_scene,
            self._hash_ids(purchased_set),
            self._hash_ids(cart_set),
            self._hash_ids(wishlist_set),
        ]
        if normalized_scene in {"wishlist", "homepage", "cart"}:
            interest_context = self._get_raw_user_interest_context(user_id)
            state_parts.append(str(interest_context.get("fingerprint", "none")).strip() or "none")
        return self._hash_parts(state_parts)

    def get_user_product_sets(self, user_id: int):
        normalized_user_id = int(user_id)
        cached = self._user_product_sets_cache.get(normalized_user_id)
        if cached is not None:
            return cached

        cache_key = recommendation_cache.build_key(
            scene="user_state",
            limit=1,
            user_id=normalized_user_id,
        )
        cached_payload = recommendation_cache.get_json(cache_key)
        if isinstance(cached_payload, dict):
            purchased_set = self._normalize_cached_id_list(cached_payload.get("purchased"))
            cart_set = self._normalize_cached_id_list(cached_payload.get("cart"))
            wishlist_set = self._normalize_cached_id_list(cached_payload.get("wishlist"))
            result = (purchased_set, cart_set, wishlist_set)
            self._user_product_sets_cache[normalized_user_id] = result
            return result

        query = text("""
        SELECT user_products.source_type, user_products.product_sys_id
        FROM (
            SELECT DISTINCT 'purchased' AS source_type, p.product_sys_id
            FROM OrderItem oi
            JOIN [Order] o ON oi.order_id = o.order_id
            JOIN Product p ON oi.product_id = p.product_id
            WHERE o.user_id = :user_id

            UNION ALL

            SELECT DISTINCT 'cart' AS source_type, p.product_sys_id
            FROM CartItem ci
            JOIN Cart c ON ci.cart_id = c.cart_id
            JOIN Product p ON ci.product_id = p.product_id
            WHERE c.user_id = :user_id

            UNION ALL

            SELECT DISTINCT 'wishlist' AS source_type, p.product_sys_id
            FROM Wishlist wl
            JOIN Product p ON wl.product_id = p.product_id
            WHERE wl.user_id = :user_id
        ) AS user_products
        WHERE user_products.product_sys_id IS NOT NULL
        """)

        purchased_set: set[str] = set()
        cart_set: set[str] = set()
        wishlist_set: set[str] = set()

        rows = self._get_connection().execute(query, {"user_id": normalized_user_id}).fetchall()
        for source_type, raw_product_sys_id in rows:
            product_sys_id = str(raw_product_sys_id or "").strip()
            if not product_sys_id:
                continue
            if source_type == "purchased":
                purchased_set.add(product_sys_id)
            elif source_type == "cart":
                cart_set.add(product_sys_id)
            elif source_type == "wishlist":
                wishlist_set.add(product_sys_id)

        result = (purchased_set, cart_set, wishlist_set)
        self._user_product_sets_cache[normalized_user_id] = result
        recommendation_cache.set_json(
            cache_key,
            {
                "purchased": sorted(purchased_set),
                "cart": sorted(cart_set),
                "wishlist": sorted(wishlist_set),
            },
            ttl_seconds=self.user_state_cache_ttl_seconds,
        )
        return result

    def get_recommendations_homepage(self, user_id: int, top_n: int = 50):
        purchased_set, cart_set, wishlist_set = self.get_user_product_sets(user_id)
        exclude = purchased_set.union(cart_set).union(wishlist_set)

        recent_viewed, interest_context = self._get_homepage_source_ids(
            user_id=user_id,
            exclude=exclude,
            fallback_sets=(purchased_set, cart_set, wishlist_set),
        )

        if not recent_viewed:
            return []

        candidate_scores: dict[str, float] = {}
        reason_code_by_id: dict[str, str] = {}
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

        self._collect_homepage_exploration_candidates(
            source_profiles=self._fetch_product_profiles(recent_viewed),
            candidate_scores=candidate_scores,
            exclude=exclude,
            reason_code_by_id=reason_code_by_id,
        )

        return self._rank_candidates(
            user_id=user_id,
            scene="homepage",
            source_ids=recent_viewed,
            candidate_scores=candidate_scores,
            exclude=exclude,
            top_n=top_n,
            interest_context=interest_context,
            initial_reason_code_by_id=reason_code_by_id,
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

        exclude = purchased_set.union(cart_set).union(wishlist_set)
        source_ids = list(cart_set)
        source_profiles = self._fetch_product_profiles(source_ids)
        candidate_scores: dict[str, float] = {}
        reason_code_by_id: dict[str, str] = {}

        self._collect_complementary_candidates(
            source_profiles=source_profiles,
            candidate_scores=candidate_scores,
            exclude=exclude,
            reason_code_by_id=reason_code_by_id,
        )

        if len(candidate_scores) < top_n:
            for pid in source_ids:
                for sim_id in recommend(pid, top_n=8):
                    if sim_id in exclude or sim_id in candidate_scores:
                        continue
                    candidate_scores[sim_id] = (
                        candidate_scores.get(sim_id, 0.0)
                        + (self.scene_similarity_weights["cart"] * 0.15)
                    )
                    reason_code_by_id.setdefault(sim_id, "cart_related_fallback")
                if len(candidate_scores) >= top_n:
                    break

        return self._rank_candidates(
            user_id=user_id,
            scene="cart",
            source_ids=source_ids,
            candidate_scores=candidate_scores,
            exclude=exclude,
            top_n=top_n,
            source_profiles=source_profiles,
            initial_reason_code_by_id=reason_code_by_id,
        )

    def _rank_candidates(
        self,
        user_id: int,
        scene: str,
        source_ids: list[str],
        candidate_scores: dict[str, float],
        exclude: set[str],
        top_n: int,
        interest_context: dict[str, Any] | None = None,
        source_profiles: list[dict[str, Any]] | None = None,
        initial_reason_code_by_id: dict[str, str] | None = None,
    ) -> list[str]:
        self.last_source_ids = [str(pid).strip() for pid in source_ids if str(pid).strip()]
        self.last_llm_latency_ms = 0.0
        self.last_llm_status = "not_used"
        self.last_recommendation_items = []
        if interest_context is None:
            interest_context = self._get_user_interest_context(user_id=user_id, source_ids=source_ids)
        reason_code_by_id: dict[str, str] = dict(initial_reason_code_by_id or {})
        self._boost_candidates_from_interest(
            scene=scene,
            candidate_scores=candidate_scores,
            interest_context=interest_context,
            exclude=exclude,
            reason_code_by_id=reason_code_by_id,
        )
        if source_profiles is None:
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
        self._apply_price_compatibility_scores(
            scene=scene,
            source_profiles=source_profiles,
            candidate_scores=candidate_scores,
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
        fallback_ids = self._diversify_candidates(
            scene=scene,
            sorted_candidate_ids=sorted_candidate_ids,
            top_n=top_n,
        )

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

    def _get_homepage_source_ids(
        self,
        user_id: int,
        exclude: set[str],
        fallback_sets: tuple[set[str], set[str], set[str]] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        if not user_id:
            return [], {
                "product_scores": {},
                "category_scores": {},
                "brand_scores": {},
                "recent_interest_products": [],
            }

        interest_context = self._get_user_interest_context(user_id=user_id, source_ids=[])
        recent_viewed = [
            pid
            for pid in interest_context.get("recent_interest_products", [])[:8]
            if pid and pid not in exclude
        ]

        if not recent_viewed:
            if fallback_sets is None:
                purchased_set, cart_set, wishlist_set = self.get_user_product_sets(user_id)
            else:
                purchased_set, cart_set, wishlist_set = fallback_sets
            recent_viewed = list(cart_set or wishlist_set or purchased_set)
            recent_viewed = [pid for pid in recent_viewed if pid not in exclude]

        return recent_viewed[:8], interest_context

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

    def _load_product_catalog(self) -> dict[str, dict[str, Any]]:
        try:
            if not PRODUCTS_PATH.exists():
                return {}
            payload = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                return {}
            catalog: dict[str, dict[str, Any]] = {}
            for product in payload:
                if not isinstance(product, dict):
                    continue
                product_id = str(product.get("product_sys_id", "")).strip()
                if not product_id:
                    continue
                catalog[product_id] = {
                    "product_id": product_id,
                    "name": str(product.get("name", "")).strip(),
                    "category": str(product.get("category", "")).strip(),
                    "brand": str(product.get("brand", "")).strip(),
                    "price": self._to_float(product.get("price")),
                    "stock": int(self._to_float(product.get("stock"))),
                    "status": str(product.get("status", "")).strip(),
                }
            return catalog
        except Exception:
            return {}

    def _load_compatibility_rules(self) -> dict[str, dict[str, Any]]:
        try:
            if not COMPATIBILITY_RULES_PATH.exists():
                return {}
            payload = json.loads(COMPATIBILITY_RULES_PATH.read_text(encoding="utf-8"))
            raw_rules = payload.get("rules", payload) if isinstance(payload, dict) else {}
            if not isinstance(raw_rules, dict):
                return {}
            rules: dict[str, dict[str, Any]] = {}
            for source, rule in raw_rules.items():
                source_slug = self._slugify(str(source))
                if not source_slug or not isinstance(rule, dict):
                    continue
                rules[source_slug] = {
                    "target_categories": self._normalize_pattern_list(rule.get("target_categories")),
                    "target_category_limits": self._normalize_limit_dict(rule.get("target_category_limits")),
                    "include_any": self._normalize_pattern_list(rule.get("include_any")),
                    "exclude_any": self._normalize_pattern_list(rule.get("exclude_any")),
                }
            return rules
        except Exception:
            return {}

    def _normalize_limit_dict(self, values: Any) -> dict[str, int]:
        if not isinstance(values, dict):
            return {}
        limits: dict[str, int] = {}
        for key, value in values.items():
            category_key = self._slugify(str(key))
            if not category_key:
                continue
            try:
                limit = int(value)
            except (TypeError, ValueError):
                continue
            if limit > 0:
                limits[category_key] = limit
        return limits

    def _normalize_pattern_list(self, values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        patterns: list[str] = []
        for value in values:
            pattern = self._slugify(str(value))
            if pattern and pattern not in patterns:
                patterns.append(pattern)
        return patterns

    def _sorted_candidate_ids(self, candidate_scores: dict[str, float], exclude: set[str]) -> list[str]:
        return [
            pid
            for pid, _score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            if pid not in exclude
        ]

    def _slugify(self, value: str) -> str:
        text = str(value or "").strip().lower()
        text = text.replace("\u0111", "d").replace("\u0110", "d")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"[^a-z0-9]+", "_", text)
        return text.strip("_")

    def _scene_interest_reason(self, scene: str) -> str:
        if scene == "wishlist":
            return "wishlist_similar_alternative"
        if scene == "cart":
            return "cart_bundle_candidate"
        return "homepage_recent_interest"

    def _scene_rule_reason(self, scene: str) -> str:
        if scene == "cart":
            return "cart_accessory_match"
        if scene == "wishlist":
            return "wishlist_similar_alternative"
        return "homepage_category_exploration"

    def _scene_affinity_reason(
        self,
        scene: str,
        *,
        same_category: bool,
        same_brand: bool,
    ) -> str:
        if scene == "cart":
            return "cart_bundle_candidate" if same_category or same_brand else "cart_accessory_match"
        if scene == "wishlist":
            if same_category and same_brand:
                return "wishlist_similar_alternative"
            if same_category:
                return "wishlist_price_match"
            return "wishlist_upgrade_option"
        if same_category:
            return "homepage_category_exploration"
        if same_brand:
            return "homepage_recent_interest"
        return "homepage_category_exploration"

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
        source_category_set: set[str] = set()
        for profile in source_profiles:
            category_key = self._slugify(str(profile.get("category", "")))
            if not category_key:
                continue
            source_category_set.add(category_key)
            target_categories.extend(self.accessory_rules.get(category_key, []))

        if not target_categories:
            return

        scene_weight = self.scene_rule_weights.get(scene, {"exact": 1.0, "partial": 0.6})
        normalized_targets = [self._slugify(value) for value in target_categories if self._slugify(value)]
        for product_id, category_key in candidate_category_map.items():
            if product_id in exclude:
                continue
            candidate_profile = self.product_catalog.get(product_id)
            if (
                scene in {"homepage", "cart"}
                and candidate_profile is not None
                and not self._is_cart_compatible_candidate(source_category_set, candidate_profile)
            ):
                continue
            rule_boost = 0.0
            for target_key in normalized_targets:
                if category_key == target_key:
                    rule_boost = max(rule_boost, float(scene_weight["exact"]))
                elif target_key and target_key in category_key:
                    rule_boost = max(rule_boost, float(scene_weight["partial"]))
            if rule_boost > 0:
                candidate_scores[product_id] = candidate_scores.get(product_id, 0.0) + rule_boost
                reason_code_by_id.setdefault(product_id, self._scene_rule_reason(scene))

    def _collect_complementary_candidates(
        self,
        source_profiles: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        exclude: set[str],
        reason_code_by_id: dict[str, str],
    ) -> None:
        if not source_profiles or not self.accessory_rules or not self.product_catalog:
            return

        source_categories = [
            self._slugify(str(profile.get("category", "")))
            for profile in source_profiles
            if self._slugify(str(profile.get("category", "")))
        ]
        source_category_set = set(source_categories)
        target_categories: set[str] = set()
        for category_key in source_categories:
            for target in self.accessory_rules.get(category_key, []):
                normalized_target = self._slugify(str(target))
                if normalized_target:
                    target_categories.add(normalized_target)
        if not target_categories:
            return

        scene_weight = self.scene_rule_weights["cart"]
        added_by_category: dict[str, int] = {}
        for product_id, profile in self.product_catalog.items():
            if product_id in exclude:
                continue
            if int(profile.get("stock", 0) or 0) <= 0:
                continue
            category_key = self._slugify(str(profile.get("category", "")))
            if not category_key:
                continue
            if not self._is_cart_compatible_candidate(source_category_set, profile):
                continue
            category_limit = self._cart_target_category_limit(source_category_set, category_key)
            if category_limit is not None and added_by_category.get(category_key, 0) >= category_limit:
                continue
            rule_score = 0.0
            for target_key in target_categories:
                if category_key == target_key:
                    rule_score = max(rule_score, float(scene_weight["exact"]))
                elif target_key in category_key:
                    rule_score = max(rule_score, float(scene_weight["partial"]))
            if rule_score <= 0:
                continue
            candidate_scores[product_id] = candidate_scores.get(product_id, 0.0) + rule_score
            reason_code_by_id.setdefault(product_id, "cart_accessory_match")
            added_by_category[category_key] = added_by_category.get(category_key, 0) + 1

    def _cart_target_category_limit(self, source_categories: set[str], candidate_category: str) -> int | None:
        limits: list[int] = []
        for source_category in source_categories:
            rule = self.compatibility_rules.get(source_category)
            if not rule:
                continue
            limit = rule.get("target_category_limits", {}).get(candidate_category)
            if isinstance(limit, int) and limit > 0:
                limits.append(limit)
        if not limits:
            return None
        return max(limits)

    def _is_cart_compatible_candidate(
        self,
        source_categories: set[str],
        candidate_profile: dict[str, Any],
    ) -> bool:
        matching_rules = [
            self.compatibility_rules[source_category]
            for source_category in source_categories
            if source_category in self.compatibility_rules
        ]
        if not matching_rules:
            return True

        candidate_category = self._slugify(str(candidate_profile.get("category", "")))
        searchable_text = self._candidate_searchable_text(candidate_profile)
        for rule in matching_rules:
            exclude_any = rule.get("exclude_any", [])
            if any(pattern and pattern in searchable_text for pattern in exclude_any):
                continue

            target_categories = rule.get("target_categories", [])
            if target_categories and candidate_category not in target_categories:
                continue

            include_any = rule.get("include_any", [])
            if include_any and not any(pattern and pattern in searchable_text for pattern in include_any):
                continue

            return True
        return False

    def _candidate_searchable_text(self, candidate_profile: dict[str, Any]) -> str:
        return self._slugify(
            " ".join(
                [
                    str(candidate_profile.get("name", "")),
                    str(candidate_profile.get("brand", "")),
                    str(candidate_profile.get("category", "")),
                ]
            )
        )

    def _collect_homepage_exploration_candidates(
        self,
        source_profiles: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        exclude: set[str],
        reason_code_by_id: dict[str, str],
    ) -> None:
        if not source_profiles or not self.accessory_rules or not self.product_catalog:
            return

        source_categories = {
            self._slugify(str(profile.get("category", "")))
            for profile in source_profiles
            if self._slugify(str(profile.get("category", "")))
        }
        target_categories: set[str] = set()
        for category_key in source_categories:
            for target in self.accessory_rules.get(category_key, []):
                normalized_target = self._slugify(str(target))
                if normalized_target:
                    target_categories.add(normalized_target)
        if not target_categories:
            return

        added_per_category: dict[str, int] = {}
        for product_id, profile in self.product_catalog.items():
            if product_id in exclude:
                continue
            if int(profile.get("stock", 0) or 0) <= 0:
                continue
            category_key = self._slugify(str(profile.get("category", "")))
            if category_key not in target_categories:
                continue
            if not self._is_cart_compatible_candidate(source_categories, profile):
                continue
            if added_per_category.get(category_key, 0) >= 4:
                continue
            candidate_scores[product_id] = max(candidate_scores.get(product_id, 0.0), 3.5)
            reason_code_by_id.setdefault(product_id, "homepage_category_exploration")
            added_per_category[category_key] = added_per_category.get(category_key, 0) + 1

    def _apply_price_compatibility_scores(
        self,
        scene: str,
        source_profiles: list[dict[str, Any]],
        candidate_scores: dict[str, float],
        candidate_profiles: list[dict[str, Any]],
        reason_code_by_id: dict[str, str],
    ) -> None:
        source_prices = [
            self._to_float(profile.get("price"))
            for profile in source_profiles
            if self._to_float(profile.get("price")) > 0
        ]
        if not source_prices:
            return
        source_price = sum(source_prices) / len(source_prices)
        for profile in candidate_profiles:
            product_id = str(profile.get("product_id", "")).strip()
            candidate_price = self._to_float(profile.get("price"))
            if not product_id or product_id not in candidate_scores or candidate_price <= 0:
                continue
            ratio = candidate_price / source_price
            boost = 0.0
            reason_code = ""
            if scene == "wishlist":
                if 0.8 <= ratio <= 1.2:
                    boost = 0.45
                    reason_code = "wishlist_price_match"
                elif 0.6 <= ratio <= 1.5:
                    boost = 0.18
                    reason_code = "wishlist_similar_alternative"
            elif scene == "cart":
                if 0.02 <= ratio <= 0.4:
                    boost = 0.7
                    reason_code = "cart_accessory_price_fit"
                elif 0.4 < ratio <= 0.8:
                    boost = 0.25
                    reason_code = "cart_bundle_candidate"
            elif scene == "homepage":
                if 0.5 <= ratio <= 1.8:
                    boost = 0.12
                    reason_code = "homepage_recent_interest"
            if boost <= 0:
                continue
            candidate_scores[product_id] = candidate_scores.get(product_id, 0.0) + boost
            reason_code_by_id.setdefault(product_id, reason_code)

    def _diversify_candidates(
        self,
        scene: str,
        sorted_candidate_ids: list[str],
        top_n: int,
    ) -> list[str]:
        if scene != "homepage" or top_n <= 0:
            return sorted_candidate_ids[:top_n]

        profiles = {
            str(profile.get("product_id", "")).strip(): profile
            for profile in self._fetch_product_profiles(sorted_candidate_ids[: max(top_n * 3, top_n)])
        }
        selected: list[str] = []
        deferred: list[str] = []
        category_counts: dict[str, int] = {}
        brand_counts: dict[str, int] = {}
        for product_id in sorted_candidate_ids:
            profile = profiles.get(product_id, {})
            category = self._slugify(str(profile.get("category", ""))) or "unknown"
            brand = self._slugify(str(profile.get("brand", ""))) or "unknown"
            if (
                category_counts.get(category, 0) >= self.max_per_category
                or brand_counts.get(brand, 0) >= self.max_per_brand
            ):
                deferred.append(product_id)
                continue
            selected.append(product_id)
            category_counts[category] = category_counts.get(category, 0) + 1
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
            if len(selected) >= top_n:
                return selected
        for product_id in deferred:
            if product_id not in selected:
                selected.append(product_id)
            if len(selected) >= top_n:
                break
        return selected[:top_n]

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
        raw_context = self._get_raw_user_interest_context(user_id)
        source_id_set = {
            str(pid).strip()
            for pid in source_ids
            if str(pid).strip()
        }
        recent_interest_products = [
            pid
            for pid in raw_context.get("recent_interest_products", [])
            if pid not in source_id_set
        ]
        return {
            "product_scores": dict(raw_context.get("product_scores", {})),
            "category_scores": dict(raw_context.get("category_scores", {})),
            "brand_scores": dict(raw_context.get("brand_scores", {})),
            "recent_interest_products": recent_interest_products,
            "fingerprint": raw_context.get("fingerprint", "none"),
        }

    def _get_raw_user_interest_context(self, user_id: int) -> dict[str, Any]:
        normalized_user_id = int(user_id)
        cached = self._interest_context_cache.get(normalized_user_id)
        if cached is not None:
            return cached

        cache_key = recommendation_cache.build_key(
            scene="interest_state",
            limit=max(int(self.interest_event_limit), 1),
            user_id=normalized_user_id,
        )
        cached_payload = recommendation_cache.get_json(cache_key)
        if isinstance(cached_payload, dict):
            normalized_context = self._normalize_interest_context_payload(cached_payload)
            self._interest_context_cache[normalized_user_id] = normalized_context
            return normalized_context

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
            rows = self._get_connection().execute(query, {"user_id": normalized_user_id}).mappings().all()
        except Exception:
            empty_context = {
                "product_scores": {},
                "category_scores": {},
                "brand_scores": {},
                "recent_interest_products": [],
                "fingerprint": "empty",
            }
            self._interest_context_cache[normalized_user_id] = empty_context
            return empty_context

        if not rows:
            empty_context = {
                "product_scores": {},
                "category_scores": {},
                "brand_scores": {},
                "recent_interest_products": [],
                "fingerprint": "empty",
            }
            self._interest_context_cache[normalized_user_id] = empty_context
            return empty_context

        latest_interacted_at = max(
            (
                value
                for value in (row.get("last_interacted_at") for row in rows)
                if isinstance(value, datetime)
            ),
            default=None,
        )

        product_scores: dict[str, float] = {}
        category_scores: dict[str, float] = {}
        brand_scores: dict[str, float] = {}
        recent_interest_products: list[str] = []
        seen_recent_interest_products: set[str] = set()

        for row in rows:
            product_sys_id = str(row.get("product_sys_id") or "").strip()
            if not product_sys_id:
                continue

            category = str(row.get("category") or "").strip()
            brand = str(row.get("brand") or "").strip()
            view_count = self._to_float(row.get("view_count"))
            add_to_cart_count = self._to_float(row.get("add_to_cart_count"))
            wishlist_count = self._to_float(row.get("wishlist_count"))
            purchase_count = self._to_float(row.get("purchase_count"))
            interaction_score = self._to_float(row.get("interaction_score"))
            last_interacted_at = row.get("last_interacted_at")

            recency_factor = 1.0
            if latest_interacted_at is not None and isinstance(last_interacted_at, datetime):
                age_days = max((latest_interacted_at - last_interacted_at).total_seconds() / 86400.0, 0.0)
                recency_factor = math.exp(-self.decay_lambda * age_days)

            aggregate_signal = (
                (interaction_score * self.implicit_weights.interaction)
                + (purchase_count * self.implicit_weights.purchase)
                + (add_to_cart_count * self.implicit_weights.cart)
                + (wishlist_count * self.implicit_weights.wishlist)
                + (view_count * self.implicit_weights.view)
            )
            interest_score = aggregate_signal * recency_factor

            product_scores[product_sys_id] = product_scores.get(product_sys_id, 0.0) + interest_score
            if category:
                category_scores[category] = category_scores.get(category, 0.0) + interest_score
            if brand:
                brand_scores[brand] = brand_scores.get(brand, 0.0) + interest_score

            if product_sys_id not in seen_recent_interest_products:
                recent_interest_products.append(product_sys_id)
                seen_recent_interest_products.add(product_sys_id)

        fingerprint_parts = [
            *(f"p:{pid}:{round(score, 3)}" for pid, score in sorted(product_scores.items(), key=lambda item: item[1], reverse=True)[:10]),
            *(f"c:{name}:{round(score, 3)}" for name, score in sorted(category_scores.items(), key=lambda item: item[1], reverse=True)[:5]),
            *(f"b:{name}:{round(score, 3)}" for name, score in sorted(brand_scores.items(), key=lambda item: item[1], reverse=True)[:5]),
            *(f"r:{pid}" for pid in recent_interest_products[:10]),
        ]

        context = {
            "product_scores": product_scores,
            "category_scores": category_scores,
            "brand_scores": brand_scores,
            "recent_interest_products": recent_interest_products,
            "fingerprint": self._hash_parts(fingerprint_parts) if fingerprint_parts else "empty",
        }
        self._interest_context_cache[normalized_user_id] = context
        recommendation_cache.set_json(
            cache_key,
            context,
            ttl_seconds=self.interest_context_cache_ttl_seconds,
        )
        return context

    def _normalize_cached_id_list(self, values: Any) -> set[str]:
        if not isinstance(values, list):
            return set()
        return {
            str(value).strip()
            for value in values
            if str(value).strip()
        }

    def _normalize_interest_context_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "product_scores": self._normalize_score_dict(payload.get("product_scores")),
            "category_scores": self._normalize_score_dict(payload.get("category_scores")),
            "brand_scores": self._normalize_score_dict(payload.get("brand_scores")),
            "recent_interest_products": [
                str(pid).strip()
                for pid in payload.get("recent_interest_products", [])
                if str(pid).strip()
            ] if isinstance(payload.get("recent_interest_products"), list) else [],
            "fingerprint": str(payload.get("fingerprint", "empty")).strip() or "empty",
        }

    def _normalize_score_dict(self, payload: Any) -> dict[str, float]:
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, float] = {}
        for key, value in payload.items():
            normalized_key = str(key).strip()
            if not normalized_key:
                continue
            normalized[normalized_key] = self._to_float(value)
        return normalized

    def _hash_ids(self, values: set[str]) -> str:
        normalized_values = sorted(str(value).strip() for value in values if str(value).strip())
        return self._hash_parts(normalized_values)

    def _hash_parts(self, parts: list[str]) -> str:
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
        return digest[:16]

    def _to_float(self, value: Any) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _boost_candidates_from_interest(
        self,
        scene: str,
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
                if scene == "cart" and pid not in candidate_scores:
                    continue
                direct_multiplier = 0.04 if scene == "cart" else 0.28
                candidate_scores[pid] = candidate_scores.get(pid, 0.0) + (base_interest_score * direct_multiplier)
                reason_code_by_id.setdefault(pid, self._scene_interest_reason(scene))

            if base_interest_score <= 0:
                continue

            if scene == "cart":
                continue

            for sim_id in recommend(pid, top_n=6):
                if sim_id in exclude:
                    continue
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0.0) + (base_interest_score * 0.12)
                reason_code_by_id.setdefault(sim_id, self._scene_interest_reason(scene))

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

            category_interest_weight = 0.015 if scene == "cart" else 0.06
            brand_interest_weight = 0.01 if scene == "cart" else 0.035
            affinity_boost = (
                category_scores.get(category, 0.0) * category_interest_weight
                + brand_scores.get(brand, 0.0) * brand_interest_weight
            )
            source_affinity_boost = 0.0
            if category in source_category_counts:
                if scene == "cart":
                    source_affinity_boost += 0.02 * source_category_counts[category]
                else:
                    source_affinity_boost += 0.22 * source_category_counts[category]
            if brand in source_brand_counts:
                source_affinity_boost += 0.08 * source_brand_counts[brand]
            affinity_boost += source_affinity_boost
            candidate_scores[product_id] = candidate_scores.get(product_id, 0.0) + affinity_boost
            if affinity_boost > 0 and product_id not in reason_code_by_id:
                reason_code_by_id[product_id] = self._scene_affinity_reason(
                    scene,
                    same_category=source_affinity_boost > 0 and category in source_category_counts,
                    same_brand=source_affinity_boost > 0 and brand in source_brand_counts,
                )

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

