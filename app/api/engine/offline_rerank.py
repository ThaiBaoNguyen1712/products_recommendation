import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from app.api.engine.content_based import get_product_profiles, recommend
from app.api.engine.index_store import (
    ACCESSORY_RULES_PATH,
    COMPATIBILITY_RULES_PATH,
    OFFLINE_RERANK_SCORES_PATH,
    PRODUCTS_PATH,
)
from app.api.engine.llm_personalization import OpenRouterPersonalizationReranker

SCENES = ("homepage", "wishlist", "cart")


def _load_products() -> list[dict[str, Any]]:
    if not PRODUCTS_PATH.exists():
        return []
    try:
        payload = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
    except Exception:
        pass
    return []


def _load_accessory_rules() -> dict[str, Any]:
    if not ACCESSORY_RULES_PATH.exists():
        return {}
    try:
        payload = json.loads(ACCESSORY_RULES_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_compatibility_rules() -> dict[str, dict[str, Any]]:
    if not COMPATIBILITY_RULES_PATH.exists():
        return {}
    try:
        payload = json.loads(COMPATIBILITY_RULES_PATH.read_text(encoding="utf-8"))
        raw_rules = payload.get("rules", payload) if isinstance(payload, dict) else {}
        if not isinstance(raw_rules, dict):
            return {}
        rules: dict[str, dict[str, Any]] = {}
        for source, rule in raw_rules.items():
            source_slug = _slugify(source)
            if not source_slug or not isinstance(rule, dict):
                continue
            rules[source_slug] = {
                "target_categories": _normalize_patterns(rule.get("target_categories")),
                "include_any": _normalize_patterns(rule.get("include_any")),
                "exclude_any": _normalize_patterns(rule.get("exclude_any")),
            }
        return rules
    except Exception:
        return {}


def _normalize_patterns(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    patterns: list[str] = []
    for value in values:
        pattern = _slugify(value)
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    return patterns
    try:
        payload = json.loads(ACCESSORY_RULES_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u0111", "d").replace("\u0110", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _price(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _write_scores(payload: dict[str, Any]) -> None:
    OFFLINE_RERANK_SCORES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_product_map(products: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    product_map: dict[str, dict[str, Any]] = {}
    for product in products:
        product_id = str(product.get("product_sys_id", "")).strip()
        if product_id:
            product_map[product_id] = product
    return product_map


def _cart_candidate_ids(
    source_id: str,
    product_map: dict[str, dict[str, Any]],
    accessory_rules: dict[str, Any],
    compatibility_rules: dict[str, dict[str, Any]],
    limit: int,
) -> list[str]:
    source = product_map.get(source_id, {})
    source_category = _slugify(source.get("category", ""))
    targets = {
        _slugify(target)
        for target in accessory_rules.get(source_category, [])
        if _slugify(target)
    }
    if not targets:
        return recommend(source_id, top_n=limit)

    source_price = _price(source.get("price"))
    scored: list[tuple[float, str]] = []
    for product_id, product in product_map.items():
        if product_id == source_id:
            continue
        if int(_price(product.get("stock"))) <= 0:
            continue
        if not _is_cart_compatible_candidate(source_category, product, compatibility_rules):
            continue
        category = _slugify(product.get("category", ""))
        if category not in targets and not any(target in category for target in targets):
            continue
        ratio = (_price(product.get("price")) / source_price) if source_price > 0 else 1.0
        price_fit = 0.4 if 0.02 <= ratio <= 0.4 else 0.15 if 0.4 < ratio <= 0.8 else 0.0
        scored.append((1.0 + price_fit, product_id))

    ranked = [product_id for _score, product_id in sorted(scored, key=lambda item: item[0], reverse=True)]
    fallback = [pid for pid in recommend(source_id, top_n=limit) if pid not in ranked]
    return (ranked + fallback)[:limit]


def _is_cart_compatible_candidate(
    source_category: str,
    product: dict[str, Any],
    compatibility_rules: dict[str, dict[str, Any]],
) -> bool:
    rule = compatibility_rules.get(source_category)
    if not rule:
        return True
    category = _slugify(product.get("category", ""))
    searchable_text = _slugify(
        " ".join(
            [
                str(product.get("name", "")),
                str(product.get("brand", "")),
                str(product.get("category", "")),
            ]
        )
    )
    if any(pattern and pattern in searchable_text for pattern in rule.get("exclude_any", [])):
        return False

    target_categories = rule.get("target_categories", [])
    if target_categories and category not in target_categories:
        return False

    include_any = rule.get("include_any", [])
    if include_any and not any(pattern and pattern in searchable_text for pattern in include_any):
        return False

    return True


def _candidate_ids_for_scene(
    scene: str,
    source_id: str,
    product_map: dict[str, dict[str, Any]],
    accessory_rules: dict[str, Any],
    compatibility_rules: dict[str, dict[str, Any]],
    limit: int,
) -> list[str]:
    if scene == "cart":
        return _cart_candidate_ids(source_id, product_map, accessory_rules, compatibility_rules, limit)
    return recommend(source_id, top_n=limit)


def _score_candidate(
    scene: str,
    rank: int,
    source_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
) -> dict[str, Any]:
    same_category = (
        str(candidate_profile.get("category", "")).strip().lower()
        == str(source_profile.get("category", "")).strip().lower()
    )
    same_brand = (
        str(candidate_profile.get("brand", "")).strip().lower()
        == str(source_profile.get("brand", "")).strip().lower()
    )
    source_price = _price(source_profile.get("price"))
    candidate_price = _price(candidate_profile.get("price"))
    ratio = (candidate_price / source_price) if source_price > 0 and candidate_price > 0 else 0.0
    llm_rank_score = round(max(0.0, 0.25 - ((rank - 1) * 0.04)), 4)
    category_fit = 0.08 if same_category else 0.0
    brand_fit = 0.04 if same_brand else 0.0
    stock_score = 0.03 if int(candidate_profile.get("stock", 0) or 0) > 0 else 0.0
    price_fit = 0.0
    reason_code = "homepage_category_exploration"
    if scene == "wishlist":
        price_fit = 0.08 if 0.8 <= ratio <= 1.2 else 0.03 if 0.6 <= ratio <= 1.5 else 0.0
        reason_code = "wishlist_similar_alternative" if same_category else "wishlist_upgrade_option"
        if price_fit >= 0.08:
            reason_code = "wishlist_price_match"
    elif scene == "cart":
        price_fit = 0.12 if 0.02 <= ratio <= 0.4 else 0.04 if 0.4 < ratio <= 0.8 else 0.0
        category_fit = 0.0
        brand_fit = 0.0
        reason_code = "cart_accessory_match" if price_fit > 0 else "cart_bundle_candidate"
    elif same_brand:
        reason_code = "homepage_recent_interest"

    final_score = round(llm_rank_score + category_fit + brand_fit + stock_score + price_fit, 4)
    return {
        "score": final_score,
        "reason_code": reason_code,
        "components": {
            "llm_rank_score": llm_rank_score,
            "category_fit": category_fit,
            "brand_fit": brand_fit,
            "stock_score": stock_score,
            "price_fit": price_fit,
        },
    }


def refresh_offline_rerank_scores(
    db_engine: Engine,
    *,
    product_sys_id: str | None = None,
    trigger: str = "manual",
) -> dict[str, Any]:
    del db_engine

    enabled = os.getenv("ENABLE_OFFLINE_LLM_REFRESH", "false").strip().lower() in {"1", "true", "yes"}
    reranker = OpenRouterPersonalizationReranker()
    if not enabled or not reranker.enabled:
        payload = {
            "meta": {
                "status": "disabled",
                "trigger": trigger,
                "reason": "offline_llm_refresh_disabled",
            },
            "homepage": {},
            "wishlist": {},
            "cart": {},
        }
        _write_scores(payload)
        return payload["meta"]

    products = _load_products()
    if not products:
        payload = {
            "meta": {
                "status": "empty",
                "trigger": trigger,
                "processed_sources": 0,
            },
            "homepage": {},
            "wishlist": {},
            "cart": {},
        }
        _write_scores(payload)
        return payload["meta"]

    product_map = _build_product_map(products)
    accessory_rules = _load_accessory_rules()
    compatibility_rules = _load_compatibility_rules()
    normalized_target_id = str(product_sys_id).strip() if product_sys_id else None
    source_ids = [
        str(product.get("product_sys_id", "")).strip()
        for product in products
        if str(product.get("product_sys_id", "")).strip()
    ]

    if normalized_target_id:
        source_ids = [product_id for product_id in source_ids if product_id == normalized_target_id]
    else:
        source_limit = int(os.getenv("OFFLINE_LLM_SOURCE_LIMIT", "80"))
        source_ids = source_ids[:source_limit]

    payload: dict[str, Any] = {
        "meta": {
            "status": "ready",
            "trigger": trigger,
            "processed_sources": 0,
            "requested_product_sys_id": normalized_target_id,
            "model": reranker.model,
        }
    }
    for scene in SCENES:
        payload[scene] = {}

    processed_sources = 0
    for source_id in source_ids:
        source_profiles = get_product_profiles([source_id])
        if not source_profiles:
            continue

        wrote_scene = False
        source_profile = source_profiles[0]
        for scene in SCENES:
            candidate_ids = _candidate_ids_for_scene(
                scene=scene,
                source_id=source_id,
                product_map=product_map,
                accessory_rules=accessory_rules,
                compatibility_rules=compatibility_rules,
                limit=8,
            )
            if not candidate_ids:
                continue
            candidate_profiles = get_product_profiles(candidate_ids)
            if not candidate_profiles:
                continue
            candidate_profile_map = {
                str(profile["product_id"]).strip(): profile
                for profile in candidate_profiles
            }
            ranked_ids = reranker.rerank(
                scene=scene,
                source_products=source_profiles,
                candidate_products=candidate_profiles,
                top_n=min(5, len(candidate_ids)),
            )
            if not ranked_ids:
                continue

            scored_candidates: dict[str, Any] = {}
            for rank, candidate_id in enumerate(ranked_ids, start=1):
                normalized_candidate_id = str(candidate_id).strip()
                candidate_profile = candidate_profile_map.get(normalized_candidate_id, {})
                scored_candidates[normalized_candidate_id] = _score_candidate(
                    scene=scene,
                    rank=rank,
                    source_profile=source_profile,
                    candidate_profile=candidate_profile,
                )
            payload[scene][source_id] = scored_candidates
            wrote_scene = True

        if wrote_scene:
            processed_sources += 1

    payload["meta"]["processed_sources"] = processed_sources
    _write_scores(payload)
    return payload["meta"]
