import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from app.api.engine.content_based import get_product_profiles, recommend
from app.api.engine.index_store import OFFLINE_RERANK_SCORES_PATH, PRODUCTS_PATH
from app.api.engine.llm_personalization import GroqPersonalizationReranker

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


def _write_scores(payload: dict[str, Any]) -> None:
    OFFLINE_RERANK_SCORES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def refresh_offline_rerank_scores(
    db_engine: Engine,
    *,
    product_sys_id: str | None = None,
    trigger: str = "manual",
) -> dict[str, Any]:
    del db_engine

    enabled = os.getenv("ENABLE_OFFLINE_LLM_REFRESH", "false").strip().lower() in {"1", "true", "yes"}
    reranker = GroqPersonalizationReranker()
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
        candidate_ids = recommend(source_id, top_n=8)
        if not candidate_ids:
            continue

        source_profiles = get_product_profiles([source_id])
        candidate_profiles = get_product_profiles(candidate_ids)
        if not source_profiles or not candidate_profiles:
            continue

        candidate_profile_map = {
            str(profile["product_id"]).strip(): profile
            for profile in candidate_profiles
        }
        ranked_ids = reranker.rerank(
            scene="homepage",
            source_products=source_profiles,
            candidate_products=candidate_profiles,
            top_n=min(5, len(candidate_ids)),
        )
        if not ranked_ids:
            continue

        source_profile = source_profiles[0]
        scored_candidates: dict[str, Any] = {}
        for rank, candidate_id in enumerate(ranked_ids, start=1):
            normalized_candidate_id = str(candidate_id).strip()
            candidate_profile = candidate_profile_map.get(normalized_candidate_id, {})
            same_category = (
                str(candidate_profile.get("category", "")).strip().lower()
                == str(source_profile.get("category", "")).strip().lower()
            )
            same_brand = (
                str(candidate_profile.get("brand", "")).strip().lower()
                == str(source_profile.get("brand", "")).strip().lower()
            )
            llm_rank_score = round(max(0.0, 0.25 - ((rank - 1) * 0.04)), 4)
            category_fit = 0.08 if same_category else 0.0
            brand_fit = 0.04 if same_brand else 0.0
            stock_score = 0.03 if int(candidate_profile.get("stock", 0) or 0) > 0 else 0.0
            final_score = round(llm_rank_score + category_fit + brand_fit + stock_score, 4)
            reason_code = "interest_match"
            if same_category and same_brand:
                reason_code = "brand_category_affinity"
            elif same_category:
                reason_code = "category_affinity"
            elif same_brand:
                reason_code = "brand_affinity"

            scored_candidates[normalized_candidate_id] = {
                "score": final_score,
                "reason_code": reason_code,
                "components": {
                    "llm_rank_score": llm_rank_score,
                    "category_fit": category_fit,
                    "brand_fit": brand_fit,
                    "stock_score": stock_score,
                },
            }

        for scene in SCENES:
            payload[scene][source_id] = scored_candidates
        processed_sources += 1

    payload["meta"]["processed_sources"] = processed_sources
    _write_scores(payload)
    return payload["meta"]
