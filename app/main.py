from time import perf_counter
from typing import Any

from fastapi import FastAPI, Response
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel, Field

from app.api.engine.accessory_rules_refresh import refresh_accessory_rules
from app.api.engine.SceneRecommendationFilter import SceneRecommendationFilter
from app.api.engine.content_based import load_all_data, recommend
from app.api.engine.index_store import build_file_index, ensure_file_index, read_index_status, sync_product_index
from app.api.engine.offline_rerank import refresh_offline_rerank_scores
from app.api.engine.recommendation_cache import get_scene_cache_ttl, recommendation_cache
from db.mssql import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing ML data...")
    ensure_file_index(engine)
    load_all_data()
    yield
    print("Shutting down application...")


app = FastAPI(lifespan=lifespan)


class SyncProductRequest(BaseModel):
    product_sys_id: str = Field(min_length=1)
    action: str = Field(default="upsert", pattern="^(upsert|delete)$")


def _set_response_headers(
    response: Response,
    response_time_ms: float,
    llm_latency_ms: float,
    llm_status: str,
    cache_status: str = "MISS",
) -> None:
    response.headers["X-Response-Time-Ms"] = f"{response_time_ms:.2f}"
    response.headers["X-LLM-Latency-Ms"] = f"{llm_latency_ms:.2f}"
    response.headers["X-LLM-Status"] = llm_status
    response.headers["X-Cache"] = cache_status


def _build_payload(
    scene: str,
    recommendation_items: list[dict[str, Any]],
    recommendations: list[str] | None = None,
    user_id: int | None = None,
    product_sys_id: str | None = None,
) -> dict[str, Any]:
    normalized_items = [
        {
            "product_sys_id": str(item.get("product_sys_id", "")).strip(),
            "score": round(float(item.get("score", 0.0) or 0.0), 4),
            "reason_code": str(item.get("reason_code", "ranking_score")).strip() or "ranking_score",
        }
        for item in recommendation_items
        if str(item.get("product_sys_id", "")).strip()
    ]
    normalized_recommendations = (
        [str(product_id).strip() for product_id in recommendations if str(product_id).strip()]
        if recommendations is not None
        else [item["product_sys_id"] for item in normalized_items]
    )
    payload: dict[str, Any] = {
        "scene": scene,
        "recommendation_items": normalized_items,
        "recommendations": normalized_recommendations,
    }
    if user_id is not None:
        payload["user_id"] = user_id
    if product_sys_id is not None:
        payload["product_sys_id"] = product_sys_id
    return payload


def _resolve_limit(limit: int | None, top_n: int | None, default: int) -> int:
    if limit is not None:
        return limit
    if top_n is not None:
        return top_n
    return default


def _build_cache_entry(
    *,
    scene: str,
    recommendation_items: list[dict[str, Any]],
    recommendations: list[str] | None,
    llm_latency_ms: float,
    llm_status: str,
    user_id: int | None = None,
    product_sys_id: str | None = None,
    source_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "payload": _build_payload(
            scene=scene,
            recommendation_items=recommendation_items,
            recommendations=recommendations,
            user_id=user_id,
            product_sys_id=product_sys_id,
        ),
        "meta": {
            "llm_latency_ms": round(float(llm_latency_ms), 2),
            "llm_status": llm_status,
            "source_ids": [str(pid).strip() for pid in (source_ids or []) if str(pid).strip()],
        },
    }


def _read_cached_entry(
    *,
    scene: str,
    response: Response,
    limit: int,
    user_id: int | None = None,
    product_sys_id: str | None = None,
) -> dict[str, Any] | None:
    cache_key = recommendation_cache.build_key(
        scene=scene,
        limit=limit,
        user_id=user_id,
        product_sys_id=product_sys_id,
    )
    cached_entry = recommendation_cache.get_json(cache_key)
    if not cached_entry:
        return None

    payload = dict(cached_entry.get("payload", {}))
    recommendation_items = payload.get("recommendation_items")
    recommendations = payload.get("recommendations", [])
    if not isinstance(recommendation_items, list):
        recommendation_items = _build_ranked_items_from_ids(
            recommendations=[str(product_id).strip() for product_id in recommendations if str(product_id).strip()],
            reason_code="ranking_score",
        )
        payload = _build_payload(
            scene=scene,
            recommendation_items=recommendation_items,
            recommendations=recommendations,
            user_id=payload.get("user_id"),
            product_sys_id=payload.get("product_sys_id"),
        )
        cached_entry["payload"] = payload

    metadata = dict(cached_entry.get("meta", {}))
    _set_response_headers(
        response=response,
        response_time_ms=0.0,
        llm_latency_ms=float(metadata.get("llm_latency_ms", 0.0) or 0.0),
        llm_status=f"cached:{metadata.get('llm_status', 'not_used')}",
        cache_status="HIT",
    )
    return cached_entry


def _write_cached_entry(
    *,
    scene: str,
    limit: int,
    entry: dict[str, Any],
    user_id: int | None = None,
    product_sys_id: str | None = None,
) -> None:
    cache_key = recommendation_cache.build_key(
        scene=scene,
        limit=limit,
        user_id=user_id,
        product_sys_id=product_sys_id,
    )
    recommendation_cache.set_json(cache_key, entry, ttl_seconds=get_scene_cache_ttl(scene))


def _build_ranked_items_from_ids(recommendations: list[str], reason_code: str) -> list[dict[str, Any]]:
    return [
        {
            "product_sys_id": str(product_id).strip(),
            "score": round(float(max(len(recommendations) - idx, 1)), 4),
            "reason_code": reason_code,
        }
        for idx, product_id in enumerate(recommendations)
    ]


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/admin/index-status")
def get_index_status():
    return read_index_status()


@app.post("/admin/rebuild-index")
def rebuild_index():
    artifact_status = build_file_index(engine, trigger="admin_rebuild")
    accessory_rules_status = refresh_accessory_rules(trigger="admin_rebuild")
    load_all_data()
    offline_refresh = refresh_offline_rerank_scores(engine, trigger="admin_rebuild")
    cache_version = recommendation_cache.invalidate_all()
    return {
        "message": "File-based recommendation index rebuilt successfully.",
        "index": artifact_status,
        "accessory_rules": accessory_rules_status,
        "offline_rerank": offline_refresh,
        "cache_version": cache_version,
    }


@app.post("/admin/sync-product")
def sync_product(request: SyncProductRequest):
    artifact_status = sync_product_index(
        db_engine=engine,
        product_sys_id=request.product_sys_id,
        action=request.action,
    )
    load_all_data()
    cache_version = recommendation_cache.invalidate_all()
    return {
        "message": "Product sync completed with incremental artifact update. Offline rerank was not executed.",
        "product_sys_id": request.product_sys_id.strip(),
        "action": request.action,
        "index": artifact_status,
        "offline_rerank": {
            "status": "skipped",
            "trigger": "sync_product",
            "reason": "offline_rerank_runs_only_on_rebuild",
        },
        "cache_version": cache_version,
    }


@app.get("/api/v1/recommendations/similar/{product_sys_id}")
async def get_similar_recommendations(product_sys_id: str, response: Response, limit: int = 15):
    started_at = perf_counter()
    cached_entry = _read_cached_entry(
        scene="similar",
        product_sys_id=product_sys_id,
        limit=limit,
        response=response,
    )
    if cached_entry is not None:
        return cached_entry["payload"]

    recommendations = recommend(product_sys_id=product_sys_id, top_n=limit)
    response_time_ms = (perf_counter() - started_at) * 1000
    _set_response_headers(
        response=response,
        response_time_ms=response_time_ms,
        llm_latency_ms=0.0,
        llm_status="not_used",
        cache_status="MISS",
    )
    cache_entry = _build_cache_entry(
        scene="similar",
        recommendation_items=_build_ranked_items_from_ids(recommendations, "content_similarity"),
        recommendations=recommendations,
        llm_latency_ms=0.0,
        llm_status="not_used",
        product_sys_id=product_sys_id,
    )
    _write_cached_entry(scene="similar", limit=limit, product_sys_id=product_sys_id, entry=cache_entry)
    return cache_entry["payload"]


@app.get("/api/v1/recommendations/users/{user_id}/detail/{product_sys_id}")
async def get_detail_recommendations(user_id: int, product_sys_id: str, response: Response, limit: int = 12):
    started_at = perf_counter()
    cached_entry = _read_cached_entry(
        scene="detail",
        user_id=user_id,
        product_sys_id=product_sys_id,
        limit=limit,
        response=response,
    )
    if cached_entry is not None:
        return cached_entry["payload"]

    recommender = SceneRecommendationFilter()
    recommendations = recommender.get_recommendations_detail(
        user_id=user_id,
        product_sys_id=product_sys_id,
        top_n=limit,
    )
    response_time_ms = (perf_counter() - started_at) * 1000
    _set_response_headers(
        response=response,
        response_time_ms=response_time_ms,
        llm_latency_ms=recommender.last_llm_latency_ms,
        llm_status=recommender.last_llm_status,
        cache_status="MISS",
    )
    cache_entry = _build_cache_entry(
        scene="detail",
        recommendation_items=recommender.last_recommendation_items,
        recommendations=recommendations,
        llm_latency_ms=recommender.last_llm_latency_ms,
        llm_status=recommender.last_llm_status,
        user_id=user_id,
        product_sys_id=product_sys_id,
        source_ids=recommender.last_source_ids,
    )
    _write_cached_entry(
        scene="detail",
        limit=limit,
        user_id=user_id,
        product_sys_id=product_sys_id,
        entry=cache_entry,
    )
    return cache_entry["payload"]


@app.get("/api/v1/recommendations/users/{user_id}/wishlist")
async def get_wishlist_recommendations(user_id: int, response: Response, limit: int = 15):
    started_at = perf_counter()
    cached_entry = _read_cached_entry(
        scene="wishlist",
        user_id=user_id,
        limit=limit,
        response=response,
    )
    if cached_entry is not None:
        return cached_entry["payload"]

    recommender = SceneRecommendationFilter()
    recommendations = recommender.get_recommendations_wishlist(
        user_id=user_id,
        top_n=limit,
    )
    response_time_ms = (perf_counter() - started_at) * 1000
    _set_response_headers(
        response=response,
        response_time_ms=response_time_ms,
        llm_latency_ms=recommender.last_llm_latency_ms,
        llm_status=recommender.last_llm_status,
        cache_status="MISS",
    )
    cache_entry = _build_cache_entry(
        scene="wishlist",
        recommendation_items=recommender.last_recommendation_items,
        recommendations=recommendations,
        llm_latency_ms=recommender.last_llm_latency_ms,
        llm_status=recommender.last_llm_status,
        user_id=user_id,
        source_ids=recommender.last_source_ids,
    )
    _write_cached_entry(scene="wishlist", limit=limit, user_id=user_id, entry=cache_entry)
    return cache_entry["payload"]


@app.get("/api/v1/recommendations/users/{user_id}/cart")
async def get_cart_recommendations(user_id: int, response: Response, limit: int = 15):
    started_at = perf_counter()
    cached_entry = _read_cached_entry(
        scene="cart",
        user_id=user_id,
        limit=limit,
        response=response,
    )
    if cached_entry is not None:
        return cached_entry["payload"]

    recommender = SceneRecommendationFilter()
    recommendations = recommender.get_recommendations_cart(
        user_id=user_id,
        top_n=limit,
    )
    response_time_ms = (perf_counter() - started_at) * 1000
    _set_response_headers(
        response=response,
        response_time_ms=response_time_ms,
        llm_latency_ms=recommender.last_llm_latency_ms,
        llm_status=recommender.last_llm_status,
        cache_status="MISS",
    )
    cache_entry = _build_cache_entry(
        scene="cart",
        recommendation_items=recommender.last_recommendation_items,
        recommendations=recommendations,
        llm_latency_ms=recommender.last_llm_latency_ms,
        llm_status=recommender.last_llm_status,
        user_id=user_id,
        source_ids=recommender.last_source_ids,
    )
    _write_cached_entry(scene="cart", limit=limit, user_id=user_id, entry=cache_entry)
    return cache_entry["payload"]


@app.get("/api/v1/recommendations/users/{user_id}/homepage")
async def get_homepage_recommendations(user_id: int, response: Response, limit: int = 15):
    started_at = perf_counter()
    cached_entry = _read_cached_entry(
        scene="homepage",
        user_id=user_id,
        limit=limit,
        response=response,
    )
    if cached_entry is not None:
        return cached_entry["payload"]

    recommender = SceneRecommendationFilter()
    recommendations = recommender.get_recommendations_homepage(
        user_id=user_id,
        top_n=limit,
    )
    response_time_ms = (perf_counter() - started_at) * 1000
    _set_response_headers(
        response=response,
        response_time_ms=response_time_ms,
        llm_latency_ms=recommender.last_llm_latency_ms,
        llm_status=recommender.last_llm_status,
        cache_status="MISS",
    )
    cache_entry = _build_cache_entry(
        scene="homepage",
        recommendation_items=recommender.last_recommendation_items,
        recommendations=recommendations,
        llm_latency_ms=recommender.last_llm_latency_ms,
        llm_status=recommender.last_llm_status,
        user_id=user_id,
        source_ids=recommender.last_source_ids,
    )
    _write_cached_entry(scene="homepage", limit=limit, user_id=user_id, entry=cache_entry)
    return cache_entry["payload"]


@app.get("/content_based_filter/{product_sys_id}", deprecated=True, include_in_schema=False)
async def get_similar_recommendations_legacy(product_sys_id: str, response: Response, top_n: int = 15):
    return await get_similar_recommendations(
        product_sys_id=product_sys_id,
        response=response,
        limit=_resolve_limit(limit=None, top_n=top_n, default=15),
    )


@app.get("/api/v1/recommendation/{user_id}/detail/{product_sys_id}", deprecated=True, include_in_schema=False)
async def get_detail_recommendations_legacy(user_id: int, product_sys_id: str, response: Response, top_n: int = 12):
    return await get_detail_recommendations(
        user_id=user_id,
        product_sys_id=product_sys_id,
        response=response,
        limit=_resolve_limit(limit=None, top_n=top_n, default=12),
    )


@app.get("/api/v1/recommendation/wishlist/{user_id}", deprecated=True, include_in_schema=False)
async def get_wishlist_recommendations_legacy_v1(user_id: int, response: Response, top_n: int = 15):
    return await get_wishlist_recommendations(
        user_id=user_id,
        response=response,
        limit=_resolve_limit(limit=None, top_n=top_n, default=15),
    )


@app.get("/api/v1/recommendation/{user_id}/wishlist/_", deprecated=True, include_in_schema=False)
async def get_wishlist_recommendations_legacy_path(user_id: int, response: Response, top_n: int = 15):
    return await get_wishlist_recommendations(
        user_id=user_id,
        response=response,
        limit=_resolve_limit(limit=None, top_n=top_n, default=15),
    )


@app.get("/api/v1/recommendation/cart/{user_id}", deprecated=True, include_in_schema=False)
async def get_cart_recommendations_legacy_v1(user_id: int, response: Response, top_n: int = 15):
    return await get_cart_recommendations(
        user_id=user_id,
        response=response,
        limit=_resolve_limit(limit=None, top_n=top_n, default=15),
    )


@app.get("/api/v1/recommendation/{user_id}/cart/_", deprecated=True, include_in_schema=False)
async def get_cart_recommendations_legacy_path(user_id: int, response: Response, top_n: int = 15):
    return await get_cart_recommendations(
        user_id=user_id,
        response=response,
        limit=_resolve_limit(limit=None, top_n=top_n, default=15),
    )


@app.get("/api/v1/recommendation/homepage/{user_id}", deprecated=True, include_in_schema=False)
async def get_homepage_recommendations_legacy_v1(user_id: int, response: Response, top_n: int = 15):
    return await get_homepage_recommendations(
        user_id=user_id,
        response=response,
        limit=_resolve_limit(limit=None, top_n=top_n, default=15),
    )
