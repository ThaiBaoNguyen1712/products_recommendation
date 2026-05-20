from time import perf_counter
from typing import Any

from fastapi import FastAPI, Response
from fastapi.concurrency import asynccontextmanager

from app.api.engine.SceneRecommendationFilter import SceneRecommendationFilter
from app.api.engine.content_based import load_all_data, recommend
from app.api.engine.recommendation_tracking import log_recommendation_impressions
from db.mssql import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing ML data...")
    load_all_data()
    yield
    print("Shutting down application...")


app = FastAPI(lifespan=lifespan)


def _set_response_headers(
    response: Response,
    response_time_ms: float,
    llm_latency_ms: float,
    llm_status: str,
) -> None:
    response.headers["X-Response-Time-Ms"] = f"{response_time_ms:.2f}"
    response.headers["X-LLM-Latency-Ms"] = f"{llm_latency_ms:.2f}"
    response.headers["X-LLM-Status"] = llm_status


def _build_payload(
    scene: str,
    recommendations: list[str],
    user_id: int | None = None,
    product_sys_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "scene": scene,
        "recommendations": recommendations,
    }
    if user_id is not None:
        payload["user_id"] = user_id
    if product_sys_id is not None:
        payload["product_sys_id"] = product_sys_id
    return payload


def _log_scene_impressions(
    user_id: int,
    scene: str,
    recommendations: list[str],
    response_time_ms: float,
    recommender: SceneRecommendationFilter,
) -> None:
    log_recommendation_impressions(
        db_engine=engine,
        user_id=user_id,
        scene=scene,
        recommendations=recommendations,
        response_time_ms=response_time_ms,
        llm_latency_ms=recommender.last_llm_latency_ms,
        source_ids=recommender.last_source_ids,
    )


def _resolve_limit(limit: int | None, top_n: int | None, default: int) -> int:
    if limit is not None:
        return limit
    if top_n is not None:
        return top_n
    return default


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/recommendations/similar/{product_sys_id}")
async def get_similar_recommendations(product_sys_id: str, response: Response, limit: int = 15):
    started_at = perf_counter()
    recommendations = recommend(product_sys_id=product_sys_id, top_n=limit)
    response_time_ms = (perf_counter() - started_at) * 1000
    _set_response_headers(
        response=response,
        response_time_ms=response_time_ms,
        llm_latency_ms=0.0,
        llm_status="not_used",
    )
    return _build_payload(
        scene="similar",
        product_sys_id=product_sys_id,
        recommendations=recommendations,
    )


@app.get("/api/v1/recommendations/users/{user_id}/detail/{product_sys_id}")
async def get_detail_recommendations(user_id: int, product_sys_id: str, response: Response, limit: int = 12):
    started_at = perf_counter()
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
    )
    _log_scene_impressions(
        user_id=user_id,
        scene="detail",
        recommendations=recommendations,
        response_time_ms=response_time_ms,
        recommender=recommender,
    )
    return _build_payload(
        scene="detail",
        user_id=user_id,
        product_sys_id=product_sys_id,
        recommendations=recommendations,
    )


@app.get("/api/v1/recommendations/users/{user_id}/wishlist")
async def get_wishlist_recommendations(user_id: int, response: Response, limit: int = 15):
    started_at = perf_counter()
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
    )
    _log_scene_impressions(
        user_id=user_id,
        scene="wishlist",
        recommendations=recommendations,
        response_time_ms=response_time_ms,
        recommender=recommender,
    )
    return _build_payload(
        scene="wishlist",
        user_id=user_id,
        recommendations=recommendations,
    )


@app.get("/api/v1/recommendations/users/{user_id}/cart")
async def get_cart_recommendations(user_id: int, response: Response, limit: int = 15):
    started_at = perf_counter()
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
    )
    _log_scene_impressions(
        user_id=user_id,
        scene="cart",
        recommendations=recommendations,
        response_time_ms=response_time_ms,
        recommender=recommender,
    )
    return _build_payload(
        scene="cart",
        user_id=user_id,
        recommendations=recommendations,
    )


@app.get("/api/v1/recommendations/users/{user_id}/homepage")
async def get_homepage_recommendations(user_id: int, response: Response, limit: int = 15):
    started_at = perf_counter()
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
    )
    _log_scene_impressions(
        user_id=user_id,
        scene="homepage",
        recommendations=recommendations,
        response_time_ms=response_time_ms,
        recommender=recommender,
    )
    return _build_payload(
        scene="homepage",
        user_id=user_id,
        recommendations=recommendations,
    )


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
