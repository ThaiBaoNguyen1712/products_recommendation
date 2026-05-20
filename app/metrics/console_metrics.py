import argparse
import math
import statistics
import time
from dataclasses import dataclass

import pandas as pd
from fastapi.testclient import TestClient

from app.api.engine.SceneRecommendationFilter import SceneRecommendationFilter
from app.api.engine.content_based import load_all_data, recommend
from app.main import app
from db.mssql import engine


POSITIVE_EVENT_TYPES = {"view_detail", "search_click", "recommendation_click", "wishlist_add", "add_cart", "purchase"}


@dataclass
class BenchmarkObservation:
    endpoint: str
    response_time_ms: float
    llm_latency_ms: float
    llm_status: str
    recommendation_count: int
    status_code: int


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _parse_ms(raw_value: str | None) -> float:
    if not raw_value:
        return 0.0
    try:
        return round(float(raw_value), 2)
    except (TypeError, ValueError):
        return 0.0


def resolve_user_id(user_id: int | None, user_email: str | None) -> int | None:
    if user_id:
        return int(user_id)
    if not user_email:
        return None

    query = "SELECT user_id FROM [User] WHERE email = ?"
    df = pd.read_sql(query, engine, params=(user_email,))
    if df.empty:
        return None
    return int(df.iloc[0]["user_id"])


def _seed_homepage_history() -> None:
    recommender = SceneRecommendationFilter()
    query = """
    SELECT TOP 50
        upe.user_id,
        p.product_sys_id,
        upe.created_at
    FROM UserProductEvent upe
    JOIN Product p ON upe.product_id = p.product_id
    WHERE upe.event_type = 'view_detail'
    ORDER BY upe.created_at DESC
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return

    for user_id, group in df.groupby("user_id"):
        key = f"user:{int(user_id)}:latest_watched"
        try:
            recommender.redis_client.delete(key)
            for product_sys_id in group["product_sys_id"].astype(str).tolist():
                recommender.redis_client.rpush(key, product_sys_id)
            recommender.redis_client.ltrim(key, -10, -1)
        except Exception:
            return


def _pick_sample_identifiers(target_user_id: int | None = None) -> dict[str, list[str | int]]:
    if target_user_id:
        detail_query = """
        SELECT TOP 5 p.product_sys_id
        FROM UserProductEvent upe
        JOIN Product p ON upe.product_id = p.product_id
        WHERE upe.event_type = 'view_detail'
          AND upe.user_id = ?
        GROUP BY p.product_sys_id
        ORDER BY COUNT(*) DESC, MAX(upe.created_at) DESC
        """
        detail_df = pd.read_sql(detail_query, engine, params=(target_user_id,))
        return {
            "detail": detail_df["product_sys_id"].tolist() if not detail_df.empty else [],
            "wishlist": [target_user_id],
            "cart": [target_user_id],
            "homepage": [target_user_id],
        }

    queries = {
        "detail": """
        SELECT TOP 5 p.product_sys_id
        FROM UserProductEvent upe
        JOIN Product p ON upe.product_id = p.product_id
        WHERE upe.event_type = 'view_detail'
        GROUP BY p.product_sys_id
        ORDER BY COUNT(*) DESC, MAX(upe.created_at) DESC
        """,
        "wishlist": """
        SELECT TOP 5 user_id
        FROM Wishlist
        GROUP BY user_id
        ORDER BY COUNT(*) DESC, MAX(added_date) DESC
        """,
        "cart": """
        SELECT TOP 5 c.user_id
        FROM CartItem ci
        JOIN Cart c ON ci.cart_id = c.cart_id
        GROUP BY c.user_id
        ORDER BY COUNT(*) DESC, MAX(ci.created_at) DESC
        """,
        "homepage": """
        SELECT TOP 5 user_id
        FROM UserProductEvent
        WHERE event_type = 'view_detail'
        GROUP BY user_id
        ORDER BY COUNT(*) DESC, MAX(created_at) DESC
        """,
    }
    result: dict[str, list[str | int]] = {}
    for key, query in queries.items():
        df = pd.read_sql(query, engine)
        if df.empty:
            result[key] = []
            continue
        column_name = df.columns[0]
        result[key] = df[column_name].tolist()
    return result


def run_benchmarks(per_endpoint_runs: int, target_user_id: int | None = None) -> list[BenchmarkObservation]:
    _seed_homepage_history()
    samples = _pick_sample_identifiers(target_user_id=target_user_id)
    observations: list[BenchmarkObservation] = []

    with TestClient(app) as client:
        for product_sys_id in samples["detail"][:per_endpoint_runs]:
            started_at = time.perf_counter()
            response = client.get(f"/content_based_filter/{product_sys_id}", params={"top_n": 5})
            wall_clock_ms = (time.perf_counter() - started_at) * 1000
            payload = response.json()
            observations.append(
                BenchmarkObservation(
                    endpoint=f"/content_based_filter/{product_sys_id}",
                    response_time_ms=_parse_ms(response.headers.get("X-Response-Time-Ms")) or round(wall_clock_ms, 2),
                    llm_latency_ms=_parse_ms(response.headers.get("X-LLM-Latency-Ms")),
                    llm_status=_safe_text(response.headers.get("X-LLM-Status")) or "not_used",
                    recommendation_count=len(payload.get("recommendations", [])),
                    status_code=response.status_code,
                )
            )

        user_paths = [
            ("wishlist", "/api/v1/recommendation/wishlist/{user_id}"),
            ("cart", "/api/v1/recommendation/cart/{user_id}"),
            ("homepage", "/api/v1/recommendation/homepage/{user_id}"),
        ]
        for scene, path_template in user_paths:
            for user_id in samples[scene][:per_endpoint_runs]:
                started_at = time.perf_counter()
                response = client.get(path_template.format(user_id=int(user_id)), params={"top_n": 5})
                wall_clock_ms = (time.perf_counter() - started_at) * 1000
                payload = response.json()
                observations.append(
                    BenchmarkObservation(
                        endpoint=path_template.format(user_id=int(user_id)),
                        response_time_ms=_parse_ms(response.headers.get("X-Response-Time-Ms")) or round(wall_clock_ms, 2),
                        llm_latency_ms=_parse_ms(response.headers.get("X-LLM-Latency-Ms")),
                        llm_status=_safe_text(response.headers.get("X-LLM-Status")) or "not_used",
                        recommendation_count=len(payload.get("recommendations", [])),
                        status_code=response.status_code,
                    )
                )

    return observations


def evaluate_ranking_metrics(top_k: int, lookahead_days: int, target_user_id: int | None = None) -> dict[str, float]:
    query = """
    SELECT
        upe.user_id,
        p.product_sys_id,
        upe.event_type,
        upe.created_at
    FROM UserProductEvent upe
    JOIN Product p ON upe.product_id = p.product_id
    WHERE upe.event_type IN (
        'view_detail',
        'search_click',
        'recommendation_click',
        'wishlist_add',
        'add_cart',
        'purchase'
    )
    ORDER BY upe.user_id, upe.created_at, upe.id
    """
    events = pd.read_sql(query, engine)
    if events.empty:
        return {"precision_at_5": 0.0, "hit_rate_at_5": 0.0, "ndcg_at_5": 0.0, "evaluated_cases": 0}

    if target_user_id:
        events = events[events["user_id"] == int(target_user_id)].copy()
        if events.empty:
            return {"precision_at_5": 0.0, "hit_rate_at_5": 0.0, "ndcg_at_5": 0.0, "evaluated_cases": 0}

    events["product_sys_id"] = events["product_sys_id"].astype(str).str.strip()
    events["created_at"] = pd.to_datetime(events["created_at"], errors="coerce")

    precisions: list[float] = []
    hit_rates: list[float] = []
    ndcgs: list[float] = []
    evaluated_cases = 0

    for user_id, user_events in events.groupby("user_id"):
        rows = user_events.reset_index(drop=True)
        for idx, row in rows.iterrows():
            if row["event_type"] != "view_detail":
                continue

            anchor_product = _safe_text(row["product_sys_id"])
            predictions = recommend(anchor_product, top_n=top_k)
            if not predictions:
                continue

            cutoff_time = row["created_at"] + pd.Timedelta(days=lookahead_days)
            future_events = rows.iloc[idx + 1 :]
            future_events = future_events[
                (future_events["created_at"] <= cutoff_time)
                & (future_events["event_type"].isin(POSITIVE_EVENT_TYPES))
            ]
            relevant_set = {
                _safe_text(pid)
                for pid in future_events["product_sys_id"].tolist()
                if _safe_text(pid) and _safe_text(pid) != anchor_product
            }
            if not relevant_set:
                continue

            hits = [1 if _safe_text(pid) in relevant_set else 0 for pid in predictions[:top_k]]
            evaluated_cases += 1
            precisions.append(sum(hits) / top_k)
            hit_rates.append(1.0 if any(hits) else 0.0)

            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal_hits = min(len(relevant_set), top_k)
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
            ndcgs.append((dcg / idcg) if idcg else 0.0)

    return {
        "precision_at_5": _mean(precisions),
        "hit_rate_at_5": _mean(hit_rates),
        "ndcg_at_5": _mean(ndcgs),
        "evaluated_cases": evaluated_cases,
    }


def evaluate_engagement_metrics(top_k: int, exact_only: bool, target_user_id: int | None = None) -> dict[str, float | str | int]:
    exact_query = """
    SELECT
        upe.user_id,
        p.product_sys_id,
        upe.event_type,
        upe.created_at
    FROM UserProductEvent upe
    JOIN Product p ON upe.product_id = p.product_id
    WHERE upe.event_type IN ('recommendation_impression', 'recommendation_click', 'add_cart')
    ORDER BY upe.user_id, upe.created_at, upe.id
    """
    exact_df = pd.read_sql(exact_query, engine)
    if target_user_id:
        exact_df = exact_df[exact_df["user_id"] == int(target_user_id)].copy()
    impressions = 0
    clicked_impressions = 0
    carted_impressions = 0

    if not exact_df.empty:
        exact_df["product_sys_id"] = exact_df["product_sys_id"].astype(str).str.strip()
        exact_df["created_at"] = pd.to_datetime(exact_df["created_at"], errors="coerce")
        impression_df = exact_df[exact_df["event_type"] == "recommendation_impression"].copy()
        click_df = exact_df[exact_df["event_type"] == "recommendation_click"].copy()
        cart_df = exact_df[exact_df["event_type"] == "add_cart"].copy()

        impressions = len(impression_df)
        for _, row in impression_df.iterrows():
            clicked_match = click_df[
                (click_df["user_id"] == row["user_id"])
                & (click_df["product_sys_id"] == row["product_sys_id"])
                & (click_df["created_at"] >= row["created_at"])
                & (click_df["created_at"] <= row["created_at"] + pd.Timedelta(hours=24))
            ]
            if clicked_match.empty:
                continue

            clicked_impressions += 1
            cart_match = cart_df[
                (cart_df["user_id"] == row["user_id"])
                & (cart_df["product_sys_id"] == row["product_sys_id"])
                & (cart_df["created_at"] >= row["created_at"])
                & (cart_df["created_at"] <= row["created_at"] + pd.Timedelta(hours=24))
            ]
            if not cart_match.empty:
                carted_impressions += 1

    if impressions > 0:
        return {
            "mode": "exact_impressions",
            "impressions": impressions,
            "ctr": round(clicked_impressions / impressions, 4),
            "add_to_cart_rate": round(carted_impressions / impressions, 4),
        }

    if exact_only:
        return {
            "mode": "exact_impressions",
            "impressions": 0,
            "ctr": 0.0,
            "add_to_cart_rate": 0.0,
        }

    replay_query = """
    SELECT
        upe.user_id,
        p.product_sys_id,
        upe.created_at
    FROM UserProductEvent upe
    JOIN Product p ON upe.product_id = p.product_id
    WHERE upe.event_type = 'view_detail'
    ORDER BY upe.user_id, upe.created_at, upe.id
    """
    click_query = """
    SELECT
        upe.user_id,
        p.product_sys_id,
        upe.created_at
    FROM UserProductEvent upe
    JOIN Product p ON upe.product_id = p.product_id
    WHERE upe.event_type = 'recommendation_click'
    ORDER BY upe.user_id, upe.created_at, upe.id
    """
    cart_query = """
    SELECT
        upe.user_id,
        p.product_sys_id,
        upe.created_at
    FROM UserProductEvent upe
    JOIN Product p ON upe.product_id = p.product_id
    WHERE upe.event_type = 'add_cart'
    ORDER BY upe.user_id, upe.created_at, upe.id
    """
    detail_df = pd.read_sql(replay_query, engine)
    click_df = pd.read_sql(click_query, engine)
    cart_df = pd.read_sql(cart_query, engine)
    if target_user_id:
        detail_df = detail_df[detail_df["user_id"] == int(target_user_id)].copy()
        click_df = click_df[click_df["user_id"] == int(target_user_id)].copy()
        cart_df = cart_df[cart_df["user_id"] == int(target_user_id)].copy()

    if detail_df.empty:
        return {
            "mode": "replayed_detail_impressions",
            "impressions": 0,
            "ctr": 0.0,
            "add_to_cart_rate": 0.0,
        }

    detail_df["product_sys_id"] = detail_df["product_sys_id"].astype(str).str.strip()
    detail_df["created_at"] = pd.to_datetime(detail_df["created_at"], errors="coerce")
    click_df["product_sys_id"] = click_df["product_sys_id"].astype(str).str.strip()
    click_df["created_at"] = pd.to_datetime(click_df["created_at"], errors="coerce")
    cart_df["product_sys_id"] = cart_df["product_sys_id"].astype(str).str.strip()
    cart_df["created_at"] = pd.to_datetime(cart_df["created_at"], errors="coerce")

    replayed_impressions = 0
    clicked = 0
    carted = 0

    for _, row in detail_df.iterrows():
        predictions = recommend(_safe_text(row["product_sys_id"]), top_n=top_k)
        if not predictions:
            continue

        replayed_impressions += len(predictions)
        clicked_products = click_df[
            (click_df["user_id"] == row["user_id"])
            & (click_df["created_at"] >= row["created_at"])
            & (click_df["created_at"] <= row["created_at"] + pd.Timedelta(hours=24))
        ]["product_sys_id"].tolist()
        clicked_products = {pid for pid in clicked_products if pid in predictions}
        clicked += len(clicked_products)

        if not clicked_products:
            continue

        cart_products = cart_df[
            (cart_df["user_id"] == row["user_id"])
            & (cart_df["created_at"] >= row["created_at"])
            & (cart_df["created_at"] <= row["created_at"] + pd.Timedelta(hours=24))
        ]["product_sys_id"].tolist()
        cart_products = {pid for pid in cart_products if pid in clicked_products}
        carted += len(cart_products)

    return {
        "mode": "replayed_detail_impressions",
        "impressions": replayed_impressions,
        "ctr": round((clicked / replayed_impressions), 4) if replayed_impressions else 0.0,
        "add_to_cart_rate": round((carted / replayed_impressions), 4) if replayed_impressions else 0.0,
    }


def summarize_latency(observations: list[BenchmarkObservation]) -> dict[str, float | int]:
    response_times = [obs.response_time_ms for obs in observations if obs.status_code == 200]
    llm_latencies = [float(obs.llm_latency_ms or 0.0) for obs in observations if float(obs.llm_latency_ms or 0.0) > 0]
    llm_statuses = {}
    for obs in observations:
        llm_statuses[obs.llm_status] = llm_statuses.get(obs.llm_status, 0) + 1
    return {
        "requests": len(observations),
        "successful_requests": sum(1 for obs in observations if obs.status_code == 200),
        "response_time_ms_p50": _median(response_times),
        "response_time_ms_avg": _mean(response_times),
        "llm_latency_ms_p50": _median(llm_latencies),
        "llm_latency_ms_avg": _mean(llm_latencies),
        "llm_status_summary": ", ".join(f"{key}={value}" for key, value in sorted(llm_statuses.items())),
    }


def print_report(
    ranking_metrics: dict[str, float],
    engagement_metrics: dict[str, float | str | int],
    latency_metrics: dict[str, float | int],
    observations: list[BenchmarkObservation],
    target_user_id: int | None,
) -> None:
    print()
    print("=== RECOMMENDATION METRICS REPORT ===")
    if target_user_id:
        print(f"Target User ID    : {target_user_id}")
    print()
    print("Ranking Metrics")
    print(f"Precision@5      : {ranking_metrics['precision_at_5']:.4f}")
    print(f"Hit Rate@5       : {ranking_metrics['hit_rate_at_5']:.4f}")
    print(f"NDCG@5           : {ranking_metrics['ndcg_at_5']:.4f}")
    print(f"Evaluated cases  : {ranking_metrics['evaluated_cases']}")
    print()
    print("Engagement Metrics")
    print(f"CTR              : {float(engagement_metrics['ctr']):.4f}")
    print(f"Add-to-Cart Rate : {float(engagement_metrics['add_to_cart_rate']):.4f}")
    print(f"Impression mode  : {engagement_metrics['mode']}")
    print(f"Impressions      : {engagement_metrics['impressions']}")
    print()
    print("Latency Metrics")
    print(f"Response Time P50: {float(latency_metrics['response_time_ms_p50']):.2f} ms")
    print(f"Response Time Avg: {float(latency_metrics['response_time_ms_avg']):.2f} ms")
    print(f"LLM Latency P50  : {float(latency_metrics['llm_latency_ms_p50']):.2f} ms")
    print(f"LLM Latency Avg  : {float(latency_metrics['llm_latency_ms_avg']):.2f} ms")
    print(f"LLM Status       : {latency_metrics['llm_status_summary']}")
    print(f"Successful Calls : {latency_metrics['successful_requests']}/{latency_metrics['requests']}")
    print()
    print("Benchmark Samples")
    for obs in observations:
        print(
            f"[{obs.status_code}] {obs.endpoint} | "
            f"response={obs.response_time_ms:.2f} ms | "
            f"llm={obs.llm_latency_ms:.2f} ms | "
            f"llm_status={obs.llm_status} | "
            f"items={obs.recommendation_count}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Console metrics for recommendation system.")
    parser.add_argument("--runs", type=int, default=2, help="Benchmark requests per endpoint.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K cutoff for ranking metrics.")
    parser.add_argument("--lookahead-days", type=int, default=14, help="Future relevance window in days.")
    parser.add_argument("--exact-impressions-only", action="store_true", help="Disable replay fallback for CTR metrics.")
    parser.add_argument("--user-id", type=int, default=None, help="Run metrics for one specific user_id.")
    parser.add_argument("--user-email", type=str, default=None, help="Resolve a user by email and run metrics for that user.")
    args = parser.parse_args()

    load_all_data()
    target_user_id = resolve_user_id(user_id=args.user_id, user_email=args.user_email)
    observations = run_benchmarks(per_endpoint_runs=max(args.runs, 1), target_user_id=target_user_id)
    ranking_metrics = evaluate_ranking_metrics(
        top_k=args.top_k,
        lookahead_days=args.lookahead_days,
        target_user_id=target_user_id,
    )
    engagement_metrics = evaluate_engagement_metrics(
        top_k=args.top_k,
        exact_only=args.exact_impressions_only,
        target_user_id=target_user_id,
    )
    latency_metrics = summarize_latency(observations)
    print_report(ranking_metrics, engagement_metrics, latency_metrics, observations, target_user_id=target_user_id)


if __name__ == "__main__":
    main()
