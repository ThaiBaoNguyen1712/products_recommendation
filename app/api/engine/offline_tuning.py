import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.api.engine.content_based import recommend
from app.api.engine.recommendation_weights import DEFAULT_WEIGHT_CONFIG_PATH
from db.mssql import engine as default_engine

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_JSON_PATH = REPO_ROOT / "data" / "offline_tuning_results.json"
DEFAULT_RESULTS_CSV_PATH = REPO_ROOT / "data" / "offline_tuning_results.csv"
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "offline_evaluation_dataset.json"

EVENT_TYPE_COLUMNS = {
    "view": "view_count",
    "cart": "add_to_cart_count",
    "wishlist": "wishlist_count",
    "purchase": "purchase_count",
}

DEFAULT_SEARCH_SPACE = {
    "purchase": [6.0, 8.0, 10.0],
    "cart": [3.0, 4.0, 5.0],
    "wishlist": [1.0, 2.0, 3.0],
    "view": [0.5, 1.0],
    "interaction": [1.0],
    "decay_lambda": [0.03, 0.05, 0.08],
    "homepage_similarity": [0.9, 1.0, 1.1],
    "wishlist_similarity": [1.2, 1.3, 1.4],
    "cart_similarity": [1.5, 1.6, 1.7],
}


@dataclass(frozen=True)
class InteractionEvent:
    user_id: int
    product_sys_id: str
    event_type: str
    event_value: float
    occurred_at: str


@dataclass(frozen=True)
class EvaluationSample:
    user_id: int
    training_events: list[dict[str, Any]]
    future_events: list[dict[str, Any]]


@dataclass(frozen=True)
class WeightCandidate:
    purchase: float
    cart: float
    wishlist: float
    view: float
    interaction: float
    decay_lambda: float
    homepage_similarity: float
    wishlist_similarity: float
    cart_similarity: float

    def event_weight(self, event_type: str) -> float:
        if event_type == "purchase":
            return self.purchase
        if event_type == "cart":
            return self.cart
        if event_type == "wishlist":
            return self.wishlist
        if event_type == "view":
            return self.view
        return self.interaction

    def scene_similarity_weight(self, scene: str) -> float:
        if scene == "wishlist":
            return self.wishlist_similarity
        if scene == "cart":
            return self.cart_similarity
        return self.homepage_similarity


def parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    text_value = str(value or "").strip()
    if not text_value:
        return None
    try:
        parsed = datetime.fromisoformat(text_value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def fetch_interaction_events(db_engine: Engine, *, limit: int = 10000) -> list[InteractionEvent]:
    query = text(f"""
    SELECT TOP ({max(int(limit), 1)})
        upe.user_id,
        p.product_sys_id,
        upe.view_count,
        upe.add_to_cart_count,
        upe.wishlist_count,
        upe.purchase_count,
        upe.interaction_score,
        upe.last_interacted_at
    FROM UserProductEvent upe
    JOIN Product p ON upe.product_id = p.product_id
    WHERE p.product_sys_id IS NOT NULL
      AND upe.last_interacted_at IS NOT NULL
    ORDER BY upe.user_id, upe.last_interacted_at
    """)
    with db_engine.connect() as connection:
        rows = connection.execute(query).mappings().all()
    events: list[InteractionEvent] = []
    for row in rows:
        user_id = int(row.get("user_id") or 0)
        product_sys_id = str(row.get("product_sys_id") or "").strip()
        occurred_at = parse_datetime(row.get("last_interacted_at"))
        if not user_id or not product_sys_id or occurred_at is None:
            continue
        for event_type, column in EVENT_TYPE_COLUMNS.items():
            event_value = _to_float(row.get(column))
            if event_value <= 0:
                continue
            events.append(
                InteractionEvent(
                    user_id=user_id,
                    product_sys_id=product_sys_id,
                    event_type=event_type,
                    event_value=event_value,
                    occurred_at=occurred_at.isoformat(),
                )
            )
        interaction_score = _to_float(row.get("interaction_score"))
        if interaction_score > 0:
            events.append(
                InteractionEvent(
                    user_id=user_id,
                    product_sys_id=product_sys_id,
                    event_type="interaction",
                    event_value=interaction_score,
                    occurred_at=occurred_at.isoformat(),
                )
            )
    return events


def build_evaluation_samples(
    events: list[InteractionEvent],
    *,
    holdout_days: int = 7,
    future_event_types: set[str] | None = None,
    min_training_events: int = 2,
    min_future_events: int = 1,
) -> list[EvaluationSample]:
    if future_event_types is None:
        future_event_types = {"purchase", "cart", "wishlist"}

    by_user: dict[int, list[InteractionEvent]] = defaultdict(list)
    for event in events:
        if parse_datetime(event.occurred_at) is None:
            continue
        by_user[event.user_id].append(event)

    samples: list[EvaluationSample] = []
    for user_id, user_events in by_user.items():
        ordered_events = sorted(user_events, key=lambda event: str(event.occurred_at))
        timestamps = [
            parsed
            for parsed in (parse_datetime(event.occurred_at) for event in ordered_events)
            if parsed is not None
        ]
        if not timestamps:
            continue
        cutoff = max(timestamps) - timedelta(days=max(int(holdout_days), 1))
        training_events = [
            asdict(event)
            for event in ordered_events
            if (parse_datetime(event.occurred_at) or cutoff) <= cutoff
        ]
        future_events = [
            asdict(event)
            for event in ordered_events
            if (parse_datetime(event.occurred_at) or cutoff) > cutoff
            and event.event_type in future_event_types
        ]
        if len(training_events) < min_training_events or len(future_events) < min_future_events:
            continue
        samples.append(
            EvaluationSample(
                user_id=user_id,
                training_events=training_events,
                future_events=future_events,
            )
        )
    return samples


def recommend_for_sample(
    sample: EvaluationSample,
    weights: WeightCandidate,
    *,
    scene: str,
    top_k: int,
    candidate_per_source: int = 12,
) -> list[str]:
    candidate_scores: dict[str, float] = {}
    training_product_ids = {
        str(event.get("product_sys_id", "")).strip()
        for event in sample.training_events
        if str(event.get("product_sys_id", "")).strip()
    }
    max_training_time = max(
        (
            parsed
            for parsed in (parse_datetime(event.get("occurred_at")) for event in sample.training_events)
            if parsed is not None
        ),
        default=None,
    )

    for event in sample.training_events:
        product_id = str(event.get("product_sys_id", "")).strip()
        if not product_id:
            continue
        event_time = parse_datetime(event.get("occurred_at"))
        age_days = 0.0
        if max_training_time is not None and event_time is not None:
            age_days = max((max_training_time - event_time).total_seconds() / 86400.0, 0.0)
        decay = math.exp(-weights.decay_lambda * age_days)
        signal = (
            weights.event_weight(str(event.get("event_type", "")).strip())
            * _to_float(event.get("event_value"))
            * decay
        )
        if signal <= 0:
            continue
        for candidate_id in recommend(product_id, top_n=candidate_per_source):
            if candidate_id in training_product_ids:
                continue
            candidate_scores[candidate_id] = (
                candidate_scores.get(candidate_id, 0.0)
                + signal * weights.scene_similarity_weight(scene)
            )

    return [
        product_id
        for product_id, _score in sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
    ][:top_k]


def precision_at_k(recommendations: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_items = recommendations[:k]
    if not top_items:
        return 0.0
    hits = sum(1 for product_id in top_items if product_id in relevant_ids)
    return hits / min(k, len(top_items))


def recall_at_k(recommendations: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for product_id in recommendations[:k] if product_id in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(recommendations: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for idx, product_id in enumerate(recommendations[:k], start=1):
        if product_id in relevant_ids:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_weights(
    samples: list[EvaluationSample],
    weights: WeightCandidate,
    *,
    scene: str,
    k: int = 10,
) -> dict[str, float]:
    if not samples:
        return {"precision@10": 0.0, "recall@10": 0.0, "ndcg@10": 0.0, "sample_count": 0.0}

    precision_sum = 0.0
    recall_sum = 0.0
    ndcg_sum = 0.0
    evaluated = 0
    for sample in samples:
        relevant_ids = {
            str(event.get("product_sys_id", "")).strip()
            for event in sample.future_events
            if str(event.get("product_sys_id", "")).strip()
        }
        if not relevant_ids:
            continue
        recommendations = recommend_for_sample(sample, weights, scene=scene, top_k=k)
        precision_sum += precision_at_k(recommendations, relevant_ids, k)
        recall_sum += recall_at_k(recommendations, relevant_ids, k)
        ndcg_sum += ndcg_at_k(recommendations, relevant_ids, k)
        evaluated += 1

    if evaluated <= 0:
        return {"precision@10": 0.0, "recall@10": 0.0, "ndcg@10": 0.0, "sample_count": 0.0}
    return {
        "precision@10": round(precision_sum / evaluated, 6),
        "recall@10": round(recall_sum / evaluated, 6),
        "ndcg@10": round(ndcg_sum / evaluated, 6),
        "sample_count": float(evaluated),
    }


def iter_weight_candidates(search_space: dict[str, list[float]]) -> list[WeightCandidate]:
    keys = [
        "purchase",
        "cart",
        "wishlist",
        "view",
        "interaction",
        "decay_lambda",
        "homepage_similarity",
        "wishlist_similarity",
        "cart_similarity",
    ]
    candidates: list[WeightCandidate] = []
    for values in itertools.product(*(search_space[key] for key in keys)):
        candidates.append(WeightCandidate(**dict(zip(keys, values, strict=True))))
    return candidates


def run_grid_search(
    samples: list[EvaluationSample],
    *,
    scene: str,
    search_space: dict[str, list[float]] | None = None,
    max_combinations: int | None = None,
    k: int = 10,
) -> list[dict[str, Any]]:
    candidate_weights = iter_weight_candidates(search_space or DEFAULT_SEARCH_SPACE)
    if max_combinations is not None:
        candidate_weights = candidate_weights[: max(int(max_combinations), 1)]

    results: list[dict[str, Any]] = []
    for idx, weights in enumerate(candidate_weights, start=1):
        metrics = evaluate_weights(samples, weights, scene=scene, k=k)
        results.append(
            {
                "experiment_name": "weight_search_v1",
                "combination_index": idx,
                "scene": scene,
                "weights": asdict(weights),
                "metrics": metrics,
            }
        )
    return sorted(
        results,
        key=lambda item: (
            item["metrics"].get("ndcg@10", 0.0),
            item["metrics"].get("recall@10", 0.0),
            item["metrics"].get("precision@10", 0.0),
        ),
        reverse=True,
    )


def save_dataset(samples: list[EvaluationSample], path: Path = DEFAULT_DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(sample) for sample in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_dataset(path: Path) -> list[EvaluationSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    samples: list[EvaluationSample] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        samples.append(
            EvaluationSample(
                user_id=int(item.get("user_id") or 0),
                training_events=list(item.get("training_events") or []),
                future_events=list(item.get("future_events") or []),
            )
        )
    return samples


def save_results(
    results: list[dict[str, Any]],
    *,
    json_path: Path = DEFAULT_RESULTS_JSON_PATH,
    csv_path: Path = DEFAULT_RESULTS_CSV_PATH,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "scene",
                "ndcg@10",
                "recall@10",
                "precision@10",
                "sample_count",
                "weights_json",
            ],
        )
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "scene": result["scene"],
                    "ndcg@10": result["metrics"].get("ndcg@10", 0.0),
                    "recall@10": result["metrics"].get("recall@10", 0.0),
                    "precision@10": result["metrics"].get("precision@10", 0.0),
                    "sample_count": result["metrics"].get("sample_count", 0.0),
                    "weights_json": json.dumps(result["weights"], ensure_ascii=False),
                }
            )


def save_best_weight_config(best_result: dict[str, Any], path: Path = DEFAULT_WEIGHT_CONFIG_PATH) -> None:
    weights = best_result.get("weights", {})
    payload = {
        "meta": {
            "source": "offline_tuning",
            "experiment_name": best_result.get("experiment_name", "weight_search_v1"),
            "scene": best_result.get("scene"),
            "metrics": best_result.get("metrics", {}),
            "selected_by": "ndcg@10_then_recall@10_then_precision@10",
        },
        "implicit": {
            "purchase": weights.get("purchase", 8.0),
            "cart": weights.get("cart", 4.0),
            "wishlist": weights.get("wishlist", 2.0),
            "view": weights.get("view", 1.0),
            "interaction": weights.get("interaction", 1.0),
        },
        "time_decay": {
            "lambda": weights.get("decay_lambda", 0.05),
        },
        "scene_similarity": {
            "homepage": weights.get("homepage_similarity", 1.0),
            "wishlist": weights.get("wishlist_similarity", 1.3),
            "cart": weights.get("cart_similarity", 1.6),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_synthetic_samples() -> list[EvaluationSample]:
    now = datetime.now(timezone.utc)
    products = [
        "prd_6114329447",
        "prd_30277b69be",
        "prd_39a1a049f8",
        "prd_5cb7e52cea",
        "prd_1f13e9c8ef",
    ]
    events = [
        InteractionEvent(1, products[0], "view", 3, (now - timedelta(days=12)).isoformat()),
        InteractionEvent(1, products[0], "cart", 1, (now - timedelta(days=10)).isoformat()),
        InteractionEvent(1, products[1], "wishlist", 1, (now - timedelta(days=9)).isoformat()),
        InteractionEvent(1, products[3], "cart", 1, (now - timedelta(days=2)).isoformat()),
        InteractionEvent(2, products[1], "view", 4, (now - timedelta(days=14)).isoformat()),
        InteractionEvent(2, products[1], "wishlist", 1, (now - timedelta(days=8)).isoformat()),
        InteractionEvent(2, products[2], "purchase", 1, (now - timedelta(days=1)).isoformat()),
    ]
    return build_evaluation_samples(events, holdout_days=7, min_training_events=1, min_future_events=1)


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline recommendation weight tuning.")
    parser.add_argument("--scene", choices=["homepage", "wishlist", "cart"], default="homepage")
    parser.add_argument("--from-db", action="store_true")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--event-limit", type=int, default=10000)
    parser.add_argument("--max-combinations", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if args.synthetic:
        samples = build_synthetic_samples()
    elif args.from_db:
        events = fetch_interaction_events(default_engine, limit=args.event_limit)
        samples = build_evaluation_samples(events, holdout_days=args.holdout_days)
        save_dataset(samples, dataset_path)
    else:
        samples = load_dataset(dataset_path)

    results = run_grid_search(
        samples,
        scene=args.scene,
        max_combinations=args.max_combinations,
    )
    if not args.no_save:
        save_results(results)
    if results and not args.no_save:
        save_best_weight_config(results[0])
    print(
        json.dumps(
            {
                "scene": args.scene,
                "sample_count": len(samples),
                "combination_count": len(results),
                "best": results[0] if results else None,
                "saved": not args.no_save,
                "results_json": str(DEFAULT_RESULTS_JSON_PATH) if not args.no_save else None,
                "results_csv": str(DEFAULT_RESULTS_CSV_PATH) if not args.no_save else None,
                "best_weight_config": str(DEFAULT_WEIGHT_CONFIG_PATH) if not args.no_save else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
