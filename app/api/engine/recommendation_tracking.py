import json
from typing import Any

import pandas as pd
from sqlalchemy import text


def lookup_product_ids(db_engine, product_sys_ids: list[str]) -> dict[str, int]:
    normalized_ids = [str(pid).strip() for pid in product_sys_ids if str(pid).strip()]
    if not normalized_ids:
        return {}

    placeholders = ",".join([f":pid_{idx}" for idx, _value in enumerate(normalized_ids)])
    params = {f"pid_{idx}": value for idx, value in enumerate(normalized_ids)}
    query = text(
        f"""
        SELECT product_id, product_sys_id
        FROM Product
        WHERE product_sys_id IN ({placeholders})
        """
    )

    df = pd.read_sql(query, db_engine, params=params)
    if df.empty:
        return {}

    return {
        str(row["product_sys_id"]).strip(): int(row["product_id"])
        for _, row in df.iterrows()
    }


def log_recommendation_impressions(
    db_engine,
    user_id: int,
    scene: str,
    recommendations: list[str],
    response_time_ms: float,
    llm_latency_ms: float,
    source_ids: list[str] | None = None,
) -> int:
    if not user_id or not recommendations:
        return 0

    product_map = lookup_product_ids(db_engine, recommendations)
    if not product_map:
        return 0

    rows: list[dict[str, Any]] = []
    for rank, product_sys_id in enumerate(recommendations, start=1):
        product_id = product_map.get(str(product_sys_id).strip())
        if product_id is None:
            continue

        metadata = {
            "scene": scene,
            "rank": rank,
            "response_time_ms": round(float(response_time_ms), 2),
            "llm_latency_ms": round(float(llm_latency_ms), 2),
            "source_product_ids": [str(pid).strip() for pid in (source_ids or []) if str(pid).strip()],
        }
        rows.append(
            {
                "user_id": int(user_id),
                "product_id": int(product_id),
                "event_type": "recommendation_impression",
                "weight": 0.0,
                "source": f"recommendation_api:{scene}",
                "metadata_json": json.dumps(metadata, ensure_ascii=True),
            }
        )

    if not rows:
        return 0

    insert_stmt = text(
        """
        INSERT INTO UserProductEvent (
            user_id,
            session_id,
            product_id,
            event_type,
            weight,
            source,
            metadata_json,
            created_at
        )
        VALUES (
            :user_id,
            NULL,
            :product_id,
            :event_type,
            :weight,
            :source,
            :metadata_json,
            SYSUTCDATETIME()
        )
        """
    )

    with db_engine.begin() as connection:
        connection.execute(insert_stmt, rows)

    return len(rows)
