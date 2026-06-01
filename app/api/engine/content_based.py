# app/api/engine/content_based.py
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

from db.mssql import engine

# Global caches
df_products = None
similarity_matrix = None
product_sys_id_to_index = None
behavior_transition_scores = None

POSITIVE_EVENT_WEIGHTS = {
    "view_detail": 1.0,
    "search_click": 1.5,
    "recommendation_click": 2.0,
    "wishlist_add": 3.0,
    "add_cart": 4.0,
    "purchase": 5.0,
}
DEFAULT_LOOKAHEAD_DAYS = 14


def load_all_data():
    global df_products, similarity_matrix, product_sys_id_to_index, behavior_transition_scores

    query = """
    SELECT
        p.product_sys_id, p.name, p.sellPrice, p.stock,
        ct.name AS category, b.name AS brand,
        STRING_AGG(CONCAT(s.name, ' ', svl.value), ' ') AS specs_text
    FROM Product p
    JOIN Brand b ON p.brandId = b.BrandId
    JOIN Category ct ON p.category_id = ct.category_id
    LEFT JOIN SpecValue svl ON p.product_id = svl.product_id
    LEFT JOIN Specs s ON svl.spec_id = s.spec_id
    GROUP BY p.product_sys_id, p.name, p.sellPrice, p.stock, ct.name, b.name
    """

    df = pd.read_sql(query, engine)
    df.fillna("", inplace=True)

    df["sellPrice"] = pd.to_numeric(df["sellPrice"], errors="coerce").fillna(0).astype(float)
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
    df["product_sys_id"] = df["product_sys_id"].astype(str).str.strip()
    df = df[df["product_sys_id"] != ""].reset_index(drop=True)

    df["combined_features"] = (
        (df["name"] + " ") * 5
        + (df["category"] + " ") * 4
        + (df["brand"] + " ") * 2
        + df["specs_text"]
    )

    tfidf = TfidfVectorizer(stop_words="english", min_df=2)
    tf_matrix = tfidf.fit_transform(df["combined_features"])

    normalizer = Normalizer()
    tf_matrix = normalizer.fit_transform(tf_matrix)

    similarity_matrix = cosine_similarity(tf_matrix)
    df_products = df
    product_sys_id_to_index = {pid: idx for idx, pid in enumerate(df["product_sys_id"])}
    behavior_transition_scores = _build_behavior_transition_scores(lookahead_days=DEFAULT_LOOKAHEAD_DAYS)

    print(f"Content-based Ready: {df.shape[0]} products.")


def recommend(
    product_sys_id: str,
    top_n: int = 5,
    price_margin: float = 0.8,
    behavior_weight: float = 5.0,
):
    if (
        product_sys_id_to_index is None
        or df_products is None
        or similarity_matrix is None
        or behavior_transition_scores is None
    ):
        load_all_data()

    if product_sys_id not in product_sys_id_to_index:
        return []

    idx = product_sys_id_to_index[product_sys_id]
    current_item = df_products.iloc[idx]

    current_price = float(current_item["sellPrice"])
    lower_bound = current_price * (1 - price_margin)
    upper_bound = current_price * (1 + price_margin)

    mask = (
        (df_products["stock"] > 0)
        & (df_products["sellPrice"] >= lower_bound)
        & (df_products["sellPrice"] <= upper_bound)
    )

    eligible_indices = np.where(mask)[0]
    eligible_indices = eligible_indices[eligible_indices != idx]

    if len(eligible_indices) == 0:
        return []

    scores = similarity_matrix[idx][eligible_indices]

    behavior_map = behavior_transition_scores.get(product_sys_id, {})
    behavior_scores = np.array(
        [float(behavior_map.get(df_products.iloc[item_idx]["product_sys_id"], 0.0)) for item_idx in eligible_indices],
        dtype=float,
    )
    if behavior_scores.size > 0 and behavior_scores.max() > 0:
        behavior_scores = behavior_scores / behavior_scores.max()

    final_scores = scores + (behavior_scores * behavior_weight)
    top_local_indices = np.argsort(final_scores)[::-1][:top_n]
    final_indices = eligible_indices[top_local_indices]

    return df_products.iloc[final_indices]["product_sys_id"].tolist()


def _build_behavior_transition_scores(lookahead_days: int) -> dict[str, dict[str, float]]:
    events_query = """
    SELECT
        upe.user_id,
        p.product_sys_id,
        upe.event_type,
        upe.created_at,
        upe.id
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

    events_df = pd.read_sql(events_query, engine)
    if events_df.empty:
        return {}

    events_df["product_sys_id"] = events_df["product_sys_id"].astype(str).str.strip()
    events_df["created_at"] = pd.to_datetime(events_df["created_at"], errors="coerce")
    transitions: defaultdict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for _user_id, user_events in events_df.groupby("user_id"):
        rows = user_events.reset_index(drop=True)
        for idx, row in rows.iterrows():
            source_id = str(row["product_sys_id"]).strip()
            if not source_id:
                continue

            cutoff_time = row["created_at"] + pd.Timedelta(days=lookahead_days)
            future_events = rows.iloc[idx + 1 :]
            future_events = future_events[
                (future_events["created_at"] <= cutoff_time)
                & (future_events["product_sys_id"] != source_id)
            ]

            for _, future_row in future_events.iterrows():
                target_id = str(future_row["product_sys_id"]).strip()
                if not target_id:
                    continue
                transitions[source_id][target_id] += float(
                    POSITIVE_EVENT_WEIGHTS.get(str(future_row["event_type"]).strip(), 0.0)
                )

    return {source_id: dict(target_scores) for source_id, target_scores in transitions.items()}
