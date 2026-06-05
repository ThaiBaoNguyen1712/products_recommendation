import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import Normalizer

from db.mssql import engine

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
PRODUCTS_PATH = DATA_DIR / "products.json"
PRODUCT_IDS_PATH = DATA_DIR / "product_ids.json"
PRODUCT_VECTORS_PATH = DATA_DIR / "product_vectors.npy"

HASH_VECTOR_DIM = int(os.getenv("HASH_VECTOR_DIM", "2048"))

_hashing_vectorizer = HashingVectorizer(
    n_features=HASH_VECTOR_DIM,
    alternate_sign=False,
    stop_words="english",
    norm=None,
)
_normalizer = Normalizer()

df_products = None
similarity_matrix = None
product_sys_id_to_index = None
product_profiles_by_id = None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _paths_ready() -> bool:
    return all(path.exists() for path in [PRODUCTS_PATH, PRODUCT_IDS_PATH, PRODUCT_VECTORS_PATH])


def _load_all_data_from_artifacts() -> bool:
    global df_products, similarity_matrix, product_sys_id_to_index, product_profiles_by_id

    if not _paths_ready():
        return False

    products = _load_json(PRODUCTS_PATH)
    product_ids = _load_json(PRODUCT_IDS_PATH)
    vector_matrix = np.load(PRODUCT_VECTORS_PATH, allow_pickle=False)

    if not isinstance(products, list) or not isinstance(product_ids, list):
        return False
    if vector_matrix.ndim != 2 or len(product_ids) != len(products) or len(product_ids) != int(vector_matrix.shape[0]):
        return False

    normalized_products: list[dict[str, Any]] = []
    for product in products:
        normalized_products.append(
            {
                "product_sys_id": str(product.get("product_sys_id", "")).strip(),
                "name": str(product.get("name", "")).strip(),
                "description": str(product.get("description", "")).strip(),
                "description_text": str(product.get("description_text", "")).strip(),
                "sellPrice": float(product.get("price", 0.0) or 0.0),
                "stock": int(product.get("stock", 0) or 0),
                "status": str(product.get("status", "")).strip(),
                "category": str(product.get("category", "")).strip(),
                "brand": str(product.get("brand", "")).strip(),
                "specs_text": str(product.get("specs_text", "")).strip(),
            }
        )

    df = pd.DataFrame(normalized_products)
    if df.empty:
        df_products = df
        similarity_matrix = np.empty((0, 0), dtype=np.float32)
        product_sys_id_to_index = {}
        product_profiles_by_id = {}
        return True

    product_sys_id_to_index = {
        str(pid).strip(): idx
        for idx, pid in enumerate(product_ids)
        if str(pid).strip()
    }
    df_products = df
    similarity_matrix = np.matmul(vector_matrix, vector_matrix.T)
    product_profiles_by_id = {
        str(row["product_sys_id"]).strip(): {
            "product_id": str(row["product_sys_id"]).strip(),
            "name": str(row["name"]).strip(),
            "category": str(row["category"]).strip(),
            "brand": str(row["brand"]).strip(),
            "price": float(row["sellPrice"]),
            "stock": int(row["stock"]),
            "status": str(row["status"]).strip(),
            "description": str(row["description"]).strip(),
            "description_text": str(row.get("description_text", "")).strip(),
            "specs_text": str(row["specs_text"]).strip(),
        }
        for _, row in df.iterrows()
    }
    print(f"Content-based Ready from artifacts: {df.shape[0]} products.")
    return True


def load_all_data():
    if _load_all_data_from_artifacts():
        return
    _load_all_data_from_database()


def _combine_product_features(row: pd.Series) -> str:
    name = str(row.get("name", "")).strip()
    category = str(row.get("category", "")).strip()
    brand = str(row.get("brand", "")).strip()
    specs_text = str(row.get("specs_text", "")).strip()
    description_text = str(row.get("description_text", "")).strip() or str(row.get("description", "")).strip()
    return ((name + " ") * 5 + (category + " ") * 4 + (brand + " ") * 2 + specs_text + " " + description_text).strip()


def _load_all_data_from_database():
    global df_products, similarity_matrix, product_sys_id_to_index, product_profiles_by_id

    query = """
    SELECT
        p.product_sys_id, p.name, p.description, p.sellPrice, p.stock, p.status,
        ct.name AS category, b.name AS brand,
        STRING_AGG(CONCAT(s.name, ' ', svl.value), ' ') AS specs_text
    FROM Product p
    JOIN Brand b ON p.brandId = b.BrandId
    JOIN Category ct ON p.category_id = ct.category_id
    LEFT JOIN SpecValue svl ON p.product_id = svl.product_id
    LEFT JOIN Specs s ON svl.spec_id = s.spec_id
    GROUP BY p.product_sys_id, p.name, p.description, p.sellPrice, p.stock, p.status, ct.name, b.name
    """

    df = pd.read_sql(query, engine)
    df.fillna("", inplace=True)
    df["sellPrice"] = pd.to_numeric(df["sellPrice"], errors="coerce").fillna(0).astype(float)
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
    df["product_sys_id"] = df["product_sys_id"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["description"] = df["description"].astype(str).str.strip()
    df["description_text"] = df["description"]
    df["category"] = df["category"].astype(str).str.strip()
    df["brand"] = df["brand"].astype(str).str.strip()
    df["specs_text"] = df["specs_text"].astype(str).str.strip()
    df = df[df["product_sys_id"] != ""].reset_index(drop=True)

    if df.empty:
        df_products = df
        similarity_matrix = np.empty((0, 0), dtype=np.float32)
        product_sys_id_to_index = {}
        product_profiles_by_id = {}
        return

    feature_texts = df.apply(_combine_product_features, axis=1).tolist()
    raw_matrix = _hashing_vectorizer.transform(feature_texts)
    normalized_matrix = _normalizer.fit_transform(raw_matrix).toarray().astype(np.float32)

    df_products = df
    similarity_matrix = np.matmul(normalized_matrix, normalized_matrix.T)
    product_sys_id_to_index = {pid: idx for idx, pid in enumerate(df["product_sys_id"])}
    product_profiles_by_id = {
        str(row["product_sys_id"]).strip(): {
            "product_id": str(row["product_sys_id"]).strip(),
            "name": str(row["name"]).strip(),
            "category": str(row["category"]).strip(),
            "brand": str(row["brand"]).strip(),
            "price": float(row["sellPrice"]),
            "stock": int(row["stock"]),
            "status": str(row["status"]).strip(),
            "description": str(row["description"]).strip(),
            "description_text": str(row.get("description_text", "")).strip(),
            "specs_text": str(row["specs_text"]).strip(),
        }
        for _, row in df.iterrows()
    }
    print(f"Content-based Ready from database: {df.shape[0]} products.")


def recommend(
    product_sys_id: str,
    top_n: int = 5,
    price_margin: float = 0.8,
):
    if product_sys_id_to_index is None or df_products is None or similarity_matrix is None:
        load_all_data()

    normalized_product_id = str(product_sys_id).strip()
    if normalized_product_id not in product_sys_id_to_index:
        return []

    idx = product_sys_id_to_index[normalized_product_id]
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
    top_local_indices = np.argsort(scores)[::-1][:top_n]
    final_indices = eligible_indices[top_local_indices]
    return df_products.iloc[final_indices]["product_sys_id"].tolist()


def get_product_profiles(
    product_ids: list[str],
    candidate_scores: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    if product_profiles_by_id is None:
        load_all_data()

    normalized_ids = [str(pid).strip() for pid in product_ids if str(pid).strip()]
    ordered_profiles: list[dict[str, Any]] = []
    for product_id in normalized_ids:
        base_profile = product_profiles_by_id.get(product_id) if product_profiles_by_id else None
        if not base_profile:
            continue
        profile = dict(base_profile)
        if candidate_scores is not None:
            profile["base_score"] = round(float(candidate_scores.get(product_id, 0.0)), 4)
        ordered_profiles.append(profile)
    return ordered_profiles
