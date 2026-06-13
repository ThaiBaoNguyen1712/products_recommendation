import json
import os
import html
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import Normalizer
from sqlalchemy import text
from sqlalchemy.engine import Engine

REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLED_DATA_DIR = REPO_ROOT / "data"
RUNTIME_DATA_DIR = Path(os.getenv("RUNTIME_DATA_DIR", "/tmp/products-rcm-sys-api/data")).resolve()
DATA_DIR = RUNTIME_DATA_DIR if os.getenv("VERCEL", "").strip() == "1" else BUNDLED_DATA_DIR
PRODUCTS_PATH = DATA_DIR / "products.json"
PRODUCT_IDS_PATH = DATA_DIR / "product_ids.json"
PRODUCT_VECTORS_PATH = DATA_DIR / "product_vectors.npy"
INDEX_META_PATH = DATA_DIR / "index_meta.json"
ACCESSORY_RULES_PATH = DATA_DIR / "accessory_rules.json"
COMPATIBILITY_RULES_PATH = DATA_DIR / "compatibility_rules.json"
OFFLINE_RERANK_SCORES_PATH = DATA_DIR / "offline_rerank_scores.json"
LEGACY_BEHAVIOR_SCORES_PATH = DATA_DIR / "behavior_scores.json"

HASH_VECTOR_DIM = int(os.getenv("HASH_VECTOR_DIM", "2048"))

DEFAULT_ACCESSORY_RULES = {
    "gaming_laptop": ["gaming_mouse", "mechanical_keyboard", "cooling_pad", "gaming_headset", "monitor"],
    "office_laptop": ["wireless_mouse", "laptop_bag", "usb_hub", "monitor"],
    "iphone": ["phone_case", "screen_protector", "charger", "airpods", "power_bank"],
    "android_phone": ["phone_case", "screen_protector", "charger", "earbuds", "power_bank"],
    "monitor": ["hdmi_cable", "displayport_cable", "monitor_arm", "keyboard", "mouse"],
    "desktop_pc": ["monitor", "keyboard", "mouse", "speaker", "ups"],
}

DEFAULT_COMPATIBILITY_RULES = {
    "meta": {
        "schema_version": 1,
        "description": "Data-driven cart compatibility filters. Domain-specific terms live in data, not Python code.",
    },
    "rules": {
        "dien_thoai": {
            "target_categories": ["phu_kien", "dong_ho_thong_minh"],
            "target_category_limits": {"phu_kien": 24, "dong_ho_thong_minh": 3},
            "include_any": [
                "airpods", "tai_nghe", "earbuds", "buds", "op_lung", "magsafe",
                "sac", "cu_sac", "sac_du_phong", "cap", "cable", "pin_du_phong", "cuong_luc", "kinh_cuong_luc",
                "screen", "protector", "adapter", "dong_ho", "watch"
            ],
            "exclude_any": [
                "camera", "may_in", "man_hinh", "screenbar", "den_man_hinh", "day_deo",
                "bon_ngam", "hut_bui", "loc_khong_khi"
            ]
        },
        "may_tinh_bang": {
            "target_categories": ["phu_kien"],
            "target_category_limits": {"phu_kien": 24},
            "include_any": [
                "airpods", "tai_nghe", "ban_phim", "but", "pencil", "hub", "gia_do",
                "op_lung", "bao_da", "sac", "cu_sac", "cap", "adapter", "pin_du_phong",
                "kinh_cuong_luc"
            ],
            "exclude_any": ["camera", "bon_ngam", "hut_bui", "loc_khong_khi", "iphone", "magsafe"]
        },
        "laptop": {
            "target_categories": ["phu_kien", "man_hinh", "may_in"],
            "target_category_limits": {"phu_kien": 18, "man_hinh": 4, "may_in": 3},
            "include_any": [
                "chuot", "mouse", "ban_phim", "keyboard", "usb", "hub", "cap",
                "cable", "tai_nghe", "headset", "loa", "cooling", "de_tan_nhiet",
                "balo", "tui_chong_soc", "adapter"
            ],
            "exclude_any": ["bon_ngam", "hut_bui", "loc_khong_khi"]
        },
        "may_tinh_de_ban": {
            "target_categories": ["phu_kien", "man_hinh", "may_in"],
            "target_category_limits": {"phu_kien": 18, "man_hinh": 4, "may_in": 3},
            "include_any": [
                "chuot", "mouse", "ban_phim", "keyboard", "usb", "hub", "cap",
                "cable", "tai_nghe", "headset", "loa", "adapter"
            ],
            "exclude_any": ["bon_ngam", "hut_bui", "loc_khong_khi"]
        },
        "man_hinh": {
            "target_categories": ["phu_kien"],
            "target_category_limits": {"phu_kien": 24},
            "include_any": [
                "gia_treo", "arm", "chan_de", "hdmi", "displayport", "cap",
                "cable", "den_man_hinh", "screenbar", "ve_sinh_man_hinh",
                "hub", "adapter"
            ],
            "exclude_any": [
                "op_lung", "iphone", "ipad", "tablet", "may_tinh_bang", "lightning", "sac",
                "bon_ngam", "hut_bui", "loc_khong_khi", "noi_chien"
            ]
        }
    }
}

_hashing_vectorizer = HashingVectorizer(
    n_features=HASH_VECTOR_DIM,
    alternate_sign=False,
    stop_words="english",
    norm=None,
)
_normalizer = Normalizer()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_DIR == BUNDLED_DATA_DIR:
        return

    for bundled_path in BUNDLED_DATA_DIR.glob("*"):
        target_path = DATA_DIR / bundled_path.name
        if target_path.exists():
            continue
        if bundled_path.is_file():
            shutil.copy2(bundled_path, target_path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_text(value: Any, max_len: int | None = None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = " ".join(str(value).split())
    if max_len is not None:
        return text[:max_len]
    return text


def _clean_html_text(value: Any, max_len: int | None = None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"<[^>]*", " ", text)
    text = re.sub(r"\b(?:class|style|href|src|id|data-[\w-]+)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s]+)?", " ", text)
    text = re.sub(r"\b(?:div|span|p|br|strong|em|ul|ol|li|table|tbody|thead|tr|td|th|h[1-6]|img|a)\b", " ", text)
    text = text.replace("<", " ").replace(">", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if max_len is not None:
        return text[:max_len]
    return text


def _fetch_products_df(db_engine: Engine) -> pd.DataFrame:
    base_query = """
    SELECT
        p.product_id,
        p.product_sys_id,
        p.name,
        p.description,
        p.sellPrice,
        p.stock,
        p.status,
        ct.name AS category,
        b.name AS brand
    FROM Product p
    JOIN Brand b ON p.brandId = b.BrandId
    JOIN Category ct ON p.category_id = ct.category_id
    """
    specs_query = """
    SELECT
        svl.product_id,
        STRING_AGG(CONCAT(s.name, ' ', svl.value), ' ') AS specs_text
    FROM SpecValue svl
    JOIN Specs s ON svl.spec_id = s.spec_id
    GROUP BY svl.product_id
    """

    base_df = pd.read_sql(base_query, db_engine)
    specs_df = pd.read_sql(specs_query, db_engine)
    if not specs_df.empty:
        base_df = base_df.merge(specs_df, on="product_id", how="left")
    if "specs_text" not in base_df.columns:
        base_df["specs_text"] = ""
    return _normalize_products_df(base_df)


def _fetch_product_df_by_id(db_engine: Engine, product_sys_id: str) -> pd.DataFrame:
    base_query = text("""
    SELECT
        p.product_id,
        p.product_sys_id,
        p.name,
        p.description,
        p.sellPrice,
        p.stock,
        p.status,
        ct.name AS category,
        b.name AS brand
    FROM Product p
    JOIN Brand b ON p.brandId = b.BrandId
    JOIN Category ct ON p.category_id = ct.category_id
    WHERE p.product_sys_id = :product_sys_id
    """)
    specs_query = text("""
    SELECT
        svl.product_id,
        STRING_AGG(CONCAT(s.name, ' ', svl.value), ' ') AS specs_text
    FROM SpecValue svl
    JOIN Specs s ON svl.spec_id = s.spec_id
    JOIN Product p ON p.product_id = svl.product_id
    WHERE p.product_sys_id = :product_sys_id
    GROUP BY svl.product_id
    """)
    params = {"product_sys_id": str(product_sys_id).strip()}
    base_df = pd.read_sql(base_query, db_engine, params=params)
    specs_df = pd.read_sql(specs_query, db_engine, params=params)
    if not specs_df.empty:
        base_df = base_df.merge(specs_df, on="product_id", how="left")
    if "specs_text" not in base_df.columns:
        base_df["specs_text"] = ""
    return _normalize_products_df(base_df)


def _normalize_products_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df.fillna("", inplace=True)
    df["sellPrice"] = pd.to_numeric(df["sellPrice"], errors="coerce").fillna(0).astype(float)
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(0).astype(int)
    df["product_sys_id"] = df["product_sys_id"].astype(str).str.strip()
    df = df[df["product_sys_id"] != ""].drop_duplicates(subset=["product_sys_id"]).reset_index(drop=True)
    if "product_id" in df.columns:
        df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").fillna(0).astype(int)
    df["name"] = df["name"].map(lambda value: _clean_text(value, 200))
    df["description"] = df["description"].map(lambda value: _clean_text(value, 1500))
    df["status"] = df["status"].map(lambda value: _clean_text(value, 60))
    df["category"] = df["category"].map(lambda value: _clean_text(value, 120))
    df["brand"] = df["brand"].map(lambda value: _clean_text(value, 120))
    df["specs_text"] = df["specs_text"].map(lambda value: _clean_text(value, 2000))
    return df


def _combine_product_features(product: dict[str, Any]) -> str:
    name = str(product.get("name", "")).strip()
    category = str(product.get("category", "")).strip()
    brand = str(product.get("brand", "")).strip()
    specs_text = str(product.get("specs_text", "")).strip()
    description = str(product.get("description_text", "")).strip()
    return ((name + " ") * 5 + (category + " ") * 4 + (brand + " ") * 2 + specs_text + " " + description).strip()


def _vectorize_products(products: list[dict[str, Any]]) -> np.ndarray:
    if not products:
        return np.empty((0, HASH_VECTOR_DIM), dtype=np.float32)
    feature_texts = [_combine_product_features(product) for product in products]
    raw_matrix = _hashing_vectorizer.transform(feature_texts)
    normalized_matrix = _normalizer.fit_transform(raw_matrix).toarray().astype(np.float32)
    return normalized_matrix


def _serialize_products(products_df: pd.DataFrame) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    for _, row in products_df.iterrows():
        products.append(
            {
                "product_sys_id": str(row["product_sys_id"]).strip(),
                "name": _clean_text(row["name"], 200),
                "description": _clean_text(row["description"], 4000),
                "description_text": _clean_html_text(row["description"], 1500),
                "category": _clean_text(row["category"], 120),
                "brand": _clean_text(row["brand"], 120),
                "price": float(row["sellPrice"]),
                "stock": int(row["stock"]),
                "status": _clean_text(row["status"], 60),
                "specs_text": _clean_text(row["specs_text"], 2000),
                "updated_at": _utc_now_iso(),
            }
        )
    return products


def _write_default_accessory_rules() -> None:
    if not ACCESSORY_RULES_PATH.exists():
        _write_json(ACCESSORY_RULES_PATH, DEFAULT_ACCESSORY_RULES)


def _write_default_compatibility_rules() -> None:
    if not COMPATIBILITY_RULES_PATH.exists():
        _write_json(COMPATIBILITY_RULES_PATH, DEFAULT_COMPATIBILITY_RULES)


def _write_default_offline_rerank_scores() -> None:
    if not OFFLINE_RERANK_SCORES_PATH.exists():
        _write_json(OFFLINE_RERANK_SCORES_PATH, {"meta": {"status": "empty"}, "homepage": {}, "wishlist": {}, "cart": {}})


def _load_products_file() -> list[dict[str, Any]]:
    if not PRODUCTS_PATH.exists():
        return []
    try:
        payload = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _load_vectors_file() -> np.ndarray:
    if not PRODUCT_VECTORS_PATH.exists():
        return np.empty((0, HASH_VECTOR_DIM), dtype=np.float32)
    return np.load(PRODUCT_VECTORS_PATH, allow_pickle=False)


def _prune_offline_rerank_scores(removed_product_id: str) -> None:
    if not OFFLINE_RERANK_SCORES_PATH.exists():
        return
    try:
        payload = json.loads(OFFLINE_RERANK_SCORES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(payload, dict):
        return

    for scene in ("homepage", "wishlist", "cart"):
        scene_payload = payload.get(scene)
        if not isinstance(scene_payload, dict):
            continue
        scene_payload.pop(removed_product_id, None)
        for source_id, candidates in list(scene_payload.items()):
            if not isinstance(candidates, dict):
                continue
            candidates.pop(removed_product_id, None)
    _write_json(OFFLINE_RERANK_SCORES_PATH, payload)


def _write_index_files(
    *,
    products: list[dict[str, Any]],
    vectors: np.ndarray,
    trigger: str,
    action: str | None,
    product_sys_id: str | None,
) -> dict[str, Any]:
    product_ids = [str(product["product_sys_id"]).strip() for product in products]
    _write_json(PRODUCTS_PATH, products)
    _write_json(PRODUCT_IDS_PATH, product_ids)
    np.save(PRODUCT_VECTORS_PATH, vectors.astype(np.float32))
    if LEGACY_BEHAVIOR_SCORES_PATH.exists():
        LEGACY_BEHAVIOR_SCORES_PATH.unlink()
    _write_default_accessory_rules()
    _write_default_compatibility_rules()
    _write_default_offline_rerank_scores()

    index_meta = {
        "status": "ready",
        "artifact_type": "file_based_vector_store",
        "vector_provider": "hashing_normalized",
        "vector_format": "numpy_float32",
        "vector_dim": int(vectors.shape[1]) if vectors.ndim == 2 else HASH_VECTOR_DIM,
        "hash_features": HASH_VECTOR_DIM,
        "product_count": int(len(products)),
        "trigger": trigger,
        "sync_action": action or "rebuild",
        "synced_product_sys_id": str(product_sys_id).strip() if product_sys_id else None,
        "built_at": _utc_now_iso(),
        "files": {
            "products": _display_path(PRODUCTS_PATH),
            "product_ids": _display_path(PRODUCT_IDS_PATH),
            "product_vectors": _display_path(PRODUCT_VECTORS_PATH),
            "accessory_rules": _display_path(ACCESSORY_RULES_PATH),
            "compatibility_rules": _display_path(COMPATIBILITY_RULES_PATH),
            "offline_rerank_scores": _display_path(OFFLINE_RERANK_SCORES_PATH),
        },
    }
    _write_json(INDEX_META_PATH, index_meta)
    return index_meta


def build_file_index(
    db_engine: Engine,
    *,
    trigger: str = "manual",
    product_sys_id: str | None = None,
    action: str | None = None,
) -> dict[str, Any]:
    _ensure_data_dir()
    products_df = _fetch_products_df(db_engine)
    products = _serialize_products(products_df)
    vectors = _vectorize_products(products)
    return _write_index_files(
        products=products,
        vectors=vectors,
        trigger=trigger,
        action=action,
        product_sys_id=product_sys_id,
    )


def ensure_file_index(db_engine: Engine) -> dict[str, Any]:
    _ensure_data_dir()
    required_paths = [
        PRODUCTS_PATH,
        PRODUCT_IDS_PATH,
        PRODUCT_VECTORS_PATH,
        INDEX_META_PATH,
    ]
    _write_default_accessory_rules()
    _write_default_compatibility_rules()
    _write_default_offline_rerank_scores()
    if all(path.exists() for path in required_paths):
        return read_index_status()
    return build_file_index(db_engine, trigger="startup_bootstrap")


def sync_product_index(db_engine: Engine, product_sys_id: str, action: str) -> dict[str, Any]:
    _ensure_data_dir()
    normalized_product_id = str(product_sys_id).strip()
    normalized_action = str(action).strip().lower()

    if not all(path.exists() for path in [PRODUCTS_PATH, PRODUCT_IDS_PATH, PRODUCT_VECTORS_PATH]):
        return build_file_index(
            db_engine,
            trigger="sync_product_bootstrap",
            product_sys_id=normalized_product_id,
            action=normalized_action,
        )

    products = _load_products_file()
    vectors = _load_vectors_file()
    product_index_map = {
        str(product.get("product_sys_id", "")).strip(): idx
        for idx, product in enumerate(products)
        if str(product.get("product_sys_id", "")).strip()
    }

    if normalized_action == "delete":
        existing_index = product_index_map.get(normalized_product_id)
        if existing_index is not None:
            products.pop(existing_index)
            vectors = np.delete(vectors, existing_index, axis=0)
        _prune_offline_rerank_scores(normalized_product_id)
        return _write_index_files(
            products=products,
            vectors=vectors,
            trigger="sync_product_incremental",
            action=normalized_action,
            product_sys_id=normalized_product_id,
        )

    product_df = _fetch_product_df_by_id(db_engine, normalized_product_id)
    if product_df.empty:
        return build_file_index(
            db_engine,
            trigger="sync_product_fallback_rebuild",
            product_sys_id=normalized_product_id,
            action=normalized_action,
        )

    product_record = _serialize_products(product_df)[0]
    product_vector = _vectorize_products([product_record])

    existing_index = product_index_map.get(normalized_product_id)
    if existing_index is None:
        products.append(product_record)
        vectors = np.vstack([vectors, product_vector]) if vectors.size else product_vector
    else:
        products[existing_index] = product_record
        vectors[existing_index] = product_vector[0]

    return _write_index_files(
        products=products,
        vectors=vectors,
        trigger="sync_product_incremental",
        action=normalized_action,
        product_sys_id=normalized_product_id,
    )


def read_index_status() -> dict[str, Any]:
    _ensure_data_dir()
    _write_default_accessory_rules()
    _write_default_compatibility_rules()
    _write_default_offline_rerank_scores()

    file_status = {
        "products": PRODUCTS_PATH.exists(),
        "product_ids": PRODUCT_IDS_PATH.exists(),
        "product_vectors": PRODUCT_VECTORS_PATH.exists(),
        "index_meta": INDEX_META_PATH.exists(),
        "accessory_rules": ACCESSORY_RULES_PATH.exists(),
        "compatibility_rules": COMPATIBILITY_RULES_PATH.exists(),
        "offline_rerank_scores": OFFLINE_RERANK_SCORES_PATH.exists(),
    }

    if INDEX_META_PATH.exists():
        try:
            metadata = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"status": "corrupt", "built_at": None}
    else:
        metadata = {"status": "missing", "built_at": None}

    metadata["data_dir"] = _display_path(DATA_DIR)
    metadata["files_exist"] = file_status
    return metadata
