# app/api/engine/content_based.py

import pandas as pd
import numpy as np
from db.mssql import engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

# Cache global
df_products = None
similarity_matrix = None
product_sys_id_to_index = None

def load_all_data():
    global df_products, similarity_matrix, product_sys_id_to_index
    
    # 1. LOAD DATA & PREPROCESSING
    query = """
    SELECT 
        p.product_sys_id, p.name, p.sellPrice, p.stock,
        ct.name AS category, b.name AS brand,
        STRING_AGG(CONCAT(s.name, ' ', svl.value), ' ') AS specs_text
    FROM Product p
    JOIN Brand b ON p.brandId = b.BrandId
    JOIN Category ct ON p.category_id = ct.category_id
    LEFT JOIN SpecValue svl ON p.product_id = svl.ProductId
    LEFT JOIN Specs s ON svl.SpecId = s.spec_id
    GROUP BY p.product_sys_id, p.name, p.sellPrice, p.stock, ct.name, b.name
    """
    
    df = pd.read_sql(query, engine)
    df.fillna("", inplace=True)
    
    # Ép kiểu dữ liệu chuẩn xác để tính toán nhanh
    df['sellPrice'] = pd.to_numeric(df['sellPrice'], errors='coerce').fillna(0).astype(float)
    df['stock'] = pd.to_numeric(df['stock'], errors='coerce').fillna(0).astype(int)
    df['product_sys_id'] = df['product_sys_id'].astype(str).str.strip()
    
    # Loại bỏ sản phẩm không có ID hoặc lỗi dữ liệu
    df = df[df['product_sys_id'] != ""].reset_index(drop=True)

    # 2. FEATURE ENGINEERING (Tăng cường trọng số văn bản)
    df['combined_features'] = (
        (df['name'] + ' ') * 5 + 
        (df['category'] + ' ') * 4 + 
        (df['brand'] + ' ') * 2 + 
        df['specs_text']
    )

    # 3. TF-IDF + NORMALIZATION
    tfidf = TfidfVectorizer(stop_words='english', min_df=2) # Loại bỏ từ quá hiếm để giảm nhiễu
    tf_matrix = tfidf.fit_transform(df['combined_features'])
    
    # Chuẩn hóa vector giúp điểm Cosine chính xác hơn
    normalizer = Normalizer()
    tf_matrix = normalizer.fit_transform(tf_matrix)
    
    similarity_matrix = cosine_similarity(tf_matrix)

    # 4. CACHE DATA
    df_products = df
    product_sys_id_to_index = {pid: idx for idx, pid in enumerate(df['product_sys_id'])}
    
    print(f"Content-based Ready: {df.shape[0]} products.")

# Load data khi khởi tạo
load_all_data()

# 5. OPTIMIZED RECOMMEND FUNCTION
def recommend(product_sys_id: str, top_n: int = 5, price_margin: float = 0.3):
    if product_sys_id not in product_sys_id_to_index:
        return []

    idx = product_sys_id_to_index[product_sys_id]
    current_item = df_products.iloc[idx]
    
    current_price = float(current_item['sellPrice'])
    current_brand = current_item['brand']
    current_cat = current_item['category']
    
    # 1. TÍNH KHOẢNG GIÁ (Sai số tối đa 30%)
    lower_bound = current_price * (1 - price_margin)
    upper_bound = current_price * (1 + price_margin)

    # 2. TẠO BỘ LỌC (MASK)
    # Lọc: Có hàng + Trong tầm giá + Không phải chính nó
    mask = (
        (df_products['stock'] > 0) & 
        (df_products['sellPrice'] >= lower_bound) & 
        (df_products['sellPrice'] <= upper_bound)
    )
    
    eligible_indices = np.where(mask)[0]
    eligible_indices = eligible_indices[eligible_indices != idx]

    if len(eligible_indices) == 0:
        return []

    # 3. TÍNH ĐIỂM VÀ ƯU TIÊN (BRAND & CATEGORY)
    # Lấy điểm tương đồng gốc
    scores = similarity_matrix[idx][eligible_indices]
    
    # Lấy thông tin Brand và Cat của những thằng đủ điều kiện
    eligible_df = df_products.iloc[eligible_indices]
    
    # Thưởng điểm: +0.2 nếu cùng Category, +0.1 nếu cùng Brand
    bonus = (
        (eligible_df['category'] == current_cat).astype(float) * 0.2 +
        (eligible_df['brand'] == current_brand).astype(float) * 0.1
    )
    
    final_scores = scores + bonus.values

    # 4. LẤY TOP KẾT QUẢ
    top_local_indices = np.argsort(final_scores)[::-1][:top_n]
    final_indices = eligible_indices[top_local_indices]

    return df_products.iloc[final_indices]['product_sys_id'].tolist()