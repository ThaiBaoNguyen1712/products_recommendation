# app/api/engine/content_based.py

import pandas as pd
from db.mssql import engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Định nghĩa các biến global để các module khác (như hybrid.py) có thể import và dùng chung
df_products = None
similarity_matrix = None
product_sys_id_to_index = None

def load_all_data():
    global df_products, similarity_matrix, product_sys_id_to_index
    
    # 1. LOAD DATA FROM SQL
    query = """
    SELECT 
        p.product_sys_id,
        p.name,
        p.sellPrice,
        p.stock,
        ct.name AS category,
        b.name AS brand,
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
    
    # Làm sạch ID
    df['product_sys_id'] = df['product_sys_id'].astype(str).str.strip()
    df = df[df['product_sys_id'] != ""].reset_index(drop=True)

    # 2. FEATURE ENGINEERING (Trọng số đã được cải tiến)
    df['combined_features'] = (
        (df['name'] + ' ') * 5 +         # Tên là quan trọng nhất
        (df['category'] + ' ') * 4 +     # Ép cùng loại sản phẩm
        (df['brand'] + ' ') * 2 +        # Thương hiệu
        df['specs_text']                 # Thông số kỹ thuật
    )

    # 3. TF-IDF + COSINE
    tfidf = TfidfVectorizer(stop_words='english')
    tf_matrix = tfidf.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(tf_matrix)

    # 4. MAP & CACHE
    # Lưu df vào biến global, set index là product_sys_id để hybrid.py tra cứu cực nhanh
    df_products = df.set_index('product_sys_id')
    
    product_sys_id_to_index = {
        pid: idx for idx, pid in enumerate(df['product_sys_id'])
    }
    
    print("Content-based Engine Loaded: ", df.shape[0], " products.")

# Thực hiện load ngay khi module được import lần đầu
load_all_data()

# 5. RECOMMEND FUNCTION
def recommend(product_sys_id: str, top_n: int = 5):
    if product_sys_id not in product_sys_id_to_index:
        return []

    product_index = product_sys_id_to_index[product_sys_id]

    # Lấy điểm tương đồng từ ma trận
    scores = list(enumerate(similarity_matrix[product_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Lấy dư ra (top_n * 2) -> Hybrid có thể lọc Category/Price mà không bị thiếu
    scores = scores[1: (top_n * 2) + 1]

    # Trả về list product_sys_id
    return [
        df_products.iloc[idx].name # .name ở đây chính là product_sys_id do ta set_index
        for idx, _ in scores
    ]