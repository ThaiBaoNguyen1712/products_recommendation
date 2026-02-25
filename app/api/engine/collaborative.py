import pandas as pd
import numpy as np
import pickle
import os
from sklearn.decomposition import TruncatedSVD

class CollaborativeFiltering:
    def __init__(self, data_path='db/csv_files/ratings.csv', model_path='models/collaborative_model.pkl'):
        self.data_path = data_path
        self.model_path = model_path
        self.predicted_matrix = None
        self.product_sys_ids = None # Lưu danh sách ID sản phẩm (string)
        self.user_ids = None
        
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.predicted_matrix = data['matrix']
                    self.product_sys_ids = data['product_sys_ids']
                    self.user_ids = data['user_ids']
                print("--- Model loaded from disk ---")
            except Exception as e:
                print(f"--- Error loading model: {e}. Training new one... ---")
                self._train_model()
        else:
            self._train_model()

    def _train_model(self):
        print("--- Training new model with Sklearn TruncatedSVD... ---")
        # Load dữ liệu từ CSV (hoặc có thể query trực tiếp từ engine SQL của bạn)
        ratings = pd.read_csv(self.data_path)
        #Ép kiểu về string cho product_sys_id
        ratings['product_sys_id'] = ratings['product_sys_id'].astype(str)
        # Loại bỏ các dòng bị thiếu ID system
        ratings = ratings.dropna(subset=['product_sys_id'])
        # 1. Tạo Pivot Table sử dụng product_sys_id (Dạng chuỗi prd_...)
        # Index là user_id, Columns là product_sys_id
        matrix = ratings.pivot(index='user_id', columns='product_sys_id', values='rating').fillna(0)
        
        self.product_sys_ids = matrix.columns.tolist() # Danh sách prd_...
        self.user_ids = matrix.index.tolist()

        # 2. Matrix Factorization (SVD)
        n_comp = min(50, len(self.product_sys_ids) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        
        user_factors = svd.fit_transform(matrix)
        item_factors = svd.components_
        
        # 3. Tính toán ma trận dự đoán (Predicted Ratings)
        self.predicted_matrix = np.dot(user_factors, item_factors)
        
        # 4. Lưu lại toàn bộ thông tin cần thiết
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'matrix': self.predicted_matrix,
                'product_sys_ids': self.product_sys_ids,
                'user_ids': self.user_ids
            }, f)
        print(f"--- Training complete. Matrix shape: {self.predicted_matrix.shape} ---")

    def get_recommendations(self, user_id: int, product_sys_id: str = None, top_n: int = 50):
        try:
            # 1. Tìm vị trí user
            user_idx = self.user_ids.index(user_id)
            user_predictions = self.predicted_matrix[user_idx]
            preds_series = pd.Series(user_predictions, index=self.product_sys_ids)

            # 2. LỌC THEO GIÁ VÀ CATEGORY NGAY TẠI ĐÂY (NẾU CÓ PRODUCT_SYS_ID)
            if product_sys_id and self.products_info is not None:
                # Tìm thông tin món đang xem
                curr_item = self.products_info[self.products_info['product_sys_id'] == str(product_sys_id)]
                
                if not curr_item.empty:
                    curr_price = curr_item['sellPrice'].values[0]
                    curr_cat = curr_item['category_id'].values[0]

                    # Lọc danh sách ID hợp lệ: Cùng category và giá trong khoảng 50% - 150%
                    valid_items = self.products_info[
                        (self.products_info['category_id'] == curr_cat) & 
                        (self.products_info['sellPrice'] >= curr_price * 0.5) & 
                        (self.products_info['sellPrice'] <= curr_price * 1.5)
                    ]['product_sys_id'].tolist()

                    # Giữ lại những dự đoán nằm trong tập valid
                    preds_series = preds_series[preds_series.index.isin(valid_items)]

            # 3. Lấy kết quả
            buffer_n = max(top_n, 50) 
            recommendations = preds_series.sort_values(ascending=False).head(buffer_n).index.tolist()
            
            # Nếu lọc xong bị ít quá, có thể lấy thêm popular bù vào (tùy chọn)
            return recommendations
            
        except (ValueError, IndexError):
            print(f"--- User {user_id} not found. Returning popular items. ---")
            return self._get_popular_items(top_n)

    def _get_popular_items(self, top_n):
        """Sử dụng SQL để lấy top sản phẩm có lượt rating cao nhất"""
        try:
            query = """
                SELECT TOP {} p.product_sys_id
                FROM Product p
                JOIN (SELECT product_sys_id, COUNT(*) as vote_count FROM Ratings GROUP BY product_sys_id) r 
                ON p.product_sys_id = r.product_sys_id
                ORDER BY r.vote_count DESC
            """.format(top_n)
            
            pop_df = pd.read_sql(query, self.engine)
            return pop_df['product_sys_id'].astype(str).tolist()
        except:
            # Fallback nếu SQL lỗi
            return self.product_sys_ids[:top_n]