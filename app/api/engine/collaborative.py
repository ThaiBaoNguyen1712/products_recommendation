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
        # Giữ n_components thấp hơn số lượng sản phẩm (1k sản phẩm -> 50-100 là ổn)
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

    def get_recommendations(self, user_id: int, top_n: int = 10):
        try:
            # Tìm vị trí của user trong danh sách đã train
            user_idx = self.user_ids.index(user_id)
            
            # Lấy vector dự đoán của user này (chứa score cho từng product_sys_id)
            user_predictions = self.predicted_matrix[user_idx]
            
            # Gắn ID vào điểm số để sắp xếp
            preds_series = pd.Series(user_predictions, index=self.product_sys_ids)
            
            # Lấy Top N sản phẩm có score cao nhất (prd_...)
            recommendations = preds_series.sort_values(ascending=False).head(top_n).index.tolist()
            
            return recommendations
            
        except (ValueError, IndexError):
            # Trường hợp User mới (Cold Start): Gợi ý các sản phẩm có nhiều rating nhất (Popularity)
            print(f"--- User {user_id} not found. Returning popular items. ---")
            return self._get_popular_items(top_n)

    def _get_popular_items(self, top_n):
        # Đây là fallback đơn giản: lấy N sản phẩm đầu tiên trong danh sách 
        # (Lý tưởng nhất là bạn nên lấy từ một list 'Hot Products' đã tính trước)
        return self.product_sys_ids[:top_n]