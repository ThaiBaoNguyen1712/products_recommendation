import pandas as pd
from db.mssql import engine
from app.api.engine.content_based import recommend
import upstash_redis as redis
from dotenv import load_dotenv
import os
load_dotenv()

class SceneRecommendationFilter:
    def __init__(self):
        self.engine = engine
        # Kết nối Upstash Redis
        self.redis_client = redis.Redis(url=os.getenv("UPSTASH_URL"), token=os.getenv("UPSTASH_TOKEN"))

    def get_exclude_idx(self, user_id: int):
        """Lấy set các SP đã mua và trong giỏ hàng để lọc."""
        purchased_query = "SELECT DISTINCT p.product_sys_id FROM OrderItem oi JOIN [Order] o ON oi.order_id = o.order_id JOIN Product p ON oi.product_id = p.product_id WHERE o.user_id = ?"
        cart_query = "SELECT DISTINCT p.product_sys_id FROM CartItem ci JOIN Cart c ON ci.cart_id = c.cart_id JOIN Product p ON ci.product_id = p.product_id WHERE c.user_id = ?"
        
        purchased_df = pd.read_sql(purchased_query, self.engine, params=(user_id,))
        cart_df = pd.read_sql(cart_query, self.engine, params=(user_id,))
        
        return (
            set(purchased_df['product_sys_id'].astype(str).str.strip()), 
            set(cart_df['product_sys_id'].astype(str).str.strip())
        )

    def get_recommendations_homepage(self, user_id: int, top_n: int = 50):
        """
        Endpoint 1: Homepage Personalization
        Lấy từ listKey = user:{userId}:latest_watched
        """
        list_key = f"user:{user_id}:latest_watched" if user_id else f"guest:{user_id}:latest_watched"
        
        if list_key is None:
            return []  # Hoặc trả về sản phẩm trending nếu user mới
        
        # 1. Lấy 10 sản phẩm xem gần nhất (List trong Redis)
        # lrange trả về list các byte/string
        recent_viewed_raw = self.redis_client.lrange(list_key, 0, 9)
        recent_viewed = [pid.decode('utf-8') if isinstance(pid, bytes) else str(pid) for pid in recent_viewed_raw]

        if not recent_viewed:
            return [] # Hoặc trả về sản phẩm trending nếu user mới

        purchased_set, cart_set = self.get_exclude_idx(user_id)
        exclude = purchased_set.union(cart_set)
        
        candidate_scores = {}
        
        # 2. Với mỗi sản phẩm đã xem, lấy top 10 tương đồng
        for pid in recent_viewed:
            similar_ids = recommend(pid, top_n=10)
            for sim_id in similar_ids:
                # Cộng dồn điểm (Accumulated Score)
                # Logic: Càng tương đồng với nhiều món đã xem càng ở trên đầu
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0) + 1
        
        # 3. Reranking & Filtering
        # Theo yêu cầu: Không loại bỏ các sản phẩm trong recent_viewed
        final_list = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [pid for pid, score in final_list if pid not in exclude][:top_n]
    
    def get_recommendations_detail(self, user_id: int, product_sys_id: str, top_n: int = 11):
        """Endpoint 2: Trang chi tiết (Similar products)"""
        # Lấy top tương đồng
        recommendations = recommend(product_sys_id, top_n=top_n + 5)
        purchased_set, cart_set = self.get_exclude_idx(user_id)
        
        # Loại bỏ chính nó, đã mua, và đã trong giỏ
        exclude = purchased_set.union(cart_set)
        exclude.add(str(product_sys_id))
        
        return [pid for pid in recommendations if pid not in exclude][:top_n]

    def get_recommendations_wishlist(self, user_id: int, top_n: int = 50):
        """Endpoint 3: Wishlist Discovery"""
        wl_query = "SELECT DISTINCT p.product_sys_id FROM Wishlist wl JOIN Product p ON wl.product_id = p.product_id WHERE wl.user_id = ?"
        wl_df = pd.read_sql(wl_query, self.engine, params=(user_id,))
        wishlist_ids = wl_df['product_sys_id'].astype(str).str.strip().tolist()
        
        if not wishlist_ids: return []

        purchased_set, cart_set = self.get_exclude_idx(user_id)
        exclude = purchased_set.union(cart_set).union(set(wishlist_ids))
        
        candidate_scores = {}
        for pid in wishlist_ids:
            for sim_id in recommend(pid, top_n=10):
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0) + 1
                
        final_list = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, score in final_list if pid not in exclude][:top_n]

    def get_recommendations_cart(self, user_id: int, top_n: int = 15):
        """Endpoint 4: Cross-selling (Giỏ hàng)"""
        purchased_set, cart_set = self.get_exclude_idx(user_id)
        
        if not cart_set: return []

        candidate_scores = {}
        for pid in cart_set:
            for sim_id in recommend(pid, top_n=10):
                candidate_scores[sim_id] = candidate_scores.get(sim_id, 0) + 1
        
        # Loại bỏ đã mua và những thứ ĐÃ có trong giỏ
        exclude = purchased_set.union(cart_set)
        
        final_list = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [pid for pid, score in final_list if pid not in exclude][:top_n]