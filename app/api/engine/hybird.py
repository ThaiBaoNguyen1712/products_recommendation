# app/api/engine/hybrid.py
from .collaborative import CollaborativeFiltering
from .content_based import df_products, recommend

class HybridRecommender:
    def __init__(self):
        self.collab_engine = CollaborativeFiltering()

    def get_hybrid_recommendations(self, user_id: int, product_sys_id: str, top_n: int = 10, scene: str = 'detail'):
        # 1. Lấy dư dữ liệu (Over-fetching) để có đủ "nguyên liệu" lọc
        cf_recs = self.collab_engine.get_recommendations(user_id, top_n=top_n * 4)
        cb_recs = recommend(product_sys_id, top_n=top_n * 4)

        # 2. Lấy metadata của sản phẩm đang xem
        try:
            current_product = df_products.loc[product_sys_id]
            current_cat = current_product['category']
            current_price = current_product['sellPrice']
        except KeyError:
            current_cat = None
            current_price = None

        # 3. Loại bỏ sản phẩm hiện tại
        cf_recs = [pid for pid in cf_recs if pid != product_sys_id]
        cb_recs = [pid for pid in cb_recs if pid != product_sys_id]

        # 4. Sắp xếp lại danh sách dựa trên Category và Price (Smart Filtering)
        if current_cat:
            # Ưu tiên sản phẩm cùng loại và lọc giá chênh lệch không quá 2 lần (ví dụ thế)
            cb_recs = self._smart_sort(cb_recs, current_cat, current_price)
            cf_recs = self._smart_sort(cf_recs, current_cat, current_price)

        final_recs = []

        # 5. Logic điều hướng theo Ngữ cảnh (Scene)
        if scene == 'detail':
            # Trang chi tiết: 70% là Content-based (Sản phẩm tương tự)
            final_recs = self._blend_results(cb_recs, cf_recs, ratio=0.7, top_n=top_n)
        elif scene == 'cart':
            # Giỏ hàng: 70% là Collab (Bán chéo/Mua cùng nhau)
            final_recs = self._blend_results(cf_recs, cb_recs, ratio=0.7, top_n=top_n)
        elif scene == 'homepage':
            # Trang chủ: Ưu tiên Collab (Cá nhân hóa)
            final_recs = self._blend_results(cf_recs, cb_recs, ratio=0.8, top_n=top_n)
        else:
            final_recs = self._blend_results(cf_recs, cb_recs, ratio=0.5, top_n=top_n)

        return final_recs[:top_n]

    def _smart_sort(self, rec_list, target_cat, target_price):
        """Đưa sản phẩm cùng category lên đầu và lọc bớt sản phẩm lệch giá quá xa"""
        same_cat = []
        diff_cat = []
        
        for pid in rec_list:
            try:
                item = df_products.loc[pid]

                if(item['sellPrice'] <=0):
                    continue
                if(item['status'] == 'outstock'):
                    continue
                if(item['stock'] <=0):
                    continue

                # Lọc giá: Ví dụ chỉ lấy máy trong khoảng 60% - 250% giá máy hiện tại
                is_reasonable_price = (item['sellPrice'] >= target_price * 0.6)
                
                if item['category'] == target_cat:
                    if is_reasonable_price:
                        same_cat.append(pid)
                    else:
                        # Giá quá lệch, đưa xuống cuối
                        diff_cat.insert(0, pid) 
                else:
                    diff_cat.append(pid)
            except KeyError:
                diff_cat.append(pid)
        
        return same_cat + diff_cat

    def _blend_results(self, primary, secondary, ratio, top_n):
        """Trộn kết quả đảm bảo tính đa dạng và đủ số lượng"""
        num_primary = int(top_n * ratio)
        res = primary[:num_primary]
        
        for item in secondary:
            if len(res) >= top_n:
                break
            if item not in res:
                res.append(item)
        
        # Nếu vẫn thiếu món (do trùng lặp hoặc lọc giá), lấy nốt từ primary
        if len(res) < top_n:
            for item in primary[num_primary:]:
                if len(res) >= top_n: break
                if item not in res: res.append(item)
                
        return res