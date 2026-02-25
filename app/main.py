#app/main.py
from fastapi.concurrency import asynccontextmanager
import pandas as pd
from app.api.engine.content_based import load_all_data, recommend
from app.api.engine.collaborative import CollaborativeFiltering
from app.api.engine.hybird import HybridRecommender
from app.api.engine.SceneRecommendationFilter import SceneRecommendationFilter
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Những gì viết ở đây sẽ chạy khi App BẮT ĐẦU
    print("Đang khởi tạo dữ liệu ML...")
    load_all_data() 
    yield
    # Những gì viết ở đây sẽ chạy khi App TẮT (nếu cần)
    print("Đang tắt ứng dụng...")

# 2. Truyền lifespan vào FastAPI
app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/content_based_filter/{product_sys_id}")
async def get_recommendations(product_sys_id: str, top_n: int = 5):
    recommendations = recommend(product_sys_id=product_sys_id, top_n=top_n)
    return {"product_sys_id": product_sys_id, "recommendations": recommendations}

# @app.get("/collaborative_filter/{user_id}")
# async def get_collaborative_recommendations(user_id: int, top_n: int = 5):
#     ids = CollaborativeFiltering().get_recommendations(user_id=user_id, top_n=top_n)
#     return {"user_id": user_id, "recommendations": ids}

# @app.get("/get_recommend/{user_id}/{product_sys_id}")
# async def get_hybrid_recommendations(user_id: int, product_sys_id: str, top_n: int = 10, scene: str = 'detail'):
#     recommender = HybridRecommender()
#     recommendations = recommender.get_hybrid_recommendations(
#         user_id=user_id,
#         product_sys_id=product_sys_id,
#         top_n=top_n,
#         scene=scene
#     )
#     return {
#         "product_sys_id": product_sys_id,
#         "scene": scene,
#         "recommendations": recommendations
#     }

@app.get("/api/v1/recommendation/{user_id}/{scene}/{product_sys_id}")
async def get_scene_recommendations(user_id: int, scene: str, product_sys_id: str = None, top_n: int = 10):
    recommender = SceneRecommendationFilter()
    
    if scene == 'detail' and product_sys_id:
        recommendations = recommender.get_recommendations_detail(
            user_id=user_id,
            product_sys_id=product_sys_id,
            top_n=top_n
        )
    elif scene == 'wishlist':
        recommendations = recommender.get_recommendations_wishlist(
            user_id=user_id,
            top_n=top_n
        )
    elif scene == 'cart':
        recommendations = recommender.get_recommendations_cart(
            user_id=user_id,
            top_n=top_n
        )
    else:
        return {"error": "Invalid scene or missing product_sys_id for detail scene."}
    
    return {
        "user_id": user_id,
        "scene": scene,
        "recommendations": recommendations
    }

@app.get("/api/v1/recommendation/homepage/{user_id}")
async def get_homepage_recommendations(user_id: int, top_n: int = 50):
    recommender = SceneRecommendationFilter()
    recommendations = recommender.get_recommendations_homepage(
        user_id=user_id,
        top_n=top_n
    )
    return {
        "user_id": user_id,
        "scene": "homepage",
        "recommendations": recommendations
    }