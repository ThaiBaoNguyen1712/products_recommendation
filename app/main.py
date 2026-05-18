#app/main.py
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from app.api.engine.SceneRecommendationFilter import SceneRecommendationFilter
from app.api.engine.content_based import load_all_data, recommend

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
async def get_recommendations(product_sys_id: str, top_n: int = 15):
    recommendations = recommend(product_sys_id=product_sys_id, top_n=top_n)
    return {"product_sys_id": product_sys_id, "recommendations": recommendations}

@app.get("/api/v1/recommendation/wishlist/{user_id}")
async def get_wishlist_recommendations(user_id: int, top_n: int = 15):
    recommender = SceneRecommendationFilter()
    recommendations = recommender.get_recommendations_wishlist(
        user_id=user_id,
        top_n=top_n
    )

    return {
        "user_id": user_id,
        "scene": "wishlist",
        "recommendations": recommendations
    }

@app.get("/api/v1/recommendation/cart/{user_id}")
async def get_cart_recommendations(user_id: int, top_n: int = 15):
    recommender = SceneRecommendationFilter()
    recommendations = recommender.get_recommendations_cart(
        user_id=user_id,
        top_n=top_n
    )

    return {
        "user_id": user_id,
        "scene": "cart",
        "recommendations": recommendations
    }

@app.get("/api/v1/recommendation/homepage/{user_id}")
async def get_homepage_recommendations(user_id: int, top_n: int = 15):
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
