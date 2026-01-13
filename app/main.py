#app/main.py
import pandas as pd
from app.api.engine.content_based import recommend
from app.api.engine.collaborative import CollaborativeFiltering
from app.api.engine.hybird import HybridRecommender
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/content_based_filter/{product_sys_id}")
async def get_recommendations(product_sys_id: str, top_n: int = 5):
    recommendations = recommend(product_sys_id=product_sys_id, top_n=top_n)
    return {"product_sys_id": product_sys_id, "recommendations": recommendations}

@app.get("/collaborative_filter/{user_id}")
async def get_collaborative_recommendations(user_id: int, top_n: int = 5):
    ids = CollaborativeFiltering().get_recommendations(user_id=user_id, top_n=top_n)
    return {"user_id": user_id, "recommendations": ids}

@app.get("/get_recommend/{user_id}/{product_sys_id}")
async def get_hybrid_recommendations(user_id: int, product_sys_id: str, top_n: int = 10, scene: str = 'detail'):
    recommender = HybridRecommender()
    recommendations = recommender.get_hybrid_recommendations(
        user_id=user_id,
        product_sys_id=product_sys_id,
        top_n=top_n,
        scene=scene
    )
    return {
        "product_sys_id": product_sys_id,
        "scene": scene,
        "recommendations": recommendations
    }