# Product Recommendation API

FastAPI service for ecommerce product recommendation.

This repository is the recommendation layer for a technology ecommerce system.  
Main backend remains ASP.NET Core.  
MSSQL remains source of truth.

## Current purpose
- Similar product recommendation with content-based filtering
- Personalized recommendation for `wishlist`, `cart`, and `homepage`
- User-interest scoring from `UserProductEvent`
- LLM reranking for personalization using Groq `llama-3.1-8b-instant`

## Stack
- FastAPI
- MSSQL via SQLAlchemy + pyodbc
- Redis via Upstash Redis
- Groq API for LLM reranking

## Main flows

### 1. Similar products
- Input: `product_sys_id`
- Strategy: content-based retrieval using category, brand, specs, price, stock
- Endpoint:
  - `GET /content_based_filter/{product_sys_id}`

### 2. Wishlist recommendation
- Input: `user_id`
- Strategy:
  - read wishlist products
  - generate candidates from similar products
  - boost by user interaction history
  - rerank with LLM
- Endpoint:
  - `GET /api/v1/recommendation/wishlist/{user_id}`

### 3. Cart recommendation
- Input: `user_id`
- Strategy:
  - read cart products
  - generate candidates from similar products
  - boost by user interaction history
  - rerank with LLM
- Endpoint:
  - `GET /api/v1/recommendation/cart/{user_id}`

### 4. Homepage recommendation
- Input: `user_id`
- Strategy:
  - read recent viewed products from Redis
  - generate candidates from similar products
  - boost by user interaction history
  - rerank with LLM
- Endpoint:
  - `GET /api/v1/recommendation/homepage/{user_id}`

## User interaction signals
The service uses `UserProductEvent` to estimate user interest and add score before LLM reranking.

Typical events:
- `view_detail`
- `search_click`
- `wishlist_add`
- `wishlist_remove`
- `add_cart`
- `remove_cart`
- `purchase`
- `homepage_click`
- `recommendation_click`

Current behavior:
- products already in `cart` or `wishlist` are excluded from personalized recommendations
- products already purchased are also excluded

## Project structure
```text
app/
  main.py
  api/
    engine/
      content_based.py
      SceneRecommendationFilter.py
      llm_personalization.py

db/
  mssql.py
```

## Environment variables
Required values are stored in `.env`.

Main variables:
- `DB_SERVER`
- `DB_NAME`
- `DB_TRUSTED_CONNECTION`
- `UPSTASH_URL`
- `UPSTASH_TOKEN`
- `GROQ_API_KEY`
- `GROQ_MODEL`
- `GROQ_TIMEOUT_SECONDS`
- `LLM_CANDIDATE_LIMIT`
- `USER_INTEREST_EVENT_LIMIT`

## Run locally
```powershell
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If virtual env is already activated:
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Notes
- This is a recommendation API, not a chatbot.
- FastAPI should return recommendation IDs only.
- ASP.NET should fetch full product details from MSSQL.
- LLM is used as reranker, not as free-form generator.
- For detail-page similar products, content-based remains the main strategy.
