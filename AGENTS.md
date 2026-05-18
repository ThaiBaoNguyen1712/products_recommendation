# Project Context: Ecommerce Recommendation System

## Repo role and current state
- This repository is the FastAPI recommendation/AI service layer, not the main ASP.NET Core backend.
- Current code already contains FastAPI entrypoints in `app/main.py`.
- Current recommendation engines live under `app/api/engine/` and include content-based, collaborative, hybrid, and scene-based logic.
- MSSQL connectivity already exists in `db/mssql.py` via SQLAlchemy + pyodbc.
- `upstash-redis` is already present in dependencies, so Redis caching can be added without introducing a heavy new platform.
- Existing routes currently return recommendation lists directly from the FastAPI service. Future cleanup should standardize contracts around product IDs, scores, and reason codes.

## Current stack
- Main backend: ASP.NET Core
- Database: MSSQL
- Existing AI/recommendation service: FastAPI
- Product domain: technology ecommerce
- Product size: around 1,000 products
- Product data includes specs, variants, descriptions, categories, brands, prices, stock.
- Deadline is short, so avoid risky database migration.
- Do not migrate MSSQL to PostgreSQL.
- Do not introduce heavy infrastructure unless necessary.

## Recommendation goal
Build a fast recommendation system for ecommerce, not a chatbot.

The system should support:
1. Similar product recommendation
2. Complementary/accessory recommendation
3. Personalized homepage recommendation

## Architecture direction
Detail page product
-> content-based retrieval using category/specs/brand/price
-> scoring/reranking
-> Redis cache
-> return similar product IDs to ASP.NET

Cart / wishlist / user behavior
-> user intent extraction
-> candidate generation from product metadata, category rules, and user context
-> optional LLM-assisted reranking or personalization
-> Redis cache
-> return product IDs to ASP.NET

## Important design principles
- MSSQL remains source of truth.
- FastAPI acts as recommendation/AI layer.
- Recommendation API returns product IDs, scores, and reason codes only.
- ASP.NET remains responsible for fetching full product details from MSSQL.
- Content-based recommendation is the primary strategy for similar-product recommendation in `sceneRcm` / product detail flows.
- LLM may be used for cart and wishlist personalization when it improves intent understanding or reranking quality.
- Do not use LLM as a chatbot-style answer generator.
- Do not let LLM return full product objects or final UI payloads.
- Prefer deterministic candidate generation first, then use LLM only as a controlled reasoning/reranking layer.
- Prefer prompt-constrained JSON output if LLM is used.
- If latency is too high, move LLM logic to async/precompute flow or cache heavily.
- Prefer embedding-based semantic retrieval and structured reranking over free-form generation.
- Prefer free/open AI model APIs instead of deploying local models.
- Redis should cache final recommendation results, not full product objects.
- For MVP, rule-based recommendation is acceptable before adding embeddings.
- Avoid overengineering.

## Recommended FastAPI module structure
```text
app/
  recommend/
    __init__.py
    schemas.py
    service.py
    scoring.py
    rules.py
    index_store.py
    cache.py

data/
  products.json
  accessory_rules.json
  index_meta.json
```

## Core endpoints
- `GET /health`
- `POST /recommend/similar`
- `POST /recommend/complementary`
- `POST /recommend/homepage`
- `POST /admin/rebuild-index`
- `POST /admin/sync-product`

## Recommendation strategies

### Similar products
Used mainly on product detail pages.
Goal: recommend alternatives or products with similar category/specs/price segment.
Primary implementation direction: content-based retrieval only.

### Complementary products
Used mainly in cart and product detail.
Goal: recommend accessories or products commonly bought together.
This should use accessory/category relationship rules first.

Example rules:
- `gaming_laptop -> gaming_mouse, mechanical_keyboard, cooling_pad, gaming_headset, monitor`
- `office_laptop -> wireless_mouse, laptop_bag, usb_hub, monitor`
- `iphone -> phone_case, screen_protector, charger, airpods, power_bank`
- `android_phone -> phone_case, screen_protector, charger, earbuds, power_bank`
- `monitor -> hdmi_cable, displayport_cable, monitor_arm, keyboard, mouse`
- `desktop_pc -> monitor, keyboard, mouse, speaker, ups`

### Personalized homepage
Use user behavior, not rating data.
Signals:
- `view_detail`
- `search_click`
- `wishlist`
- `add_cart`
- `purchase`
- `remove_cart`

Suggested weights:
- `view_detail = 1`
- `search_click = 2`
- `wishlist = 4`
- `add_cart = 6`
- `purchase = 8`
- `remove_cart = -3`

Homepage should mix:
- user interest/category affinity
- recently viewed similar products
- complementary candidates from cart/wishlist
- popular/latest fallback

### Cart and wishlist personalization
Use cart and wishlist as high-intent signals.
Recommended flow:
- generate candidates from related categories, accessory rules, and similar products
- summarize user intent from cart/wishlist contents
- optionally use LLM to rerank candidates and assign reason codes
- keep final output bounded, structured, and cached

## Scoring ideas

Complementary score:
```text
score = ruleWeight + priceFit + stockScore + popularityScore
```

Personalized score:
```text
score =
  userInterestSimilarity/categoryAffinity
  + recentIntentScore
  + priceFit
  + complementaryScore
  + popularity
  + stockScore
```

Penalties:
- product already in cart
- product already in wishlist when inappropriate
- out of stock
- deleted/inactive product
- extreme price mismatch

## Data sync design
When a product is added, updated, or deleted in MSSQL:
- mark product as pending AI sync
- FastAPI/admin job syncs product into AI data files
- rebuild or update index
- clear affected Redis cache

For MVP with ~1k products, full rebuild is acceptable.

## AI model direction
Do not deploy local model for now.
Prefer provider-based free/open APIs:
- Gemini Embedding API
- Hugging Face Inference Providers
- OpenRouter/Groq/Gemini for optional cart/wishlist personalization and offline product profile generation

Embedding should be added later behind an interface:
- `EmbeddingProvider`
- `DummyEmbeddingProvider` for testing
- `GeminiEmbeddingProvider TODO`
- `HuggingFaceEmbeddingProvider TODO`

If LLM reranking is added, place it behind an interface such as:
- `PersonalizationProvider`
- `DummyPersonalizationProvider` for testing
- `LLMRerankProvider TODO`

## Cache design
Use Redis if available, otherwise fallback in-memory cache for development.

Cache keys:
- `rec:similar:product:{productId}`
- `rec:complementary:user:{userId}`
- `rec:homepage:user:{userId}`
- `rec:homepage:session:{sessionId}`

Suggested TTL:
- `similar: 6 hours`
- `complementary: 5-10 minutes`
- `homepage: 15-30 minutes`

## Constraints
- Do not change database provider.
- Do not build chatbot.
- Do not return full product objects from recommendation API.
- Do not introduce LangChain unless clearly necessary.
- Keep code simple and easy to demo.
- Prioritize working MVP over perfect architecture.

LLM-specific constraints:
- Do not send the full catalog to the LLM.
- Retrieve a small candidate set first, then ask the LLM to rerank that bounded set.
- Prefer cached or asynchronous personalization if provider latency is unstable.
- Keep prompts product-centric and structured, not conversational.

## Implementation guidance for future agents
- Treat this repo as a focused recommendation service that integrates with ASP.NET, not as a standalone ecommerce backend.
- Preserve MSSQL compatibility in all implementation decisions.
- Prefer additive refactors over big rewrites because current deadline is short.
- When extending APIs, keep response contracts small and backend-friendly: product ID, score, reason code.
- For detail-page similar products, prefer content-based retrieval over LLM.
- For cart and wishlist flows, use deterministic candidate generation first, then optionally LLM reranking.
- If embeddings are not ready, ship rule-based and metadata-based heuristics first.
- If a feature can be solved with simple category/spec/price logic for MVP, prefer that over introducing new infra.
- Avoid introducing local vector databases unless product scale or latency proves they are necessary.
