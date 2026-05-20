import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class GroqPersonalizationReranker:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
        self.timeout = float(os.getenv("GROQ_TIMEOUT_SECONDS", "12"))
        self.source_limit = int(os.getenv("GROQ_SOURCE_LIMIT", "3"))
        self.candidate_limit = int(os.getenv("GROQ_RERANK_CANDIDATE_LIMIT", "8"))
        self.description_limit = int(os.getenv("GROQ_DESCRIPTION_LIMIT", "80"))
        self.enabled = bool(self.api_key)
        self.last_latency_ms = 0.0
        self.last_status = "disabled" if not self.enabled else "idle"
        self.last_error = ""

    def rerank(
        self,
        scene: str,
        source_products: list[dict[str, Any]],
        candidate_products: list[dict[str, Any]],
        top_n: int,
    ) -> list[str]:
        self.last_latency_ms = 0.0
        self.last_error = ""
        self.last_status = "disabled" if not self.enabled else "idle"
        if not self.enabled or not candidate_products:
            if not candidate_products:
                self.last_status = "skipped_no_candidates"
            return []

        compact_sources = self._compact_products(
            products=source_products,
            limit=self.source_limit,
            include_base_score=False,
        )
        compact_candidates = self._compact_products(
            products=candidate_products,
            limit=self.candidate_limit,
            include_base_score=True,
        )
        payload = self._build_payload(scene, compact_sources, compact_candidates, top_n)
        started_at = time.perf_counter()

        try:
            response = self._post_with_retry(payload)
            self.last_latency_ms = (time.perf_counter() - started_at) * 1000
            self.last_status = "http_ok"
        except Exception as exc:
            self.last_latency_ms = (time.perf_counter() - started_at) * 1000
            self.last_status = "http_error"
            self.last_error = str(exc)[:300]
            return []

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            ranked_ids = parsed.get("ranked_product_ids", [])
            if not isinstance(ranked_ids, list):
                self.last_status = "parse_error"
                self.last_error = "ranked_product_ids_not_list"
                return []
            self.last_status = "success"
            return [str(pid) for pid in ranked_ids]
        except Exception:
            self.last_status = "parse_error"
            self.last_error = "invalid_json_content"
            return []

    def _build_payload(
        self,
        scene: str,
        source_products: list[dict[str, Any]],
        candidate_products: list[dict[str, Any]],
        top_n: int,
    ) -> dict[str, Any]:
        system_prompt = (
            "Rank ecommerce candidate product IDs for a technology store. "
            "Use only provided candidate IDs. Favor relevance, accessory fit, price fit, brand/category fit, and stock. "
            "Return JSON only: {\"ranked_product_ids\": [\"...\"]}."
        )

        user_prompt = {
            "scene": scene,
            "top_n": top_n,
            "rules": [
                "cart/wishlist: favor complementary add-ons",
                "homepage: favor interest relevance and diversity",
                "prefer in-stock items",
                "penalize extreme price mismatch",
            ],
            "sources": source_products,
            "candidates": candidate_products,
        }

        return {
            "model": self.model,
            "temperature": 0.2,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
            ],
        }

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if response.status_code == 413:
                reduced_payload = self._shrink_payload(payload)
                retry_response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=reduced_payload,
                )
                retry_response.raise_for_status()
                return retry_response
            response.raise_for_status()
            return response

    def _shrink_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        reduced_payload = dict(payload)
        reduced_messages = list(payload["messages"])
        user_prompt = json.loads(reduced_messages[1]["content"])
        user_prompt["sources"] = user_prompt.get("sources", [])[:2]
        user_prompt["candidates"] = user_prompt.get("candidates", [])[:5]
        reduced_messages[1] = {
            "role": "user",
            "content": json.dumps(user_prompt, ensure_ascii=True),
        }
        reduced_payload["messages"] = reduced_messages
        return reduced_payload

    def _compact_products(
        self,
        products: list[dict[str, Any]],
        limit: int,
        include_base_score: bool,
    ) -> list[dict[str, Any]]:
        compacted = []
        for product in products[:limit]:
            compact = {
                "id": str(product.get("product_id", "")).strip(),
                "n": self._truncate_text(product.get("name", "")),
                "c": self._truncate_text(product.get("category", ""), 40),
                "b": self._truncate_text(product.get("brand", ""), 30),
                "p": round(float(product.get("price", 0.0) or 0.0), 2),
                "st": int(product.get("stock", 0) or 0),
            }
            description = self._truncate_text(product.get("description", ""), self.description_limit)
            if description:
                compact["d"] = description
            if include_base_score:
                compact["bs"] = round(float(product.get("base_score", 0.0) or 0.0), 4)
            compacted.append(compact)
        return compacted

    def _truncate_text(self, value: Any, limit: int | None = None) -> str:
        text = str(value or "").strip()
        max_len = limit if limit is not None else self.description_limit
        if len(text) <= max_len:
            return text
        return text[: max_len - 3].rstrip() + "..."
