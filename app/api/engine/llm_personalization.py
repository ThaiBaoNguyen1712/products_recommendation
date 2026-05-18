import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class GroqPersonalizationReranker:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()
        self.timeout = float(os.getenv("GROQ_TIMEOUT_SECONDS", "12"))
        self.enabled = bool(self.api_key)

    def rerank(
        self,
        scene: str,
        source_products: list[dict[str, Any]],
        candidate_products: list[dict[str, Any]],
        top_n: int,
    ) -> list[str]:
        if not self.enabled or not candidate_products:
            return []

        payload = self._build_payload(scene, source_products, candidate_products, top_n)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except Exception:
            return []

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            ranked_ids = parsed.get("ranked_product_ids", [])
            if not isinstance(ranked_ids, list):
                return []
            return [str(pid) for pid in ranked_ids]
        except Exception:
            return []

    def _build_payload(
        self,
        scene: str,
        source_products: list[dict[str, Any]],
        candidate_products: list[dict[str, Any]],
        top_n: int,
    ) -> dict[str, Any]:
        system_prompt = (
            "You are an ecommerce recommendation reranker for a technology store. "
            "Your job is to rank candidate product IDs for a user based on the current scene. "
            "Prefer practical relevance, category fit, accessory fit, brand fit, price fit, and in-stock items. "
            "Do not invent product IDs. Only rank from the provided candidates. "
            "Return JSON only with one key: ranked_product_ids."
        )

        user_prompt = {
            "scene": scene,
            "top_n": top_n,
            "source_products": source_products,
            "candidate_products": candidate_products,
            "ranking_rules": [
                "Favor products that match the user's current intent.",
                "For cart and wishlist, prioritize complementary and useful add-on products.",
                "For homepage, prioritize relevant exploration across the user's recent interests.",
                "Prefer in-stock products.",
                "Penalize extreme price mismatch.",
                "Avoid returning products too similar to each other when better diversity is possible.",
            ],
            "output_format": {
                "ranked_product_ids": ["candidate_id_1", "candidate_id_2"],
            },
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
