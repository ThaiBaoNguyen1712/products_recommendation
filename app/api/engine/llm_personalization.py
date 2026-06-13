import json
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


class OpenRouterPersonalizationReranker:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = (
            os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash").strip()
            or "deepseek/deepseek-v4-flash"
        )
        self.base_url = os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1/chat/completions",
        ).strip()
        self.timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "20").strip() or "20")
        self.source_limit = int(os.getenv("OPENROUTER_SOURCE_LIMIT", "3").strip() or "3")
        self.candidate_limit = int(
            os.getenv("OPENROUTER_RERANK_CANDIDATE_LIMIT", "8").strip() or "8"
        )
        self.description_limit = int(
            os.getenv("OPENROUTER_DESCRIPTION_LIMIT", "80").strip() or "80"
        )
        self.reasoning_enabled = (
            os.getenv("OPENROUTER_REASONING_ENABLED", "true").strip().lower()
            in {"1", "true", "yes"}
        )
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "products-rcm-sys-api").strip()
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
            parsed = self._parse_json_content(content)
            ranked_ids = parsed.get("ranked_product_ids", [])
            if not isinstance(ranked_ids, list):
                self.last_status = "parse_error"
                self.last_error = "ranked_product_ids_not_list"
                return []
            self.last_status = "success"
            return [str(pid).strip() for pid in ranked_ids if str(pid).strip()]
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
        scene_objective = self._scene_objective(scene)
        system_prompt = (
            "Rank ecommerce candidate product IDs for a technology store. "
            f"Current scene objective: {scene_objective}. "
            "Use only provided candidate IDs. Do not invent IDs. "
            "Return strict JSON only with this shape: {\"ranked_product_ids\": [\"...\"]}."
        )

        user_prompt = {
            "scene": scene,
            "top_n": top_n,
            "rules": self._scene_rules(scene),
            "sources": source_products,
            "candidates": candidate_products,
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
            ],
            "temperature": 0.2,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        return payload

    def _scene_objective(self, scene: str) -> str:
        normalized_scene = str(scene or "").strip().lower()
        if normalized_scene == "cart":
            return "cross-sell, bundle completion, and checkout purchase completion"
        if normalized_scene == "wishlist":
            return "similar alternatives, upgrade or downgrade options, and preference fit"
        return "discovery, broader interest exploration, and diverse relevant products"

    def _scene_rules(self, scene: str) -> list[str]:
        normalized_scene = str(scene or "").strip().lower()
        base_rules = [
            "prefer in-stock candidates",
            "penalize extreme price mismatch",
            "return the strongest product IDs first",
        ]
        if normalized_scene == "cart":
            return [
                "favor accessories and products likely bought together with the source products",
                "favor bundle-completion products over same-category alternatives",
                "for expensive source products, cheaper compatible accessories are usually better",
                *base_rules,
            ]
        if normalized_scene == "wishlist":
            return [
                "favor close substitutes and similar alternatives to saved products",
                "include sensible upgrade or downgrade options when they match the source category",
                "prefer candidates near the source product price unless clearly better",
                *base_rules,
            ]
        return [
            "favor recent interest and category affinity",
            "include some category or brand exploration instead of near-duplicates only",
            "prefer a diverse set of useful discovery candidates",
            *base_rules,
        ]

    def _post_with_retry(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url, headers=headers, json=payload)
            if response.status_code == 413:
                reduced_payload = self._shrink_payload(payload)
                retry_response = client.post(self.base_url, headers=headers, json=reduced_payload)
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
        compacted: list[dict[str, Any]] = []
        for product in products[:limit]:
            compact = {
                "id": str(product.get("product_id", "")).strip(),
                "n": self._truncate_text(product.get("name", "")),
                "c": self._truncate_text(product.get("category", ""), 40),
                "b": self._truncate_text(product.get("brand", ""), 30),
                "p": round(float(product.get("price", 0.0) or 0.0), 2),
                "st": int(product.get("stock", 0) or 0),
            }
            description = self._truncate_text(
                product.get("description_text") or product.get("description", ""),
                self.description_limit,
            )
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

    def _parse_json_content(self, content: Any) -> dict[str, Any]:
        text = str(content or "").strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            fenced = "\n".join(lines).strip()
            if fenced:
                return json.loads(fenced)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise json.JSONDecodeError("No JSON object found", text, 0)
