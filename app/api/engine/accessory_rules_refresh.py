import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Any

import httpx

from app.api.engine.index_store import ACCESSORY_RULES_PATH, PRODUCTS_PATH


def _load_products() -> list[dict[str, Any]]:
    if not PRODUCTS_PATH.exists():
        return []
    try:
        payload = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _load_existing_rules() -> dict[str, list[str]]:
    if not ACCESSORY_RULES_PATH.exists():
        return {}
    try:
        payload = json.loads(ACCESSORY_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, list[str]] = {}
    for key, values in payload.items():
        if not isinstance(values, list):
            continue
        normalized_key = _slugify(key)
        normalized_values = []
        for value in values:
            slug = _slugify(value)
            if slug and slug not in normalized_values and slug != normalized_key:
                normalized_values.append(slug)
        if normalized_key:
            cleaned[normalized_key] = normalized_values
    return cleaned


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _build_category_context(products: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    sample_counts: dict[str, int] = defaultdict(int)

    for product in products:
        category_label = str(product.get("category", "")).strip()
        category_slug = _slugify(category_label)
        if not category_slug:
            continue

        category_payload = grouped.setdefault(
            category_slug,
            {
                "category": category_label or category_slug,
                "sample_products": [],
                "brands": [],
            },
        )

        if sample_counts[category_slug] < 3:
            product_name = str(product.get("name", "")).strip()
            if product_name:
                category_payload["sample_products"].append(product_name[:120])
                sample_counts[category_slug] += 1

        brand = str(product.get("brand", "")).strip()
        if brand and brand not in category_payload["brands"] and len(category_payload["brands"]) < 3:
            category_payload["brands"].append(brand[:60])

    return grouped


class OpenRouterAccessoryRuleGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = (
            os.getenv("OPENROUTER_ACCESSORY_RULE_MODEL", "").strip()
            or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash").strip()
            or "deepseek/deepseek-v4-flash"
        )
        self.base_url = os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1/chat/completions",
        ).strip()
        self.timeout = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "20").strip() or "20")
        self.reasoning_enabled = (
            os.getenv("OPENROUTER_REASONING_ENABLED", "true").strip().lower()
            in {"1", "true", "yes"}
        )
        self.http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "products-rcm-sys-api").strip()
        self.enabled = bool(self.api_key)

    def generate_rules(self, category_context: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
        payload = self._build_payload(category_context)
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
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        parsed = self._parse_json_content(content)
        rules = parsed.get("rules", {})
        return rules if isinstance(rules, dict) else {}

    def _build_payload(self, category_context: dict[str, dict[str, Any]]) -> dict[str, Any]:
        system_prompt = (
            "You design complementary-category rules for a general ecommerce recommendation engine. "
            "Given a set of category slugs and sample products, infer which categories are useful accessories, bundles, "
            "consumables, add-ons, or natural cross-sell companions for each source category. "
            "Use only the provided category slugs. Do not invent new category slugs. "
            "Do not include the source category itself. "
            "Return strict JSON only in the format: "
            "{\"rules\": {\"source_category_slug\": [\"target_category_slug_1\", \"target_category_slug_2\"]}}."
        )
        user_prompt = {
            "task": "Generate reusable complementary-category rules for an ecommerce catalog.",
            "constraints": [
                "Only use category slugs that appear in the provided category_context.",
                "Prefer broad cross-sell relationships that generalize beyond a single brand.",
                "For each source category, return at most 5 complementary target categories.",
                "Do not return substitute categories unless they are clearly part of a useful bundle.",
            ],
            "category_context": category_context,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
            ],
            "temperature": 0.1,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"enabled": True}
        return payload

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


def _sanitize_rules(
    generated_rules: dict[str, Any],
    allowed_categories: set[str],
    fallback_rules: dict[str, list[str]],
) -> dict[str, list[str]]:
    sanitized: dict[str, list[str]] = {}

    for source_category, targets in generated_rules.items():
        source_slug = _slugify(source_category)
        if source_slug not in allowed_categories or not isinstance(targets, list):
            continue
        normalized_targets: list[str] = []
        for target in targets:
            target_slug = _slugify(target)
            if (
                target_slug
                and target_slug in allowed_categories
                and target_slug != source_slug
                and target_slug not in normalized_targets
            ):
                normalized_targets.append(target_slug)
        if normalized_targets:
            sanitized[source_slug] = normalized_targets[:5]

    for source_slug, targets in fallback_rules.items():
        if source_slug in sanitized:
            continue
        if source_slug not in allowed_categories:
            continue
        normalized_targets = [
            target for target in targets
            if target in allowed_categories and target != source_slug
        ][:5]
        if normalized_targets:
            sanitized[source_slug] = normalized_targets

    return sanitized


def refresh_accessory_rules(*, trigger: str = "manual") -> dict[str, Any]:
    enabled = os.getenv("ENABLE_LLM_ACCESSORY_RULE_REFRESH", "false").strip().lower() in {"1", "true", "yes"}
    existing_rules = _load_existing_rules()
    if not enabled:
        return {
            "status": "skipped",
            "trigger": trigger,
            "reason": "llm_accessory_rule_refresh_disabled",
            "category_count": len(existing_rules),
        }

    products = _load_products()
    if not products:
        return {
            "status": "empty",
            "trigger": trigger,
            "reason": "no_products_available",
            "category_count": 0,
        }

    category_context = _build_category_context(products)
    allowed_categories = set(category_context.keys())
    if not allowed_categories:
        return {
            "status": "empty",
            "trigger": trigger,
            "reason": "no_categories_available",
            "category_count": 0,
        }

    generator = OpenRouterAccessoryRuleGenerator()
    if not generator.enabled:
        return {
            "status": "disabled",
            "trigger": trigger,
            "reason": "openrouter_api_key_missing",
            "category_count": len(existing_rules),
        }

    try:
        generated_rules = generator.generate_rules(category_context)
    except Exception as exc:
        return {
            "status": "failed",
            "trigger": trigger,
            "reason": str(exc)[:300],
            "category_count": len(existing_rules),
            "model": generator.model,
        }

    sanitized_rules = _sanitize_rules(
        generated_rules=generated_rules,
        allowed_categories=allowed_categories,
        fallback_rules=existing_rules,
    )

    if not sanitized_rules:
        return {
            "status": "failed",
            "trigger": trigger,
            "reason": "generated_rules_empty_after_sanitization",
            "category_count": len(existing_rules),
            "model": generator.model,
        }

    ACCESSORY_RULES_PATH.write_text(
        json.dumps(sanitized_rules, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "ready",
        "trigger": trigger,
        "category_count": len(sanitized_rules),
        "model": generator.model,
    }
