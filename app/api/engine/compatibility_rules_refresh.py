import json
import os
import re
import unicodedata
from collections import defaultdict
from typing import Any

import httpx

from app.api.engine.index_store import (
    COMPATIBILITY_RULES_PATH,
    DEFAULT_COMPATIBILITY_RULES,
    ACCESSORY_RULES_PATH,
    PRODUCTS_PATH,
)


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\u0111", "d").replace("\u0110", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _load_products() -> list[dict[str, Any]]:
    if not PRODUCTS_PATH.exists():
        return []
    try:
        payload = json.loads(PRODUCTS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


def _load_existing_rules() -> dict[str, Any]:
    if not COMPATIBILITY_RULES_PATH.exists():
        return DEFAULT_COMPATIBILITY_RULES
    try:
        payload = json.loads(COMPATIBILITY_RULES_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else DEFAULT_COMPATIBILITY_RULES
    except Exception:
        return DEFAULT_COMPATIBILITY_RULES


def _load_accessory_rules() -> dict[str, list[str]]:
    if not ACCESSORY_RULES_PATH.exists():
        return {}
    try:
        payload = json.loads(ACCESSORY_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, list[str]] = {}
    for source, targets in payload.items():
        source_slug = _slugify(source)
        if not source_slug or not isinstance(targets, list):
            continue
        normalized_targets: list[str] = []
        for target in targets:
            target_slug = _slugify(target)
            if target_slug and target_slug != source_slug and target_slug not in normalized_targets:
                normalized_targets.append(target_slug)
        if normalized_targets:
            cleaned[source_slug] = normalized_targets
    return cleaned


def _build_category_context(products: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    sample_counts: dict[str, int] = defaultdict(int)
    for product in products:
        category = str(product.get("category", "")).strip()
        category_slug = _slugify(category)
        if not category_slug:
            continue
        payload = grouped.setdefault(
            category_slug,
            {
                "category": category,
                "sample_products": [],
                "brands": [],
            },
        )
        if sample_counts[category_slug] < 8:
            name = str(product.get("name", "")).strip()
            if name:
                payload["sample_products"].append(name[:120])
                sample_counts[category_slug] += 1
        brand = str(product.get("brand", "")).strip()
        if brand and brand not in payload["brands"] and len(payload["brands"]) < 8:
            payload["brands"].append(brand[:60])
    return grouped


class OpenRouterCompatibilityRuleGenerator:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = (
            os.getenv("OPENROUTER_COMPATIBILITY_RULE_MODEL", "").strip()
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

    def generate_rules(self, category_context: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        payload = self._build_payload(category_context)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        parsed = self._parse_json_content(content)
        rules = parsed.get("rules", {})
        return rules if isinstance(rules, dict) else {}

    def _build_payload(self, category_context: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You design reusable cart cross-sell compatibility rules for a general ecommerce recommender. "
            "Given category slugs and sample product names, infer for each source category which target categories "
            "and product-name patterns are valid cart add-ons. The output must be domain-config data, not code. "
            "Use only provided category slugs. Return strict JSON only."
        )
        user_prompt = {
            "task": "Generate cart compatibility filters.",
            "schema": {
                "rules": {
                    "source_category_slug": {
                        "target_categories": ["provided_category_slug"],
                        "include_any": ["normalized pattern that should appear in candidate name/category/brand"],
                        "exclude_any": ["normalized pattern that should disqualify candidate"],
                    }
                }
            },
            "constraints": [
                "Use only category slugs from category_context for target_categories.",
                "Patterns must be lowercase ASCII-ish slugs, e.g. op_lung, sac, camera.",
                "include_any means candidate is valid if any pattern appears.",
                "exclude_any means candidate is invalid if any pattern appears.",
                "Prefer reusable product-type patterns, not exact SKUs.",
                "Do not force include_any when a whole target category is safely compatible.",
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


def _sanitize_rules(generated_rules: dict[str, Any], allowed_categories: set[str]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for source, payload in generated_rules.items():
        source_slug = _slugify(source)
        if source_slug not in allowed_categories or not isinstance(payload, dict):
            continue
        target_categories = []
        for target in payload.get("target_categories", []):
            target_slug = _slugify(target)
            if target_slug in allowed_categories and target_slug != source_slug and target_slug not in target_categories:
                target_categories.append(target_slug)
        include_any = _sanitize_patterns(payload.get("include_any", []))
        exclude_any = _sanitize_patterns(payload.get("exclude_any", []))
        target_category_limits: dict[str, int] = {}
        for category_key, raw_limit in payload.get("target_category_limits", {}).items():
            normalized_key = _slugify(category_key)
            try:
                limit_value = int(raw_limit)
            except (TypeError, ValueError):
                continue
            if normalized_key in allowed_categories and limit_value > 0:
                target_category_limits[normalized_key] = min(limit_value, 24)
        if target_categories:
            sanitized[source_slug] = {
                "target_categories": target_categories[:8],
                "target_category_limits": target_category_limits,
                "include_any": include_any[:30],
                "exclude_any": exclude_any[:30],
            }
    return sanitized


def _sanitize_patterns(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    patterns: list[str] = []
    for value in values:
        pattern = _slugify(value)
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    return patterns


def refresh_compatibility_rules(*, trigger: str = "manual") -> dict[str, Any]:
    enabled = os.getenv("ENABLE_LLM_COMPATIBILITY_RULE_REFRESH", "false").strip().lower() in {"1", "true", "yes"}
    existing_payload = _load_existing_rules()
    existing_rules = existing_payload.get("rules", {}) if isinstance(existing_payload, dict) else {}
    if not enabled:
        if not COMPATIBILITY_RULES_PATH.exists():
            COMPATIBILITY_RULES_PATH.write_text(
                json.dumps(DEFAULT_COMPATIBILITY_RULES, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return {
            "status": "skipped",
            "trigger": trigger,
            "reason": "llm_compatibility_rule_refresh_disabled",
            "rule_count": len(existing_rules),
        }

    products = _load_products()
    if not products:
        return {
            "status": "empty",
            "trigger": trigger,
            "reason": "no_products_available",
            "rule_count": len(existing_rules),
        }
    category_context = _build_category_context(products)
    allowed_categories = set(category_context.keys())
    generator = OpenRouterCompatibilityRuleGenerator()
    if not generator.enabled:
        return {
            "status": "disabled",
            "trigger": trigger,
            "reason": "openrouter_api_key_missing",
            "rule_count": len(existing_rules),
        }
    try:
        generated_rules = generator.generate_rules(category_context)
        sanitized_rules = _sanitize_rules(generated_rules, allowed_categories)
    except Exception as exc:
        return {
            "status": "failed",
            "trigger": trigger,
            "reason": str(exc)[:300],
            "rule_count": len(existing_rules),
            "model": generator.model,
        }
    if not sanitized_rules:
        return {
            "status": "failed",
            "trigger": trigger,
            "reason": "generated_rules_empty_after_sanitization",
            "rule_count": len(existing_rules),
            "model": generator.model,
        }
    payload = {
        "meta": {
            "schema_version": 1,
            "source": "llm",
            "model": generator.model,
            "trigger": trigger,
        },
        "rules": sanitized_rules,
    }
    COMPATIBILITY_RULES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "status": "ready",
        "trigger": trigger,
        "rule_count": len(sanitized_rules),
        "model": generator.model,
    }
