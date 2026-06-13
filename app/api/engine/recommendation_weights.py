import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WEIGHT_CONFIG_PATH = REPO_ROOT / "data" / "offline_tuning_best_weights.json"


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    return _to_float(os.getenv(name), default)


def _load_json_config() -> dict[str, Any]:
    raw_path = os.getenv("RECOMMEND_WEIGHT_CONFIG_PATH", "").strip()
    path = Path(raw_path) if raw_path else DEFAULT_WEIGHT_CONFIG_PATH
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _config_float(config: dict[str, Any], section: str, name: str, default: float) -> float:
    section_payload = config.get(section, {})
    if isinstance(section_payload, dict) and name in section_payload:
        return _to_float(section_payload.get(name), default)
    return default


@dataclass(frozen=True)
class ImplicitFeedbackWeights:
    interaction: float = 1.0
    purchase: float = 8.0
    cart: float = 4.0
    wishlist: float = 2.0
    view: float = 1.0


@dataclass(frozen=True)
class SceneSimilarityWeights:
    homepage: float = 1.0
    wishlist: float = 1.3
    cart: float = 1.6

    def as_dict(self) -> dict[str, float]:
        return {
            "homepage": self.homepage,
            "wishlist": self.wishlist,
            "cart": self.cart,
        }


@dataclass(frozen=True)
class SceneRuleWeight:
    exact: float
    partial: float


@dataclass(frozen=True)
class RecommendationWeightConfig:
    implicit: ImplicitFeedbackWeights = field(default_factory=ImplicitFeedbackWeights)
    scene_similarity: SceneSimilarityWeights = field(default_factory=SceneSimilarityWeights)
    scene_rule: dict[str, SceneRuleWeight] = field(default_factory=dict)
    decay_lambda: float = 0.05

    def scene_rule_as_dict(self) -> dict[str, dict[str, float]]:
        return {
            scene: {"exact": weight.exact, "partial": weight.partial}
            for scene, weight in self.scene_rule.items()
        }


def load_recommendation_weight_config() -> RecommendationWeightConfig:
    file_config = _load_json_config()

    implicit = ImplicitFeedbackWeights(
        interaction=_env_float(
            "RECOMMEND_INTERACTION_WEIGHT",
            _config_float(file_config, "implicit", "interaction", 1.0),
        ),
        purchase=_env_float(
            "RECOMMEND_PURCHASE_WEIGHT",
            _config_float(file_config, "implicit", "purchase", 8.0),
        ),
        cart=_env_float(
            "RECOMMEND_CART_WEIGHT",
            _config_float(file_config, "implicit", "cart", 4.0),
        ),
        wishlist=_env_float(
            "RECOMMEND_WISHLIST_WEIGHT",
            _config_float(file_config, "implicit", "wishlist", 2.0),
        ),
        view=_env_float(
            "RECOMMEND_VIEW_WEIGHT",
            _config_float(file_config, "implicit", "view", 1.0),
        ),
    )

    scene_similarity = SceneSimilarityWeights(
        homepage=_env_float(
            "RECOMMEND_HOMEPAGE_SIMILARITY_WEIGHT",
            _config_float(file_config, "scene_similarity", "homepage", 1.0),
        ),
        wishlist=_env_float(
            "RECOMMEND_WISHLIST_SIMILARITY_WEIGHT",
            _config_float(file_config, "scene_similarity", "wishlist", 1.3),
        ),
        cart=_env_float(
            "RECOMMEND_CART_SIMILARITY_WEIGHT",
            _config_float(file_config, "scene_similarity", "cart", 1.6),
        ),
    )

    scene_rule = {
        "homepage": SceneRuleWeight(
            exact=_env_float(
                "RECOMMEND_HOMEPAGE_RULE_EXACT_WEIGHT",
                _config_float(file_config, "scene_rule_homepage", "exact", 1.1),
            ),
            partial=_env_float(
                "RECOMMEND_HOMEPAGE_RULE_PARTIAL_WEIGHT",
                _config_float(file_config, "scene_rule_homepage", "partial", 0.65),
            ),
        ),
        "wishlist": SceneRuleWeight(
            exact=_env_float(
                "RECOMMEND_WISHLIST_RULE_EXACT_WEIGHT",
                _config_float(file_config, "scene_rule_wishlist", "exact", 1.55),
            ),
            partial=_env_float(
                "RECOMMEND_WISHLIST_RULE_PARTIAL_WEIGHT",
                _config_float(file_config, "scene_rule_wishlist", "partial", 0.95),
            ),
        ),
        "cart": SceneRuleWeight(
            exact=_env_float(
                "RECOMMEND_CART_RULE_EXACT_WEIGHT",
                _config_float(file_config, "scene_rule_cart", "exact", 1.95),
            ),
            partial=_env_float(
                "RECOMMEND_CART_RULE_PARTIAL_WEIGHT",
                _config_float(file_config, "scene_rule_cart", "partial", 1.2),
            ),
        ),
    }

    return RecommendationWeightConfig(
        implicit=implicit,
        scene_similarity=scene_similarity,
        scene_rule=scene_rule,
        decay_lambda=_env_float(
            "RECOMMEND_DECAY_LAMBDA",
            _config_float(file_config, "time_decay", "lambda", 0.05),
        ),
    )
