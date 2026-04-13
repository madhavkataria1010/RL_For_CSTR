"""Named environment builders for paper-faithful and extended scenarios."""

from __future__ import annotations

from typing import Any, Mapping

from .cstr import CSTRConfig
from .wrappers import make_cstr_env


MODERN_RL_METHODS = {"sac", "td3", "tqc", "ppo"}
DIRECT_ACTION_METHODS = MODERN_RL_METHODS | {"pure_rl_paper"}


def _base_method_overrides(method: str) -> dict[str, Any]:
    return {"norm_rl": method in DIRECT_ACTION_METHODS}


def _scenario_overrides(scenario: str) -> dict[str, Any]:
    if scenario == "nominal":
        return {"test": False, "dist": False, "highop": False}
    if scenario == "nominal_test":
        return {"test": True, "dist": False, "highop": False}
    if scenario == "disturbance":
        return {"test": False, "dist": True, "highop": False}
    if scenario == "disturbance_test":
        return {"test": True, "dist": True, "highop": False}
    if scenario == "highop":
        return {"test": False, "dist": False, "highop": True}
    if scenario == "highop_test":
        return {"test": True, "dist": False, "highop": True}
    if scenario == "uncertainty_pm10":
        return {
            "test": True,
            "dist": False,
            "highop": False,
            "uncertainty": {"caf_scale": 1.10, "ua_scale": 1.10, "k0_scale": 1.10},
        }
    if scenario == "uncertainty_pm20":
        return {
            "test": True,
            "dist": False,
            "highop": False,
            "uncertainty": {"caf_scale": 1.20, "ua_scale": 1.20, "k0_scale": 1.20},
        }
    if scenario == "noise":
        return {
            "test": True,
            "dist": False,
            "highop": False,
            "measurement_noise": {"enabled": True, "ca": 0.003, "cb": 0.003, "cc": 0.003, "t": 0.3, "v": 0.03},
        }
    if scenario == "saturation":
        return {
            "test": True,
            "dist": False,
            "highop": False,
            "rate_limit": (5.0, 0.2),
        }
    if scenario == "unseen_setpoints":
        return {"test": True, "dist": False, "highop": False}
    raise ValueError(f"Unsupported scenario '{scenario}'.")


def make_paper_exact_env(
    method: str,
    scenario: str,
    config: CSTRConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
):
    """Create a paper-faithful environment with method-aware action mode."""

    merged: dict[str, Any] = {}
    if isinstance(config, CSTRConfig):
        merged.update(config.to_dict())
    else:
        merged.update(dict(config or {}))
    merged.update(_base_method_overrides(method))
    merged.update(_scenario_overrides(scenario))
    merged.update(overrides)
    return make_cstr_env(merged)


def make_benchmark_env(
    method: str,
    scenario: str,
    config: CSTRConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
):
    """Create an environment for the standardized benchmark layer."""

    merged: dict[str, Any] = {}
    if isinstance(config, CSTRConfig):
        merged.update(config.to_dict())
    else:
        merged.update(dict(config or {}))
    merged.update(_base_method_overrides(method))
    merged.update(_scenario_overrides(scenario))
    merged["paper_exact"] = False
    merged.update(overrides)
    return make_cstr_env(merged)


def build_standardized_env(config: Mapping[str, Any]):
    method_id = str(config.get("method_id", config.get("method", {}).get("id", "sac")))
    scenario_id = str(config.get("scenario_id", config.get("scenario", {}).get("id", "nominal")))
    env_config = dict(config.get("env_overrides", {}))
    return make_benchmark_env(method_id, scenario_id, config=env_config)
