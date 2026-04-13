"""Scenario perturbation utilities for the nonlinear CSTR."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Mapping

import numpy as np


def sample_uniform_multiplier(
    rng: np.random.Generator,
    pct: float,
) -> float:
    """Sample a multiplicative factor within ``±pct`` around nominal."""

    return float(rng.uniform(1.0 - pct, 1.0 + pct))


def sample_domain_randomization(
    rng: np.random.Generator,
    ranges: Mapping[str, float],
) -> Dict[str, float]:
    """Sample multiplicative factors for each randomized parameter group."""

    return {
        key: sample_uniform_multiplier(rng, float(value))
        for key, value in ranges.items()
    }


def uncertainty_grid(
    percentages: Iterable[float],
    keys: Iterable[str] = ("caf_scale", "ua_scale", "k0_scale"),
) -> Iterator[Dict[str, float]]:
    """Generate one-at-a-time uncertainty sweeps for benchmark stress tests."""

    for key in keys:
        for pct in percentages:
            yield {
                "label": f"{key}_{pct:+.0%}",
                "caf_scale": 1.0,
                "ua_scale": 1.0,
                "k0_scale": 1.0,
                key: 1.0 + pct,
            }


def apply_rate_limit(
    previous: np.ndarray,
    proposed: np.ndarray,
    rate_limit: np.ndarray | None,
) -> np.ndarray:
    """Apply per-step rate limits if configured."""

    if rate_limit is None:
        return proposed
    delta = np.clip(proposed - previous, -rate_limit, rate_limit)
    return previous + delta
