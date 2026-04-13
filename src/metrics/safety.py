"""Safety and robustness metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .tracking import _as_1d_array


def constraint_violation_profile(
    signal: Iterable[float] | np.ndarray,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> np.ndarray:
    signal_arr = _as_1d_array(signal)
    lower_violation = np.zeros_like(signal_arr)
    upper_violation = np.zeros_like(signal_arr)
    if lower_bound is not None:
        lower_violation = np.maximum(lower_bound - signal_arr, 0.0)
    if upper_bound is not None:
        upper_violation = np.maximum(signal_arr - upper_bound, 0.0)
    return lower_violation + upper_violation


def saturation_profile(
    action: Iterable[float] | np.ndarray,
    lower_bound: float | np.ndarray | None = None,
    upper_bound: float | np.ndarray | None = None,
    atol: float = 1e-8,
) -> np.ndarray:
    action_arr = np.asarray(action, dtype=float)
    if action_arr.ndim == 1:
        action_arr = action_arr.reshape(-1, 1)
    saturated = np.zeros(action_arr.shape[0], dtype=bool)
    if lower_bound is not None:
        lower = np.asarray(lower_bound, dtype=float)
        saturated |= np.any(np.isclose(action_arr, lower, atol=atol) | (action_arr < lower), axis=1)
    if upper_bound is not None:
        upper = np.asarray(upper_bound, dtype=float)
        saturated |= np.any(np.isclose(action_arr, upper, atol=atol) | (action_arr > upper), axis=1)
    return saturated


def recovery_time(
    error: Iterable[float] | np.ndarray,
    time: Iterable[float] | np.ndarray,
    disturbance_end_time: float,
    tolerance: float,
) -> float:
    error_arr = np.abs(_as_1d_array(error))
    time_arr = _as_1d_array(time)
    if error_arr.size != time_arr.size:
        raise ValueError("error and time lengths must match")
    post_disturbance = np.where(time_arr >= disturbance_end_time)[0]
    if post_disturbance.size == 0:
        return float("inf")
    start = int(post_disturbance[0])
    for idx in range(start, error_arr.size):
        if np.all(error_arr[idx:] <= tolerance):
            return float(time_arr[idx] - disturbance_end_time)
    return float("inf")


def compute_safety_metrics(
    signal: Iterable[float] | np.ndarray,
    action: Iterable[float] | np.ndarray | None = None,
    signal_lower_bound: float | None = None,
    signal_upper_bound: float | None = None,
    action_lower_bound: float | np.ndarray | None = None,
    action_upper_bound: float | np.ndarray | None = None,
) -> dict[str, float]:
    violations = constraint_violation_profile(
        signal=signal,
        lower_bound=signal_lower_bound,
        upper_bound=signal_upper_bound,
    )
    metrics = {
        "constraint_violations": float(np.count_nonzero(violations > 0)),
        "total_constraint_violation_magnitude": float(np.sum(violations)),
    }
    if action is not None:
        saturated = saturation_profile(
            action=action,
            lower_bound=action_lower_bound,
            upper_bound=action_upper_bound,
        )
        metrics["saturation_count"] = float(np.count_nonzero(saturated))
        metrics["saturation_duration"] = float(np.sum(saturated))
    return metrics
