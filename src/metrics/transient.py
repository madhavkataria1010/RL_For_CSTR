"""Transient-response metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .tracking import _as_1d_array


def _resolve_directional_signal(
    reference: np.ndarray,
    output: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    initial_reference = float(reference[0])
    final_reference = float(reference[-1])
    span = final_reference - initial_reference
    if np.isclose(span, 0.0):
        span = np.sign(final_reference - float(output[0])) or 1.0
    direction = 1.0 if span >= 0 else -1.0
    normalized = direction * output
    return normalized, direction, initial_reference, final_reference


def overshoot(reference: Iterable[float], output: Iterable[float]) -> float:
    reference_arr = _as_1d_array(reference)
    output_arr = _as_1d_array(output)
    normalized, direction, _, final_reference = _resolve_directional_signal(reference_arr, output_arr)
    final = direction * final_reference
    peak = float(np.max(normalized))
    overshoot_value = peak - final
    return float(max(0.0, overshoot_value))


def rise_time(
    reference: Iterable[float],
    output: Iterable[float],
    time: Iterable[float] | None = None,
    lower: float = 0.1,
    upper: float = 0.9,
) -> float:
    reference_arr = _as_1d_array(reference)
    output_arr = _as_1d_array(output)
    time_arr = np.arange(output_arr.size, dtype=float) if time is None else _as_1d_array(time)
    normalized, direction, initial_reference, final_reference = _resolve_directional_signal(reference_arr, output_arr)
    low_target = direction * (initial_reference + lower * (final_reference - initial_reference))
    high_target = direction * (initial_reference + upper * (final_reference - initial_reference))

    low_hits = np.where(normalized >= low_target)[0]
    high_hits = np.where(normalized >= high_target)[0]
    if low_hits.size == 0 or high_hits.size == 0:
        return float("inf")
    return float(time_arr[high_hits[0]] - time_arr[low_hits[0]])


def settling_time(
    reference: Iterable[float],
    output: Iterable[float],
    time: Iterable[float] | None = None,
    tolerance: float = 0.02,
) -> float:
    reference_arr = _as_1d_array(reference)
    output_arr = _as_1d_array(output)
    time_arr = np.arange(output_arr.size, dtype=float) if time is None else _as_1d_array(time)
    final_reference = float(reference_arr[-1])
    band = tolerance * max(abs(final_reference), 1.0)
    within_band = np.abs(output_arr - final_reference) <= band
    for idx in range(output_arr.size):
        if np.all(within_band[idx:]):
            return float(time_arr[idx])
    return float("inf")


def compute_transient_metrics(
    reference: Iterable[float] | np.ndarray,
    output: Iterable[float] | np.ndarray,
    time: Iterable[float] | np.ndarray | None = None,
    tolerance: float = 0.02,
) -> dict[str, float]:
    return {
        "overshoot": overshoot(reference, output),
        "rise_time": rise_time(reference, output, time=time),
        "settling_time": settling_time(reference, output, time=time, tolerance=tolerance),
    }
