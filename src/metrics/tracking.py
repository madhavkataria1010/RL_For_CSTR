"""Tracking-quality metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_1d_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def _integration_weights(time: np.ndarray | None, n: int) -> np.ndarray:
    if n == 0:
        return np.asarray([], dtype=float)
    if time is None:
        return np.ones(n, dtype=float)
    time = _as_1d_array(time)
    if time.size != n:
        raise ValueError("time and signal lengths must match")
    if n == 1:
        return np.ones(1, dtype=float)

    diffs = np.diff(time)
    if np.any(diffs <= 0):
        raise ValueError("time must be strictly increasing")
    weights = np.empty(n, dtype=float)
    weights[0] = diffs[0]
    weights[1:] = diffs
    return weights


def compute_tracking_metrics(
    reference: Iterable[float] | np.ndarray,
    output: Iterable[float] | np.ndarray,
    time: Iterable[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute core tracking metrics from reference and output trajectories."""

    reference_arr = _as_1d_array(reference)
    output_arr = _as_1d_array(output)
    if reference_arr.size != output_arr.size:
        raise ValueError("reference and output lengths must match")

    error = reference_arr - output_arr
    weights = _integration_weights(None if time is None else _as_1d_array(time), error.size)
    abs_error = np.abs(error)
    squared_error = error**2

    iae = float(np.sum(abs_error * weights))
    ise = float(np.sum(squared_error * weights))
    if time is None:
        itae_weights = np.arange(error.size, dtype=float)
    else:
        itae_weights = _as_1d_array(time)
    itae = float(np.sum(itae_weights * abs_error * weights))
    rmse = float(np.sqrt(np.mean(squared_error))) if error.size else 0.0

    window = max(1, min(10, error.size))
    steady_state_error = float(np.mean(error[-window:])) if error.size else 0.0

    return {
        "iae": iae,
        "ise": ise,
        "itae": itae,
        "rmse": rmse,
        "steady_state_error": steady_state_error,
    }
