"""Control-effort metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .tracking import _as_1d_array


def _as_action_matrix(action: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(action, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("action must be 1D or 2D")
    return array


def compute_control_effort_metrics(action: Iterable[float] | np.ndarray) -> dict[str, float]:
    action_matrix = _as_action_matrix(action)
    magnitudes = np.linalg.norm(action_matrix, ord=1, axis=1)
    diffs = np.diff(action_matrix, axis=0)
    total_variation = float(np.sum(np.abs(diffs))) if diffs.size else 0.0
    max_control_magnitude = float(np.max(np.abs(action_matrix))) if action_matrix.size else 0.0
    mean_absolute_input = float(np.mean(np.abs(action_matrix))) if action_matrix.size else 0.0
    return {
        "mean_absolute_input": mean_absolute_input,
        "total_variation": total_variation,
        "max_control_magnitude": max_control_magnitude,
        "mean_l1_magnitude": float(np.mean(magnitudes)) if magnitudes.size else 0.0,
    }
