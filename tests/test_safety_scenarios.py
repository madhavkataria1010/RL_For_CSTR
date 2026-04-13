from __future__ import annotations

import math

import numpy as np

from src.metrics.safety import compute_safety_metrics, recovery_time


def test_recovery_time_after_disturbance():
    time = np.arange(6, dtype=float)
    error = np.array([0.0, 0.8, 0.7, 0.2, 0.05, 0.01])
    value = recovery_time(error=error, time=time, disturbance_end_time=2.0, tolerance=0.1)
    assert math.isclose(value, 2.0)


def test_safety_metrics_count_violation_magnitude_and_duration():
    signal = np.array([0.9, 1.2, 1.1, 0.8])
    action = np.array([[0.0], [1.0], [1.0], [0.4]])
    metrics = compute_safety_metrics(
        signal=signal,
        action=action,
        signal_upper_bound=1.0,
        action_lower_bound=np.array([-1.0]),
        action_upper_bound=np.array([1.0]),
    )

    assert metrics["constraint_violations"] == 2.0
    assert math.isclose(metrics["total_constraint_violation_magnitude"], 0.3)
    assert metrics["saturation_duration"] == 2.0
