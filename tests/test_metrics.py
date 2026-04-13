from __future__ import annotations

import math

import numpy as np

from src.metrics.control_effort import compute_control_effort_metrics
from src.metrics.safety import compute_safety_metrics
from src.metrics.tracking import compute_tracking_metrics
from src.metrics.transient import compute_transient_metrics


def test_tracking_metrics_on_toy_signal():
    reference = np.array([1.0, 1.0, 1.0, 1.0])
    output = np.array([0.0, 0.5, 1.0, 1.0])
    metrics = compute_tracking_metrics(reference=reference, output=output, time=np.arange(4))

    assert math.isclose(metrics["iae"], 1.5)
    assert math.isclose(metrics["ise"], 1.25)
    assert math.isclose(metrics["rmse"], math.sqrt(1.25 / 4))


def test_transient_metrics_capture_overshoot_and_settling():
    reference = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    output = np.array([0.0, 0.3, 1.1, 1.02, 1.0])
    metrics = compute_transient_metrics(reference=reference, output=output, time=np.arange(5), tolerance=0.05)

    assert math.isclose(metrics["overshoot"], 0.1, rel_tol=1e-6)
    assert metrics["rise_time"] >= 0.0
    assert metrics["settling_time"] == 3.0


def test_control_and_safety_metrics():
    action = np.array([[0.0], [0.5], [1.0]])
    signal = np.array([0.2, 1.3, 0.8])
    effort = compute_control_effort_metrics(action)
    safety = compute_safety_metrics(
        signal=signal,
        action=action,
        signal_lower_bound=0.0,
        signal_upper_bound=1.0,
        action_lower_bound=np.array([-1.0]),
        action_upper_bound=np.array([1.0]),
    )

    assert math.isclose(effort["total_variation"], 1.0)
    assert effort["max_control_magnitude"] == 1.0
    assert safety["constraint_violations"] == 1.0
    assert safety["saturation_count"] == 1.0
