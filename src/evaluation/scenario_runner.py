"""Scenario-level evaluation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.metrics import (
    compute_control_effort_metrics,
    recovery_time,
    compute_safety_metrics,
    compute_tracking_metrics,
    compute_transient_metrics,
)

from .rollout import RolloutEpisode, rollout_episode


@dataclass(slots=True)
class EvaluationRun:
    """One rollout plus computed metrics."""

    rollout: RolloutEpisode
    metrics: dict[str, float]
    artifact_paths: dict[str, str] = field(default_factory=dict)


def default_metric_bundle(
    rollout: RolloutEpisode,
    *,
    signal_index: int = 0,
    safety_bounds: tuple[float | None, float | None] = (None, None),
    action_bounds: tuple[np.ndarray | float | None, np.ndarray | float | None] = (None, None),
) -> dict[str, float]:
    output = rollout.observations[:, signal_index]
    reference = (
        rollout.references[:, min(signal_index, rollout.references.shape[1] - 1)]
        if rollout.references is not None
        else np.zeros_like(output)
    )
    action_signal = rollout.applied_actions if rollout.applied_actions is not None else rollout.actions
    metrics = {}
    metrics.update(compute_tracking_metrics(reference=reference, output=output, time=rollout.time))
    metrics.update(compute_transient_metrics(reference=reference, output=output, time=rollout.time))
    metrics.update(compute_control_effort_metrics(action_signal))
    metadata_action_bounds = (
        np.asarray(rollout.metadata["action_lower_bound"], dtype=float)
        if "action_lower_bound" in rollout.metadata
        else action_bounds[0],
        np.asarray(rollout.metadata["action_upper_bound"], dtype=float)
        if "action_upper_bound" in rollout.metadata
        else action_bounds[1],
    )
    metrics.update(
        compute_safety_metrics(
            signal=output,
            action=action_signal,
            signal_lower_bound=safety_bounds[0],
            signal_upper_bound=safety_bounds[1],
            action_lower_bound=metadata_action_bounds[0],
            action_upper_bound=metadata_action_bounds[1],
        )
    )
    if "disturbance_start_step" in rollout.metadata and rollout.time.size:
        disturbance_time = rollout.time[min(int(rollout.metadata["disturbance_start_step"]), rollout.time.size - 1)]
        metrics["recovery_time"] = recovery_time(
            error=reference - output,
            time=rollout.time,
            disturbance_end_time=float(disturbance_time),
            tolerance=0.02 * max(abs(reference[-1]), 1.0),
        )
    else:
        metrics["recovery_time"] = 0.0
    metrics["episode_return"] = float(np.sum(rollout.rewards))
    metrics["episode_cost"] = float(np.sum(rollout.costs))
    return metrics


def evaluate_scenario(
    *,
    env_factory: Callable[[int], Any],
    controller_factory: Callable[[int], Any],
    method: str,
    scenario: str,
    seeds: list[int],
    horizon: int,
    output_dir: str | Path | None = None,
    deterministic: bool = True,
    metric_fn: Callable[[RolloutEpisode], dict[str, float]] | None = None,
) -> list[EvaluationRun]:
    """Evaluate a controller across seeds for one scenario."""

    output_path = Path(output_dir) if output_dir is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    results: list[EvaluationRun] = []
    for seed in seeds:
        env = env_factory(seed)
        controller = controller_factory(seed)
        rollout = rollout_episode(
            env=env,
            controller=controller,
            method=method,
            scenario=scenario,
            seed=seed,
            horizon=horizon,
            deterministic=deterministic,
        )
        metrics = metric_fn(rollout) if metric_fn is not None else default_metric_bundle(rollout)
        artifacts: dict[str, str] = {}
        if output_path is not None:
            csv_path = output_path / f"{method}__{scenario}__seed{seed}.csv"
            rollout.save_csv(csv_path)
            artifacts["trajectory_csv"] = str(csv_path)
        results.append(EvaluationRun(rollout=rollout, metrics=metrics, artifact_paths=artifacts))
    return results
