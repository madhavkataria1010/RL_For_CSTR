"""Generic rollout helpers for controller evaluation."""

from __future__ import annotations

from csv import DictWriter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class SupportsReset(Protocol):
    def reset(self, seed: int | None = None) -> Any:
        ...


class SupportsStep(Protocol):
    def step(self, action: np.ndarray) -> Any:
        ...


def _infer_time_step(env: Any) -> float:
    time_grid = getattr(env, "time_grid", None)
    if time_grid is None:
        return 1.0
    time_array = np.asarray(time_grid, dtype=float).reshape(-1)
    if time_array.size < 2:
        return 1.0
    diffs = np.diff(time_array)
    positive = diffs[diffs > 0]
    if positive.size == 0:
        return 1.0
    return float(positive[0])


def _extract_reset(reset_result: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(reset_result, tuple) and len(reset_result) == 2 and isinstance(reset_result[1], dict):
        observation, info = reset_result
        return np.asarray(observation, dtype=float), info
    return np.asarray(reset_result, dtype=float), {}


def _extract_step(step_result: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return (
            np.asarray(observation, dtype=float),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info),
        )
    if len(step_result) == 4:
        observation, reward, done, info = step_result
        return (
            np.asarray(observation, dtype=float),
            float(reward),
            bool(done),
            False,
            dict(info),
        )
    raise ValueError("step() must return a 4-tuple or 5-tuple")


def _controller_action(controller: Any, observation: np.ndarray, deterministic: bool) -> np.ndarray:
    if hasattr(controller, "act"):
        action = controller.act(observation, deterministic=deterministic)
    elif hasattr(controller, "predict"):
        prediction = controller.predict(observation, deterministic=deterministic)
        action = prediction[0] if isinstance(prediction, tuple) else prediction
    elif callable(controller):
        action = controller(observation)
    else:
        raise TypeError("controller must define act(), predict(), or be callable")
    return np.asarray(action, dtype=float).reshape(-1)


@dataclass(slots=True)
class RolloutEpisode:
    """Container for one evaluation rollout."""

    method: str
    scenario: str
    seed: int
    time: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    applied_actions: np.ndarray | None
    rewards: np.ndarray
    costs: np.ndarray
    references: np.ndarray | None = None
    terminated: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, time_value in enumerate(self.time):
            row: dict[str, Any] = {
                "method": self.method,
                "scenario": self.scenario,
                "seed": self.seed,
                "step": index,
                "time": float(time_value),
                "reward": float(self.rewards[index]),
                "cost": float(self.costs[index]),
            }
            for obs_idx, value in enumerate(self.observations[index]):
                row[f"observation_{obs_idx}"] = float(value)
            for action_idx, value in enumerate(self.actions[index]):
                row[f"action_{action_idx}"] = float(value)
            if self.applied_actions is not None:
                for action_idx, value in enumerate(np.asarray(self.applied_actions[index], dtype=float).reshape(-1)):
                    row[f"applied_action_{action_idx}"] = float(value)
            if self.references is not None:
                reference_row = np.asarray(self.references[index], dtype=float).reshape(-1)
                for ref_idx, value in enumerate(reference_row):
                    row[f"reference_{ref_idx}"] = float(value)
            rows.append(row)
        return rows

    def save_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_rows()
        fieldnames = list(rows[0].keys()) if rows else ["method", "scenario", "seed", "step", "time", "reward", "cost"]
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_path


def rollout_episode(
    env: SupportsReset & SupportsStep,
    controller: Any,
    *,
    method: str,
    scenario: str,
    seed: int,
    horizon: int,
    deterministic: bool = True,
) -> RolloutEpisode:
    """Run one environment/controller rollout with a gym-like API."""

    if hasattr(controller, "reset"):
        controller.reset()

    observation, reset_info = _extract_reset(env.reset(seed=seed))
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    costs: list[float] = []
    applied_actions: list[np.ndarray] = []
    times: list[float] = []
    references: list[np.ndarray] = []
    terminated = False
    truncated = False
    time_step = _infer_time_step(env)

    for step in range(horizon):
        action = _controller_action(controller, observation, deterministic=deterministic)
        next_observation, reward, terminated, truncated, info = _extract_step(env.step(action))

        observations.append(observation.copy())
        actions.append(action.copy())
        rewards.append(float(reward))
        costs.append(float(info.get("cost", -reward)))
        applied_actions.append(np.asarray(info.get("u", action), dtype=float).reshape(-1))
        # Build a monotonic evaluation axis from the environment step size. The raw
        # env info time can repeat at the terminal step and can reset when the paper
        # training environment internally advances to a new setpoint segment.
        times.append(float(step * time_step))
        if "reference" in info:
            references.append(np.asarray(info["reference"], dtype=float).reshape(-1))

        observation = next_observation
        if terminated or truncated:
            break

    reference_array = np.vstack(references) if references else None
    metadata = {**reset_info, "num_steps": len(times), "time_step": time_step}
    return RolloutEpisode(
        method=method,
        scenario=scenario,
        seed=seed,
        time=np.asarray(times, dtype=float),
        observations=np.vstack(observations) if observations else np.empty((0, 0)),
        actions=np.vstack(actions) if actions else np.empty((0, 0)),
        applied_actions=np.vstack(applied_actions) if applied_actions else None,
        rewards=np.asarray(rewards, dtype=float),
        costs=np.asarray(costs, dtype=float),
        references=reference_array,
        terminated=terminated,
        truncated=truncated,
        metadata=metadata,
    )
