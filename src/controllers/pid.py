"""Static PID helpers aligned with the official CIRL implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.environments.cstr import (
    PID_GAIN_HIGH,
    PID_GAIN_LOW,
    pid_velocity_update,
)


def normalize_pid_gains(gains: Sequence[float]) -> np.ndarray:
    """Map physical PID gains to the normalized policy space."""

    gains = np.asarray(gains, dtype=float)
    return 2.0 * (gains - PID_GAIN_LOW) / (PID_GAIN_HIGH - PID_GAIN_LOW) - 1.0


def denormalize_pid_gains(normalized_gains: Sequence[float]) -> np.ndarray:
    """Map normalized gains back to the physical controller space."""

    normalized_gains = np.asarray(normalized_gains, dtype=float)
    return ((normalized_gains + 1.0) / 2.0) * (
        PID_GAIN_HIGH - PID_GAIN_LOW
    ) + PID_GAIN_LOW


@dataclass
class StaticPIDController:
    """A frozen gain controller that emits the same normalized gains each step."""

    normalized_gains: np.ndarray

    @classmethod
    def from_physical_gains(cls, gains: Iterable[float]) -> "StaticPIDController":
        return cls(normalized_gains=normalize_pid_gains(np.asarray(list(gains), dtype=float)))

    @property
    def gains(self) -> np.ndarray:
        return denormalize_pid_gains(self.normalized_gains)

    def act(self, _observation: np.ndarray | None = None, deterministic: bool = True) -> np.ndarray:
        del deterministic
        return np.asarray(self.normalized_gains, dtype=float).copy()


def rollout_pid_velocity(
    gains: Sequence[float],
    error: Sequence[float],
    error_history: np.ndarray,
    action_history: list[np.ndarray],
    time_window: Sequence[float],
) -> np.ndarray:
    """Thin wrapper around the official velocity-form update."""

    return pid_velocity_update(
        np.asarray(gains, dtype=float),
        np.asarray(error, dtype=float),
        np.asarray(error_history, dtype=float),
        action_history,
        np.asarray(time_window, dtype=float),
    )
