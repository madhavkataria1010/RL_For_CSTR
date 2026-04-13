from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class DummyTrackingEnv:
    horizon: int = 12
    disturbance_step: int | None = None
    noise_scale: float = 0.0

    def reset(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.state = np.array([0.0, 0.0], dtype=float)
        return self.state.copy(), {"reference": np.array([0.0], dtype=float)}

    def step(self, action):
        action = np.asarray(action, dtype=float).reshape(-1)
        reference = 1.0 if self.step_count >= 2 else 0.0
        disturbance = 0.25 if self.disturbance_step is not None and self.step_count >= self.disturbance_step else 0.0
        noise = self.rng.normal(0.0, self.noise_scale)
        self.state[0] = 0.82 * self.state[0] + 0.18 * reference + 0.12 * action[0] + disturbance + noise
        self.state[1] = reference
        error = reference - self.state[0]
        cost = float(error**2 + 0.05 * action[0] ** 2)
        self.step_count += 1
        terminated = self.step_count >= self.horizon
        info = {
            "time": float(self.step_count - 1),
            "cost": cost,
            "reference": np.array([reference], dtype=float),
        }
        return self.state.copy(), -cost, terminated, False, info


class DummyLinearController:
    def reset(self):
        self.last_action = np.array([0.0], dtype=float)

    def act(self, observation, deterministic: bool = True):
        obs = np.asarray(observation, dtype=float).reshape(-1)
        error = obs[1] - obs[0]
        self.last_action = np.array([0.8 * error], dtype=float)
        return self.last_action


class DummyPredictController:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic: bool = True):
        obs = np.asarray(observation, dtype=float).reshape(-1)
        error = obs[1] - obs[0]
        if deterministic:
            action = np.array([0.5 * error], dtype=float)
        else:
            action = np.array([0.5 * error + self.rng.normal(0.0, 0.05)], dtype=float)
        return action, None


@pytest.fixture
def dummy_env_factory():
    return lambda seed: DummyTrackingEnv()


@pytest.fixture
def dummy_controller_factory():
    return lambda seed: DummyLinearController()


@pytest.fixture
def dummy_predict_controller_factory():
    return lambda seed: DummyPredictController(seed=seed)
