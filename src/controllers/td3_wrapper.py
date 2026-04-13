from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class TD3Controller:
    method_id = "td3"

    def __init__(self, model=None):
        self.model = model

    @staticmethod
    def _algorithm_class():
        try:  # pragma: no cover - optional dependency
            from stable_baselines3 import TD3

            return TD3
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "stable-baselines3 is required for TD3. Install project requirements first."
            ) from exc

    @classmethod
    def build(cls, env, policy_kwargs: dict[str, Any], **kwargs):
        algo = cls._algorithm_class()
        model = algo("MlpPolicy", env, policy_kwargs=policy_kwargs, **kwargs)
        return cls(model=model)

    def train(self, total_timesteps: int, callback=None) -> None:
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def predict(self, observation, deterministic: bool = True):
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=float)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path):
        algo = cls._algorithm_class()
        return cls(model=algo.load(str(path)))

