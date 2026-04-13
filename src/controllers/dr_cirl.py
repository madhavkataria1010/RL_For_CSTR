"""Domain-randomized CIRL controller definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.controllers.cirl_wrapper import CIRLController, CIRLPolicy


class DRCIRLPolicy(CIRLPolicy):
    """Same architecture as CIRL, reserved for domain-randomized training."""


@dataclass
class DRCIRLController(CIRLController):
    """Inference wrapper for DR-CIRL."""

    policy: DRCIRLPolicy

    def act(self, observation: Iterable[float], deterministic: bool = True) -> np.ndarray:
        return np.asarray(super().act(observation, deterministic=deterministic), dtype=float)
