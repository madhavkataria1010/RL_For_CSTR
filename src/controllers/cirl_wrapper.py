"""CIRL policy wrappers for paper-faithful reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from src.controllers.pid import denormalize_pid_gains
from src.controllers.pure_rl_paper import PureRLPaperPolicy


class CIRLPolicy(PureRLPaperPolicy):
    """Official MLP policy that emits normalized PID gains."""

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 16,
        output_size: int = 6,
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        extra_hidden_layers: int = 0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation,
            extra_hidden_layers=extra_hidden_layers,
        )

    def predict_physical_gains(self, observation: Iterable[float]) -> np.ndarray:
        obs = torch.as_tensor(list(observation), dtype=torch.float32)
        with torch.no_grad():
            normalized = self.forward(obs).cpu().numpy()
        return denormalize_pid_gains(normalized)


@dataclass
class CIRLController:
    """Inference wrapper returning normalized gains for the environment."""

    policy: CIRLPolicy
    device: torch.device | str = "cpu"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.policy.to(self.device)
        self.policy.eval()

    def act(self, observation: Iterable[float], deterministic: bool = True) -> np.ndarray:
        del deterministic  # Policy is deterministic by construction.
        obs = torch.as_tensor(list(observation), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.policy(obs).detach().cpu().numpy()
        return np.asarray(action, dtype=float)

    def gains(self, observation: Iterable[float]) -> np.ndarray:
        return denormalize_pid_gains(self.act(observation))
