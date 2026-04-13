"""Paper baseline direct-action RL controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


class PureRLPaperPolicy(torch.nn.Module):
    """Official MLP policy architecture with direct normalized actions."""

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        output_size: int = 2,
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        extra_hidden_layers: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        # The official repo declares this module list as ``n_layers`` and includes
        # it in saved state_dicts even though the forward pass effectively uses the
        # network as a two-hidden-layer MLP for the paper experiments.
        self.n_layers = torch.nn.ModuleList(
            torch.nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(extra_hidden_layers)
        )
        self.output = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = self.activation(self.hidden1(x))
        y = self.activation(self.hidden2(y))
        for layer in self.n_layers:
            y = self.activation(layer(y))
        return F.tanh(self.output(y))


class OfficialReplayPolicy(torch.nn.Module):
    """Compatibility policy for official visualization artifacts with mu/std heads."""

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        output_size: int = 2,
        activation: type[torch.nn.Module] = torch.nn.ReLU,
        extra_hidden_layers: int = 0,
        *,
        pid_output: bool,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.hidden2 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.n_layers = torch.nn.ModuleList(
            torch.nn.Linear(hidden_size, hidden_size, bias=True)
            for _ in range(extra_hidden_layers)
        )
        self.output_mu = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.output_std = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.activation = activation()
        self.pid_output = pid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        y = self.activation(self.hidden1(x))
        y = self.activation(self.hidden2(y))
        for layer in self.n_layers:
            y = self.activation(layer(y))
        mu = self.output_mu(y)
        if self.pid_output:
            return torch.tanh(mu)
        return torch.clamp(mu, -1.0, 1.0)


@dataclass
class PureRLPaperController:
    """Inference wrapper for the paper baseline policy."""

    policy: PureRLPaperPolicy
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
