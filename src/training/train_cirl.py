"""Paper-faithful CIRL training with random search followed by PSO."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.cirl_wrapper import CIRLPolicy
from src.environments.standardized_env import make_paper_exact_env
from src.training.common_interface import build_paper_training_payload, load_project_config

try:
    import yaml
except ImportError:  # pragma: no cover - optional until environment setup.
    yaml = None

try:  # pragma: no cover - optional only in thin test envs
    from torch_pso import ParticleSwarmOptimizer
except ModuleNotFoundError:  # pragma: no cover
    ParticleSwarmOptimizer = None


@dataclass
class PaperTrainingConfig:
    """Config shared by the paper-faithful trainable methods."""

    method_id: str = "cirl_reproduced"
    scenario: str = "nominal"
    seed: int = 0
    ns: int = 120
    input_size: int = 15
    output_size: int = 6
    hidden_size: int = 16
    extra_hidden_layers: int = 0
    activation: str = "ReLU"
    episodes_per_eval: int = 3
    random_search_candidates: int = 30
    pso_particles: int = 30
    pso_iterations: int = 150
    pso_inertial_weight: float = 0.6
    pso_cognitive_coefficient: float = 1.0
    pso_social_coefficient: float = 1.0
    param_min: float = -0.1
    param_max: float = 0.1
    pid_maxiter: int = 150
    pid_popsize: int = 15
    results_root: str = "results/raw"
    experiment_name: str | None = None
    checkpoint_name: str = "policy.pt"
    device: str = "auto"
    env_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None, **overrides: Any) -> "PaperTrainingConfig":
        payload = dict(data or {})
        payload.update(overrides)
        return cls(**payload)


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load training configs.")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_run_dir(results_root: str, experiment_name: str | None, method_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = experiment_name or method_id
    run_dir = ROOT / results_root / method_id / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def flatten_parameters(model: torch.nn.Module) -> np.ndarray:
    return (
        torch.cat([param.detach().flatten().cpu() for param in model.parameters()])
        .numpy()
        .astype(np.float64)
    )


def assign_parameters(model: torch.nn.Module, flat_vector: Sequence[float], device: torch.device) -> None:
    vector = torch.as_tensor(np.asarray(flat_vector, dtype=np.float32), device=device)
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            size = param.numel()
            param.copy_(vector[offset : offset + size].view_as(param))
            offset += size


def env_factory_from_config(config: PaperTrainingConfig) -> Callable[[int | None], Any]:
    def _factory(seed: int | None = None):
        env_kwargs = dict(config.env_overrides)
        env_kwargs.update({"seed": seed if seed is not None else config.seed, "ns": config.ns})
        return make_paper_exact_env(
            method=config.method_id,
            scenario=config.scenario,
            config=env_kwargs,
        )

    return _factory


def evaluate_policy(
    policy: torch.nn.Module,
    env_factory: Callable[[int | None], Any],
    episodes: int,
    base_seed: int,
    device: torch.device,
) -> tuple[float, list[float]]:
    """Return mean episode cost over a fixed set of seeds."""

    policy.eval()
    returns: list[float] = []
    for offset in range(episodes):
        env = env_factory(base_seed + offset)
        state, _ = env.reset(seed=base_seed + offset)
        episode_cost = 0.0
        done = False
        while not done:
            obs = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                action = policy(obs).detach().cpu().numpy()
            state, reward, done, _, _ = env.step(action)
            episode_cost += float(reward)
        returns.append(episode_cost)
    return float(np.mean(returns)), returns


class RandomSearchPSOTrainer:
    """Reimplementation of the official random-search + PSO training family."""

    def __init__(
        self,
        config: PaperTrainingConfig,
        policy_factory: Callable[[], torch.nn.Module],
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.rng = np.random.default_rng(config.seed)
        self.policy = policy_factory().to(self.device)
        self.template_vector = flatten_parameters(self.policy)
        self.dimension = int(self.template_vector.size)
        self.env_factory = env_factory_from_config(config)
        self.env = self.env_factory(config.seed)
        self.all_scores: list[float] = []
        self.all_state_dicts: list[dict[str, Any]] = []
        self.iter_scores: list[float] = []
        self.iter_state_dicts: list[dict[str, Any]] = []

    def sample_uniform_vector(self) -> np.ndarray:
        return self.rng.uniform(
            self.config.param_min,
            self.config.param_max,
            size=self.dimension,
        ).astype(np.float64)

    def _clone_state_dict(self) -> dict[str, Any]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.policy.state_dict().items()
        }

    def criterion(self) -> float:
        """Evaluate the current policy like the official training loop."""

        self.policy.eval()
        returns: list[float] = []
        for _ in range(self.config.episodes_per_eval):
            state, _ = self.env.reset()
            episode_cost = 0.0
            while True:
                obs = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action = self.policy(obs).detach().cpu().numpy()
                state, reward, done, _, _ = self.env.step(action)
                episode_cost += float(reward)
                if done:
                    break
            returns.append(float(episode_cost))

        score = float(np.mean(returns))
        snapshot = self._clone_state_dict()
        self.all_scores.append(score)
        self.all_state_dicts.append(snapshot)
        self.iter_scores.append(score)
        self.iter_state_dicts.append(snapshot)
        return score

    def objective(self, param_vector: Sequence[float], seed_offset: int = 0) -> float:
        del seed_offset
        assign_parameters(self.policy, param_vector, self.device)
        return self.criterion()

    def _run_manual_fallback(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        best_vector = self.template_vector.copy()
        best_score = float("inf")
        random_search_scores: list[float] = []

        for candidate_idx in range(self.config.random_search_candidates):
            vector = self.sample_uniform_vector()
            score = self.objective(vector, seed_offset=1000 + candidate_idx * 10)
            random_search_scores.append(score)
            if score < best_score:
                best_score = score
                best_vector = vector.copy()

        particles = np.stack(
            [self.sample_uniform_vector() for _ in range(self.config.pso_particles)],
            axis=0,
        )
        particles[0] = best_vector.copy()
        velocities = self.rng.uniform(
            -(self.config.param_max - self.config.param_min),
            self.config.param_max - self.config.param_min,
            size=particles.shape,
        )
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.config.pso_particles, np.inf, dtype=np.float64)
        global_best_position = best_vector.copy()
        global_best_score = best_score
        iteration_best_scores: list[float] = []

        for _ in range(self.config.pso_iterations):
            for particle_idx in range(self.config.pso_particles):
                score = self.objective(particles[particle_idx])
                if score < personal_best_scores[particle_idx]:
                    personal_best_scores[particle_idx] = score
                    personal_best_positions[particle_idx] = particles[particle_idx].copy()
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[particle_idx].copy()

            iteration_best_scores.append(float(global_best_score))

            r_personal = self.rng.random(size=particles.shape)
            r_global = self.rng.random(size=particles.shape)
            velocities = (
                self.config.pso_inertial_weight * velocities
                + self.config.pso_cognitive_coefficient
                * r_personal
                * (personal_best_positions - particles)
                + self.config.pso_social_coefficient
                * r_global
                * (global_best_position - particles)
            )
            particles = np.clip(
                particles + velocities,
                self.config.param_min,
                self.config.param_max,
            )

        assign_parameters(self.policy, global_best_position, self.device)
        summary = {
            "best_score": float(global_best_score),
            "random_search_scores": random_search_scores,
            "iteration_best_scores": iteration_best_scores,
            "device": str(self.device),
            "optimizer_backend": "manual_fallback",
        }
        return self.policy, summary

    def run(self) -> tuple[torch.nn.Module, dict[str, Any]]:
        if ParticleSwarmOptimizer is None:
            return self._run_manual_fallback()

        best_score = float("inf")
        best_state_dict = self._clone_state_dict()
        random_search_scores: list[float] = []

        for _ in range(self.config.random_search_candidates):
            vector = self.sample_uniform_vector()
            score = self.objective(vector)
            random_search_scores.append(score)
            if score < best_score:
                best_score = score
                best_state_dict = self._clone_state_dict()

        optim = ParticleSwarmOptimizer(
            self.policy.parameters(),
            inertial_weight=self.config.pso_inertial_weight,
            cognitive_coefficient=self.config.pso_cognitive_coefficient,
            social_coefficient=self.config.pso_social_coefficient,
            num_particles=self.config.pso_particles,
            max_param_value=self.config.param_max,
            min_param_value=self.config.param_min,
        )
        iteration_best_scores: list[float] = []

        for _ in range(self.config.pso_iterations):
            self.iter_scores = []
            self.iter_state_dicts = []

            def closure():
                optim.zero_grad()
                return self.criterion()

            optim.step(closure)
            if not self.iter_scores:
                continue
            iter_best_index = int(np.argmin(np.asarray(self.iter_scores, dtype=float)))
            iter_best_score = float(self.iter_scores[iter_best_index])
            iteration_best_scores.append(iter_best_score)
            if iter_best_score < best_score:
                best_score = iter_best_score
                best_state_dict = self.iter_state_dicts[iter_best_index]

        self.policy.load_state_dict(best_state_dict)
        summary = {
            "best_score": float(best_score),
            "random_search_scores": random_search_scores,
            "iteration_best_scores": iteration_best_scores,
            "device": str(self.device),
            "optimizer_backend": "torch_pso",
        }
        return self.policy, summary


def save_training_outputs(
    run_dir: Path,
    policy: torch.nn.Module,
    config: PaperTrainingConfig,
    summary: Mapping[str, Any],
) -> dict[str, str]:
    checkpoint_path = run_dir / config.checkpoint_name
    torch.save(policy.state_dict(), checkpoint_path)

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(json.dumps(dict(summary), indent=2), encoding="utf-8")

    artifact_paths = {
        "method_id": config.method_id,
        "scenario_id": config.scenario,
        "seed": config.seed,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "summary": str(summary_path),
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(artifact_paths, indent=2), encoding="utf-8")
    return artifact_paths


def build_cirl_policy(config: PaperTrainingConfig) -> CIRLPolicy:
    activation = getattr(torch.nn, config.activation)
    return CIRLPolicy(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        activation=activation,
        extra_hidden_layers=config.extra_hidden_layers,
    )


def train_cirl(
    config: PaperTrainingConfig | Mapping[str, Any] | Sequence[str | Path] | str | Path | None = None,
    **overrides: Any,
) -> dict[str, str]:
    if isinstance(config, Sequence) and not isinstance(config, (str, Path, bytes)):
        merged = load_project_config([str(path) for path in config])
        cfg = PaperTrainingConfig.from_mapping(
            build_paper_training_payload(merged, default_method_id="cirl_reproduced"),
            **overrides,
        )
    elif isinstance(config, (str, Path)):
        cfg = PaperTrainingConfig.from_mapping(load_yaml_config(config), **overrides)
    elif isinstance(config, PaperTrainingConfig):
        cfg = PaperTrainingConfig.from_mapping(asdict(config), **overrides)
    else:
        cfg = PaperTrainingConfig.from_mapping(config, **overrides)

    run_dir = ensure_run_dir(cfg.results_root, cfg.experiment_name, cfg.method_id)
    trainer = RandomSearchPSOTrainer(cfg, lambda: build_cirl_policy(cfg))
    policy, summary = trainer.run()
    return save_training_outputs(run_dir, policy, cfg, summary)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the reproduced CIRL method.")
    parser.add_argument("--config", nargs="+")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(list(argv or sys.argv[1:]))
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    artifacts = train_cirl(args.config if args.config else None, **overrides)
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
