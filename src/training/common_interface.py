from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.config import load_and_merge_yaml


@dataclass
class TrainingArtifacts:
    method_id: str
    best_model_path: Path
    history_path: Path | None
    manifest_path: Path
    run_dir: Path
    metadata: dict[str, Any]


class TrainableController:
    """Minimal protocol-like base class for trainable controllers."""

    method_id: str

    def predict(self, observation, deterministic: bool = True):
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path):
        raise NotImplementedError


DEFAULT_TRAINING_BUDGETS = {
    "debug_local": {
        "total_env_steps": 200,
        "evaluation_frequency": 50,
        "evaluation_episodes": 1,
    },
    "cuda_validation": {
        "total_env_steps": 2_000,
        "evaluation_frequency": 250,
        "evaluation_episodes": 3,
    },
    "cuda_full": {
        "total_env_steps": 10_000,
        "evaluation_frequency": 1_000,
        "evaluation_episodes": 5,
    },
    "accelerated_8h": {
        "total_env_steps": 10_000,
        "evaluation_frequency": 1_000,
        "evaluation_episodes": 5,
    },
    "results_repair": {
        "total_env_steps": 50_000,
        "evaluation_frequency": 5_000,
        "evaluation_episodes": 10,
    },
}

DEFAULT_PAPER_EXECUTION_OVERRIDES = {
    "debug_local": {
        "episodes_per_eval": 1,
        "random_search_candidates": 3,
        "pso_particles": 4,
        "pso_iterations": 5,
        "pid_maxiter": 3,
        "pid_popsize": 4,
    },
    "cuda_validation": {
        "episodes_per_eval": 2,
        "random_search_candidates": 10,
        "pso_particles": 10,
        "pso_iterations": 25,
        "pid_maxiter": 20,
        "pid_popsize": 8,
    },
    "cuda_full": {},
    "accelerated_8h": {
        "episodes_per_eval": 2,
        "random_search_candidates": 10,
        "pso_particles": 10,
        "pso_iterations": 25,
        "pid_maxiter": 20,
        "pid_popsize": 8,
    },
    "results_repair": {},
}

_KNOWN_ENV_KEYS = {
    "ns",
    "test",
    "ds_mode",
    "norm_rl",
    "dist",
    "dist_train",
    "dist_obs",
    "highop",
    "paper_exact",
    "seed",
    "time_start",
    "time_end",
    "ca_ss",
    "t_ss",
    "v_ss",
    "initial_pid_action",
    "direct_action_bounds",
    "pid_action_bounds",
    "disturbance_start_step",
    "training_disturbances",
    "evaluation_disturbance",
    "rate_limit",
    "measurement_noise",
    "process",
    "domain_randomization",
    "uncertainty",
}


def load_project_config(config_paths: list[str]) -> dict[str, Any]:
    return load_and_merge_yaml(config_paths)


def resolve_method_id(config: dict[str, Any], fallback: str | None = None) -> str:
    method_id = config.get("method", {}).get("id")
    if method_id:
        return str(method_id)
    if fallback is not None:
        return fallback
    raise KeyError("No method.id found in configuration.")


def resolve_seed(config: dict[str, Any]) -> int:
    execution = config.get("execution", {})
    seeds = execution.get("seeds", [0])
    return int(seeds[0])


def resolve_execution_id(config: dict[str, Any]) -> str:
    return str(config.get("execution", {}).get("id", "debug_local"))


def resolve_scenario_id(config: dict[str, Any]) -> str:
    return str(config.get("scenario", {}).get("id", "nominal"))


def _normalize_env_overrides(env_block: dict[str, Any] | None) -> dict[str, Any]:
    env_block = dict(env_block or {})
    if "disturbance_activation_step" in env_block:
        env_block["disturbance_start_step"] = env_block.pop("disturbance_activation_step")
    if "training_ca_in_values" in env_block:
        env_block["training_disturbances"] = tuple(env_block.pop("training_ca_in_values"))
    if "eval_ca_in_value" in env_block:
        env_block["evaluation_disturbance"] = env_block.pop("eval_ca_in_value")
    return {key: value for key, value in env_block.items() if key in _KNOWN_ENV_KEYS}


def build_paper_training_payload(config: dict[str, Any], *, default_method_id: str) -> dict[str, Any]:
    method = config.get("method", {})
    paper_exact = config.get("paper_exact", {})
    policy = paper_exact.get("policy", {})
    execution_id = resolve_execution_id(config)
    scenario_id = resolve_scenario_id(config)
    execution = config.get("execution", {})
    smoke_horizon = execution.get("smoke_test_horizon_steps")
    execution_overrides = DEFAULT_PAPER_EXECUTION_OVERRIDES.get(execution_id, {})
    ns = int(paper_exact.get("train_env", {}).get("ns", 120))
    if smoke_horizon:
        ns = min(ns, int(smoke_horizon))

    payload = {
        "method_id": str(method.get("id", default_method_id)),
        "scenario": scenario_id,
        "seed": resolve_seed(config),
        "ns": ns,
        "input_size": int(policy.get("input_size", 15)),
        "output_size": int(policy.get("output_size", 6 if "cirl" in default_method_id else 2)),
        "hidden_size": int((policy.get("hidden_sizes") or [16])[0]),
        "extra_hidden_layers": int(
            policy.get(
                "extra_hidden_layers_declared",
                max(len(policy.get("hidden_sizes", [16, 16])) - 2, 0),
            )
        ),
        "activation": str(policy.get("activation", "ReLU")),
        "episodes_per_eval": int(
            execution_overrides.get(
                "episodes_per_eval",
                paper_exact.get("optimizer_family", {}).get("objective_rollout_repetitions", 3),
            )
        ),
        "random_search_candidates": int(
            execution_overrides.get(
                "random_search_candidates",
                paper_exact.get("optimizer_family", {}).get("random_search_initializations", 30),
            )
        ),
        "pso_particles": int(
            execution_overrides.get(
                "pso_particles",
                paper_exact.get("optimizer_family", {}).get("pso", {}).get("num_particles", 30),
            )
        ),
        "pso_iterations": int(
            execution_overrides.get(
                "pso_iterations",
                paper_exact.get("optimizer_family", {}).get("pso", {}).get("max_iter", 150),
            )
        ),
        "pso_inertial_weight": float(
            paper_exact.get("optimizer_family", {}).get("pso", {}).get("inertial_weight", 0.6)
        ),
        "pso_cognitive_coefficient": float(
            paper_exact.get("optimizer_family", {}).get("pso", {}).get("cognitive_coefficient", 1.0)
        ),
        "pso_social_coefficient": float(
            paper_exact.get("optimizer_family", {}).get("pso", {}).get("social_coefficient", 1.0)
        ),
        "param_min": float(
            (paper_exact.get("optimizer_family", {}).get("random_init_uniform_range") or [-0.1, 0.1])[0]
        ),
        "param_max": float(
            (paper_exact.get("optimizer_family", {}).get("random_init_uniform_range") or [-0.1, 0.1])[1]
        ),
        "results_root": str(config.get("paths", {}).get("results_raw", "results/raw")),
        "experiment_name": execution_id,
        "env_overrides": _normalize_env_overrides(paper_exact.get("train_env", {})),
        "pid_maxiter": int(execution_overrides.get("pid_maxiter", 150)),
        "pid_popsize": int(execution_overrides.get("pid_popsize", 15)),
    }
    return payload


def build_modern_rl_payload(config: dict[str, Any], *, method_id: str) -> dict[str, Any]:
    execution_id = resolve_execution_id(config)
    budgets = DEFAULT_TRAINING_BUDGETS.get(execution_id, DEFAULT_TRAINING_BUDGETS["debug_local"])
    benchmark = config.get("benchmark_standardization", {})
    scenario_id = resolve_scenario_id(config)

    return {
        "method_id": method_id,
        "scenario_id": scenario_id,
        "seed": resolve_seed(config),
        "device_preference": list(config.get("execution", {}).get("device_preference", ["cpu"])),
        "results_root": str(config.get("paths", {}).get("results_raw", "results/raw")),
        "total_env_steps": int(benchmark.get("training_budget", {}).get("total_environment_steps") or budgets["total_env_steps"]),
        "evaluation_frequency": int(benchmark.get("training_budget", {}).get("evaluation_frequency") or budgets["evaluation_frequency"]),
        "evaluation_episodes": int(benchmark.get("training_budget", {}).get("evaluation_episodes") or budgets["evaluation_episodes"]),
        "deterministic_eval": bool(benchmark.get("final_evaluation", {}).get("deterministic_inference", True)),
    }
