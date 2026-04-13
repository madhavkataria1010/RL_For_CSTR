"""Static PID tuning aligned with the official differential-evolution setup."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import differential_evolution

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.pid import denormalize_pid_gains
from src.environments.standardized_env import make_paper_exact_env

try:
    import yaml
except ImportError:  # pragma: no cover - optional until environment setup.
    yaml = None

from src.training.common_interface import build_paper_training_payload, load_project_config

@dataclass
class PIDTrainingConfig:
    method_id: str = "static_pid"
    scenario: str = "nominal_test"
    seed: int = 0
    ns: int = 360
    maxiter: int = 150
    popsize: int = 15
    polish: bool = False
    workers: int = 1
    results_root: str = "results/raw"
    experiment_name: str | None = None
    env_overrides: dict[str, Any] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None, **overrides: Any) -> "PIDTrainingConfig":
        payload = dict(data or {})
        payload.update(overrides)
        if payload.get("env_overrides") is None:
            payload["env_overrides"] = {}
        return cls(**payload)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load PID training configs.")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_run_dir(results_root: str, experiment_name: str | None, method_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = experiment_name or method_id
    run_dir = ROOT / results_root / method_id / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def rollout_pid(normalized_gains: Sequence[float], config: PIDTrainingConfig) -> float:
    env_kwargs = dict(config.env_overrides)
    env_kwargs.update({"seed": config.seed, "ns": config.ns})
    env = make_paper_exact_env("static_pid", config.scenario, config=env_kwargs)
    state, _ = env.reset(seed=config.seed)
    total_cost = 0.0
    done = False
    gains = np.asarray(normalized_gains, dtype=float)
    while not done:
        state, reward, done, _, _ = env.step(gains)
        total_cost += float(reward)
    return total_cost


def train_pid(
    config: PIDTrainingConfig | Mapping[str, Any] | Sequence[str | Path] | str | Path | None = None,
    **overrides: Any,
) -> dict[str, str]:
    if isinstance(config, Sequence) and not isinstance(config, (str, Path, bytes)):
        merged = load_project_config([str(path) for path in config])
        payload = build_paper_training_payload(merged, default_method_id="static_pid")
        cfg = PIDTrainingConfig.from_mapping(
            {
                "method_id": "static_pid",
                "scenario": f"{payload['scenario']}_test" if not payload["scenario"].endswith("_test") else payload["scenario"],
                "seed": payload["seed"],
                "ns": int(payload["env_overrides"].get("ns", 360)),
                "maxiter": int(payload.get("pid_maxiter", 150)),
                "popsize": int(payload.get("pid_popsize", 15)),
                "results_root": payload["results_root"],
                "experiment_name": payload["experiment_name"],
                "env_overrides": payload["env_overrides"],
            },
            **overrides,
        )
    elif isinstance(config, (str, Path)):
        cfg = PIDTrainingConfig.from_mapping(load_yaml_config(config), **overrides)
    elif isinstance(config, PIDTrainingConfig):
        cfg = PIDTrainingConfig.from_mapping(asdict(config), **overrides)
    else:
        cfg = PIDTrainingConfig.from_mapping(config, **overrides)

    run_dir = ensure_run_dir(cfg.results_root, cfg.experiment_name, cfg.method_id)
    bounds = [(-1.0, 1.0)] * 6
    result = differential_evolution(
        rollout_pid,
        bounds=bounds,
        args=(cfg,),
        maxiter=cfg.maxiter,
        popsize=cfg.popsize,
        seed=cfg.seed,
        polish=cfg.polish,
        workers=cfg.workers,
    )

    normalized_gains = np.asarray(result.x, dtype=float)
    gains = denormalize_pid_gains(normalized_gains)
    summary = {
        "normalized_gains": normalized_gains.tolist(),
        "physical_gains": gains.tolist(),
        "objective_cost": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
    }

    (run_dir / "pid_tuning.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    processed_target = ROOT / "results" / "processed" / "pid_tuning.json"
    processed_target.parent.mkdir(parents=True, exist_ok=True)
    processed_target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "pid_tuning": str(run_dir / "pid_tuning.json"),
                "config": str(run_dir / "config.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "pid_tuning": str(run_dir / "pid_tuning.json"),
        "config": str(run_dir / "config.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train/tune the static PID baseline.")
    parser.add_argument("--config", nargs="+")
    args = parser.parse_args(list(argv or sys.argv[1:]))
    artifacts = train_pid(args.config if args.config else None)
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
