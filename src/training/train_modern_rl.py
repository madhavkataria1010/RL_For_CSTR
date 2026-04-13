from __future__ import annotations

import argparse
from pathlib import Path

from src.controllers.ppo_wrapper import PPOController
from src.controllers.sac_wrapper import SACController
from src.controllers.td3_wrapper import TD3Controller
from src.controllers.tqc_wrapper import TQCController
from src.training.checkpointing import save_pickle
from src.training.common_interface import build_modern_rl_payload, load_project_config
from src.utils.device import get_best_device
from src.utils.logging import create_run_directory
from src.utils.manifests import ExperimentManifest
from src.utils.paths import CONFIGS_DIR
from src.utils.seeding import seed_everything


CONTROLLER_REGISTRY = {
    "sac": SACController,
    "td3": TD3Controller,
    "tqc": TQCController,
    "ppo": PPOController,
}


def train_modern_rl(method_id: str, config_paths: list[str], *, seed_override: int | None = None) -> None:
    from src.environments.standardized_env import build_standardized_env
    from stable_baselines3.common.callbacks import EvalCallback

    merged = load_project_config(config_paths)
    runtime = build_modern_rl_payload(merged, method_id=method_id)
    if seed_override is not None:
        runtime["seed"] = int(seed_override)
    seed = int(runtime["seed"])
    scenario_id = str(runtime["scenario_id"])
    seed_everything(seed)
    run_dir = create_run_directory(method_id, scenario_id, seed)
    device = get_best_device(prefer_accelerator=True)

    env = build_standardized_env(
        {
            "method_id": method_id,
            "scenario_id": scenario_id,
            "env_overrides": {},
        }
    )
    eval_env = build_standardized_env(
        {
            "method_id": method_id,
            "scenario_id": scenario_id,
            "env_overrides": {},
        }
    )
    controller_cls = CONTROLLER_REGISTRY[method_id]
    controller = controller_cls.build(
        env=env,
        policy_kwargs={},
        seed=seed,
        device=device,
        verbose=1,
    )
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir),
        eval_freq=max(int(runtime["evaluation_frequency"]), 1),
        n_eval_episodes=max(int(runtime["evaluation_episodes"]), 1),
        deterministic=bool(runtime["deterministic_eval"]),
        verbose=1,
    )
    controller.train(total_timesteps=int(runtime["total_env_steps"]), callback=callback)

    best_model_path = run_dir / "best_model.zip"
    if best_model_path.exists():
        model_path = best_model_path
    else:
        model_path = run_dir / "policy"
        controller.save(model_path)
    manifest = ExperimentManifest(
        method_id=method_id,
        scenario_id=scenario_id,
        seed=seed,
        config_paths=config_paths,
        run_dir=str(run_dir),
        device=device,
        metadata={
            "total_env_steps": int(runtime["total_env_steps"]),
            "evaluation_frequency": int(runtime["evaluation_frequency"]),
            "evaluation_episodes": int(runtime["evaluation_episodes"]),
            "deterministic_eval": bool(runtime["deterministic_eval"]),
            "selected_model_path": str(model_path),
            "model_selection_rule": "best_eval_checkpoint" if best_model_path.exists() else "final_checkpoint",
        },
    )
    manifest.write(run_dir / "manifest.json")
    save_pickle(merged, run_dir / "resolved_config.pkl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a modern RL baseline.")
    parser.add_argument("--method", required=True, choices=sorted(CONTROLLER_REGISTRY))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--config",
        nargs="+",
        default=[
            str(CONFIGS_DIR / "base.yaml"),
            str(CONFIGS_DIR / "execution" / "debug_local.yaml"),
            str(CONFIGS_DIR / "scenarios" / "nominal.yaml"),
        ],
    )
    args = parser.parse_args()
    train_modern_rl(args.method, args.config, seed_override=args.seed)


if __name__ == "__main__":
    main()
