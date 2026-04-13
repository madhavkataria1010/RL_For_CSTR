from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.controllers.pid import StaticPIDController


OFFICIAL_STATIC_PID_PHYSICAL_GAINS = np.array(
    [3.09717127, 0.03626456, 0.83202401, 0.84267329, 1.84896398, 0.08209610],
    dtype=float,
)


def _load_torch_state_dict(path: str | Path):
    try:  # pragma: no cover - optional dependency
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch is required to load learned paper-faithful controllers.") from exc
    return torch.load(Path(path), map_location="cpu")


def _paper_extra_hidden_layers_from_state_dict(state_dict: dict[str, Any]) -> int:
    prefixes = ("n_layers.", "extra_layers.")
    indices: set[int] = set()
    for key in state_dict:
        for prefix in prefixes:
            if key.startswith(prefix):
                remainder = key[len(prefix) :]
                index_token = remainder.split(".", 1)[0]
                if index_token.isdigit():
                    indices.add(int(index_token))
    return (max(indices) + 1) if indices else 0


def _paper_hidden_size_from_state_dict(state_dict: dict[str, Any], default: int) -> int:
    hidden_weight = state_dict.get("hidden1.weight")
    if hidden_weight is None:
        return default
    shape = getattr(hidden_weight, "shape", None)
    if not shape:
        return default
    return int(shape[0])


def _normalize_paper_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith("n_layers."):
            normalized["n_layers." + key[len("n_layers.") :]] = value
        elif key.startswith("extra_layers."):
            normalized["n_layers." + key[len("extra_layers.") :]] = value
        else:
            normalized[key] = value
    return normalized


def _is_visualization_policy_state_dict(state_dict: dict[str, Any]) -> bool:
    return "output_mu.weight" in state_dict and "output_std.weight" in state_dict


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    run_dir = checkpoint.parent
    config_path = run_dir / "config.json"
    if config_path.exists():
        config_payload = _read_json(config_path)
        return {
            "scenario_id": config_payload.get("scenario"),
            "seed": config_payload.get("seed"),
        }

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        return {
            "scenario_id": manifest.get("scenario_id"),
            "seed": manifest.get("seed"),
        }
    return {}


def latest_checkpoint(
    method_id: str,
    root: str | Path = "results/raw",
    *,
    scenario_id: str | None = None,
    seed: int | None = None,
) -> Path | None:
    base = Path(root) / method_id
    if not base.exists():
        return None
    patterns = ("**/policy.pt", "**/policy.zip", "**/*_model.zip", "**/policy")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(base.glob(pattern)))
    candidates = sorted({path.resolve() for path in candidates if path.exists()})
    if scenario_id is not None or seed is not None:
        filtered: list[Path] = []
        for candidate in candidates:
            metadata = _checkpoint_metadata(candidate)
            metadata_scenario = metadata.get("scenario_id")
            metadata_seed = metadata.get("seed")
            if scenario_id is not None and metadata_scenario != scenario_id:
                continue
            if seed is not None and metadata_seed != seed:
                continue
            filtered.append(candidate)
        if filtered:
            candidates = filtered
    return candidates[-1] if candidates else None


def load_controller(method_id: str, checkpoint: str | Path | None = None, **kwargs: Any):
    if method_id == "static_pid":
        gains = np.asarray(kwargs.get("physical_gains", OFFICIAL_STATIC_PID_PHYSICAL_GAINS), dtype=float)
        return StaticPIDController.from_physical_gains(gains)

    if checkpoint is None:
        checkpoint = latest_checkpoint(
            method_id,
            scenario_id=kwargs.get("training_scenario"),
            seed=kwargs.get("seed"),
        )
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found for method '{method_id}'.")

    if method_id == "pure_rl_paper":
        from src.controllers.pure_rl_paper import (
            OfficialReplayPolicy,
            PureRLPaperController,
            PureRLPaperPolicy,
        )

        state_dict = _load_torch_state_dict(checkpoint)
        normalized_state_dict = _normalize_paper_state_dict_keys(state_dict)
        extra_hidden_layers = _paper_extra_hidden_layers_from_state_dict(
            normalized_state_dict
        )
        hidden_size = _paper_hidden_size_from_state_dict(
            normalized_state_dict, default=128
        )
        if _is_visualization_policy_state_dict(normalized_state_dict):
            policy = OfficialReplayPolicy(
                hidden_size=hidden_size,
                output_size=2,
                extra_hidden_layers=extra_hidden_layers,
                pid_output=True,
            )
        else:
            policy = PureRLPaperPolicy(
                hidden_size=hidden_size,
                extra_hidden_layers=extra_hidden_layers,
            )
        policy.load_state_dict(normalized_state_dict)
        return PureRLPaperController(policy=policy)

    if method_id in {"cirl_reproduced", "cirl_highop_extended_paper"}:
        from src.controllers.cirl_wrapper import CIRLController, CIRLPolicy
        from src.controllers.pure_rl_paper import OfficialReplayPolicy

        state_dict = _load_torch_state_dict(checkpoint)
        normalized_state_dict = _normalize_paper_state_dict_keys(state_dict)
        extra_hidden_layers = _paper_extra_hidden_layers_from_state_dict(
            normalized_state_dict
        )
        hidden_size = _paper_hidden_size_from_state_dict(
            normalized_state_dict, default=16
        )
        if _is_visualization_policy_state_dict(normalized_state_dict):
            policy = OfficialReplayPolicy(
                hidden_size=hidden_size,
                output_size=6,
                extra_hidden_layers=extra_hidden_layers,
                pid_output=True,
            )
        else:
            policy = CIRLPolicy(
                hidden_size=hidden_size,
                extra_hidden_layers=extra_hidden_layers,
            )
        policy.load_state_dict(normalized_state_dict)
        return CIRLController(policy=policy)

    if method_id == "dr_cirl":
        from src.controllers.dr_cirl import DRCIRLController, DRCIRLPolicy

        state_dict = _load_torch_state_dict(checkpoint)
        normalized_state_dict = _normalize_paper_state_dict_keys(state_dict)
        extra_hidden_layers = _paper_extra_hidden_layers_from_state_dict(
            normalized_state_dict
        )
        hidden_size = _paper_hidden_size_from_state_dict(
            normalized_state_dict, default=16
        )
        policy = DRCIRLPolicy(
            hidden_size=hidden_size,
            extra_hidden_layers=extra_hidden_layers,
        )
        policy.load_state_dict(normalized_state_dict)
        return DRCIRLController(policy=policy)

    if method_id == "sac":
        from src.controllers.sac_wrapper import SACController

        return SACController.load(checkpoint)

    if method_id == "td3":
        from src.controllers.td3_wrapper import TD3Controller

        return TD3Controller.load(checkpoint)

    if method_id == "tqc":
        from src.controllers.tqc_wrapper import TQCController

        return TQCController.load(checkpoint)

    if method_id == "ppo":
        from src.controllers.ppo_wrapper import PPOController

        return PPOController.load(checkpoint)

    raise KeyError(f"Unsupported controller method: {method_id}")
