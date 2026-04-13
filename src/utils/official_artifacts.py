from __future__ import annotations

import pickle
import ssl
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np

from src.controllers.pid import denormalize_pid_gains
from src.utils.paths import RAW_RESULTS_DIR


OFFICIAL_REPO = "OptiMaL-PSE-Lab/CIRL"
OFFICIAL_COMMIT = "c3fd22580a15d1e008570c08e78b38cdd887ef2c"
OFFICIAL_RAW_BASE = (
    f"https://raw.githubusercontent.com/{OFFICIAL_REPO}/{OFFICIAL_COMMIT}"
)
OFFICIAL_CACHE_DIR = RAW_RESULTS_DIR / "official_artifacts"


def _download_official_file(relative_path: str) -> Path:
    target = OFFICIAL_CACHE_DIR / relative_path
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{OFFICIAL_RAW_BASE}/{relative_path}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", None)
        if not isinstance(reason, ssl.SSLCertVerificationError):
            raise
        with urllib.request.urlopen(
            url,
            timeout=60,
            context=ssl._create_unverified_context(),
        ) as response:
            target.write_bytes(response.read())
    return target


def _import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "Torch is required to materialize official paper artifacts."
        ) from exc
    return torch


def _pickle_state_dict_selector(
    method_id: str,
    scenario_id: str,
    payload: Any,
) -> dict[str, Any]:
    if method_id == "pure_rl_paper" and scenario_id == "nominal":
        return payload[1]["p_list"][149]
    if method_id == "cirl_reproduced" and scenario_id in {"nominal", "highop"}:
        return payload[0]["p_list"][149]
    raise KeyError(
        f"No official pickle selector defined for method={method_id!r}, scenario={scenario_id!r}."
    )


def _materialize_pickle_checkpoint(
    *,
    method_id: str,
    scenario_id: str,
    relative_path: str,
) -> Path:
    torch = _import_torch()
    source = _download_official_file(relative_path)
    target = (
        OFFICIAL_CACHE_DIR
        / "materialized"
        / method_id
        / scenario_id
        / "policy.pt"
    )
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    state_dict = _pickle_state_dict_selector(method_id, scenario_id, payload)
    torch.save(state_dict, target)
    return target


def resolve_paper_reproduction_artifact(
    method_id: str,
    scenario_id: str,
) -> dict[str, Any] | None:
    if method_id == "static_pid" and scenario_id == "nominal":
        normalized = np.load(_download_official_file("data/constant_gains.npy"))
        return {
            "type": "static_pid",
            "physical_gains": denormalize_pid_gains(normalized),
            "source": "official constant_gains.npy",
        }
    if method_id == "static_pid" and scenario_id == "highop":
        normalized = np.load(
            _download_official_file("data/constant_gains_highop.npy")
        )
        return {
            "type": "static_pid",
            "physical_gains": denormalize_pid_gains(normalized),
            "source": "official constant_gains_highop.npy",
        }
    if method_id == "pure_rl_paper" and scenario_id == "nominal":
        return {
            "type": "checkpoint",
            "checkpoint_path": _materialize_pickle_checkpoint(
                method_id=method_id,
                scenario_id=scenario_id,
                relative_path="data/results_rl_network_rep_newobs_0.pkl",
            ),
            "source": "official results_rl_network_rep_newobs_0.pkl -> inter[1]['p_list'][149]",
        }
    if method_id == "cirl_reproduced" and scenario_id == "nominal":
        return {
            "type": "checkpoint",
            "checkpoint_path": _materialize_pickle_checkpoint(
                method_id=method_id,
                scenario_id=scenario_id,
                relative_path="data/results_pid_network_rep_newobs_1.pkl",
            ),
            "source": "official results_pid_network_rep_newobs_1.pkl -> inter[0]['p_list'][149]",
        }
    if method_id == "pure_rl_paper" and scenario_id == "disturbance":
        return {
            "type": "checkpoint",
            "checkpoint_path": _download_official_file("data/best_policy_rl_dist.pth"),
            "source": "official best_policy_rl_dist.pth",
        }
    if method_id == "cirl_reproduced" and scenario_id == "disturbance":
        return {
            "type": "checkpoint",
            "checkpoint_path": _download_official_file("data/best_policy_pid_dist_0.pth"),
            "source": "official best_policy_pid_dist_0.pth",
        }
    if method_id == "cirl_reproduced" and scenario_id == "highop":
        return {
            "type": "checkpoint",
            "checkpoint_path": _materialize_pickle_checkpoint(
                method_id=method_id,
                scenario_id=scenario_id,
                relative_path="data/results_pid_network_rep_newobs_1.pkl",
            ),
            "source": "official low-op CIRL from results_pid_network_rep_newobs_1.pkl -> inter[0]['p_list'][149]",
        }
    if method_id == "cirl_highop_extended_paper" and scenario_id == "highop":
        return {
            "type": "checkpoint",
            "checkpoint_path": _download_official_file(
                "data/best_policy_pid_highop_0.pth"
            ),
            "source": "official best_policy_pid_highop_0.pth",
        }
    return None
