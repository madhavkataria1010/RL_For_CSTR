from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only in thin envs
    yaml = None


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load configuration files. "
            "Install it with `pip install pyyaml` or `pip install -r requirements.txt`."
        )

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at top-level of config {path}")
    return loaded


def load_and_merge_yaml(paths: list[str | Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = _deep_update(merged, load_yaml(path))
    return merged


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to save configuration files.")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

