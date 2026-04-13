"""Convenience wrappers for constructing configured CSTR environments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .cstr import CSTRConfig, CSTRReactorEnv

try:
    import yaml
except ImportError:  # pragma: no cover - optional until environment setup is complete.
    yaml = None


def _read_yaml(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load environment config files.")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def make_cstr_env(
    config: CSTRConfig | Mapping[str, Any] | str | Path | None = None,
    **overrides: Any,
) -> CSTRReactorEnv:
    """Build an environment from a dataclass, mapping, or YAML path."""

    if isinstance(config, (str, Path)):
        data = _read_yaml(config)
        cfg = CSTRConfig.from_mapping(data, **overrides)
    elif isinstance(config, CSTRConfig):
        cfg = CSTRConfig.from_mapping(config.to_dict(), **overrides)
    else:
        cfg = CSTRConfig.from_mapping(config, **overrides)
    return CSTRReactorEnv(cfg)
