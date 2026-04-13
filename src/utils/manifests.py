from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logging import save_json


@dataclass
class ExperimentManifest:
    method_id: str
    scenario_id: str
    seed: int
    config_paths: list[str]
    run_dir: str
    device: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def write(self, target_path: str | Path) -> None:
        save_json(asdict(self), target_path)

