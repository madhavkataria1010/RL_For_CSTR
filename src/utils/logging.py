from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.paths import RAW_RESULTS_DIR, ensure_project_directories


def timestamp_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def create_run_directory(method_id: str, scenario_id: str, seed: int) -> Path:
    ensure_project_directories()
    run_dir = RAW_RESULTS_DIR / method_id / f"{timestamp_slug()}_{method_id}_{scenario_id}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
