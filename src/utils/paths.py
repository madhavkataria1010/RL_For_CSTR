from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
PROCESSED_RESULTS_DIR = RESULTS_DIR / "processed"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def ensure_project_directories() -> None:
    """Create the core result directories when missing."""
    for path in (
        RESULTS_DIR,
        RAW_RESULTS_DIR,
        PROCESSED_RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

