"""Sync the curated figure set into the paper/figures folder."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.utils.paths import FIGURES_DIR, PROJECT_ROOT


CURATED_FIGURES = [
    "paper_rl_loop_diagram",
    "paper_policy_network_diagram",
    "paper_cirl_architecture_diagram",
    "paper_training_progress",
    "report_nominal_tracking",
    "report_disturbance_rejection",
    "report_highop_transfer",
    "uncertainty_robustness_bars",
    "uncertainty_robustness_pm10",
    "uncertainty_robustness_pm20",
]


def _copy_if_exists(source: Path, target: Path) -> bool:
    if not source.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def main() -> None:
    paper_figures_dir = PROJECT_ROOT / "paper" / "figures"
    paper_figures_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for stem in CURATED_FIGURES:
        for suffix in (".png", ".pdf"):
            source = FIGURES_DIR / f"{stem}{suffix}"
            target = paper_figures_dir / f"{stem}{suffix}"
            if _copy_if_exists(source, target):
                copied.append(target)

    for path in copied:
        print(f"copied: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
