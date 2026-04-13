"""Shared plotting style helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_STYLE = {
    "figure.figsize": (8.0, 5.0),
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "savefig.bbox": "tight",
    "savefig.dpi": 180,
}


def apply_publication_style() -> None:
    plt.rcParams.update(DEFAULT_STYLE)


def finalize_figure(fig: plt.Figure, *, title: str | None = None) -> plt.Figure:
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, path_stem: str | Path) -> tuple[Path, Path]:
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    return png_path, pdf_path
