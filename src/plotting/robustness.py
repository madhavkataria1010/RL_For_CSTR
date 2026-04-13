"""Robustness plotting helpers."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from .publication_style import apply_publication_style, finalize_figure


def plot_metric_bars(
    methods: list[str],
    means: list[float],
    stds: list[float] | None = None,
    *,
    ylabel: str,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    apply_publication_style()
    fig, ax = plt.subplots()
    positions = np.arange(len(methods))
    errors = stds if stds is not None else None
    ax.bar(positions, means, yerr=errors, capsize=4, color="#457b9d")
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    return finalize_figure(fig, title=title), ax


def plot_rank_heatmap(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.1), max(4, len(row_labels) * 0.6)))
    image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label="Rank")
    return finalize_figure(fig, title=title), ax
