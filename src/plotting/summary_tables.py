"""Helpers for summary-table visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.plotting.publication_style import save_figure
from src.utils.paths import FIGURES_DIR, TABLES_DIR

from .publication_style import apply_publication_style, finalize_figure


def plot_summary_heatmap(
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str | None = None,
    colorbar_label: str = "Value",
) -> tuple[plt.Figure, plt.Axes]:
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.1), max(4, len(row_labels) * 0.6)))
    image = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color="white")
    fig.colorbar(image, ax=ax, label=colorbar_label)
    return finalize_figure(fig, title=title), ax


def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    import csv

    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def _numeric(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _make_master_heatmap(master_path: Path) -> tuple[Path, Path] | None:
    if not master_path.exists():
        return None
    columns, rows = _read_csv(master_path)
    if not rows:
        return None
    metric_columns = [column for column in columns if column != "method"]
    matrix = np.asarray([[_numeric(row[column]) for column in metric_columns] for row in rows], dtype=float)
    fig, _ = plot_summary_heatmap(
        values=np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0),
        row_labels=[row["method"] for row in rows],
        col_labels=metric_columns,
        title="Master Comparison Table",
        colorbar_label="Metric value",
    )
    return save_figure(fig, FIGURES_DIR / "master_comparison_heatmap")


def _make_rank_heatmap(rank_path: Path) -> tuple[Path, Path] | None:
    if not rank_path.exists():
        return None
    columns, rows = _read_csv(rank_path)
    if not rows:
        return None
    metric_columns = [column for column in columns if column != "method"]
    matrix = np.asarray([[_numeric(row[column]) for column in metric_columns] for row in rows], dtype=float)
    fig, _ = plot_summary_heatmap(
        values=np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0),
        row_labels=[row["method"] for row in rows],
        col_labels=metric_columns,
        title="Average Rank Table",
        colorbar_label="Rank",
    )
    return save_figure(fig, FIGURES_DIR / "average_rank_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary figures from benchmark CSVs.")
    parser.add_argument(
        "--master-table",
        default=str(TABLES_DIR / "master_comparison_table.csv"),
    )
    parser.add_argument(
        "--rank-table",
        default=str(TABLES_DIR / "average_rank_table.csv"),
    )
    args = parser.parse_args()

    generated: list[tuple[Path, Path]] = []
    for result in (_make_master_heatmap(Path(args.master_table)), _make_rank_heatmap(Path(args.rank_table))):
        if result is not None:
            generated.append(result)

    if not generated:
        print("No figures generated; required CSV tables were not found.")
        return
    for png_path, pdf_path in generated:
        print(f"generated: {png_path}")
        print(f"generated: {pdf_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
