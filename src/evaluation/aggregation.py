"""Aggregation and tabulation helpers."""

from __future__ import annotations

from csv import DictWriter
from pathlib import Path
from typing import Iterable

import numpy as np


def aggregate_run_metrics(
    runs: Iterable[dict[str, object]],
    *,
    group_by: tuple[str, ...] = ("method", "scenario"),
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    runs_list = list(runs)
    if not runs_list:
        return []
    metric_keys = [key for key, value in runs_list[0].items() if isinstance(value, (int, float)) and key not in group_by]
    for run in runs_list:
        key = tuple(run[name] for name in group_by)
        grouped.setdefault(key, []).append(run)

    aggregated: list[dict[str, object]] = []
    for key, rows in sorted(grouped.items()):
        row: dict[str, object] = {name: value for name, value in zip(group_by, key, strict=True)}
        row["n_seeds"] = len(rows)
        for metric in metric_keys:
            values = np.asarray([float(item[metric]) for item in rows], dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            if values.size <= 1:
                row[f"{metric}_std"] = 0.0
            elif np.all(np.isposinf(values)) or np.all(np.isneginf(values)):
                row[f"{metric}_std"] = 0.0
            else:
                row[f"{metric}_std"] = float(np.std(values, ddof=1))
        aggregated.append(row)
    return aggregated


def write_csv_rows(rows: Iterable[dict[str, object]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    fieldnames = list(rows_list[0].keys()) if rows_list else []
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_list)
    return output_path


def render_markdown_table(rows: Iterable[dict[str, object]]) -> str:
    rows_list = list(rows)
    if not rows_list:
        return ""
    columns = list(rows_list[0].keys())
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows_list
    ]
    return "\n".join([header, separator, *body])


def write_markdown_table(rows: Iterable[dict[str, object]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown_table(rows), encoding="utf-8")
    return output_path
