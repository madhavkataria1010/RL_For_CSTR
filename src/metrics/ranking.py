"""Ranking utilities for benchmark summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def rank_summary_rows(
    rows: Iterable[dict[str, object]],
    metric_name: str,
    lower_is_better: bool = True,
) -> list[dict[str, object]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: float(row[metric_name]),
        reverse=not lower_is_better,
    )
    ranked_rows: list[dict[str, object]] = []
    last_value: float | None = None
    current_rank = 0
    for index, row in enumerate(sorted_rows, start=1):
        value = float(row[metric_name])
        if last_value is None or value != last_value:
            current_rank = index
            last_value = value
        ranked_rows.append({**row, "rank": current_rank})
    return ranked_rows


def average_ranks(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row["rank"]))
    return [
        {
            "method": method,
            "overall_average_rank": sum(values) / len(values),
        }
        for method, values in sorted(grouped.items())
    ]
