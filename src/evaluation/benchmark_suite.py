"""Shared benchmark suite definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .aggregation import aggregate_run_metrics, write_csv_rows, write_markdown_table
from .scenario_runner import EvaluationRun, evaluate_scenario


@dataclass(slots=True)
class BenchmarkMethod:
    name: str
    controller_factory: Callable[[int], Any]
    deterministic_eval: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkScenario:
    name: str
    env_factory: Callable[[int], Any]
    seeds: list[int]
    horizon: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkSuite:
    """Minimal shared benchmark runner."""

    methods: list[BenchmarkMethod]
    scenarios: list[BenchmarkScenario]
    output_dir: Path | None = None

    def run(self) -> dict[str, list[EvaluationRun]]:
        all_results: dict[str, list[EvaluationRun]] = {}
        for method in self.methods:
            for scenario in self.scenarios:
                key = f"{method.name}::{scenario.name}"
                scenario_output = None
                if self.output_dir is not None:
                    scenario_output = self.output_dir / method.name / scenario.name
                all_results[key] = evaluate_scenario(
                    env_factory=scenario.env_factory,
                    controller_factory=method.controller_factory,
                    method=method.name,
                    scenario=scenario.name,
                    seeds=scenario.seeds,
                    horizon=scenario.horizon,
                    output_dir=scenario_output,
                    deterministic=method.deterministic_eval,
                )
        return all_results

    @staticmethod
    def flatten(results: dict[str, list[EvaluationRun]]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for run_group in results.values():
            for run in run_group:
                rows.append(
                    {
                        "method": run.rollout.method,
                        "scenario": run.rollout.scenario,
                        "seed": run.rollout.seed,
                        **run.metrics,
                    }
                )
        return rows

    def summarize(self, results: dict[str, list[EvaluationRun]]) -> list[dict[str, object]]:
        return aggregate_run_metrics(self.flatten(results))

    def write_summary(self, results: dict[str, list[EvaluationRun]], *, stem: str = "benchmark_summary") -> dict[str, Path]:
        if self.output_dir is None:
            raise ValueError("output_dir is required to write summaries")
        summary_rows = self.summarize(results)
        csv_path = write_csv_rows(summary_rows, self.output_dir / f"{stem}.csv")
        md_path = write_markdown_table(summary_rows, self.output_dir / f"{stem}.md")
        return {"csv": csv_path, "markdown": md_path}
