from __future__ import annotations

from pathlib import Path

import numpy as np

from src.evaluation.benchmark_suite import BenchmarkMethod, BenchmarkScenario, BenchmarkSuite
from src.plotting.publication_style import save_figure
from src.plotting.trajectories import plot_rollout_trajectory


def test_benchmark_suite_writes_rollouts_and_summary(tmp_path, dummy_env_factory, dummy_controller_factory):
    suite = BenchmarkSuite(
        methods=[BenchmarkMethod(name="Static PID", controller_factory=dummy_controller_factory)],
        scenarios=[
            BenchmarkScenario(
                name="nominal_tracking",
                env_factory=dummy_env_factory,
                seeds=[0, 1],
                horizon=6,
            )
        ],
        output_dir=tmp_path / "benchmark",
    )

    results = suite.run()
    summary_paths = suite.write_summary(results, stem="smoke_summary")

    csv_files = list((tmp_path / "benchmark").rglob("*.csv"))
    assert csv_files
    assert summary_paths["csv"].exists()
    assert summary_paths["markdown"].exists()


def test_plotting_helpers_save_png_and_pdf(tmp_path):
    time = np.arange(5, dtype=float)
    output = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    reference = np.ones_like(output)
    action = np.linspace(0.0, 1.0, num=5)

    fig, _ = plot_rollout_trajectory(
        time=time,
        output=output,
        reference=reference,
        action=action,
        title="Smoke Plot",
    )
    png_path, pdf_path = save_figure(fig, tmp_path / "trajectory_smoke")

    assert png_path.exists()
    assert pdf_path.exists()
