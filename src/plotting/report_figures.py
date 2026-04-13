"""Generate report-ready plots from benchmark tables and trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.plotting.publication_style import apply_publication_style, finalize_figure, save_figure
from src.utils.paths import FIGURES_DIR, RAW_RESULTS_DIR, TABLES_DIR


METHOD_ORDER = [
    ("static_pid", "Static PID"),
    ("pure_rl_paper", "Pure-RL (paper baseline)"),
    ("cirl_reproduced", "CIRL reproduced"),
    ("sac", "SAC"),
    ("td3", "TD3"),
    ("tqc", "TQC"),
    ("ppo", "PPO"),
    ("dr_cirl", "DR-CIRL"),
]

METHOD_COLORS = {
    "Static PID": "#4c6a92",
    "Pure-RL (paper baseline)": "#9b5de5",
    "CIRL reproduced": "#0f766e",
    "SAC": "#d97706",
    "TD3": "#ef4444",
    "TQC": "#7c3aed",
    "PPO": "#2563eb",
    "DR-CIRL": "#16a34a",
    "CIRL high-op extended (paper reproduction only)": "#0b4f4a",
}


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _latest_nominal_training_summaries(method_id: str) -> list[dict[str, object]]:
    method_dir = RAW_RESULTS_DIR / method_id
    if not method_dir.exists():
        return []

    latest_by_seed: dict[int, tuple[str, dict[str, object], Path]] = {}
    for manifest_path in sorted(method_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("scenario_id") != "nominal":
            continue
        seed = int(manifest.get("seed", -1))
        if seed < 0:
            continue
        run_name = manifest_path.parent.name
        previous = latest_by_seed.get(seed)
        if previous is None or run_name > previous[0]:
            summary_path = manifest_path.parent / "training_summary.json"
            if summary_path.exists():
                latest_by_seed[seed] = (run_name, manifest, summary_path)

    payloads: list[dict[str, object]] = []
    for _, manifest, summary_path in latest_by_seed.values():
        summary = json.loads(summary_path.read_text())
        payloads.append(
            {
                "seed": int(manifest["seed"]),
                "run_name": summary_path.parent.name,
                "iteration_best_scores": summary.get("iteration_best_scores", []),
                "random_search_scores": summary.get("random_search_scores", []),
                "best_score": summary.get("best_score"),
            }
        )
    return sorted(payloads, key=lambda item: int(item["seed"]))


def _load_trajectory_ensemble(scenario: str, method_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    folder = RAW_RESULTS_DIR / "extended_benchmark" / scenario / method_dir
    if not folder.exists():
        return None
    csv_paths = sorted(folder.glob("*.csv"))
    if not csv_paths:
        return None

    runs: list[pd.DataFrame] = [pd.read_csv(path) for path in csv_paths]
    min_len = min(len(df) for df in runs)
    if min_len == 0:
        return None
    runs = [df.iloc[:min_len].copy() for df in runs]

    time = runs[0]["time"].to_numpy(dtype=float)
    output = np.stack([df["observation_0"].to_numpy(dtype=float) for df in runs], axis=0)
    reference = np.stack([df["reference_0"].to_numpy(dtype=float) for df in runs], axis=0)
    action = np.stack([df["applied_action_0"].to_numpy(dtype=float) for df in runs], axis=0)

    return time, output, reference, action


def _plot_tracking_scenario(
    *,
    scenario: str,
    methods: list[tuple[str, str]],
    title: str,
    stem: str,
) -> tuple[Path, Path] | None:
    apply_publication_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.2]},
    )

    plotted = False
    reference_time: np.ndarray | None = None
    reference_mean: np.ndarray | None = None

    for method_dir, method_name in methods:
        loaded = _load_trajectory_ensemble(scenario, method_dir)
        if loaded is None:
            continue
        time, output, reference, action = loaded
        plotted = True

        output_mean = output.mean(axis=0)
        output_std = output.std(axis=0)
        action_mean = action.mean(axis=0)
        action_std = action.std(axis=0)

        color = METHOD_COLORS.get(method_name, "#333333")
        axes[0].plot(time, output_mean, label=method_name, color=color, linewidth=2.2)
        axes[0].fill_between(time, output_mean - output_std, output_mean + output_std, color=color, alpha=0.15)

        axes[1].plot(time, action_mean, label=method_name, color=color, linewidth=2.0)
        axes[1].fill_between(time, action_mean - action_std, action_mean + action_std, color=color, alpha=0.12)

        reference_time = time
        reference_mean = reference.mean(axis=0)

    if not plotted:
        plt.close(fig)
        return None

    if reference_time is not None and reference_mean is not None:
        axes[0].plot(reference_time, reference_mean, color="#111827", linestyle="--", linewidth=1.8, label="Setpoint")

    axes[0].set_ylabel(r"$C_B$")
    axes[1].set_ylabel(r"$T_c$ [K]")
    axes[1].set_xlabel("Time")
    axes[0].legend(loc="upper right", ncol=2)

    finalize_figure(fig, title=title)
    return save_figure(fig, FIGURES_DIR / stem)


def _plot_paper_training_progress() -> tuple[Path, Path] | None:
    method_specs = [
        ("cirl_reproduced", "CIRL reproduced"),
        ("pure_rl_paper", "Pure-RL (paper baseline)"),
        ("dr_cirl", "DR-CIRL"),
    ]

    curves: dict[str, np.ndarray] = {}
    for method_id, method_name in method_specs:
        payloads = _latest_nominal_training_summaries(method_id)
        if not payloads:
            continue
        length = min(len(item["iteration_best_scores"]) for item in payloads if item["iteration_best_scores"])
        if length == 0:
            continue
        matrix = np.asarray(
            [
                -np.asarray(item["iteration_best_scores"][:length], dtype=float)
                for item in payloads
            ],
            dtype=float,
        )
        curves[method_name] = matrix

    if not curves:
        return None

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for method_name in ["CIRL reproduced", "Pure-RL (paper baseline)", "DR-CIRL"]:
        matrix = curves.get(method_name)
        if matrix is None:
            continue
        x = np.arange(1, matrix.shape[1] + 1)
        clipped = np.clip(matrix, -100.0, 0.0)
        mean = clipped.mean(axis=0)
        std = clipped.std(axis=0)
        if mean.shape[0] >= 5:
            kernel = np.ones(5) / 5.0
            mean = np.convolve(mean, kernel, mode="same")
            std = np.convolve(std, kernel, mode="same")
        color = METHOD_COLORS.get(method_name, "#333333")
        linestyle = "--" if method_name == "Pure-RL (paper baseline)" else "-"
        ax.plot(x, mean, label=method_name, color=color, linewidth=2.2, linestyle=linestyle)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Reward proxy (-best cost, clipped)")
    ax.set_title("Paper-family training progress on the nominal task")
    ax.set_ylim(-100, 2)
    ax.legend(loc="lower right")
    finalize_figure(fig)
    return save_figure(fig, FIGURES_DIR / "paper_training_progress")


def _plot_reproduction_gap_bars(reproduction_path: Path) -> tuple[Path, Path] | None:
    if not reproduction_path.exists():
        return None
    rows = _read_csv_dicts(reproduction_path)
    if not rows:
        return None

    labels = [f"{row['method']} | {row['scenario']}" for row in rows]
    gaps = [float(row["relative_gap_percent"]) for row in rows]
    colors = [
        METHOD_COLORS.get("CIRL high-op extended (paper reproduction only)" if row["method"].startswith("CIRL high-op") else row["method"], "#457b9d")
        for row in rows
    ]

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    positions = np.arange(len(labels))
    ax.bar(positions, gaps, color=colors)
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Relative gap to paper [%]")
    ax.set_title("Paper-faithful reproduction gaps")
    finalize_figure(fig)
    return save_figure(fig, FIGURES_DIR / "reproduction_gap_bars")


def _plot_average_rank_bars(rank_path: Path) -> tuple[Path, Path] | None:
    if not rank_path.exists():
        return None
    rows = _read_csv_dicts(rank_path)
    if not rows:
        return None

    sorted_rows = sorted(rows, key=lambda row: float(row["overall_average_rank"]))
    methods = [row["method"] for row in sorted_rows]
    values = [float(row["overall_average_rank"]) for row in sorted_rows]
    colors = [METHOD_COLORS.get(method, "#457b9d") for method in methods]

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    positions = np.arange(len(methods))
    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title("Overall benchmark ranking")
    finalize_figure(fig)
    return save_figure(fig, FIGURES_DIR / "average_rank_bars")


def _plot_uncertainty_robustness(master_path: Path) -> tuple[Path, Path] | None:
    if not master_path.exists():
        return None
    rows = _read_csv_dicts(master_path)
    if not rows:
        return None

    methods = [row["method"] for row in rows]
    iae_pm10 = [float(row["uncertainty_pm10_iae"]) for row in rows]
    iae_pm20 = [float(row["uncertainty_pm20_iae"]) for row in rows]
    colors = [METHOD_COLORS.get(method, "#457b9d") for method in methods]

    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=False)
    positions = np.arange(len(methods))

    axes[0].bar(positions, iae_pm10, color=colors)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(methods, rotation=30, ha="right")
    axes[0].set_ylabel("IAE")
    axes[0].set_title(r"Uncertainty robustness ($\pm 10\%$)")

    axes[1].bar(positions, iae_pm20, color=colors)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(methods, rotation=30, ha="right")
    axes[1].set_ylabel("IAE")
    axes[1].set_title(r"Uncertainty robustness ($\pm 20\%$)")

    finalize_figure(fig)
    return save_figure(fig, FIGURES_DIR / "uncertainty_robustness_bars")


def _plot_single_uncertainty_robustness(master_path: Path, *, column: str, title: str, stem: str) -> tuple[Path, Path] | None:
    if not master_path.exists():
        return None
    rows = _read_csv_dicts(master_path)
    if not rows:
        return None

    methods = [row["method"] for row in rows]
    values = [float(row[column]) for row in rows]
    colors = [METHOD_COLORS.get(method, "#457b9d") for method in methods]

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    positions = np.arange(len(methods))
    ax.bar(positions, values, color=colors)
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("IAE")
    ax.set_title(title)
    finalize_figure(fig)
    return save_figure(fig, FIGURES_DIR / stem)


def main() -> None:
    global TABLES_DIR, RAW_RESULTS_DIR

    parser = argparse.ArgumentParser(description="Generate report-ready benchmark plots.")
    parser.add_argument("--tables-dir", default=str(TABLES_DIR))
    parser.add_argument("--raw-dir", default=str(RAW_RESULTS_DIR))
    args = parser.parse_args()

    TABLES_DIR = Path(args.tables_dir)
    RAW_RESULTS_DIR = Path(args.raw_dir)

    generated: list[tuple[Path, Path]] = []

    trajectory_specs = [
        (
            "nominal",
            [("static_pid", "Static PID"), ("cirl_reproduced", "CIRL reproduced"), ("dr_cirl", "DR-CIRL"), ("ppo", "PPO")],
            "Nominal tracking comparison",
            "report_nominal_tracking",
        ),
        (
            "disturbance",
            [("static_pid", "Static PID"), ("pure_rl_paper", "Pure-RL (paper baseline)"), ("cirl_reproduced", "CIRL reproduced"), ("dr_cirl", "DR-CIRL")],
            "Disturbance rejection comparison",
            "report_disturbance_rejection",
        ),
        (
            "highop",
            [("static_pid", "Static PID"), ("cirl_reproduced", "CIRL reproduced"), ("dr_cirl", "DR-CIRL"), ("ppo", "PPO")],
            "High-operating-point comparison",
            "report_highop_transfer",
        ),
    ]

    for scenario, methods, title, stem in trajectory_specs:
        result = _plot_tracking_scenario(scenario=scenario, methods=methods, title=title, stem=stem)
        if result is not None:
            generated.append(result)

    for result in (
        _plot_reproduction_gap_bars(TABLES_DIR / "reproduction_summary.csv"),
        _plot_average_rank_bars(TABLES_DIR / "average_rank_table.csv"),
        _plot_uncertainty_robustness(TABLES_DIR / "master_comparison_table.csv"),
        _plot_single_uncertainty_robustness(
            TABLES_DIR / "master_comparison_table.csv",
            column="uncertainty_pm10_iae",
            title=r"Uncertainty robustness ($\pm 10\%$)",
            stem="uncertainty_robustness_pm10",
        ),
        _plot_single_uncertainty_robustness(
            TABLES_DIR / "master_comparison_table.csv",
            column="uncertainty_pm20_iae",
            title=r"Uncertainty robustness ($\pm 20\%$)",
            stem="uncertainty_robustness_pm20",
        ),
        _plot_paper_training_progress(),
    ):
        if result is not None:
            generated.append(result)

    if not generated:
        print("No report figures generated.")
        return

    for png_path, pdf_path in generated:
        print(f"generated: {png_path}")
        print(f"generated: {pdf_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
