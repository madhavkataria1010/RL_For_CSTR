"""Extended benchmark CLI and table builders."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.controllers.controller_factory import load_controller
from src.environments.standardized_env import make_benchmark_env
from src.evaluation.aggregation import aggregate_run_metrics, write_csv_rows, write_markdown_table
from src.evaluation.scenario_runner import EvaluationRun, evaluate_scenario
from src.plotting.publication_style import save_figure
from src.plotting.trajectories import plot_rollout_trajectory
from src.training.common_interface import DEFAULT_TRAINING_BUDGETS, load_project_config
from src.utils.config import load_yaml
from src.utils.paths import CONFIGS_DIR, FIGURES_DIR, PROCESSED_RESULTS_DIR, TABLES_DIR, ensure_project_directories


EXTENDED_BENCHMARK_METHODS = (
    "Static PID",
    "Pure-RL (paper baseline)",
    "CIRL reproduced",
    "SAC",
    "TD3",
    "TQC",
    "PPO",
    "DR-CIRL",
)

TIER1_SCENARIOS = ("nominal", "disturbance", "highop", "uncertainty_pm10", "uncertainty_pm20")
TIER2_SCENARIOS = ("noise", "saturation", "unseen_setpoints")

MASTER_COMPARISON_COLUMNS = (
    "method",
    "nominal_iae",
    "nominal_overshoot",
    "nominal_settling_time",
    "disturbance_iae",
    "disturbance_recovery_time",
    "highop_iae",
    "highop_overshoot",
    "uncertainty_pm10_iae",
    "uncertainty_pm10_constraint_violations",
    "uncertainty_pm20_iae",
    "uncertainty_pm20_constraint_violations",
    "average_rank",
)

REPORT_NAME_BY_METHOD = {
    "static_pid": "Static PID",
    "pure_rl_paper": "Pure-RL (paper baseline)",
    "cirl_reproduced": "CIRL reproduced",
    "sac": "SAC",
    "td3": "TD3",
    "tqc": "TQC",
    "ppo": "PPO",
    "dr_cirl": "DR-CIRL",
}

METHOD_ORDER = ["static_pid", "pure_rl_paper", "cirl_reproduced", "sac", "td3", "tqc", "ppo", "dr_cirl"]
MODERN_METHOD_IDS = {"sac", "td3", "tqc", "ppo"}
RANK_METRICS = {
    "nominal": ["iae_mean", "ise_mean", "itae_mean", "rmse_mean", "overshoot_mean", "settling_time_mean", "total_variation_mean", "constraint_violations_mean"],
    "disturbance": ["iae_mean", "ise_mean", "itae_mean", "rmse_mean", "overshoot_mean", "settling_time_mean", "total_variation_mean", "constraint_violations_mean", "recovery_time_mean"],
    "highop": ["iae_mean", "ise_mean", "itae_mean", "rmse_mean", "overshoot_mean", "settling_time_mean", "total_variation_mean", "constraint_violations_mean"],
    "uncertainty_pm10": ["iae_mean", "ise_mean", "itae_mean", "rmse_mean", "overshoot_mean", "settling_time_mean", "total_variation_mean", "constraint_violations_mean", "total_constraint_violation_magnitude_mean"],
    "uncertainty_pm20": ["iae_mean", "ise_mean", "itae_mean", "rmse_mean", "overshoot_mean", "settling_time_mean", "total_variation_mean", "constraint_violations_mean", "total_constraint_violation_magnitude_mean"],
}


def _benchmark_training_scenario(method_id: str) -> str | None:
    if method_id == "static_pid":
        return None
    return "nominal"


def _scenario_config_path(scenario_id: str) -> Path:
    return CONFIGS_DIR / "scenarios" / f"{scenario_id}.yaml"


def _load_scenario_config(scenario_id: str) -> dict[str, Any]:
    path = _scenario_config_path(scenario_id)
    if not path.exists():
        return {}
    return load_yaml(path)


def _eval_overrides_for_scenario(scenario_id: str) -> dict[str, Any]:
    scenario_cfg = _load_scenario_config(scenario_id)
    eval_env = dict(scenario_cfg.get("paper_exact", {}).get("eval_env", {}))
    if scenario_id.startswith("uncertainty") and not eval_env:
        eval_env = {"test": True}
    if scenario_id in {"noise", "saturation", "unseen_setpoints"} and not eval_env:
        eval_env = {"test": True}
    if "disturbance_activation_step" in eval_env:
        eval_env["disturbance_start_step"] = eval_env.pop("disturbance_activation_step")
    if "eval_ca_in_value" in eval_env:
        eval_env["evaluation_disturbance"] = float(eval_env.pop("eval_ca_in_value"))
    return eval_env


def _horizon_for_scenario(merged_config: dict[str, Any], scenario_id: str) -> int:
    smoke = merged_config.get("execution", {}).get("smoke_test_horizon_steps")
    eval_env = _eval_overrides_for_scenario(scenario_id)
    default_horizon = int(eval_env.get("ns", 120))
    return min(int(smoke), default_horizon) if smoke else default_horizon


def _execution_scenarios(merged_config: dict[str, Any], include_tier2: bool) -> list[str]:
    execution = merged_config.get("execution", {})
    if execution.get("scenarios_tier_1"):
        scenarios = list(execution.get("scenarios_tier_1", []))
        if include_tier2:
            scenarios.extend(execution.get("scenarios_tier_2", []))
        return scenarios
    if execution.get("scenarios"):
        return list(execution.get("scenarios", []))
    return [str(merged_config.get("scenario", {}).get("id", "nominal"))]


def _plot_run(run: EvaluationRun, figure_dir: Path, method_id: str) -> None:
    if run.rollout.observations.size == 0 or run.rollout.time.size == 0:
        return
    output = run.rollout.observations[:, 0]
    reference = run.rollout.references[:, 0] if run.rollout.references is not None else None
    fig, _ = plot_rollout_trajectory(
        time=run.rollout.time,
        output=output,
        reference=reference,
        action=run.rollout.actions,
        output_label="C_B",
        output_unit="mol/L",
        action_label="Control input",
        title=f"{REPORT_NAME_BY_METHOD[method_id]} - {run.rollout.scenario}",
    )
    save_figure(fig, figure_dir / f"{method_id}_{run.rollout.scenario}_trajectory")


def _rank_methods_by_metric(rows: list[dict[str, object]], metric_key: str) -> dict[str, float]:
    valid = [row for row in rows if metric_key in row and not math.isnan(float(row[metric_key]))]
    ranked = sorted(valid, key=lambda row: float(row[metric_key]))
    return {str(row["method"]): float(rank) for rank, row in enumerate(ranked, start=1)}


def _build_average_rank_rows(aggregated_rows: list[dict[str, object]], scenario_ids: Iterable[str]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aggregated_rows:
        grouped[str(row["scenario"])].append(row)

    scenario_rank_by_method: dict[str, dict[str, float]] = defaultdict(dict)
    for scenario_id in scenario_ids:
        scenario_rows = grouped.get(scenario_id, [])
        if not scenario_rows:
            continue
        per_method_ranks: dict[str, list[float]] = defaultdict(list)
        for metric_key in RANK_METRICS.get(scenario_id, []):
            for method, rank in _rank_methods_by_metric(scenario_rows, metric_key).items():
                per_method_ranks[method].append(rank)
        for method, ranks in per_method_ranks.items():
            scenario_rank_by_method[method][scenario_id] = sum(ranks) / len(ranks)

    rows: list[dict[str, object]] = []
    for method_id in METHOD_ORDER:
        report_name = REPORT_NAME_BY_METHOD[method_id]
        rank_map = scenario_rank_by_method.get(report_name, {})
        row = {"method": report_name}
        collected: list[float] = []
        for scenario_id in ("nominal", "disturbance", "highop", "uncertainty_pm10", "uncertainty_pm20"):
            value = rank_map.get(scenario_id, math.nan)
            row[f"{scenario_id}_rank"] = value
            if not math.isnan(float(value)):
                collected.append(float(value))
        row["overall_average_rank"] = sum(collected) / len(collected) if collected else math.nan
        rows.append(row)
    return rows


def _build_master_table(
    aggregated_rows: list[dict[str, object]],
    average_rank_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_key = {(str(row["method"]), str(row["scenario"])): row for row in aggregated_rows}
    rank_by_method = {str(row["method"]): row for row in average_rank_rows}

    def value(method_name: str, scenario_id: str, metric_key: str) -> float:
        row = by_key.get((method_name, scenario_id))
        if row is None:
            return math.nan
        return float(row.get(metric_key, math.nan))

    rows: list[dict[str, object]] = []
    for method_id in METHOD_ORDER:
        method_name = REPORT_NAME_BY_METHOD[method_id]
        rows.append(
            {
                "method": method_name,
                "nominal_iae": value(method_name, "nominal", "iae_mean"),
                "nominal_overshoot": value(method_name, "nominal", "overshoot_mean"),
                "nominal_settling_time": value(method_name, "nominal", "settling_time_mean"),
                "disturbance_iae": value(method_name, "disturbance", "iae_mean"),
                "disturbance_recovery_time": value(method_name, "disturbance", "recovery_time_mean"),
                "highop_iae": value(method_name, "highop", "iae_mean"),
                "highop_overshoot": value(method_name, "highop", "overshoot_mean"),
                "uncertainty_pm10_iae": value(method_name, "uncertainty_pm10", "iae_mean"),
                "uncertainty_pm10_constraint_violations": value(method_name, "uncertainty_pm10", "constraint_violations_mean"),
                "uncertainty_pm20_iae": value(method_name, "uncertainty_pm20", "iae_mean"),
                "uncertainty_pm20_constraint_violations": value(method_name, "uncertainty_pm20", "constraint_violations_mean"),
                "average_rank": float(rank_by_method.get(method_name, {}).get("overall_average_rank", math.nan)),
            }
        )
    return rows


def _build_modern_rl_rows(
    aggregated_rows: list[dict[str, object]],
    execution_id: str,
) -> list[dict[str, object]]:
    budgets = DEFAULT_TRAINING_BUDGETS.get(execution_id, DEFAULT_TRAINING_BUDGETS["debug_local"])
    long_rows: list[dict[str, object]] = []
    for row in aggregated_rows:
        method_name = str(row["method"])
        if method_name not in {REPORT_NAME_BY_METHOD[method_id] for method_id in MODERN_METHOD_IDS}:
            continue
        metric_keys = [key[:-5] for key in row.keys() if key.endswith("_mean")]
        for metric in metric_keys:
            long_rows.append(
                {
                    "method": method_name,
                    "scenario": row["scenario"],
                    "metric": metric,
                    "mean": row.get(f"{metric}_mean", math.nan),
                    "std": row.get(f"{metric}_std", math.nan),
                    "n_seeds": row.get("n_seeds", 0),
                    "train_steps": budgets["total_env_steps"],
                    "eval_frequency": budgets["evaluation_frequency"],
                    "eval_episodes": budgets["evaluation_episodes"],
                    "model_selection_rule": "lowest_eval_cost",
                    "deterministic_eval": True,
                    "notes": "",
                }
            )
    return long_rows


def _write_method_summaries(
    aggregated_rows: list[dict[str, object]],
    average_rank_rows: list[dict[str, object]],
    scenario_ids: list[str],
) -> dict[str, Path]:
    method_summary = PROCESSED_RESULTS_DIR / "method_summary.md"
    method_summary.write_text(
        "\n".join(
            [
                "# Method Summary",
                "",
                "- Paper-faithful reproduction methods: Static PID, Pure-RL (paper baseline), CIRL reproduced, and the high-op extended reproduction row.",
                "- Standardized modern RL baselines: SAC, TD3, TQC, PPO.",
                "- DR-CIRL is the only new method and differs from CIRL only by domain randomization during training.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    modern_rl_summary = PROCESSED_RESULTS_DIR / "modern_rl_summary.md"
    modern_rows = [row for row in average_rank_rows if row["method"] in {"SAC", "TD3", "TQC", "PPO"}]
    best_modern = min(
        (row for row in modern_rows if not math.isnan(float(row["overall_average_rank"]))),
        key=lambda row: float(row["overall_average_rank"]),
        default=None,
    )
    modern_lines = ["# Modern RL Summary", ""]
    modern_lines.append(f"- Evaluated Tier 1 scenarios: {', '.join(scenario_ids)}")
    if best_modern is not None:
        modern_lines.append(
            f"- Best modern RL baseline by average rank in this run: {best_modern['method']} "
            f"(average rank {float(best_modern['overall_average_rank']):.2f})."
        )
    else:
        modern_lines.append("- No modern RL baseline completed with usable checkpoints in this run.")
    modern_rl_summary.write_text("\n".join(modern_lines) + "\n", encoding="utf-8")

    extension_summary = PROCESSED_RESULTS_DIR / "extension_summary.md"
    by_key = {(str(row["method"]), str(row["scenario"])): row for row in aggregated_rows}
    cirl_nominal = by_key.get(("CIRL reproduced", "nominal"), {})
    dr_nominal = by_key.get(("DR-CIRL", "nominal"), {})
    cirl_unc = by_key.get(("CIRL reproduced", "uncertainty_pm10"), {})
    dr_unc = by_key.get(("DR-CIRL", "uncertainty_pm10"), {})
    extension_lines = ["# DR-CIRL Summary", ""]
    extension_lines.append(
        "- DR-CIRL keeps the reproduced CIRL architecture, optimizer family, reward/cost, and observation design fixed."
    )
    if cirl_nominal and dr_nominal:
        extension_lines.append(
            f"- Nominal IAE: CIRL {float(cirl_nominal.get('iae_mean', math.nan)):.4f}, "
            f"DR-CIRL {float(dr_nominal.get('iae_mean', math.nan)):.4f}."
        )
    if cirl_unc and dr_unc:
        extension_lines.append(
            f"- Uncertainty (+/-10%) IAE: CIRL {float(cirl_unc.get('iae_mean', math.nan)):.4f}, "
            f"DR-CIRL {float(dr_unc.get('iae_mean', math.nan)):.4f}."
        )
    extension_summary.write_text("\n".join(extension_lines) + "\n", encoding="utf-8")
    return {
        "method_summary": method_summary,
        "modern_rl_summary": modern_rl_summary,
        "extension_summary": extension_summary,
    }


def run_extended_benchmark(config_paths: list[str], *, include_tier2: bool = False) -> dict[str, Path]:
    ensure_project_directories()
    merged = load_project_config(config_paths)
    execution_id = str(merged.get("execution", {}).get("id", "debug_local"))
    scenario_ids = _execution_scenarios(merged, include_tier2=include_tier2)
    seeds = [int(seed) for seed in merged.get("execution", {}).get("seeds", [0])]
    raw_root = Path(merged.get("paths", {}).get("results_raw", "results/raw")) / "extended_benchmark"

    run_results: list[EvaluationRun] = []
    missing_notes: list[str] = []

    for scenario_id in scenario_ids:
        horizon = _horizon_for_scenario(merged, scenario_id)
        env_overrides = _eval_overrides_for_scenario(scenario_id)
        figure_dir = FIGURES_DIR / "extended_benchmark" / scenario_id
        for method_id in METHOD_ORDER:
            try:
                training_scenario = _benchmark_training_scenario(method_id)
                controller_factory = lambda seed, method_id=method_id, training_scenario=training_scenario: load_controller(
                    method_id,
                    training_scenario=training_scenario,
                    seed=seed,
                )
                env_factory = lambda seed, method_id=method_id, scenario_id=scenario_id, env_overrides=env_overrides: make_benchmark_env(
                    method=method_id,
                    scenario=scenario_id,
                    config=env_overrides,
                    seed=seed,
                )
                method_runs = evaluate_scenario(
                    env_factory=env_factory,
                    controller_factory=controller_factory,
                    method=REPORT_NAME_BY_METHOD[method_id],
                    scenario=scenario_id,
                    seeds=seeds,
                    horizon=horizon,
                    output_dir=raw_root / scenario_id / method_id,
                    deterministic=True,
                )
                if method_runs:
                    _plot_run(method_runs[0], figure_dir, method_id)
                run_results.extend(method_runs)
            except FileNotFoundError:
                missing_notes.append(f"{REPORT_NAME_BY_METHOD[method_id]} missing checkpoint for scenario `{scenario_id}`.")
            except RuntimeError as exc:
                missing_notes.append(f"{REPORT_NAME_BY_METHOD[method_id]} unavailable for scenario `{scenario_id}`: {exc}")

    flat_rows = [
        {"method": run.rollout.method, "scenario": run.rollout.scenario, "seed": run.rollout.seed, **run.metrics}
        for run in run_results
    ]
    aggregated_rows = aggregate_run_metrics(flat_rows)
    average_rank_rows = _build_average_rank_rows(aggregated_rows, scenario_ids)
    master_rows = _build_master_table(aggregated_rows, average_rank_rows)
    modern_rl_rows = _build_modern_rl_rows(aggregated_rows, execution_id)

    outputs = {
        "benchmark_summary_csv": write_csv_rows(aggregated_rows, TABLES_DIR / "benchmark_summary.csv"),
        "benchmark_summary_md": write_markdown_table(aggregated_rows, TABLES_DIR / "benchmark_summary.md"),
        "modern_rl_csv": write_csv_rows(modern_rl_rows, TABLES_DIR / "modern_rl_baselines.csv"),
        "modern_rl_md": write_markdown_table(modern_rl_rows, TABLES_DIR / "modern_rl_baselines.md"),
        "master_csv": write_csv_rows(master_rows, TABLES_DIR / "master_comparison_table.csv"),
        "master_md": write_markdown_table(master_rows, TABLES_DIR / "master_comparison_table.md"),
        "rank_csv": write_csv_rows(average_rank_rows, TABLES_DIR / "average_rank_table.csv"),
        "rank_md": write_markdown_table(average_rank_rows, TABLES_DIR / "average_rank_table.md"),
    }
    outputs.update(_write_method_summaries(aggregated_rows, average_rank_rows, scenario_ids))

    notes_path = PROCESSED_RESULTS_DIR / "benchmark_notes.md"
    if missing_notes:
        notes_path.write_text("# Benchmark Notes\n\n" + "\n".join(f"- {note}" for note in missing_notes) + "\n", encoding="utf-8")
        outputs["notes"] = notes_path
    elif notes_path.exists():
        notes_path.unlink()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the extended benchmark suite.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=[
            str(CONFIGS_DIR / "base.yaml"),
            str(CONFIGS_DIR / "execution" / "debug_local.yaml"),
            str(CONFIGS_DIR / "scenarios" / "nominal.yaml"),
        ],
    )
    parser.add_argument("--include-tier-2", action="store_true")
    args = parser.parse_args()
    artifacts = run_extended_benchmark(args.config, include_tier2=args.include_tier_2)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
