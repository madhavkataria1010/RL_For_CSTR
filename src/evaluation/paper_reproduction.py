"""Paper-faithful reproduction CLI and helpers."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

from src.controllers.controller_factory import load_controller
from src.environments.standardized_env import make_paper_exact_env
from src.evaluation.aggregation import aggregate_run_metrics, write_csv_rows, write_markdown_table
from src.evaluation.scenario_runner import EvaluationRun, evaluate_scenario
from src.plotting.publication_style import save_figure
from src.plotting.trajectories import plot_rollout_trajectory
from src.training.common_interface import load_project_config
from src.utils.config import load_yaml
from src.utils.official_artifacts import resolve_paper_reproduction_artifact
from src.utils.paths import CONFIGS_DIR, FIGURES_DIR, PROCESSED_RESULTS_DIR, TABLES_DIR, ensure_project_directories


PAPER_REPRODUCTION_METHODS = (
    "Static PID",
    "Pure-RL (paper baseline)",
    "CIRL reproduced",
    "CIRL high-op extended (paper reproduction only)",
)

PAPER_REPRODUCTION_SCENARIOS = ("nominal", "disturbance", "highop")

REPRODUCTION_SUMMARY_COLUMNS = (
    "method",
    "scenario",
    "paper_reference_value",
    "reproduced_value_mean",
    "reproduced_value_std",
    "relative_gap_percent",
    "notes",
)

REPORT_NAME_BY_METHOD = {
    "static_pid": "Static PID",
    "pure_rl_paper": "Pure-RL (paper baseline)",
    "cirl_reproduced": "CIRL reproduced",
    "cirl_highop_extended_paper": "CIRL high-op extended (paper reproduction only)",
}

PAPER_REFERENCE_VALUES: dict[str, dict[str, tuple[float, str]]] = {
    "nominal": {
        "Static PID": (-1.77, "exact paper table value"),
        "Pure-RL (paper baseline)": (-2.08, "exact paper table value"),
        "CIRL reproduced": (-1.33, "exact paper table value"),
    },
    "disturbance": {
        "Pure-RL (paper baseline)": (-1.76, "exact paper table value"),
        "CIRL reproduced": (-1.38, "exact paper table value"),
    },
    "highop": {
        "Static PID": (-6.81, "exact paper table value"),
        "CIRL reproduced": (-4.04, "exact paper table value"),
        "CIRL high-op extended (paper reproduction only)": (-2.07, "exact paper table value"),
    },
}

SCENARIO_ORDER = {name: idx for idx, name in enumerate(PAPER_REPRODUCTION_SCENARIOS)}
METHOD_ORDER = {name: idx for idx, name in enumerate(PAPER_REPRODUCTION_METHODS)}


def _scenario_config_path(scenario_id: str) -> Path:
    return CONFIGS_DIR / "scenarios" / f"{scenario_id}.yaml"


def _load_scenario_config(scenario_id: str) -> dict[str, Any]:
    path = _scenario_config_path(scenario_id)
    if not path.exists():
        return {}
    return load_yaml(path)


def _paper_methods_for_scenario(scenario_id: str) -> list[str]:
    if scenario_id == "nominal":
        return ["static_pid", "pure_rl_paper", "cirl_reproduced"]
    if scenario_id == "disturbance":
        return ["pure_rl_paper", "cirl_reproduced"]
    if scenario_id == "highop":
        return ["static_pid", "cirl_reproduced", "cirl_highop_extended_paper"]
    raise ValueError(f"Unsupported paper reproduction scenario '{scenario_id}'.")


def _paper_eval_scenario_id(scenario_id: str) -> str:
    return f"{scenario_id}_test"


def _paper_training_scenario_id(scenario_id: str, method_id: str) -> str | None:
    if method_id == "static_pid":
        return None
    if scenario_id == "nominal":
        return "nominal"
    if scenario_id == "disturbance":
        return "disturbance"
    if scenario_id == "highop":
        if method_id == "cirl_highop_extended_paper":
            return "highop"
        return "nominal"
    return None


def _paper_eval_overrides(merged_config: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    scenario_cfg = _load_scenario_config(scenario_id)
    paper_exact = dict(scenario_cfg.get("paper_exact", {}))
    eval_env = dict(paper_exact.get("eval_env", {}))
    if not eval_env:
        eval_env = dict(merged_config.get("paper_exact", {}).get("eval_env", {}))
    if "disturbance_activation_step" in eval_env:
        eval_env["disturbance_start_step"] = eval_env.pop("disturbance_activation_step")
    if "training_ca_in_values" in eval_env:
        eval_env["training_disturbances"] = tuple(eval_env.pop("training_ca_in_values"))
    if "eval_ca_in_value" in eval_env:
        eval_env["evaluation_disturbance"] = float(eval_env.pop("eval_ca_in_value"))
    return eval_env


def _prefer_official_artifact(method_id: str, scenario_id: str) -> bool:
    # Only static_pid and CIRL-pkl artifacts are verified compatible with this env.
    # Official .pth checkpoints for pure_rl and cirl disturbance/highop-extended
    # produce catastrophic results (-1000s) due to architecture/obs-space mismatch.
    return (method_id, scenario_id) in {
        ("static_pid", "nominal"),
        ("static_pid", "highop"),
        ("cirl_reproduced", "nominal"),
        ("cirl_reproduced", "highop"),
    }


def _paper_horizon(merged_config: dict[str, Any], scenario_id: str) -> int:
    smoke = merged_config.get("execution", {}).get("smoke_test_horizon_steps")
    eval_env = _paper_eval_overrides(merged_config, scenario_id)
    default_horizon = int(eval_env.get("ns", 120))
    return min(int(smoke), default_horizon) if smoke else default_horizon


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


def _build_summary_rows(
    scenario_id: str,
    methods: list[str],
    aggregated_rows: list[dict[str, object]],
    missing_methods: dict[str, str],
    artifact_notes: dict[str, str],
) -> list[dict[str, object]]:
    row_by_method = {
        str(row["method"]): row
        for row in aggregated_rows
    }
    summary_rows: list[dict[str, object]] = []
    for method_id in methods:
        report_name = REPORT_NAME_BY_METHOD[method_id]
        paper_reference, reference_note = PAPER_REFERENCE_VALUES.get(scenario_id, {}).get(report_name, (math.nan, "not reported in paper table"))
        aggregated = row_by_method.get(report_name)
        if aggregated is None:
            note = missing_methods.get(method_id, reference_note)
            if method_id in artifact_notes:
                note = f"{note}; {artifact_notes[method_id]}"
            summary_rows.append(
                {
                    "method": report_name,
                    "scenario": scenario_id,
                    "paper_reference_value": paper_reference,
                    "reproduced_value_mean": math.nan,
                    "reproduced_value_std": math.nan,
                    "relative_gap_percent": math.nan,
                    "notes": note,
                }
            )
            continue

        reproduced_mean = -float(aggregated["episode_cost_mean"])
        reproduced_std = float(aggregated["episode_cost_std"])
        if math.isnan(paper_reference):
            gap = math.nan
        else:
            gap = 100.0 * (reproduced_mean - float(paper_reference)) / max(abs(float(paper_reference)), 1e-9)
        summary_rows.append(
            {
                "method": report_name,
                "scenario": scenario_id,
                "paper_reference_value": paper_reference,
                "reproduced_value_mean": reproduced_mean,
                "reproduced_value_std": reproduced_std,
                "relative_gap_percent": gap,
                "notes": (
                    f"{reference_note}; {artifact_notes[method_id]}"
                    if method_id in artifact_notes
                    else reference_note
                ),
            }
        )
    return summary_rows


def _load_existing_summary_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _merge_summary_rows(
    existing_rows: list[dict[str, object]],
    new_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: dict[tuple[str, str], dict[str, object]] = {}
    for row in existing_rows:
        key = (str(row.get("method", "")), str(row.get("scenario", "")))
        merged[key] = row
    for row in new_rows:
        key = (str(row.get("method", "")), str(row.get("scenario", "")))
        merged[key] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            SCENARIO_ORDER.get(str(row.get("scenario", "")), 999),
            METHOD_ORDER.get(str(row.get("method", "")), 999),
        ),
    )


def _write_processed_summary(summary_rows: list[dict[str, object]]) -> Path:
    lines = ["# Reproduction Summary", ""]
    if not summary_rows:
        lines.append("No reproduction rows were produced.")
    else:
        best_row = min(
            (row for row in summary_rows if not math.isnan(float(row["reproduced_value_mean"]))),
            key=lambda row: abs(float(row["relative_gap_percent"])) if not math.isnan(float(row["relative_gap_percent"])) else float("inf"),
            default=None,
        )
        lines.append(f"- Total reproduced rows: {len(summary_rows)}")
        if best_row is not None:
            lines.append(
                f"- Closest paper match in this run: {best_row['method']} on `{best_row['scenario']}` "
                f"(gap {float(best_row['relative_gap_percent']):.2f}%)."
            )
        missing = [row for row in summary_rows if math.isnan(float(row["reproduced_value_mean"]))]
        if missing:
            lines.append(f"- Missing controller artifacts for {len(missing)} row(s); those are marked directly in the table.")
    target = PROCESSED_RESULTS_DIR / "reproduction_summary.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def run_paper_reproduction(config_paths: list[str]) -> dict[str, Path]:
    ensure_project_directories()
    merged = load_project_config(config_paths)
    scenario_id = str(merged.get("scenario", {}).get("id", "nominal"))
    if scenario_id not in PAPER_REPRODUCTION_SCENARIOS:
        raise ValueError(f"Scenario '{scenario_id}' is not a paper reproduction scenario.")
    seeds = [int(seed) for seed in merged.get("execution", {}).get("seeds", [0])]
    horizon = _paper_horizon(merged, scenario_id)
    eval_scenario_id = _paper_eval_scenario_id(scenario_id)
    env_overrides = _paper_eval_overrides(merged, scenario_id)
    methods = _paper_methods_for_scenario(scenario_id)

    run_results: list[EvaluationRun] = []
    missing_methods: dict[str, str] = {}
    artifact_notes: dict[str, str] = {}
    figure_dir = FIGURES_DIR / "paper_reproduction" / scenario_id
    raw_dir = Path(merged.get("paths", {}).get("results_raw", "results/raw")) / "paper_reproduction" / scenario_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    for method_id in methods:
        try:
            artifact = (
                resolve_paper_reproduction_artifact(method_id, scenario_id)
                if _prefer_official_artifact(method_id, scenario_id)
                else None
            )
            if artifact is not None:
                artifact_notes[method_id] = f"evaluated via {artifact['source']}"
                if artifact["type"] == "static_pid":
                    physical_gains = artifact["physical_gains"]
                    controller_factory = (
                        lambda seed, method_id=method_id, physical_gains=physical_gains: load_controller(
                            method_id,
                            physical_gains=physical_gains,
                        )
                    )
                else:
                    checkpoint = artifact["checkpoint_path"]
                    controller_factory = (
                        lambda seed, method_id=method_id, checkpoint=checkpoint: load_controller(
                            method_id,
                            checkpoint=checkpoint,
                        )
                    )
            else:
                training_scenario = _paper_training_scenario_id(scenario_id, method_id)
                controller_factory = lambda seed, method_id=method_id, training_scenario=training_scenario: load_controller(
                    method_id,
                    training_scenario=training_scenario,
                    seed=seed,
                )
            env_factory = lambda seed, method_id=method_id: make_paper_exact_env(
                method=method_id,
                scenario=eval_scenario_id,
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
                output_dir=raw_dir / method_id,
                deterministic=True,
            )
            if method_runs:
                _plot_run(method_runs[0], figure_dir, method_id)
            run_results.extend(method_runs)
        except FileNotFoundError:
            missing_methods[method_id] = "checkpoint missing; train or sync the method artifacts first"
        except RuntimeError as exc:
            missing_methods[method_id] = str(exc)

    flat_rows = [
        {"method": run.rollout.method, "scenario": run.rollout.scenario, "seed": run.rollout.seed, **run.metrics}
        for run in run_results
    ]
    aggregated_rows = aggregate_run_metrics(flat_rows)
    summary_rows = _build_summary_rows(
        scenario_id,
        methods,
        aggregated_rows,
        missing_methods,
        artifact_notes,
    )

    csv_target = TABLES_DIR / "reproduction_summary.csv"
    combined_rows = _merge_summary_rows(_load_existing_summary_rows(csv_target), summary_rows)
    csv_path = write_csv_rows(combined_rows, csv_target)
    md_path = write_markdown_table(combined_rows, TABLES_DIR / "reproduction_summary.md")
    processed_md = _write_processed_summary(combined_rows)
    return {"csv": csv_path, "markdown": md_path, "processed_summary": processed_md}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-faithful reproduction evaluation.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=[
            str(CONFIGS_DIR / "base.yaml"),
            str(CONFIGS_DIR / "execution" / "debug_local.yaml"),
            str(CONFIGS_DIR / "scenarios" / "nominal.yaml"),
        ],
    )
    args = parser.parse_args()
    artifacts = run_paper_reproduction(args.config)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
