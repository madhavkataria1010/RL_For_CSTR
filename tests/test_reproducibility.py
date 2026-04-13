from __future__ import annotations

import numpy as np

from src.evaluation.scenario_runner import evaluate_scenario


def test_evaluation_is_reproducible_for_fixed_seeds(dummy_env_factory, dummy_controller_factory, tmp_path):
    kwargs = {
        "env_factory": dummy_env_factory,
        "controller_factory": dummy_controller_factory,
        "method": "Static PID",
        "scenario": "nominal_tracking",
        "seeds": [1, 2],
        "horizon": 8,
        "output_dir": tmp_path / "artifacts",
        "deterministic": True,
    }
    first = evaluate_scenario(**kwargs)
    second = evaluate_scenario(**kwargs)

    first_costs = [run.metrics["episode_cost"] for run in first]
    second_costs = [run.metrics["episode_cost"] for run in second]
    assert np.allclose(first_costs, second_costs)
