from __future__ import annotations

import numpy as np

from src.controllers.controller_factory import OFFICIAL_STATIC_PID_PHYSICAL_GAINS
from src.controllers.pid import StaticPIDController
from src.environments.standardized_env import make_paper_exact_env
from src.evaluation.rollout import rollout_episode


def test_rollout_episode_runs_with_dummy_env(dummy_env_factory, dummy_controller_factory):
    env = dummy_env_factory(seed=0)
    controller = dummy_controller_factory(seed=0)

    rollout = rollout_episode(
        env=env,
        controller=controller,
        method="Static PID",
        scenario="nominal_tracking",
        seed=0,
        horizon=8,
        deterministic=True,
    )

    assert rollout.time.shape == (8,)
    assert rollout.observations.shape[0] == 8
    assert rollout.actions.shape == (8, 1)
    assert rollout.references is not None
    assert np.isfinite(rollout.rewards).all()
    assert np.isfinite(rollout.costs).all()


def test_rollout_time_is_strictly_increasing_for_paper_env():
    env = make_paper_exact_env("static_pid", "nominal_test", config={"ns": 12})
    controller = StaticPIDController.from_physical_gains(OFFICIAL_STATIC_PID_PHYSICAL_GAINS)

    rollout = rollout_episode(
        env=env,
        controller=controller,
        method="Static PID",
        scenario="nominal",
        seed=0,
        horizon=12,
        deterministic=True,
    )

    assert rollout.time.shape == (12,)
    assert np.all(np.diff(rollout.time) > 0)
