from __future__ import annotations

import numpy as np

from src.evaluation.rollout import rollout_episode


def test_deterministic_eval_uses_predict_contract(dummy_env_factory, dummy_predict_controller_factory):
    controller_a = dummy_predict_controller_factory(seed=11)
    controller_b = dummy_predict_controller_factory(seed=99)

    rollout_a = rollout_episode(
        env=dummy_env_factory(seed=0),
        controller=controller_a,
        method="SAC",
        scenario="nominal_tracking",
        seed=0,
        horizon=8,
        deterministic=True,
    )
    rollout_b = rollout_episode(
        env=dummy_env_factory(seed=0),
        controller=controller_b,
        method="SAC",
        scenario="nominal_tracking",
        seed=0,
        horizon=8,
        deterministic=True,
    )

    assert np.allclose(rollout_a.actions, rollout_b.actions)


def test_stochastic_eval_can_vary_when_requested(dummy_env_factory, dummy_predict_controller_factory):
    rollout_a = rollout_episode(
        env=dummy_env_factory(seed=0),
        controller=dummy_predict_controller_factory(seed=1),
        method="TD3",
        scenario="nominal_tracking",
        seed=0,
        horizon=8,
        deterministic=False,
    )
    rollout_b = rollout_episode(
        env=dummy_env_factory(seed=0),
        controller=dummy_predict_controller_factory(seed=2),
        method="TD3",
        scenario="nominal_tracking",
        seed=0,
        horizon=8,
        deterministic=False,
    )

    assert not np.allclose(rollout_a.actions, rollout_b.actions)
