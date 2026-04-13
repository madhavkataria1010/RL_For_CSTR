from __future__ import annotations

import numpy as np

from src.evaluation.rollout import rollout_episode


def test_predict_style_controller_is_supported(dummy_env_factory, dummy_predict_controller_factory):
    rollout = rollout_episode(
        env=dummy_env_factory(seed=5),
        controller=dummy_predict_controller_factory(seed=5),
        method="SAC",
        scenario="nominal_tracking",
        seed=5,
        horizon=6,
        deterministic=True,
    )

    assert rollout.actions.shape == (6, 1)
    assert np.isfinite(rollout.actions).all()
