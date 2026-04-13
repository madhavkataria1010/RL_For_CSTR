from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class VectorPolicy(Protocol):
    def get_parameters(self) -> np.ndarray: ...
    def set_parameters(self, params: np.ndarray) -> None: ...
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray: ...


@dataclass
class EvolutionStrategyConfig:
    random_search_candidates: int = 30
    pso_particles: int = 30
    pso_iterations: int = 150
    eval_episodes: int = 3
    init_min: float = -0.1
    init_max: float = 0.1
    inertia_weight: float = 0.6
    cognitive_weight: float = 1.0
    social_weight: float = 1.0
    velocity_scale: float = 0.05


def rollout_policy(policy: VectorPolicy, env_factory, episodes: int) -> float:
    total_cost = 0.0
    for _ in range(episodes):
        env = env_factory()
        observation, _ = env.reset()
        done = False
        while not done:
            action = policy.predict(observation, deterministic=True)
            observation, cost, done, _, _ = env.step(action)
            total_cost += float(cost)
    return total_cost / max(episodes, 1)


def optimize_with_random_search_and_pso(
    policy: VectorPolicy,
    env_factory,
    config: EvolutionStrategyConfig,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    base_params = policy.get_parameters()
    best_params = base_params.copy()
    best_cost = np.inf
    history: list[float] = []

    # Random search warm start.
    for _ in range(config.random_search_candidates):
        candidate = rng.uniform(config.init_min, config.init_max, size=base_params.shape)
        policy.set_parameters(candidate)
        cost = rollout_policy(policy, env_factory, config.eval_episodes)
        history.append(cost)
        if cost < best_cost:
            best_cost = cost
            best_params = candidate.copy()

    # Particle swarm over flat parameter vectors.
    particles = rng.uniform(
        config.init_min,
        config.init_max,
        size=(config.pso_particles, base_params.size),
    )
    velocities = rng.normal(
        loc=0.0,
        scale=config.velocity_scale,
        size=(config.pso_particles, base_params.size),
    )
    personal_best = particles.copy()
    personal_best_costs = np.full(config.pso_particles, np.inf)
    global_best = best_params.copy()
    global_best_cost = best_cost

    for _ in range(config.pso_iterations):
        for idx in range(config.pso_particles):
            policy.set_parameters(particles[idx])
            cost = rollout_policy(policy, env_factory, config.eval_episodes)
            history.append(cost)
            if cost < personal_best_costs[idx]:
                personal_best_costs[idx] = cost
                personal_best[idx] = particles[idx].copy()
            if cost < global_best_cost:
                global_best_cost = cost
                global_best = particles[idx].copy()

        for idx in range(config.pso_particles):
            r1 = rng.random(base_params.size)
            r2 = rng.random(base_params.size)
            velocities[idx] = (
                config.inertia_weight * velocities[idx]
                + config.cognitive_weight * r1 * (personal_best[idx] - particles[idx])
                + config.social_weight * r2 * (global_best - particles[idx])
            )
            particles[idx] = particles[idx] + velocities[idx]

    policy.set_parameters(global_best)
    return global_best, history

