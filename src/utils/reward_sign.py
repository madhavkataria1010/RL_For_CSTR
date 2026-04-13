from __future__ import annotations


def cost_to_reward(cost: float) -> float:
    return -float(cost)


def reward_to_cost(reward: float) -> float:
    return -float(reward)

