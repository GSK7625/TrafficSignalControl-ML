"""Reward cua Q-Learning; SumoEnv va cac baseline van do queue theo giay."""

import math


STARVATION_SECONDS = 300


def reward_config(weight):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
        raise ValueError("starvation lambda must be finite and nonnegative")
    if weight == 0:
        return {"version": "mean-queue-v1", "lambda": 0.0}
    return {"version": "mean-queue-edge-age-v1", "lambda": weight,
            "threshold_seconds": STARVATION_SECONDS, "ramp_seconds": STARVATION_SECONDS}


def model_reward(metadata):
    if metadata["schema_version"] == 1 and "reward" not in metadata:
        return reward_config(0)
    config = metadata.get("reward")
    if (metadata["schema_version"] != 2 or not isinstance(config, dict)
            or config != reward_config(config.get("lambda"))):
        raise ValueError("Unsupported or inconsistent Q-learning reward metadata")
    return config


def reward_components(mean_queue, oldest_age, weight):
    """Tuoi cuoi action tinh bang giay. Phat chuan hoa tu 0 den 1."""
    if not math.isfinite(mean_queue) or not math.isfinite(oldest_age) or min(mean_queue, oldest_age) < 0:
        raise ValueError("Invalid queue or observed demand age")
    penalty = min(1.0, max(0.0, (oldest_age - STARVATION_SECONDS) / STARVATION_SECONDS))
    reward = -mean_queue - weight * penalty
    if not math.isfinite(reward) or reward > 0:
        raise ValueError("Invalid reward")
    return penalty, reward
