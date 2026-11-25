"""Lich theo thu tu episode (bat dau tu 1), dung chung cho train va replay offline."""


def episode_epsilon(episode, decay_episodes, epsilon_start, epsilon_min, schedule="linear"):
    if type(episode) is not int or episode < 1 or type(decay_episodes) is not int or decay_episodes < 1:
        raise ValueError("Episode and decay length must be positive integers")
    if not 0 <= epsilon_min <= epsilon_start <= 1:
        raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1")
    if schedule not in ("linear", "exponential"):
        raise ValueError("Unknown epsilon schedule")
    if schedule == "exponential" and epsilon_min == 0 and epsilon_start > 0:
        raise ValueError("Exponential decay requires a positive minimum")
    if episode == 1:
        return epsilon_start
    if episode > decay_episodes or epsilon_start == epsilon_min:
        return epsilon_min
    if schedule == "linear":
        # Giu thu tu phep tinh de replay dung ca cac checkpoint lich su.
        return max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min)
                   * (episode - 1) / (decay_episodes - 1))
    fraction = (episode - 1) / (decay_episodes - 1)
    return epsilon_start * (epsilon_min / epsilon_start) ** fraction


def episode_sumo_seed(episode, seeds):
    if type(episode) is not int or episode < 1:
        raise ValueError("Episode must be a positive integer")
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds):
        raise ValueError("Provide SUMO seeds in [0, 2**31)")
    return seeds[(episode - 1) % len(seeds)]
