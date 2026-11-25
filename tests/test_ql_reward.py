import math

import pytest

import analyze_ql as audit
import evaluate
import q_learning as ql
from ql_progress import progress
from ql_reward import model_reward, reward_components, reward_config
from sumo_env import ROOT, SumoEnv


@pytest.mark.parametrize("area,elapsed,age,weight,expected", [
    (0, 30, 0, 15, (0, 0)), (3000, 30, 300, 15, (0, -100)),
    (300, 30, 450, 5, (.5, -12.5)), (400, 40, 450, 0, (.5, -10)),
    (400, 40, 900, 15, (1, -25))])
def test_reward_by_hand(area, elapsed, age, weight, expected):
    assert reward_components(area / elapsed, age, weight) == expected


def test_service_empty_queue_hold_and_switch():
    env = SumoEnv("cologne1")
    enc = ql.build_encoding(env)
    demand = ql.EdgeDemand(env, enc)
    # Two occupied lanes on the same edge. Other edges have no observed demand.
    lanes = enc["incoming_groups"][0]["lanes"]
    state = {"incoming": [int(lane in lanes) for lane in enc["incoming_lanes"]]}
    demand.observe(state, 0)
    serving = next(sig for sig in env.green_states.values() if any(sig[i] == "G" for i in demand.links[0]))
    holding = next(sig for sig in env.green_states.values() if all(sig[i] != "G" for i in demand.links[0]))
    demand.record_green([{"signal": holding}])
    _, age = demand.observe(state, 450)
    assert reward_components(10, age, 5) == (.5, -12.5)
    # A switch cut short in clearance still leaves the queued edge unserved.
    demand.record_green([{"signal": holding.replace("G", "y")}, {"signal": "r" * len(holding)}])
    assert demand.observe(state, 460) == (0, 460)
    demand.record_green([{"signal": serving}])
    assert reward_components(10, demand.observe(state, 490)[1], 5) == (0, -10)
    assert demand.observe({"incoming": [0] * len(state["incoming"])}, 1000) == (None, 0)


def test_reward_metadata_is_explicit_and_legacy_models_keep_original_reward():
    assert model_reward({"schema_version": 1}) == reward_config(0)
    for config in (None, {}, {"version": "mean-queue-v1", "lambda": 5},
                   {**reward_config(5), "ramp_seconds": 30}):
        with pytest.raises(ValueError):
            model_reward({"schema_version": 2, "reward": config})
    for weight in (-1, math.nan, math.inf):
        with pytest.raises(ValueError):
            reward_config(weight)
    with pytest.raises(ValueError):
        reward_components(math.inf, 0, 0)


@pytest.mark.parametrize("weight", [0, 5, 15])
def test_reward_validation_smoke(tmp_path, monkeypatch, weight):
    def short_env(*args, **kwargs):
        env = SumoEnv(*args, **kwargs)
        env.end = env.begin + 900
        return env

    monkeypatch.setattr(ql, "SumoEnv", short_env)
    monkeypatch.setattr(evaluate, "SumoEnv", short_env)
    folder = tmp_path / "train"
    table, meta = ql.train("cologne1", folder, 3, overdue_state=True, decay_episodes=3,
                           starvation_lambda=weight)
    model = folder / "episode_003/q_table.json"
    assert ql.load_q_table(model, short_env("cologne1")) == (table, meta)
    with pytest.raises(ValueError, match="reward"):
        evaluate.run_evaluation("cologne1", "QL", 42, tmp_path / "wrong", model, expect_starvation_lambda=weight+1)
    runs = []
    for seed in (42, 12345):
        out = tmp_path / f"validation_{seed}"
        result = evaluate.run_evaluation("cologne1", "QL", seed, out, model, expect_starvation_lambda=weight)
        assert result["model_sha256_before"] == result["model_sha256_after"]
        runs.append(out)
    audit.replay_training(ROOT, "cologne1", audit.read_json(model), audit.read_json(runs[0] / "metadata.json"),
                          training_dir=folder)
    metrics = progress(folder, runs)[-1]
    assert metrics["validation_runs"] == 2 and 0 <= metrics["max_observed_starvation_s"] <= 900
