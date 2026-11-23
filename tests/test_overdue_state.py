import json

import pytest

import analyze_ql as audit
import evaluate
import q_learning as ql
from sumo_env import ROOT, SumoEnv


def test_overdue_edge_threshold_tie_queue_reset_and_observed_green():
    env = SumoEnv("cologne1")
    encoding = ql.build_encoding(env, overdue_state=True)
    demand = ql.EdgeDemand(env, encoding)
    state = dict(phase=0, incoming=[1] * len(env.incoming_lanes),
                 outgoing=[0] * len(env.outgoing_lanes))
    for now, expected in [(10, 0), (309, 0), (310, 1)]:
        demand.observe(state, now)
        assert ql.encode_state(state, encoding)[-1] == expected

    # Lenh doi pha bi cat trong vang/do: chua co mau G de reset tuoi.
    signal = env.green_states[1]
    demand.record_green([{"signal": signal.replace("G", "y")}, {"signal": "r" * len(signal)}])
    demand.observe(state, 320)
    assert state["overdue_edge"] == 1
    demand.record_green([{"signal": signal}])
    demand.observe(state, 350)
    assert state["overdue_edge"] == 2
    for lane in encoding["incoming_groups"][1]["lanes"]:
        state["incoming"][env.incoming_lanes.index(lane)] = 0
    demand.observe(state, 351)
    assert state["overdue_edge"] == 3
    ql.EdgeDemand(env, encoding).observe(state, 351)
    assert state["overdue_edge"] == 0


def test_overdue_model_checks_metadata_and_edge_code(tmp_path):
    env = SumoEnv("cologne1")
    metadata = ql.make_metadata(env, episodes=1, alpha=.1, gamma=.95, epsilon_start=1,
                                epsilon_min=.05, exploration_seed=7, overdue_state=True)
    key = (2, 2, 2, 2, 0, 0, 0, 0, 7, 4)
    table = {key: [0.] * 8}
    model = tmp_path / "model.json"
    ql.save_q_table(model, table, metadata)
    assert ql.load_q_table(model, env) == (table, metadata)
    for change in ("code", "threshold", "version", "edge_order", "length"):
        payload = audit.read_json(model)
        encoding = payload["metadata"]["encoding"]
        if change == "code":
            payload["rows"][0]["state"][-1] = 5
        elif change == "threshold":
            encoding["overdue_seconds"] = 299
        elif change == "version":
            encoding["state_version"] = "unknown"
        elif change == "edge_order":
            encoding["incoming_groups"].reverse()
        else:
            payload["rows"][0]["state"].pop()
        bad = tmp_path / f"{change}.json"
        bad.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            ql.load_q_table(bad, env)


@pytest.mark.parametrize("busiest", [False, True])
def test_overdue_short_sumo_train_and_greedy_replay(tmp_path, monkeypatch, busiest):
    def short_env(*args, **kwargs):
        env = SumoEnv(*args, **kwargs)
        env.end = env.begin + 420
        return env

    monkeypatch.setattr(ql, "SumoEnv", short_env)
    monkeypatch.setattr(evaluate, "SumoEnv", short_env)
    folder = tmp_path / "train"
    _, metadata = ql.train("cologne1", folder, 1, overdue_state=True, busiest_state=busiest)
    model = folder / "cologne1_q_table.json"
    out = tmp_path / "eval"
    result = evaluate.run_evaluation("cologne1", "QL", 42, out, model)
    assert result["model_sha256_before"] == result["model_sha256_after"]
    mapping = audit.read_json(out / "metadata.json")
    assert mapping["guard_seconds"] is None and mapping["epsilon"] == 0
    encoding = metadata["encoding"]
    _, _, _, observations = audit.replay_training(ROOT, "cologne1", audit.read_json(model), mapping,
                                                    training_dir=folder)
    audit.observation_analysis(observations, encoding)
    rows = audit.read_csv(out / "actions.csv")
    audit.audit_overdue_states(rows, audit.read_csv(out / "signals.csv"), mapping, encoding)
    for row in rows:
        values = json.loads(row["q_values"])
        assert int(row["action"]) == mapping["actions"][values.index(max(values))]
        assert list(ql.encode_state(json.loads(row["state"]), encoding)) == json.loads(row["encoded_state"])


def test_busiest_state_sums_lanes_breaks_ties_and_checks_model(tmp_path):
    env = SumoEnv("cologne1")
    metadata = ql.make_metadata(env, episodes=1, alpha=.1, gamma=.95, epsilon_start=1,
                                epsilon_min=.05, exploration_seed=7, busiest_state=True)
    encoding = metadata["encoding"]
    assert encoding["state_version"] == "overdue-busiest-v1"
    incoming = [0] * len(env.incoming_lanes)
    state = dict(incoming=incoming, outgoing=[1] * len(env.outgoing_lanes), phase=7, overdue_edge=0)
    assert ql.encode_state(state, encoding) == (0, 0, 0, 0, 0, 1, 1, 1, 1, 7, 0)
    for group, queues in zip(encoding["incoming_groups"][:2], [(3, 4), (7, 0)]):
        for lane, queue in zip(group["lanes"], queues, strict=True):
            incoming[env.incoming_lanes.index(lane)] = queue
    assert ql.encode_state(state, encoding)[:5] == (1, 1, 0, 0, 1)
    incoming[env.incoming_lanes.index(encoding["incoming_groups"][1]["lanes"][0])] += 1
    key = ql.encode_state(state, encoding)
    assert key[:5] == (1, 1, 0, 0, 2)
    for i, bad_prefix in enumerate([(2, 1, 0, 0, 1), (1, 1, 0, 0, 3), (1, 1, 0, 0, 0),
                                    (0, 0, 0, 0, 1), (1, 1, 0, 0, 5)]):
        path = tmp_path / f"bad_{i}.json"
        ql.save_q_table(path, {bad_prefix + key[5:]: [0.0] * 8}, metadata)
        with pytest.raises(ValueError, match="busiest edge"):
            ql.load_q_table(path, env)
