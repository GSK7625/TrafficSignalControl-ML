import json

import pytest

import analyze_ql as audit
import evaluate
import q_learning as ql
from sumo_env import ROOT, SumoEnv


@pytest.mark.parametrize("relative", [False, True])
def test_overdue_edge_threshold_tie_queue_reset_and_observed_green(relative):
    env = SumoEnv("cologne1")
    encoding = ql.build_encoding(env, overdue_state=True, relative_state=relative)
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


@pytest.mark.parametrize("totals,expected", [
    ((0, 0, 0, 0), (0, 0, 0, 0)), ((15, 0, 4, 0), (2, 0, 1, 0)),
    ((4, 10, 0, 0), (1, 2, 0, 0)), ((5, 10, 4, 0), (2, 3, 1, 0)),
    ((5, 9, 4, 0), (2, 2, 1, 0)), ((9, 9, 4, 0), (2, 2, 1, 0)),
    ((10, 7, 7, 24), (2, 2, 2, 3)), ((1, 1, 1, 1), (1, 1, 1, 1))])
def test_relative_queues(totals, expected):
    env = SumoEnv("cologne1")
    enc = ql.build_encoding(env, relative_state=True)
    queues = {}
    for total, group in zip(totals, enc["incoming_groups"], strict=True):
        left, right = group["lanes"]
        queues[left], queues[right] = total // 2, total - total // 2
    assert len(queues) == len(enc["incoming_lanes"]) == 8
    state = dict(incoming=[queues[lane] for lane in enc["incoming_lanes"]],
                 outgoing=[3] * len(enc["outgoing_lanes"]), phase=7, overdue_edge=0)
    assert ql.encode_state(state, enc) == expected + (2, 2, 2, 2, 7, 0)
    assert enc["phase_ids"] == list(env.actions)


def test_relative_model_schema_and_columns(tmp_path):
    env = SumoEnv("cologne1")
    meta = ql.make_metadata(env, episodes=1, alpha=.1, gamma=.95, epsilon_start=1,
                            epsilon_min=.05, exploration_seed=7, relative_state=True)
    assert meta["encoding"]["state_version"] == "overdue-relative-v1"
    assert meta["encoding"]["lead_gap"] == 5
    state = (3, 2, 1, 0, 0, 0, 0, 0, 7, 0)
    model = tmp_path / 'relative.json'
    table = {state: list(range(8))}
    ql.save_q_table(model, table, meta)
    assert ql.load_q_table(model, env) == (table, meta)
    for i, prefix in enumerate([(3, 0, 1, 0), (3, 3, 2, 2), (4, 2, 0, 0)]):
        bad = tmp_path / f'bad_{i}.json'
        ql.save_q_table(bad, {prefix + state[4:]: list(range(8))}, meta)
        with pytest.raises(ValueError):
            ql.load_q_table(bad, env)
    meta["encoding"]["lead_gap"] = 4
    bad = tmp_path / 'wrong_gap.json'
    ql.save_q_table(bad, table, meta)
    with pytest.raises(ValueError, match="encoding"):
        ql.load_q_table(bad, env)
    with pytest.raises(ValueError, match="cologne1"):
        ql.build_encoding(SumoEnv("ingolstadt1"), relative_state=True)
    with pytest.raises(ValueError):
        ql.build_encoding(env, relative_state=True, busiest_state=True)


@pytest.mark.parametrize("relative", [False, True], ids=["overdue", "relative"])
def test_relative_state_comparison_smoke(tmp_path, monkeypatch, relative):
    from ql_progress import progress

    def short_env(*args, **kwargs):
        env = SumoEnv(*args, **kwargs)
        env.end = env.begin + 600
        return env

    monkeypatch.setattr(ql, "SumoEnv", short_env)
    monkeypatch.setattr(evaluate, "SumoEnv", short_env)
    folder = tmp_path / "train"
    table, meta = ql.train("cologne1", folder, 3, overdue_state=True, relative_state=relative, decay_episodes=3)
    model = folder / "episode_003/q_table.json"
    assert ql.load_q_table(model, short_env("cologne1")) == (table, meta)
    payload = audit.read_json(model)
    validation = []
    for seed in (42, 12345):
        out = tmp_path / f"validation_{seed}"
        result = evaluate.run_evaluation("cologne1", "QL", seed, out, model)
        assert result["model_sha256_before"] == result["model_sha256_after"]
        validation.append(out)
    mapping = audit.read_json(validation[0] / "metadata.json")
    counts, _, _, observations = audit.replay_training(ROOT, "cologne1", payload, mapping, training_dir=folder)
    bins, aliases = audit.observation_analysis(observations, meta["encoding"])
    assert aliases["encoded_states"] == len(table)
    if relative:
        assert any(b["bins"][3] for b in bins if b["side"] == "incoming")
    for out in validation:
        rows = audit.read_csv(out / "actions.csv")
        audit.audit_overdue_states(rows, audit.read_csv(out / "signals.csv"), mapping, meta["encoding"])
        _, val_obs = audit.evaluation_policy(rows, payload, counts, "validation")
        audit.observation_analysis(val_obs, meta["encoding"])
    assert progress(folder, validation)[-1]["updated_pairs"] == len(counts)
