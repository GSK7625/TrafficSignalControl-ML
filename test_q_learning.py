"""Small, hand-calculated checks for tabular Q-Learning."""

import copy
import csv
import json
import random
import xml.etree.ElementTree as ET

import pytest

import q_learning as ql
from sumo_env import ROOT, SumoEnv


def test_configurable_bins_boundaries_and_new_model_requires_matching_encoding(tmp_path):
    env = SumoEnv('cologne1')
    encoding = ql.build_encoding(env, (0, 4, 14))
    for queue, expected in [(0, 0), (1, 1), (4, 1), (5, 2), (14, 2), (15, 3), (113, 3)]:
        state = dict(incoming=[queue] + [0]*7, outgoing=[0]*8, phase=7)
        assert ql.encode_state(state, encoding) == (expected, 0, 0, 0, 0, 0, 0, 0, 7)
    metadata = ql.make_metadata(env, episodes=15, alpha=.1, gamma=.95,
                               epsilon_start=1, epsilon_min=.05, exploration_seed=7,
                               queue_bins=(0, 4, 14))
    table = {(3, 2, 1, 0, 0, 0, 0, 0, 7): [-1.]*8}
    path = tmp_path / 'new.json'
    ql.save_q_table(path, table, metadata)
    assert ql.load_q_table(path, env, (0, 4, 14)) == (table, metadata)
    with pytest.raises(ValueError, match='encoding'):
        ql.load_q_table(path, env)
    old = ROOT / 'reports/stage7/training/cologne1/exploration_7/cologne1_q_table.json'
    assert len(ql.load_q_table(old, env)[0]) == 129
    with pytest.raises(ValueError, match='encoding'):
        ql.load_q_table(old, env, (0, 4, 14))


@pytest.mark.parametrize('bins', [(), (4,), (0, 4, 4), (0, -1), (0, 4.5), (False, 4)])
def test_invalid_queue_thresholds_rejected(bins):
    with pytest.raises(ValueError, match='queue_bins'):
        ql.build_encoding(SumoEnv('cologne1'), bins)


@pytest.mark.parametrize("name,groups", [("cologne1", 4), ("ingolstadt1", 3)])
def test_encoding_real_edges_unique_lanes_and_bins(name, groups):
    env = SumoEnv(name)
    encoding = ql.build_encoding(env)
    for side in ("incoming", "outgoing"):
        lanes = getattr(env, side + "_lanes")
        grouped = encoding[side + "_groups"]
        assert len(grouped) == groups
        assert [g["edge"] for g in grouped] == sorted(g["edge"] for g in grouped)
        flattened = [lane for g in grouped for lane in g["lanes"]]
        assert sorted(flattened) == sorted(lanes)
        assert len(flattened) == len(set(flattened))
        assert all(lane.rsplit("_", 1)[0] == g["edge"]
                   for g in grouped for lane in g["lanes"])
    for count, level in [(0, 0), (1, 1), (4, 1), (5, 2), (100, 2)]:
        state = {"incoming": [0] * len(env.incoming_lanes),
                 "outgoing": [0] * len(env.outgoing_lanes), "phase": env.actions[-1]}
        state["incoming"][0] = count
        encoded = ql.encode_state(state, encoding)
        assert encoded == (level,) + (0,) * (2 * groups - 1) + (env.actions[-1],)


def test_encoding_sums_before_binning_and_respects_input_lane_order():
    env = SumoEnv("cologne1")
    env.incoming_lanes = tuple(reversed(env.incoming_lanes))
    encoding = ql.build_encoding(env)
    state = {"incoming": (0, 0, 0, 0, 0, 0, 2, 3),
             "outgoing": (0, 1, 0, 4, 0, 5, 0, 0), "phase": 7}
    assert ql.encode_state(state, encoding) == (2, 0, 0, 0, 1, 1, 2, 0, 7)
    env.incoming_lanes += (env.incoming_lanes[0],)
    with pytest.raises(ValueError, match="duplicate"):
        ql.build_encoding(env)


@pytest.mark.parametrize("change", ["length", "negative", "phase", "fraction"])
def test_encoding_rejects_invalid_state(change):
    env = SumoEnv("ingolstadt1")
    state = {"incoming": [0] * 7, "outgoing": [0] * 6, "phase": 0}
    if change == "length":
        state["incoming"] = [0]
    elif change == "phase":
        state["phase"] = None
    else:
        state["incoming"][0] = -1 if change == "negative" else 0.5
    with pytest.raises(ValueError):
        ql.encode_state(state, ql.build_encoding(env))


def test_q_initialization_independent_rows_and_read_only_unknown():
    table = {}
    actions = (0, 2, 4)
    assert ql.get_q_values(table, (0,), actions, create=False) == [0, 0, 0]
    assert table == {}
    first = ql.get_q_values(table, (0,), actions)
    second = ql.get_q_values(table, (1,), actions)
    first[0] = -2
    assert second == [0, 0, 0]
    assert table[(0,)] == [-2, 0, 0]


def test_epsilon_greedy_action_ids_negative_values_and_ties():
    actions = (4, 2, 0)
    state = (0,)
    table = {state: [-5, -1, -1]}
    assert ql.choose_action(table, state, actions) == 2
    rng = random.Random(7)
    assert {ql.choose_action(table, state, actions, 1, rng) for _ in range(100)} == set(actions)
    assert ql.choose_action({}, state, actions) == 4
    assert table == {state: [-5, -1, -1]}
    a, b = random.Random(7), random.Random(7)
    assert [ql.choose_action(table, state, actions, 0.5, a) for _ in range(20)] == [
        ql.choose_action(table, state, actions, 0.5, b) for _ in range(20)]


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, float("nan")])
def test_invalid_epsilon(epsilon):
    with pytest.raises(ValueError):
        ql.choose_action({}, (0,), (0, 2, 4), epsilon, random.Random(7))


def test_q_update_hand_calculation_correct_column_and_negative_q():
    table = {(0,): [0, 2, -3], (1,): [-4, 6, 1]}
    # Q(s,2) = 2 + .1 * (-5 + .95*6 - 2) = 1.87.
    ql.update_q_value(table, (0,), 2, -5, (1,), False, (0, 2, 4), .1, .95)
    assert table[(0,)] == pytest.approx([0, 1.87, -3])
    assert table[(1,)] == [-4, 6, 1]
    ql.update_q_value(table, (0,), 4, -10, None, True, (0, 2, 4), 1, 1)
    assert table[(0,)][2] == -10
    assert None not in table


def test_terminal_does_not_read_or_create_next_state():
    table = {(0,): [1, 2, 3], (1,): [1000, 2000, 3000]}
    ql.update_q_value(table, (0,), 0, -5, (1,), True, (0, 2, 4), .5, 1)
    assert table[(0,)][0] == -2
    ql.update_q_value(table, (0,), 0, -5, None, True, (0, 2, 4), 1, 0)
    assert set(table) == {(0,), (1,)}


@pytest.mark.parametrize("alpha,gamma", [(0, .95), (-1, .95), (1.1, .95),
                                             (.1, -1), (.1, 1.1), (float("nan"), .95)])
def test_invalid_learning_parameters_do_not_mutate(alpha, gamma):
    table = {}
    with pytest.raises(ValueError):
        ql.update_q_value(table, (0,), 2, -1, None, True, (0, 2, 4), alpha, gamma)
    assert table == {}


def test_save_load_metadata_and_overwrite_protection(tmp_path):
    env = SumoEnv("ingolstadt1")
    metadata = ql.make_metadata(env, episodes=3, alpha=.1, gamma=.95,
                                epsilon_start=1, epsilon_min=.05, exploration_seed=7)
    state = (0, 1, 2, 0, 1, 2, 4)
    table = {state: [-1.5, 0, 2.25]}
    path = tmp_path / "model.json"
    ql.save_q_table(path, table, metadata)
    assert ql.load_q_table(path, env) == (table, metadata)
    with pytest.raises(FileExistsError):
        ql.save_q_table(path, {}, metadata)
    with pytest.raises(ValueError, match="map"):
        ql.load_q_table(path, SumoEnv("cologne1"))
    payload = json.loads(path.read_text())
    payload["metadata"]["actions"] = [0, 1, 2]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="actions"):
        ql.load_q_table(bad, env)


@pytest.mark.parametrize("change", ["encoding", "q_length", "q_nan", "state", "duplicate", "timing"])
def test_load_rejects_incompatible_or_malformed_model(tmp_path, change):
    env = SumoEnv("ingolstadt1")
    meta = ql.make_metadata(env, episodes=1, alpha=.1, gamma=.95,
                            epsilon_start=1, epsilon_min=.05, exploration_seed=7)
    path = tmp_path / "model.json"
    ql.save_q_table(path, {(0, 0, 0, 0, 0, 0, 0): [0, 0, 0]}, meta)
    data = json.loads(path.read_text())
    if change == "encoding":
        data["metadata"]["encoding"]["incoming_lanes"].reverse()
    elif change == "timing":
        data["metadata"]["environment"]["green_seconds"] = 20
    elif change == "q_length":
        data["rows"][0]["values"] = [0]
    elif change == "q_nan":
        data["rows"][0]["values"][0] = float("nan")
    elif change == "duplicate":
        data["rows"].append(data["rows"][0])
    else:
        data["rows"][0]["state"][-1] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        ql.load_q_table(path, env)


def test_linear_epsilon_schedule_and_one_episode():
    assert [ql.episode_epsilon(i, 3, 1, .05) for i in range(1, 4)] == pytest.approx([1, .525, .05])
    assert ql.episode_epsilon(1, 1, 1, .05) == 1


def test_real_train_three_episodes_save_load_and_read_only_evaluation(tmp_path, monkeypatch):
    # Short horizon exercises the pipeline; delivered reports use the original full hour.
    real_env = ql.SumoEnv
    environments = []

    def short_env(*args, **kwargs):
        env = real_env(*args, **kwargs)
        env.end = env.begin + 47
        environments.append(env)
        return env

    monkeypatch.setattr(ql, "SumoEnv", short_env)
    output = tmp_path / "train"
    table, metadata = ql.train("ingolstadt1", output, episodes=3)
    assert metadata["training"]["episodes_trained"] == 3
    assert table and any(value < 0 for row in table.values() for value in row)
    model = output / "ingolstadt1_q_table.json"
    assert ql.load_q_table(model, short_env("ingolstadt1"))[0] == table
    before = copy.deepcopy(table)
    model_bytes = model.read_bytes()
    real_load = ql.load_q_table
    evaluated_tables = []

    def capture_load(*args):
        loaded, loaded_metadata = real_load(*args)
        evaluated_tables.append((loaded, copy.deepcopy(loaded)))
        return loaded, loaded_metadata

    monkeypatch.setattr(ql, "load_q_table", capture_load)
    with (output / "training.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert [float(row["epsilon"]) for row in rows] == pytest.approx([1, .525, .05])
    result = ql.evaluate("ingolstadt1", model, tmp_path / "eval", seed=42)
    assert result["epsilon"] == 0 and result["simulation_seconds"] == 47
    assert result["training_sumo_seed"] == 23423 and result["sumo_seed"] == 42
    assert result["process_closed"]
    assert model.read_bytes() == model_bytes
    assert all(loaded == original for loaded, original in evaluated_tables)
    assert ql.load_q_table(model, short_env("ingolstadt1"))[0] == before
    assert all(env.conn is None for env in environments)
    with pytest.raises(FileExistsError):
        ql.train("ingolstadt1", output, episodes=1)
    with pytest.raises(FileExistsError):
        ql.evaluate("ingolstadt1", model, tmp_path / "eval", seed=42)


def test_episode_error_closes_real_sumo(tmp_path, monkeypatch):
    processes, envs = [], []
    real_env = ql.SumoEnv

    def capture(*args, **kwargs):
        env = real_env(*args, **kwargs)
        envs.append(env)
        return env

    def fail(*args, **kwargs):
        processes.append(envs[-1].conn._process)
        raise ValueError("decision failure")

    monkeypatch.setattr(ql, "SumoEnv", capture)
    monkeypatch.setattr(ql, "choose_action", fail)
    with pytest.raises(ValueError, match="decision failure"):
        ql.train("ingolstadt1", tmp_path / "failed", episodes=1)
    assert all(env.conn is None for env in envs)
    assert processes[0].poll() == 0


@pytest.mark.parametrize("seconds,stage", [(2, "yellow"), (7, "all_red")])
def test_terminal_transition_is_not_encoded_and_evaluation_does_not_insert(tmp_path, seconds, stage):
    env = SumoEnv("ingolstadt1", tmp_path / stage)
    env.end = env.begin + seconds
    meta = ql.make_metadata(env, episodes=1, alpha=.1, gamma=.95,
                            epsilon_start=1, epsilon_min=.05, exploration_seed=7)
    # Reset starts at phase 0; column 2 means real phase 4, forcing a transition.
    table = {(0, 0, 0, 0, 0, 0, 0): [-2, -1, 1]}
    before = copy.deepcopy(table)
    result = ql.run_episode(env, table, meta, episode=1, training=False)
    assert table == before
    assert result["simulation_seconds"] == seconds and result["process_closed"]
    with (env.output_dir / "actions.csv").open(newline="") as f:
        row, = list(csv.DictReader(f))
    assert row["next_encoded_state"] == "null" and row["done"] == "True"
    assert float(row["green"]) == 0
    with (env.output_dir / "signals.csv").open(newline="") as f:
        assert list(csv.DictReader(f))[-1]["stage"] == stage


def test_unseen_greedy_episode_keeps_empty_table(tmp_path):
    env = SumoEnv("ingolstadt1", tmp_path / "unseen")
    env.end = env.begin + 31
    meta = ql.make_metadata(env, episodes=1, alpha=.1, gamma=.95,
                            epsilon_start=1, epsilon_min=.05, exploration_seed=7)
    table = {}
    result = ql.run_episode(env, table, meta, episode=1, training=False)
    assert table == {} and result["q_states"] == 0
    assert result["unseen_decisions"] == result["control_calls"] == 2
    assert result["action_counts"] == {0: 2, 2: 0, 4: 0}


@pytest.mark.parametrize("kwargs", [{"episodes": 0}, {"episodes": 1, "alpha": 0},
                                    {"episodes": 1, "gamma": 2},
                                    {"episodes": 1, "epsilon_min": 2},
                                    {"episodes": 1, "exploration_seed": -1}])
def test_invalid_training_settings_create_no_output(tmp_path, kwargs):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        ql.train("ingolstadt1", output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("path", sorted((ROOT / "reports/stage5").glob("**/result.json")))
def test_recorded_full_episode_timing_reward_safety_and_greedy_policy(path):
    result = json.loads(path.read_text())
    meta = json.loads(path.with_name("metadata.json").read_text())
    with path.with_name("actions.csv").open(newline="") as f:
        decisions = list(csv.DictReader(f))
    with path.with_name("signals.csv").open(newline="") as f:
        signals = list(csv.DictReader(f))
    assert len(signals) == 3600 and len(decisions) == result["control_calls"]
    assert float(signals[0]["start"]) == meta["begin"]
    assert float(signals[-1]["end"]) == meta["end"]
    assert all(p["end"] == s["start"] for p, s in zip(signals, signals[1:]))
    assert result["process_closed"] and result["algorithm"] == "Q-Learning"
    by_decision = {i: [] for i in range(len(decisions))}
    for sample in signals:
        by_decision[int(sample["decision"])].append(sample)
    if result["mode"] == "evaluate":
        model = ROOT / f"reports/stage5/train/{meta['map']}/{meta['map']}_q_table.json"
        payload = json.loads(model.read_text())
        table = {tuple(row["state"]): row["values"] for row in payload["rows"]}
    for index, row in enumerate(decisions):
        action = int(row["action"])
        assert action in meta["actions"]
        samples = by_decision[index]
        assert len(samples) == float(row["elapsed"])
        assert float(samples[0]["start"]) == float(row["start"])
        assert float(samples[-1]["end"]) == float(row["end"])
        assert row["done"] == str(index == len(decisions) - 1)
        assert [s["stage"] for s in samples] == [
            stage for stage in ("yellow", "all_red", "green")
            for s in samples if s["stage"] == stage]
        groups = {stage: [s for s in samples if s["stage"] == stage]
                  for stage in ("yellow", "all_red", "green")}
        for stage, rows in groups.items():
            assert float(row[stage]) == len(rows)
        greens, yellows, reds = groups["green"], groups["yellow"], groups["all_red"]
        assert len(greens) == 30 if index < len(decisions) - 1 else len(greens) <= 30
        runtime = dict(meta["environment"]["runtime_green_states"])
        if int(row["previous_phase"]) == action:
            assert not yellows and not reds
        else:
            yellow = "".join("y" if c == "G" else "r" for c in runtime[int(row["previous_phase"])])
            assert all(s["signal"] == yellow for s in yellows)
            if greens:
                assert len(yellows) == meta["environment"]["yellow_seconds"]
                assert len(reds) >= 2 and json.loads(greens[0]["junction_before"]) == []
        assert all(set(s["signal"]) == {"r"} for s in reds)
        assert all(s["signal"] == runtime[action] for s in greens)
        queue_seconds = sum(int(s["queue"]) for s in samples)
        assert float(row["queue_seconds"]) == queue_seconds
        assert float(row["reward"]) == pytest.approx(-queue_seconds / len(samples))
        # Independent encoder reference from lane IDs in the recorded state.
        encoded = []
        for side in ("incoming", "outgoing"):
            queues = json.loads(row[side])
            totals = {}
            for lane, queue in zip(meta[side + "_lanes"], queues):
                edge = lane.rsplit("_", 1)[0]
                totals[edge] = totals.get(edge, 0) + queue
            encoded.extend(0 if totals[e] == 0 else 1 if totals[e] < 5 else 2
                           for e in sorted(totals))
        encoded.append(int(row["previous_phase"]))
        assert json.loads(row["encoded_state"]) == encoded
        if index:
            assert sum(json.loads(row["incoming"])) == int(signals[int(float(row["start"]) - meta["begin"]) - 1]["queue"])
        if result["mode"] == "evaluate":
            values = table.get(tuple(encoded), [0] * len(meta["actions"]))
            best = max(range(len(values)), key=lambda i: values[i])
            assert action == meta["actions"][best]
            assert json.loads(row["q_before"]) == json.loads(row["q_after"]) == values
            assert float(row["epsilon"]) == 0
    assert result["queue_seconds"] == sum(int(s["queue"]) for s in signals)
    assert result["total_reward"] == pytest.approx(sum(float(row["reward"]) for row in decisions))
    assert result["traci_collision_events"] == sum(len(json.loads(s["collisions"])) for s in signals)
    assert result["traci_teleport_starts"] == sum(len(json.loads(s["teleports"])) for s in signals)
    statistics = ET.parse(path.with_name("statistics.xml")).getroot()
    assert int(statistics.find("safety").get("collisions")) == result["traci_collision_events"]
    assert int(statistics.find("teleports").get("total")) == result["traci_teleport_starts"]
    assert result["action_counts"] == {str(a): sum(int(r["action"]) == a for r in decisions)
                                       for a in meta["actions"]}


@pytest.mark.parametrize("map_dir", sorted((ROOT / "reports/stage5/train").glob("*")))
def test_replay_every_training_update_and_rng_against_checkpoints(map_dir):
    table = {}
    rng = random.Random(7)
    for episode_dir in sorted(map_dir.glob("episode_*")):
        if not (episode_dir / "q_table.json").exists():
            continue
        model = json.loads((episode_dir / "q_table.json").read_text())
        meta = model["metadata"]
        actions = meta["actions"]
        with (episode_dir / "actions.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            state = tuple(json.loads(row["encoded_state"]))
            values = table.setdefault(state, [0.] * len(actions))
            assert json.loads(row["q_before"]) == pytest.approx(values)
            epsilon = float(row["epsilon"])
            if epsilon > 0 and rng.random() < epsilon:
                expected_action = rng.choice(actions)
            else:
                expected_action = actions[max(range(len(values)), key=lambda i: values[i])]
            assert int(row["action"]) == expected_action
            target = float(row["reward"])
            if row["done"] == "False":
                next_state = tuple(json.loads(row["next_encoded_state"]))
                next_values = table.setdefault(next_state, [0.] * len(actions))
                assert float(row["next_max"]) == max(next_values)
                target += meta["training"]["gamma"] * max(next_values)
            else:
                assert row["next_encoded_state"] == "null" and row["next_max"] == ""
            column = actions.index(expected_action)
            values[column] += meta["training"]["alpha"] * (target - values[column])
            assert json.loads(row["q_after"]) == pytest.approx(values)
        assert table == {tuple(r["state"]): r["values"] for r in model["rows"]}
