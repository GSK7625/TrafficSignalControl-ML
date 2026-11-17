"""Kiem tra pressure theo lane tinh tay va kiem tra trace SUMO thuc te."""

import csv
from fractions import Fraction
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

import mp
from sumo_env import ROOT, SumoEnv
from test_sumo_env import clock_env


def small_env():
    # Thu tu lane/action khong sap xep de bat loi vo tinh sort ID state/action.
    return SimpleNamespace(
        incoming_lanes=("in_b", "in_a"), outgoing_lanes=("out_y", "out_x"),
        links=((0, "in_b", "out_y", "v0"), (1, "in_b", "out_x", "v1"),
               (2, "in_b", "out_y", "v2"), (3, "in_a", "out_y", "v3")),
        actions=(4, 2, 0), green_states={4: "GrrG", 2: "rGrr", 0: "GGGG"},
    )


def test_hand_pressure_uses_both_queues_and_state_lane_order():
    env = small_env()
    state = {"incoming": (12, 5), "outgoing": (2, 8), "phase": 4}
    # a4=(12-2)/2+(5-2)=8; a2=(12-8)/2=2; a0=8+2=10 (tinh tay).
    assert mp.calculate_pressures(env, state) == {4: 8, 2: 2, 0: 10}
    assert mp.choose_action(env, state) == (0, {4: 8, 2: 2, 0: 10})


def test_shared_lane_weights_and_duplicate_link_do_not_multiply_queue():
    env = small_env()
    weights = mp.movement_weights(env)
    assert weights == {("in_b", "out_y"): 0.5,
                       ("in_b", "out_x"): 0.5, ("in_a", "out_y"): 1}
    state = {"incoming": (12, 0), "outgoing": (0, 0), "phase": 4}
    assert mp.calculate_pressures(env, state) == {4: 6, 2: 6, 0: 12}
    # Den G cua link trung lap van phuc vu dung cap do khi link con lai dang do.
    env.green_states[4] = "rrGG"
    assert mp.calculate_pressures(env, state)[4] == 6


@pytest.mark.parametrize("incoming,outgoing,expected", [
    ((10, 10), (0, 0), (0, {4: 15, 2: 5, 0: 20})),
    ((0, 0), (10, 20), (2, {4: -15, 2: -10, 0: -25})),
    ((0, 0), (0, 0), (4, {4: 0, 2: 0, 0: 0})),
    ((12, 0), (0, 12), (4, {4: 6, 2: 0, 0: 6})),
])
def test_maximum_negative_zero_and_tie(incoming, outgoing, expected):
    env = small_env()
    action, pressures = mp.choose_action(env, {"incoming": incoming, "outgoing": outgoing})
    assert (action, pressures) == expected
    assert action in env.actions


@pytest.mark.parametrize("side", ["incoming", "outgoing"])
def test_reject_wrong_state_length(side):
    state = {"incoming": (1, 2), "outgoing": (3, 4)}
    state[side] = (1,)
    with pytest.raises(ValueError, match="lane order"):
        mp.calculate_pressures(small_env(), state)


@pytest.mark.parametrize("name", ["cologne1", "ingolstadt1"])
def test_real_mapping_matches_audited_links_and_only_runtime_G(name):
    env = SumoEnv(name)
    with (ROOT / f"reports/stage1/{name}/controlled_links.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert {(i, x, y) for i, x, y, _ in env.links} == {
        (int(r["linkIndex"]), r["incoming"], r["outgoing"]) for r in rows}
    weights = mp.movement_weights(env)
    for lane in env.incoming_lanes:
        assert sum(w for (i, _), w in weights.items() if i == lane) == pytest.approx(1)
    audit = json.loads((ROOT / "reports/stage3/network_audit.json").read_text())
    assert env.green_states == {int(a): s for a, s in audit[name]["runtime_states"].items()}
    if name == "cologne1":
        state = {"incoming": tuple(12 if i == "-32038056#3_1" else 0
                                   for i in env.incoming_lanes), "outgoing": (0,) * 8}
        p = mp.calculate_pressures(env, state)
        assert (p[1], p[2], p[3]) == pytest.approx((12, 4, 8))
    else:
        state = {"incoming": tuple(12 if i == "201963537#1_3" else 0
                                   for i in env.incoming_lanes), "outgoing": (0,) * 6}
        assert mp.calculate_pressures(env, state) == {0: 0, 2: 12, 4: 0}


def test_fractional_weights_do_not_break_an_exact_pressure_tie():
    env = SumoEnv("cologne1")
    state = {"incoming": (5, 5, 1, 9, 0, 0, 0, 9), "outgoing": (0,) * 8}
    action, pressures = mp.choose_action(env, state)
    # a1=5/2+5/2+5/3+5/3+5/3=10; a4=1+9=10 (hai pha hoa nhau).
    assert pressures[1] == pressures[4] == 10
    assert action == 1


@pytest.mark.parametrize("reward", [-10000, 10000])
def test_loop_ignores_reward_and_repeated_green_has_no_fake_yellow(reward):
    env = clock_env()
    env.end = env.begin + 65
    real_step = env.step

    def step(action):
        state, _, done, info = real_step(action)
        return state, reward, done, info

    env.step = step
    results = list(mp.max_pressure(env))
    assert [r["action"] for r in results] == [0, 0, 0]
    assert [r["durations"] for r in results] == [
        {"green": 30, "yellow": 0, "all_red": 0},
        {"green": 30, "yellow": 0, "all_red": 0},
        {"green": 5, "yellow": 0, "all_red": 0}]
    assert [r["done"] for r in results] == [False, False, True]
    assert all(r["reward"] == reward for r in results)
    assert env.conn.time == env.end


@pytest.mark.parametrize("seconds,green,last_stage", [(2, 0, "yellow"),
                                                     (9, 0, "all_red"), (47, 7, "green")])
def test_loop_horizon_can_cut_any_stage(seconds, green, last_stage):
    env = clock_env((12,))
    env.end = env.begin + seconds
    results = list(mp.max_pressure(env))
    assert all(r["action"] == 1 for r in results)
    assert results[-1]["durations"]["green"] == green
    assert results[-1]["samples"][-1]["stage"] == last_stage
    assert results[-1]["done"] and env.conn.time == env.end


def test_runner_closes_real_sumo_on_controller_error_and_refuses_overwrite(tmp_path, monkeypatch):
    env = SumoEnv("ingolstadt1", tmp_path / "failed")
    monkeypatch.setattr(mp, "SumoEnv", lambda *args: env)
    processes = []

    def fail_at_decision(*args):
        processes.append(env.conn._process)
        raise ValueError("controller failure probe")

    monkeypatch.setattr(mp, "choose_action", fail_at_decision)
    with pytest.raises(ValueError, match="controller failure probe"):
        mp.run_mp("ingolstadt1", tmp_path / "failed")
    assert env.conn is None and processes[0].poll() == 0
    with pytest.raises(FileExistsError):
        mp.run_mp("ingolstadt1", tmp_path / "failed")


@pytest.mark.parametrize("path", sorted((ROOT / "reports/stage4/mp").glob("*/seed_*/result.json")))
def test_recorded_mp_decisions_pressures_timing_reward_and_safety(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    meta = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    with path.with_name("signals.csv").open(newline="", encoding="utf-8") as f:
        signals = list(csv.DictReader(f))
    with path.with_name("actions.csv").open(newline="", encoding="utf-8") as f:
        actions = list(csv.DictReader(f))
    assert result["algorithm"] == meta["algorithm"] == "MP"
    assert result["process_closed"]
    assert len(signals) == 3600 and len(actions) == result["control_calls"]
    assert float(signals[0]["start"]) == meta["begin"]
    assert float(signals[-1]["end"]) == meta["end"]
    assert all(p["end"] == s["start"] for p, s in zip(signals, signals[1:]))
    # Xay dung tham chieu doc lap tu CSV da kiem dinh, khong dung ham/trong so cua MP.
    with (ROOT / f"reports/stage1/{meta['map']}/controlled_links.csv").open(newline="") as f:
        links = list(csv.DictReader(f))
    destinations = {}
    for link in links:
        destinations.setdefault(link["incoming"], set()).add(link["outgoing"])
    for n, row in enumerate(actions):
        incoming = json.loads(row["incoming"])
        outgoing = json.loads(row["outgoing"])
        q_in = dict(zip(meta["incoming_lanes"], incoming))
        q_out = dict(zip(meta["outgoing_lanes"], outgoing))
        assert len(incoming) == len(meta["incoming_lanes"])
        assert len(outgoing) == len(meta["outgoing_lanes"])
        reference = {}
        for a in meta["action_order"]:
            pairs = {(r["incoming"], r["outgoing"]) for r in links
                     if meta["runtime_green_states"][str(a)][int(r["linkIndex"])] == "G"}
            reference[str(a)] = sum(Fraction(q_in[i] - q_out[j], len(destinations[i]))
                                    for i, j in pairs)
        assert json.loads(row["pressures"]) == pytest.approx(reference)
        best = max(meta["action_order"], key=lambda a: reference[str(a)])
        assert int(row["action"]) == best
        samples = [s for s in signals if int(s["decision"]) == n]
        assert float(samples[0]["start"]) == float(row["start"])
        assert float(samples[-1]["end"]) == float(row["end"])
        if n:
            assert float(row["start"]) == float(actions[n - 1]["end"])
            assert sum(incoming) == int(signals[int(float(row["start"]) - meta["begin"]) - 1]["queue"])
        assert all(float(s["end"]) - float(s["start"]) == 1 for s in samples)
        assert all(int(s["action"]) == best for s in samples)
        by_stage = {stage: [s for s in samples if s["stage"] == stage]
                    for stage in ("yellow", "all_red", "green")}
        assert [s["stage"] for s in samples] == [
            stage for stage, rows in by_stage.items() for _ in rows]
        for stage, rows in by_stage.items():
            assert float(row[stage]) == len(rows)
        greens, yellows, reds = by_stage["green"], by_stage["yellow"], by_stage["all_red"]
        assert row["done"] == str(n == len(actions) - 1)
        assert len(greens) == 30 if n < len(actions) - 1 else len(greens) <= 30
        if row["previous_phase"] == row["action"]:
            assert not yellows and not reds
        else:
            source = meta["runtime_green_states"][row["previous_phase"]]
            yellow = "".join("y" if c == "G" else "r" for c in source)
            assert all(s["signal"] == yellow for s in yellows)
            if greens:
                assert len(yellows) == meta["yellow_seconds"] and len(reds) >= 2
                assert json.loads(greens[0]["junction_before"]) == []
        assert all(set(s["signal"]) == {"r"} for s in reds)
        assert all(s["signal"] == meta["runtime_green_states"][str(best)] for s in greens)
        assert float(row["queue_seconds"]) == sum(int(s["queue"]) for s in samples)
        assert float(row["reward"]) == pytest.approx(-float(row["queue_seconds"]) / len(samples))
        assert float(row["elapsed"]) == len(samples)
    for stage in ("green", "yellow", "all_red"):
        assert result["total_seconds"][stage] == sum(s["stage"] == stage for s in signals)
    assert result["queue_seconds"] == sum(int(s["queue"]) for s in signals)
    assert sum(len(json.loads(s["collisions"])) for s in signals) == result["traci_collision_events"]
    assert sum(len(json.loads(s["teleports"])) for s in signals) == result["traci_teleport_starts"]
    stats = ET.parse(path.with_name("statistics.xml")).getroot()
    assert int(stats.find("safety").get("collisions")) == result["traci_collision_events"]
    assert int(stats.find("teleports").get("total")) == result["traci_teleport_starts"]
