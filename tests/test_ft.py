"""Kiem tra lich pha FT; cac luot SUMO da ghi duoc kiem tra o duoi sau khi sinh ra."""

import pytest
import csv
import json

import ft
from sumo_env import ROOT
from test_sumo_env import clock_env


@pytest.mark.parametrize("queues", [(0,), (100, 0, 40, 2)])
def test_fixed_time_cycles_without_using_queue_or_reward(queues):
    env = clock_env(queues)
    env.end = env.begin + 350
    results = list(ft.fixed_time(env))
    assert [info["action"] for info in results] == [0, 1, 2, 3, 4, 5, 6, 7, 0]
    assert [info["durations"]["green"] for info in results] == [30] * 9
    assert results[0]["durations"] == {"yellow": 0, "all_red": 0, "green": 30}
    assert all(info["durations"] == {"yellow": 8, "all_red": 2, "green": 30}
               for info in results[1:])
    assert env.conn.time == env.end


@pytest.mark.parametrize("path", sorted((ROOT / "reports/stage3/ft").glob("*/seed_*/result.json")))
def test_recorded_sumo_signals_match_ft_schedule(path):
    """Doc lai tin hieu va moc thoi gian TraCI thuc te, doc lap voi tong theo action."""
    result = json.loads(path.read_text(encoding="utf-8"))
    metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    with path.with_name("signals.csv").open(encoding="utf-8", newline="") as f:
        signals = list(csv.DictReader(f))
    with path.with_name("actions.csv").open(encoding="utf-8", newline="") as f:
        actions = list(csv.DictReader(f))
    assert len(signals) == 3600
    assert float(signals[0]["start"]) == result["begin"]
    assert float(signals[-1]["end"]) == result["end"]
    for previous, current in zip(signals, signals[1:]):
        assert previous["end"] == current["start"]
    for i, action in enumerate(actions):
        expected = metadata["action_order"][i % len(metadata["action_order"])]
        assert int(action["action"]) == expected
        samples = [s for s in signals if int(s["decision"]) == i]
        assert float(samples[0]["start"]) == float(action["start"])
        assert float(samples[-1]["end"]) == float(action["end"])
        for sample in samples:
            assert float(sample["end"]) - float(sample["start"]) == 1
            assert int(sample["action"]) == expected
        greens = [s for s in samples if s["stage"] == "green"]
        yellows = [s for s in samples if s["stage"] == "yellow"]
        reds = [s for s in samples if s["stage"] == "all_red"]
        assert len(samples) == len(greens) + len(yellows) + len(reds)
        assert [s["stage"] for s in samples] == (
            ["yellow"] * len(yellows) + ["all_red"] * len(reds) + ["green"] * len(greens))
        if i < len(actions) - 1:
            assert len(greens) == 30
        else:
            assert 0 <= len(greens) <= 30
        if i == 0:
            assert not yellows and not reds
        else:
            previous_green = metadata["runtime_green_states"][actions[i - 1]["action"]]
            yellow = "".join("y" if c == "G" else "r" for c in previous_green)
            assert all(s["signal"] == yellow for s in yellows)
            if greens:
                assert len(yellows) == metadata["yellow_seconds"]
                assert len(reds) >= 2
                assert json.loads(greens[0]["junction_before"]) == []
        assert all(set(s["signal"]) == {"r"} for s in reds)
        assert all(s["signal"] == metadata["runtime_green_states"][str(expected)] for s in greens)
        for stage, rows in (("green", greens), ("yellow", yellows), ("all_red", reds)):
            assert float(action[stage]) == sum(float(s["end"]) - float(s["start"]) for s in rows)
    assert len(actions) == result["control_calls"]
    for stage in ("green", "yellow", "all_red"):
        assert result["total_seconds"][stage] == sum(s["stage"] == stage for s in signals)
    assert sum(len(json.loads(s["collisions"])) for s in signals) == result["traci_collision_events"]
    assert sum(len(json.loads(s["teleports"])) for s in signals) == result["traci_teleport_starts"]


@pytest.mark.parametrize("seconds,green,last_stage", [
    (32, 0, "yellow"), (39, 0, "all_red"), (47, 7, "green"),
])
def test_fixed_time_allows_only_the_terminal_interval_to_be_short(seconds, green, last_stage):
    env = clock_env()
    env.end = env.begin + seconds
    results = list(ft.fixed_time(env))
    assert [info["action"] for info in results] == [0, 1]
    assert results[-1]["durations"]["green"] == green
    assert results[-1]["samples"][-1]["stage"] == last_stage
    assert env.conn.time == env.end
