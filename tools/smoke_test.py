"""Chuoi action chan doan cua Stage 2, KHONG phai thuat toan dieu khien giao thong."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sumo_env import ROOT, SumoEnv


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for name in ("cologne1", "ingolstadt1")
            for p in sorted((ROOT / "data" / name).iterdir()) if p.is_file()}


def all_pairs_sequence(actions):
    """Duong di Euler tham moi cap co thu tu dung mot lan, ke ca cap giu nguyen pha."""
    remaining = {action: list(actions) for action in actions}
    stack = [actions[0]]
    sequence = []
    while stack:
        if remaining[stack[-1]]:
            stack.append(remaining[stack[-1]].pop())
        else:
            sequence.append(stack.pop())
    return list(reversed(sequence))


def check_reset(name, output):
    env = SumoEnv(name, output)
    try:
        initial = env.reset()
        first_process = env.conn._process
        first_state, first_reward, _, _ = env.step(env.actions[0])
        assert env.reset() == initial
        assert first_process.poll() == 0
        state, reward, _, _ = env.step(env.actions[0])
        assert (state, reward) == (first_state, first_reward)
        last_process = env.conn._process
    finally:
        env.close()
    assert last_process.poll() == 0


def run_map(name, output):
    check_reset(name, output / "reset_probe")
    env = SumoEnv(name, output)
    sequence = all_pairs_sequence(env.actions)
    random_actions = random.Random(23423)
    pairs = set()
    collisions = []
    teleports = 0
    steps = 0
    extra_red_steps = 0
    yellow_seconds = set()
    red_seconds = set()
    green_seconds = set()
    try:
        initial = env.reset()
        metadata = {
            "map": name, "begin": env.begin, "end": env.end,
            "sumo_version": env.conn.getVersion(), "simulation_seed": 23423,
            "actions": env.green_states, "foe_pairs_with_yield": env.phase_foes,
            "incoming_lanes": env.incoming_lanes, "outgoing_lanes": env.outgoing_lanes,
            "internal_distance_to_exit": env.distance_to_exit, "initial_state": initial,
            "yellow_seconds_provisional": env.yellow_seconds,
            "minimum_all_red_seconds_provisional": env.minimum_red_seconds,
            "diagnostic_sequence": sequence,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (output / "signals.csv").open("w", newline="", encoding="utf-8") as sf, \
                (output / "actions.csv").open("w", newline="", encoding="utf-8") as af:
            signals = csv.DictWriter(sf, fieldnames=(
                "start", "end", "stage", "signal", "queue", "junction_before", "collisions", "teleports"))
            actions = csv.DictWriter(af, fieldnames=(
                "start", "end", "previous_phase", "action", "elapsed", "yellow", "all_red", "green",
                "queue_seconds", "reward", "done", "observed_phase", "incoming", "outgoing"))
            signals.writeheader()
            actions.writeheader()
            done = False
            while not done:
                action = sequence[steps] if steps < len(sequence) else random_actions.choice(env.actions)
                state, reward, done, info = env.step(action)
                pairs.add((info["previous_phase"], action))
                samples = info["samples"]
                durations = info["durations"]
                assert info["elapsed"] == len(samples) > 0
                assert info["end"] <= env.end
                assert reward == -sum(s["queue"] for s in samples) / len(samples)
                assert len(state["incoming"]) == len(env.incoming_lanes)
                assert len(state["outgoing"]) == len(env.outgoing_lanes)
                assert all(type(q) is int and q >= 0 for q in state["incoming"] + state["outgoing"])
                assert state == env.get_state()
                if not done:
                    assert durations["green"] == 30
                    green_seconds.add(durations["green"])
                    if info["previous_phase"] == action:
                        assert durations["yellow"] == durations["all_red"] == 0
                    else:
                        assert durations["yellow"] == env.yellow_seconds
                        assert durations["all_red"] >= 2
                        yellow_seconds.add(durations["yellow"])
                        red_seconds.add(durations["all_red"])
                        extra_red_steps += durations["all_red"] > 2
                        first_green = next(s for s in samples if s["stage"] == "green")
                        assert first_green["junction_before"] == ()
                for sample in samples:
                    assert sample["end"] - sample["start"] == 1
                    if sample["stage"] == "all_red":
                        assert set(sample["signal"]) == {"r"}
                    elif sample["stage"] == "green":
                        assert sample["signal"] == env.green_states[action]
                    else:
                        assert set(sample["signal"]) <= {"y", "r"}
                    for collision in sample["collisions"]:
                        collisions.append({"interval_start": sample["start"], "observed_at": sample["end"],
                                           "stage": sample["stage"], "action": action,
                                           "signal": sample["signal"], **collision})
                    teleports += len(sample["teleports"])
                    signals.writerow({key: json.dumps(value) if isinstance(value, (tuple, list)) else value
                                      for key, value in sample.items()})
                actions.writerow({
                    **{k: info[k] for k in ("start", "end", "previous_phase", "action", "elapsed", "queue_seconds")},
                    **durations, "reward": reward, "done": done, "observed_phase": state["phase"],
                    "incoming": json.dumps(state["incoming"]), "outgoing": json.dumps(state["outgoing"]),
                })
                steps += 1
            assert env.conn.simulation.getTime() == env.end
            assert pairs == {(a, b) for a in env.actions for b in env.actions}
            final_action = {"elapsed": info["elapsed"], "durations": durations, "phase": state["phase"]}
            process = env.conn._process
    finally:
        env.close()
    assert process.poll() == 0
    statistics = ET.parse(output / "statistics.xml").getroot()
    safety = statistics.find("safety").attrib
    teleport_statistics = statistics.find("teleports").attrib
    warnings = (output / "warnings.log").read_text(encoding="utf-8")
    result = {
        "map": name, "status": "prototype", "full_horizon_seconds": env.end - env.begin,
        "reset_and_process_close_checked": True, "steps": steps, "covered_pairs": len(pairs),
        "observed_green_seconds_nonterminal": sorted(green_seconds),
        "observed_yellow_seconds_nonterminal": sorted(yellow_seconds),
        "observed_all_red_seconds_nonterminal": sorted(red_seconds),
        "transitions_with_extended_red": extra_red_steps,
        "last_action": final_action, "safety_statistics": safety,
        "teleport_statistics": teleport_statistics,
        "traci_collision_events": len(collisions), "traci_teleport_starts": teleports,
        "warning_lines": warnings.count("Warning:"),
        "missing_yellow_warnings": warnings.count("Missing yellow phase"),
        "emergency_braking_warnings": warnings.count("performs emergency braking"),
        "emergency_stop_warnings": warnings.count("performs emergency stop"),
    }
    assert int(safety["collisions"]) == len(collisions)
    assert int(teleport_statistics["total"]) == teleports
    (output / "collision_context.json").write_text(json.dumps(collisions, indent=2), encoding="utf-8")
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/stage2/smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    results = []
    try:
        for name in ("cologne1", "ingolstadt1"):
            results.append(run_map(name, args.output / name))
    finally:
        after = source_hashes()
        (args.output / "source_hashes.json").write_text(
            json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8")
        assert before == after, "Original data changed"
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
