"""Fixed-Time: xoay vong qua cac pha xanh hop le, moi luot 30 giay xanh."""

import argparse
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sumo_env import ROOT, SumoEnv


def fixed_time(env):
    """Dung moi truong da reset san; hang doi va reward khong bao gio chon action."""
    action_index = 0
    done = False
    while not done:
        action = env.actions[action_index]
        _, _, done, info = env.step(action)
        yield info
        action_index = (action_index + 1) % len(env.actions)


def run_ft(map_name, output, seed=23423):
    """Chay tron mot episode va giu lai tin hieu tung giay cung log cua SUMO."""
    env = SumoEnv(map_name, output, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    steps = 0
    totals = {"green": 0, "yellow": 0, "all_red": 0}
    collisions = 0
    teleports = 0
    complete_greens = []
    try:
        env.reset()
        metadata = {
            "algorithm": "FT", "map": map_name, "seed": seed,
            "sumo_version": env.conn.getVersion(), "begin": env.begin, "end": env.end,
            "action_order": env.actions, "original_green_states": env.original_green_states,
            "runtime_green_states": env.green_states, "phase_foes": env.phase_foes,
            "green_seconds": env.green_seconds, "yellow_seconds": env.yellow_seconds,
            "minimum_all_red_seconds": env.minimum_red_seconds,
            "incoming_lanes": env.incoming_lanes, "outgoing_lanes": env.outgoing_lanes,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (output / "signals.csv").open("w", newline="", encoding="utf-8") as sf, \
                (output / "actions.csv").open("w", newline="", encoding="utf-8") as af:
            signals = csv.DictWriter(sf, fieldnames=(
                "decision", "action", "start", "end", "stage", "signal", "queue",
                "junction_before", "collisions", "teleports"))
            actions = csv.DictWriter(af, fieldnames=(
                "decision", "start", "end", "previous_phase", "action", "elapsed",
                "green", "yellow", "all_red", "queue_seconds", "reward", "done"))
            signals.writeheader()
            actions.writeheader()
            for info in fixed_time(env):
                durations = info["durations"]
                done = info["end"] == env.end
                assert info["action"] == env.actions[steps % len(env.actions)]
                assert info["end"] <= env.end
                if not done:
                    assert durations["green"] == 30
                    complete_greens.append(durations["green"])
                for stage in totals:
                    totals[stage] += durations[stage]
                for sample in info["samples"]:
                    collisions += len(sample["collisions"])
                    teleports += len(sample["teleports"])
                    signals.writerow({
                        "decision": steps, "action": info["action"],
                        **{k: json.dumps(v) if isinstance(v, (tuple, list)) else v
                           for k, v in sample.items()},
                    })
                actions.writerow({
                    "decision": steps,
                    **{k: info[k] for k in (
                        "start", "end", "previous_phase", "action", "elapsed", "queue_seconds")},
                    **durations, "reward": -info["mean_queue"], "done": done,
                })
                steps += 1
            assert env.conn.simulation.getTime() == env.end
            assert sum(totals.values()) == env.end - env.begin == 3600
            process = env.conn._process
    finally:
        env.close()
    assert process.poll() == 0
    statistics = ET.parse(output / "statistics.xml").getroot()
    safety = statistics.find("safety").attrib
    teleport_statistics = statistics.find("teleports").attrib
    warnings = (output / "warnings.log").read_text(encoding="utf-8")
    assert int(safety["collisions"]) == collisions
    assert len(ET.parse(output / "collisions.xml").getroot().findall("collision")) == collisions
    assert int(teleport_statistics["total"]) == teleports
    result = {
        "algorithm": "FT", "map": map_name, "seed": seed,
        "begin": env.begin, "end": env.end, "simulation_seconds": env.end - env.begin,
        "control_calls": steps, "total_seconds": totals,
        "nonterminal_green_seconds": sorted(set(complete_greens)),
        "last_action": {"action": info["action"], "durations": durations},
        "safety_statistics": safety, "teleport_statistics": teleport_statistics,
        "traci_collision_events": collisions, "traci_teleport_starts": teleports,
        "warning_lines": warnings.count("Warning:"),
        "missing_yellow_warnings": warnings.count("Missing yellow phase"),
        "emergency_braking_warnings": warnings.count("performs emergency braking"),
        "emergency_stop_warnings": warnings.count("performs emergency stop"),
        "vehicles": statistics.find("vehicles").attrib,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", choices=("cologne1", "ingolstadt1"))
    parser.add_argument("--seed", type=int, default=23423)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "reports/stage3/ft" / args.map / f"seed_{args.seed}"
    run_ft(args.map, output, args.seed)


if __name__ == "__main__":
    main()
