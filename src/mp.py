"""Max-Pressure theo lane voi xap xi chia deu ro rang tren cac lane dung chung."""

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sumo_env import ROOT, SumoEnv


def movement_weights(env):
    """Chia dong gop cua mot lane dau vao cho cac lane dich duy nhat cua no."""
    movements = sorted({(incoming, outgoing) for _, incoming, outgoing, _ in env.links})
    counts = {lane: 0 for lane in env.incoming_lanes}
    for incoming, _ in movements:
        counts[incoming] += 1
    # shortcut: chia deu theo cau truc, thay bang ty le do thuc te neu co.
    return {(incoming, outgoing): Fraction(1, counts[incoming])
            for incoming, outgoing in movements}


def calculate_pressures(env, state):
    """P(a) = sum w_ij * (q_i - q_j), chi tinh cho cac cap duy nhat co G khi chay.

    q_i/q_j la so xe dung tren toan lane, KHONG phai hang doi do rieng theo tung huong re.
    w_ij=1/d_i la phan bo xap xi, KHONG phai dong bao hoa do duoc.
    d_i gom tat ca movement xuat phat tu i, ke ca movement dang do trong pha dang xet.
    """
    if (len(state["incoming"]) != len(env.incoming_lanes)
            or len(state["outgoing"]) != len(env.outgoing_lanes)):
        raise ValueError("State must match the environment's lane order and lengths")
    incoming_queue = dict(zip(env.incoming_lanes, state["incoming"]))
    outgoing_queue = dict(zip(env.outgoing_lanes, state["outgoing"]))
    weights = movement_weights(env)
    pressures = {}
    for action in env.actions:
        served = {(incoming, outgoing) for index, incoming, outgoing, _ in env.links
                  if env.green_states[action][index] == "G"}
        # Dung phan so chinh xac de giu nguyen truong hop hoa nhau, khong bi sai so dau phay dong.
        pressure = 0
        for incoming, outgoing in sorted(served):
            pressure += weights[incoming, outgoing] * (
                incoming_queue[incoming] - outgoing_queue[outgoing])
        pressures[action] = float(pressure)
    return pressures


def choose_action(env, state):
    """Giu nguyen pressure am; neu hoa nhau thi chon action dau tien trong env.actions."""
    pressures = calculate_pressures(env, state)
    action = max(env.actions, key=pressures.__getitem__)
    return action, pressures


def max_pressure(env):
    """Dung moi truong da reset san; chi SumoEnv thuc thi tin hieu den."""
    state = env.get_state()
    done = False
    while not done:
        action, pressures = choose_action(env, state)
        next_state, reward, done, info = env.step(action)
        yield {**info, "state": state, "pressures": pressures,
               "reward": reward, "done": done}
        state = next_state


def run_mp(map_name, output, seed=23423):
    """Chay mot gio, luu lai dau vao quyet dinh, tin hieu tung giay va log SUMO."""
    env = SumoEnv(map_name, output, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    steps = 0
    totals = {"green": 0, "yellow": 0, "all_red": 0}
    collisions = 0
    teleports = 0
    queue_seconds = 0
    action_counts = {action: 0 for action in env.actions}
    streak = 0
    longest_streak = 0
    last_action = None
    try:
        env.reset()
        metadata = {
            "algorithm": "MP", "pressure_model": "equal-share lane approximation",
            "formula": "sum_{(i,j) with G in a} (q_i-q_j)/d_i",
            "weight_meaning": "structural allocation, not measured turn ratio or saturation flow",
            "tie_break": "first in action_order", "map": map_name, "seed": seed,
            "sumo_version": env.conn.getVersion(), "begin": env.begin, "end": env.end,
            "action_order": env.actions, "original_green_states": env.original_green_states,
            "runtime_green_states": env.green_states, "phase_foes": env.phase_foes,
            "green_seconds": env.green_seconds, "yellow_seconds": env.yellow_seconds,
            "minimum_all_red_seconds": env.minimum_red_seconds,
            "incoming_lanes": env.incoming_lanes, "outgoing_lanes": env.outgoing_lanes,
            "movements": [
                {"incoming": incoming, "outgoing": outgoing, "weight": float(weight),
                 "link_indices": [i for i, x, y, _ in env.links
                                  if (x, y) == (incoming, outgoing)],
                 "actions": [a for a in env.actions if any(
                     env.green_states[a][i] == "G" for i, x, y, _ in env.links
                     if (x, y) == (incoming, outgoing))]}
                for (incoming, outgoing), weight in movement_weights(env).items()],
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (output / "signals.csv").open("w", newline="", encoding="utf-8") as sf, \
                (output / "actions.csv").open("w", newline="", encoding="utf-8") as af:
            signals = csv.DictWriter(sf, fieldnames=(
                "decision", "action", "start", "end", "stage", "signal", "queue",
                "junction_before", "collisions", "teleports"))
            actions = csv.DictWriter(af, fieldnames=(
                "decision", "start", "end", "previous_phase", "action", "elapsed",
                "incoming", "outgoing", "pressures", "green", "yellow", "all_red",
                "queue_seconds", "reward", "done"))
            signals.writeheader()
            actions.writeheader()
            for info in max_pressure(env):
                durations = info["durations"]
                assert info["action"] in env.actions
                assert info["end"] <= env.end
                if not info["done"]:
                    assert durations["green"] == env.green_seconds == 30
                if info["action"] == info["previous_phase"]:
                    assert durations["yellow"] == durations["all_red"] == 0
                for stage in totals:
                    totals[stage] += durations[stage]
                queue_seconds += info["queue_seconds"]
                action_counts[info["action"]] += 1
                streak = streak + 1 if last_action == info["action"] else 1
                longest_streak = max(longest_streak, streak)
                last_action = info["action"]
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
                        "start", "end", "previous_phase", "action", "elapsed",
                        "queue_seconds", "reward", "done")},
                    **durations,
                    "incoming": json.dumps(info["state"]["incoming"]),
                    "outgoing": json.dumps(info["state"]["outgoing"]),
                    "pressures": json.dumps(info["pressures"]),
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
        "algorithm": "MP", "map": map_name, "seed": seed,
        "begin": env.begin, "end": env.end, "simulation_seconds": env.end - env.begin,
        "control_calls": steps, "total_seconds": totals, "queue_seconds": queue_seconds,
        "action_counts": action_counts, "longest_action_streak": longest_streak,
        "last_action": {"action": info["action"], "durations": durations},
        "process_closed": process.poll() == 0,
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
    output = args.output or ROOT / "reports/stage4/mp" / args.map / f"seed_{args.seed}"
    run_mp(args.map, output, args.seed)


if __name__ == "__main__":
    main()
