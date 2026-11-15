"""Tabular Q-Learning: discrete edge queues, original green IDs, shared SumoEnv."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from sumo_env import ROOT, SumoEnv


def build_encoding(env):
    """Use XML edge membership, not movement links (which repeat lanes)."""
    config = ET.parse(env.config).getroot()
    net = ET.parse(env.config.parent / config.find("input/net-file").get("value"))
    lane_edges = {lane.get("id"): edge.get("id")
                  for edge in net.findall("edge") for lane in edge.findall("lane")}
    encoding = {"queue_bins": [0, 4], "phase_ids": list(env.actions)}
    for side in ("incoming", "outgoing"):
        lanes = getattr(env, side + "_lanes")
        if len(lanes) != len(set(lanes)):
            raise ValueError("Encoding cannot contain duplicate lanes")
        groups = {}
        for lane in lanes:
            groups.setdefault(lane_edges[lane], []).append(lane)
        encoding[side + "_lanes"] = list(lanes)
        encoding[side + "_groups"] = [
            {"edge": edge, "lanes": groups[edge]} for edge in sorted(groups)]
    return encoding


def encode_state(state, encoding):
    """Sum each edge's unique boundary lanes, bin 0 / 1..4 / >=5, append phase ID."""
    encoded = []
    for side in ("incoming", "outgoing"):
        lanes = encoding[side + "_lanes"]
        queues = state[side]
        if len(queues) != len(lanes):
            raise ValueError("State must match the saved lane order and lengths")
        if any(type(q) is not int or q < 0 for q in queues):
            raise ValueError("Queues must be nonnegative integer halting counts")
        lane_queue = dict(zip(lanes, queues))
        for group in encoding[side + "_groups"]:
            queue = sum(lane_queue[lane] for lane in group["lanes"])
            encoded.append(0 if queue == 0 else 1 if queue <= 4 else 2)
    if type(state["phase"]) is not int or state["phase"] not in encoding["phase_ids"]:
        raise ValueError("Only nonterminal states with a valid green phase can be encoded")
    return tuple(encoded) + (state["phase"],)


def get_q_values(q_table, state, actions, create=True):
    """Evaluation can read an unseen state as zeros without inserting a row."""
    if state not in q_table:
        values = [0.0] * len(actions)
        if create:
            q_table[state] = values
        return values
    return q_table[state]


def choose_action(q_table, state, actions, epsilon=0.0, rng=None):
    """Epsilon-greedy; ties always use the first column in actions."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if epsilon > 0:
        if rng is None:
            raise ValueError("Exploration requires a separate random.Random instance")
        if rng.random() < epsilon:
            return rng.choice(actions)
    values = get_q_values(q_table, state, actions, create=False)
    column = max(range(len(actions)), key=values.__getitem__)
    return actions[column]


def update_q_value(q_table, state, action, reward, next_state, done, actions, alpha, gamma):
    """One online Q(s,a) update; gamma discounts decisions, not seconds."""
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("alpha must be in (0, 1], gamma in [0, 1]")
    if type(action) is not int or action not in actions or not math.isfinite(reward):
        raise ValueError("Update needs a valid action ID and finite reward")
    if not done and next_state is None:
        raise ValueError("A nonterminal update requires next_state")
    column = actions.index(action)
    values = get_q_values(q_table, state, actions)
    target = reward
    if not done:
        target += gamma * max(get_q_values(q_table, next_state, actions))
    values[column] += alpha * (target - values[column])


def episode_epsilon(episode, episodes, epsilon_start, epsilon_min):
    """Linear schedule: start at episode 1, reach min at the last episode."""
    if not 0 <= epsilon_min <= epsilon_start <= 1:
        raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1")
    if type(episodes) is not int or episodes < 1 or not 1 <= episode <= episodes:
        raise ValueError("Require 1 <= episode <= episodes")
    if episodes == 1:
        return epsilon_start
    return max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min)
               * (episode - 1) / (episodes - 1))


def environment_config(env):
    return {"begin": env.begin, "end": env.end, "green_seconds": env.green_seconds,
            "yellow_seconds": env.yellow_seconds, "minimum_all_red_seconds": env.minimum_red_seconds,
            "runtime_green_states": [[a, env.green_states[a]] for a in env.actions]}


def source_hashes(env):
    """Record original inputs including demand; never modify them."""
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(env.config.parent.iterdir()) if path.is_file()}


def make_metadata(env, *, episodes, alpha, gamma, epsilon_start, epsilon_min, exploration_seed):
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("alpha must be in (0, 1], gamma in [0, 1]")
    episode_epsilon(1, episodes, epsilon_start, epsilon_min)
    if type(exploration_seed) is not int or not 0 <= exploration_seed < 2**31:
        raise ValueError("exploration_seed must be an integer in [0, 2**31)")
    return {
        "schema_version": 1, "algorithm": "Q-Learning", "map": env.map_name,
        "actions": list(env.actions), "encoding": build_encoding(env),
        "environment": environment_config(env), "source_hashes": source_hashes(env),
        "sumo_version": None,
        "training": {"alpha": alpha, "gamma": gamma, "discount_unit": "decision",
                     "tie_break": "first in actions", "episodes_trained": 0,
                     "sumo_seed": env.seed, "exploration_seed": exploration_seed,
                     "epsilon_schedule": {"name": "linear per episode", "start": epsilon_start,
                                          "minimum": epsilon_min, "episodes": episodes}},
    }


def save_q_table(path, q_table, metadata):
    """JSON rows retain tuple keys and action-column order; exclusive create."""
    payload = {"metadata": metadata,
               "rows": [{"state": list(state), "values": values}
                        for state, values in sorted(q_table.items())]}
    content = json.dumps(payload, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(content + "\n")


def load_q_table(path, env):
    """Reject a different map, encoding, timing, input data or malformed row."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    expected = {"schema_version": 1, "algorithm": "Q-Learning", "map": env.map_name,
                "actions": list(env.actions), "encoding": build_encoding(env),
                "environment": environment_config(env), "source_hashes": source_hashes(env)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Model {key} does not match the environment")
    encoding = metadata["encoding"]
    dimensions = len(encoding["incoming_groups"]) + len(encoding["outgoing_groups"])
    q_table = {}
    for row in payload["rows"]:
        state, values = row["state"], row["values"]
        if (len(state) != dimensions + 1 or any(type(x) is not int for x in state)
                or any(x not in (0, 1, 2) for x in state[:-1]) or state[-1] not in env.actions):
            raise ValueError("Invalid encoded state in model")
        if (len(values) != len(env.actions)
                or any(type(q) not in (int, float) or not math.isfinite(q) for q in values)):
            raise ValueError("Invalid Q-values in model")
        state = tuple(state)
        if state in q_table:
            raise ValueError("Duplicate state in model")
        q_table[state] = values
    return q_table, metadata


def run_episode(env, q_table, metadata, *, episode, training, epsilon=0.0, rng=None):
    """Own one SUMO process; log common FT/MP measurements and Q update evidence."""
    output = env.output_dir
    output.mkdir(parents=True, exist_ok=False)
    encoding = metadata["encoding"]
    settings = metadata["training"]
    total_reward, queue_seconds, collisions, teleports, unseen = 0, 0, 0, 0, 0
    totals = {"green": 0, "yellow": 0, "all_red": 0}
    action_counts = {action: 0 for action in env.actions}
    steps = 0
    try:
        state = env.reset()
        metadata["sumo_version"] = list(env.conn.getVersion())
        run_metadata = {**metadata, "mode": "train" if training else "evaluate",
                        "episode": episode, "sumo_seed": env.seed, "epsilon": epsilon,
                        "exploration_seed": settings["exploration_seed"] if training else None,
                        "action_order": list(env.actions), "begin": env.begin, "end": env.end,
                        "incoming_lanes": env.incoming_lanes, "outgoing_lanes": env.outgoing_lanes}
        (output / "metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        with (output / "signals.csv").open("x", newline="", encoding="utf-8") as sf, \
                (output / "actions.csv").open("x", newline="", encoding="utf-8") as af:
            signals = csv.DictWriter(sf, fieldnames=(
                "decision", "action", "start", "end", "stage", "signal", "queue",
                "junction_before", "collisions", "teleports"))
            actions = csv.DictWriter(af, fieldnames=(
                "decision", "start", "end", "previous_phase", "action", "elapsed",
                "incoming", "outgoing", "encoded_state", "next_encoded_state", "epsilon",
                "q_before", "next_max", "q_after", "green", "yellow", "all_red",
                "queue_seconds", "reward", "done"))
            signals.writeheader()
            actions.writeheader()
            done = False
            while not done:
                encoded = encode_state(state, encoding)
                unseen += encoded not in q_table
                action = choose_action(q_table, encoded, env.actions, epsilon, rng)
                before = list(get_q_values(q_table, encoded, env.actions, create=False))
                next_state, reward, done, info = env.step(action)
                # Terminal may have phase=None during yellow/all-red; do not encode it.
                next_encoded = None if done else encode_state(next_state, encoding)
                next_max = None if done else max(get_q_values(q_table, next_encoded, env.actions, create=False))
                if training:
                    update_q_value(q_table, encoded, action, reward, next_encoded, done,
                                   env.actions, settings["alpha"], settings["gamma"])
                after = get_q_values(q_table, encoded, env.actions, create=False)
                assert info["action"] == action and info["end"] <= env.end
                if not done:
                    assert info["durations"]["green"] == env.green_seconds == 30
                for stage in totals:
                    totals[stage] += info["durations"][stage]
                total_reward += reward
                queue_seconds += info["queue_seconds"]
                action_counts[action] += 1
                for sample in info["samples"]:
                    collisions += len(sample["collisions"])
                    teleports += len(sample["teleports"])
                    signals.writerow({"decision": steps, "action": action,
                                      **{k: json.dumps(v) if isinstance(v, (tuple, list)) else v
                                         for k, v in sample.items()}})
                actions.writerow({
                    "decision": steps, **{k: info[k] for k in (
                        "start", "end", "previous_phase", "action", "elapsed", "queue_seconds")},
                    **info["durations"], "incoming": json.dumps(state["incoming"]),
                    "outgoing": json.dumps(state["outgoing"]), "encoded_state": json.dumps(encoded),
                    "next_encoded_state": json.dumps(next_encoded), "epsilon": epsilon,
                    "q_before": json.dumps(before), "next_max": next_max,
                    "q_after": json.dumps(after), "reward": reward, "done": done})
                steps += 1
                state = next_state
            assert env.conn.simulation.getTime() == env.end
            assert sum(totals.values()) == env.end - env.begin
            process = env.conn._process
    finally:
        env.close()
    assert process.poll() == 0
    statistics = ET.parse(output / "statistics.xml").getroot()
    safety = statistics.find("safety").attrib
    teleport_statistics = statistics.find("teleports").attrib
    assert int(safety["collisions"]) == collisions
    assert len(ET.parse(output / "collisions.xml").getroot().findall("collision")) == collisions
    assert int(teleport_statistics["total"]) == teleports
    warnings = (output / "warnings.log").read_text(encoding="utf-8")
    result = {
        "algorithm": "Q-Learning", "mode": "train" if training else "evaluate",
        "map": env.map_name, "episode": episode, "sumo_seed": env.seed,
        "training_sumo_seed": settings["sumo_seed"],
        "exploration_seed": settings["exploration_seed"] if training else None,
        "training_exploration_seed": settings["exploration_seed"], "epsilon": epsilon,
        "total_reward": total_reward, "queue_seconds": queue_seconds, "q_states": len(q_table),
        "unseen_decisions": unseen, "control_calls": steps, "action_counts": action_counts,
        "begin": env.begin, "end": env.end, "simulation_seconds": env.end - env.begin,
        "total_seconds": totals, "process_closed": process.poll() == 0,
        "safety_statistics": safety, "teleport_statistics": teleport_statistics,
        "traci_collision_events": collisions, "traci_teleport_starts": teleports,
        "warning_lines": warnings.count("Warning:"),
        "missing_yellow_warnings": warnings.count("Missing yellow phase"),
        "emergency_braking_warnings": warnings.count("performs emergency braking"),
        "emergency_stop_warnings": warnings.count("performs emergency stop"),
        "vehicles": statistics.find("vehicles").attrib,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_summary_row(writer, result):
    writer.writerow({key: json.dumps(result[key]) if isinstance(result[key], dict) else result[key]
                     for key in writer.fieldnames})


SUMMARY_FIELDS = ("map", "mode", "episode", "sumo_seed", "exploration_seed", "training_sumo_seed",
                  "training_exploration_seed", "epsilon", "total_reward", "queue_seconds", "q_states",
                  "unseen_decisions", "control_calls", "action_counts", "traci_collision_events",
                  "traci_teleport_starts", "simulation_seconds")


def train(map_name, output, episodes, *, seed=23423, exploration_seed=7,
          alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_min=0.05):
    """Start a fresh table; keep one exploration RNG across independent SUMO episodes."""
    output = Path(output)
    env = SumoEnv(map_name, seed=seed)
    metadata = make_metadata(env, episodes=episodes, alpha=alpha, gamma=gamma,
                             epsilon_start=epsilon_start, epsilon_min=epsilon_min,
                             exploration_seed=exploration_seed)
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(exploration_seed)
    q_table = {}
    with (output / "training.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for episode in range(1, episodes + 1):
            epsilon = episode_epsilon(episode, episodes, epsilon_start, epsilon_min)
            env = SumoEnv(map_name, output / f"episode_{episode:03d}", seed)
            result = run_episode(env, q_table, metadata, episode=episode, training=True,
                                 epsilon=epsilon, rng=rng)
            metadata["training"]["episodes_trained"] = episode
            save_q_table(env.output_dir / "q_table.json", q_table, metadata)
            write_summary_row(writer, result)
            file.flush()
            print(json.dumps(result), flush=True)
    save_q_table(output / f"{map_name}_q_table.json", q_table, metadata)
    return q_table, metadata


def evaluate(map_name, model, output, *, seed=42):
    """Load a compatible table, use epsilon=0, never update or insert a Q row."""
    env = SumoEnv(map_name, output, seed)
    q_table, metadata = load_q_table(model, env)
    result = run_episode(env, q_table, metadata, episode=1, training=False)
    with (env.output_dir / "evaluation.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        write_summary_row(writer, result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    training = commands.add_parser("train", help="Train a fresh Q-table; no resume or overwrite")
    evaluation = commands.add_parser("evaluate", help="Load a Q-table and run greedy evaluation")
    for command in (training, evaluation):
        command.add_argument("map", choices=("cologne1", "ingolstadt1"))
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=23423 if command is training else 42)
    training.add_argument("--episodes", type=int, required=True)
    training.add_argument("--exploration-seed", type=int, default=7)
    training.add_argument("--alpha", type=float, default=.1)
    training.add_argument("--gamma", type=float, default=.95)
    training.add_argument("--epsilon-start", type=float, default=1)
    training.add_argument("--epsilon-min", type=float, default=.05)
    evaluation.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.map, args.output, args.episodes, seed=args.seed,
              exploration_seed=args.exploration_seed, alpha=args.alpha, gamma=args.gamma,
              epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min)
    else:
        evaluate(args.map, args.model, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
