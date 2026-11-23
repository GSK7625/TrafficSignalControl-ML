"""Tabular Q-Learning: roi rac hoa hang doi theo edge, ID pha xanh goc, dung chung SumoEnv."""

import argparse
from bisect import bisect_left
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sumo_env import ROOT, SumoEnv
from evaluation_metrics import SIGNAL_FIELDS, write_signal_sample


EXPLORATION_STRATEGIES = ("uniform", "untried-first")
BUSIEST_STATE = "overdue-busiest-v1"


def build_encoding(env, queue_bins=(0, 4), *, overdue_state=False, busiest_state=False):
    """Dung quan he edge trong XML, khong dung link movement (vi bi lap lane)."""
    if (not queue_bins or queue_bins[0] != 0
            or any(type(x) is not int or x < 0 for x in queue_bins)
            or any(a >= b for a, b in zip(queue_bins, queue_bins[1:]))):
        raise ValueError("queue_bins must be increasing integer upper bounds starting at 0")
    config = ET.parse(env.config).getroot()
    net = ET.parse(env.config.parent / config.find("input/net-file").get("value"))
    lane_edges = {lane.get("id"): edge.get("id")
                  for edge in net.findall("edge") for lane in edge.findall("lane")}
    encoding = {"queue_bins": list(queue_bins), "phase_ids": list(env.actions)}
    if overdue_state or busiest_state:
        if env.map_name != "cologne1":
            raise ValueError("The overdue-edge state experiment is for cologne1")
        version = BUSIEST_STATE if busiest_state else "overdue-edge-v1"
        encoding.update(state_version=version, overdue_seconds=300)
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


class EdgeDemand:
    """Theo doi tuoi nhu cau cua tung edge trong mot episode, theo thu tu incoming_groups."""

    def __init__(self, env, encoding):
        self.encoding = encoding
        self.groups = encoding["incoming_groups"]
        self.since = [None] * len(self.groups)
        self.links = [[i for i, lane, _, _ in env.links if lane in group["lanes"]]
                      for group in self.groups]

    def observe(self, state, now):
        queues = dict(zip(self.encoding["incoming_lanes"], state["incoming"]))
        oldest = None
        for i, group in enumerate(self.groups):
            if not any(queues[lane] for lane in group["lanes"]):
                self.since[i] = None
                continue
            if self.since[i] is None:
                self.since[i] = now
            # Dau bang nhau thi giu edge dung truoc trong metadata.
            if oldest is None or self.since[i] < self.since[oldest]:
                oldest = i
        age = 0 if oldest is None else now - self.since[oldest]
        if "overdue_seconds" in self.encoding:
            state["overdue_edge"] = oldest + 1 if age >= self.encoding["overdue_seconds"] else 0
        return oldest, age

    def record_green(self, samples):
        for edge, indices in enumerate(self.links):
            if any(sample["signal"][i] == "G" for sample in samples for i in indices):
                self.since[edge] = None


def encode_state(state, encoding):
    """Gom queue theo edge, chia bin, noi ID pha va ma edge qua han neu model dung."""
    encoded = []
    for side in ("incoming", "outgoing"):
        lanes = encoding[side + "_lanes"]
        queues = state[side]
        if len(queues) != len(lanes):
            raise ValueError("State must match the saved lane order and lengths")
        if any(type(q) is not int or q < 0 for q in queues):
            raise ValueError("Queues must be nonnegative integer halting counts")
        lane_queue = dict(zip(lanes, queues))
        totals = [sum(lane_queue[lane] for lane in group["lanes"])
                  for group in encoding[side + "_groups"]]
        if side == "incoming" and encoding.get("state_version") == BUSIEST_STATE:
            encoded.extend(int(q > 0) for q in totals)
            peak = max(totals)
            encoded.append(totals.index(peak) + 1 if peak else 0)
        else:
            encoded.extend(bisect_left(encoding["queue_bins"], q) for q in totals)
    if type(state["phase"]) is not int or state["phase"] not in encoding["phase_ids"]:
        raise ValueError("Only nonterminal states with a valid green phase can be encoded")
    encoded.append(state["phase"])
    if "overdue_seconds" in encoding:
        edge = state.get("overdue_edge")
        if type(edge) is not int or not 0 <= edge <= len(encoding["incoming_groups"]):
            raise ValueError("Invalid or missing overdue edge code")
        encoded.append(edge)
    return tuple(encoded)


def get_q_values(q_table, state, actions, create=True):
    """Khi evaluate co the doc state chua gap nhu hang toan 0 ma khong chen them hang."""
    if state not in q_table:
        values = [0.0] * len(actions)
        if create:
            q_table[state] = values
        return values
    return q_table[state]


def choose_action(q_table, state, actions, epsilon=0.0, rng=None, *, update_counts=None):
    """Khi kham pha, uu tien cap (state, action) chua cap nhat neu co bo dem."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if epsilon > 0:
        if rng is None:
            raise ValueError("Exploration requires a separate random.Random instance")
        if rng.random() < epsilon:
            if update_counts is not None:
                untried = [a for a in actions if update_counts.get((state, a), 0) == 0]
                if untried:
                    return rng.choice(untried)
            return rng.choice(actions)
    values = get_q_values(q_table, state, actions, create=False)
    column = max(range(len(actions)), key=values.__getitem__)
    return actions[column]


def update_q_value(q_table, state, action, reward, next_state, done, actions, alpha, gamma):
    """Mot buoc cap nhat Q(s,a) truc tuyen; gamma giam gia theo quyet dinh, khong theo giay."""
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
    """Lich tuyen tinh: bat dau o episode 1, dat min o episode cuoi."""
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
    """Ghi lai cac file dau vao goc gom ca demand; khong bao gio sua chung."""
    return {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(env.config.parent.iterdir()) if path.is_file()}


def make_metadata(env, *, episodes, alpha, gamma, epsilon_start, epsilon_min, exploration_seed,
                  queue_bins=(0, 4), exploration_strategy="uniform", overdue_state=False, busiest_state=False):
    if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError("alpha must be in (0, 1], gamma in [0, 1]")
    episode_epsilon(1, episodes, epsilon_start, epsilon_min)
    if type(exploration_seed) is not int or not 0 <= exploration_seed < 2**31:
        raise ValueError("exploration_seed must be an integer in [0, 2**31)")
    if exploration_strategy not in EXPLORATION_STRATEGIES:
        raise ValueError(f"exploration_strategy must be one of {EXPLORATION_STRATEGIES}")
    return {
        "schema_version": 1, "algorithm": "Q-Learning", "map": env.map_name,
        "actions": list(env.actions), "encoding": build_encoding(
            env, queue_bins, overdue_state=overdue_state, busiest_state=busiest_state),
        "environment": environment_config(env), "source_hashes": source_hashes(env),
        "sumo_version": None,
        "training": {"alpha": alpha, "gamma": gamma, "discount_unit": "decision",
                     "tie_break": "first in actions", "episodes_trained": 0,
                     "sumo_seed": env.seed, "exploration_seed": exploration_seed,
                     "exploration_strategy": exploration_strategy,
                     "epsilon_schedule": {"name": "linear per episode", "start": epsilon_start,
                                          "minimum": epsilon_min, "episodes": episodes}},
    }


def save_q_table(path, q_table, metadata):
    """Cac dong JSON giu khoa tuple va thu tu cot action; chi tao file moi."""
    payload = {"metadata": metadata,
               "rows": [{"state": list(state), "values": values}
                        for state, values in sorted(q_table.items())]}
    content = json.dumps(payload, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(content + "\n")


def load_q_table(path, env, queue_bins=(0, 4)):
    """Tu choi neu khac map, encoding, thoi gian, du lieu dau vao hoac dong sai dinh dang."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    if not isinstance(metadata.get("encoding"), dict):
        raise ValueError("Missing or invalid model encoding")
    overdue_state = "state_version" in metadata["encoding"]
    busiest_state = metadata["encoding"].get("state_version") == BUSIEST_STATE
    expected = {"schema_version": 1, "algorithm": "Q-Learning", "map": env.map_name,
                "actions": list(env.actions), "encoding": build_encoding(
                    env, queue_bins, overdue_state=overdue_state, busiest_state=busiest_state),
                "environment": environment_config(env), "source_hashes": source_hashes(env)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Model {key} does not match the environment")
    encoding = metadata["encoding"]
    incoming_count = len(encoding["incoming_groups"])
    phase_index = incoming_count + int(busiest_state) + len(encoding["outgoing_groups"])
    q_table = {}
    for row in payload["rows"]:
        state, values = row["state"], row["values"]
        if (len(state) != phase_index + 1 + int(overdue_state) or any(type(x) is not int for x in state)
                or state[phase_index] not in env.actions
                or (overdue_state and not 0 <= state[-1] <= incoming_count)):
            raise ValueError("Invalid encoded state in model")
        bins = state[:phase_index]
        if busiest_state:
            flags, peak = state[:incoming_count], state[incoming_count]
            if (any(x not in (0, 1) for x in flags) or not 0 <= peak <= incoming_count
                    or bool(peak) != any(flags) or (peak and not flags[peak - 1])):
                raise ValueError("Invalid incoming queues or busiest edge in model")
            bins = state[incoming_count + 1:phase_index]
        if any(not 0 <= x <= len(queue_bins) for x in bins):
            raise ValueError("Invalid encoded state in model")
        if (len(values) != len(env.actions)
                or any(type(q) not in (int, float) or not math.isfinite(q) for q in values)):
            raise ValueError("Invalid Q-values in model")
        state = tuple(state)
        if state in q_table:
            raise ValueError("Duplicate state in model")
        q_table[state] = values
    return q_table, metadata


def run_episode(env, q_table, metadata, *, episode, training, epsilon=0.0, rng=None,
                update_counts=None):
    """Quan ly mot tien trinh SUMO; ghi nhan cac chi so chung nhu FT/MP va bang chung cap nhat Q."""
    encoding = metadata["encoding"]
    settings = metadata["training"]
    needs_counts = training and settings.get("exploration_strategy", "uniform") == "untried-first"
    if needs_counts != (update_counts is not None):
        raise ValueError("Supply update_counts only for untried-first training")
    output = env.output_dir
    output.mkdir(parents=True, exist_ok=False)
    total_reward, queue_seconds, collisions, teleports, unseen = 0, 0, 0, 0, 0
    totals = {"green": 0, "yellow": 0, "all_red": 0}
    action_counts = {action: 0 for action in env.actions}
    steps = 0
    try:
        state = env.reset()
        demand = EdgeDemand(env, encoding) if "overdue_seconds" in encoding else None
        if demand is not None:
            demand.observe(state, env.begin)
        metadata["sumo_version"] = list(env.conn.getVersion())
        run_metadata = {**metadata, "mode": "train" if training else "evaluate",
                        "episode": episode, "sumo_seed": env.seed, "epsilon": epsilon,
                        "exploration_seed": settings["exploration_seed"] if training else None,
                        "action_order": list(env.actions), "begin": env.begin, "end": env.end,
                        "incoming_lanes": env.incoming_lanes, "outgoing_lanes": env.outgoing_lanes}
        (output / "metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
        with (output / "signals.csv").open("x", newline="", encoding="utf-8") as sf, \
                (output / "actions.csv").open("x", newline="", encoding="utf-8") as af:
            signals = csv.DictWriter(sf, fieldnames=SIGNAL_FIELDS)
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
                action = choose_action(q_table, encoded, env.actions, epsilon, rng,
                                       update_counts=update_counts)
                before = list(get_q_values(q_table, encoded, env.actions, create=False))
                next_state, reward, done, info = env.step(action)
                if demand is not None and not done:
                    demand.record_green(info["samples"])
                    demand.observe(next_state, info["end"])
                # Trang thai ket thuc co the co phase=None trong luc vang/do tat ca; khong ma hoa no.
                next_encoded = None if done else encode_state(next_state, encoding)
                next_max = None if done else max(get_q_values(q_table, next_encoded, env.actions, create=False))
                if training:
                    update_q_value(q_table, encoded, action, reward, next_encoded, done,
                                   env.actions, settings["alpha"], settings["gamma"])
                    if update_counts is not None:
                        update_counts[encoded, action] += 1
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
                    write_signal_sample(signals, steps, action, sample)
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
          alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_min=0.05, queue_bins=(0, 4),
          exploration_strategy="uniform", overdue_state=False, busiest_state=False):
    """Bat dau bang Q moi; giu mot RNG kham pha xuyen suot cac episode SUMO doc lap."""
    output = Path(output)
    env = SumoEnv(map_name, seed=seed)
    metadata = make_metadata(env, episodes=episodes, alpha=alpha, gamma=gamma,
                             epsilon_start=epsilon_start, epsilon_min=epsilon_min,
                             exploration_seed=exploration_seed, queue_bins=queue_bins,
                             exploration_strategy=exploration_strategy, overdue_state=overdue_state,
                             busiest_state=busiest_state)
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(exploration_seed)
    update_counts = Counter() if exploration_strategy == "untried-first" else None
    q_table = {}
    with (output / "training.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for episode in range(1, episodes + 1):
            epsilon = episode_epsilon(episode, episodes, epsilon_start, epsilon_min)
            env = SumoEnv(map_name, output / f"episode_{episode:03d}", seed)
            result = run_episode(env, q_table, metadata, episode=episode, training=True,
                                 epsilon=epsilon, rng=rng, update_counts=update_counts)
            metadata["training"]["episodes_trained"] = episode
            save_q_table(env.output_dir / "q_table.json", q_table, metadata)
            write_summary_row(writer, result)
            file.flush()
            print(json.dumps(result), flush=True)
    save_q_table(output / f"{map_name}_q_table.json", q_table, metadata)
    return q_table, metadata


def evaluate(map_name, model, output, *, seed=42, queue_bins=(0, 4)):
    """Tai bang Q tuong thich, dung epsilon=0, khong cap nhat hay chen them dong Q."""
    env = SumoEnv(map_name, output, seed)
    q_table, metadata = load_q_table(model, env, queue_bins)
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
        command.add_argument("--queue-bins", type=int, nargs="+", default=[0, 4],
                             help="Inclusive queue upper bounds; Stage8: 0 4 14")
    training.add_argument("--episodes", type=int, required=True)
    training.add_argument("--overdue-state", action="store_true",
                          help="Cologne1: append the oldest queued edge without green for at least 300 s")
    training.add_argument("--busiest-state", action="store_true",
                          help="Cologne1: use incoming queue presence and busiest edge with the overdue edge")
    training.add_argument("--exploration-seed", type=int, default=7)
    training.add_argument("--exploration-strategy", choices=EXPLORATION_STRATEGIES, default="uniform",
                          help="untried-first explores actions not yet updated in the current state first")
    training.add_argument("--alpha", type=float, default=.1)
    training.add_argument("--gamma", type=float, default=.95)
    training.add_argument("--epsilon-start", type=float, default=1)
    training.add_argument("--epsilon-min", type=float, default=.05)
    evaluation.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.map, args.output, args.episodes, seed=args.seed,
              exploration_seed=args.exploration_seed, alpha=args.alpha, gamma=args.gamma,
              exploration_strategy=args.exploration_strategy, overdue_state=args.overdue_state,
              busiest_state=args.busiest_state,
              epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min, queue_bins=args.queue_bins)
    else:
        evaluate(args.map, args.model, args.output, seed=args.seed, queue_bins=args.queue_bins)


if __name__ == "__main__":
    main()
