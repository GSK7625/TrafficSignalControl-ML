"""Danh gia cong bang FT/MP/QL dong bang. Khong train hay xuat model."""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ft import fixed_time
from mp import max_pressure
import q_learning as ql
from sumo_env import ROOT, SumoEnv
from evaluation_metrics import SIGNAL_FIELDS, read_metrics, requested_trips, write_signal_sample


SEEDS = (42, 12345, 2026, 2027, 2028)
MAPS = ('cologne1', 'ingolstadt1')
ALGORITHMS = ('FT', 'MP', 'QL')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def greedy_steps(env, table, encoding, guard_seconds=None, *, initial_q=0.0, starvation_lambda=0.0):
    """QL tham lam dong bang, co tuy chon rang buoc phuc vu edge dau vao da quan sat."""
    if guard_seconds is not None and (type(guard_seconds) is not int or guard_seconds <= 0):
        raise ValueError('guard_seconds must be a positive integer or None')
    demand = ql.EdgeDemand(env, encoding)
    state = env.get_state()
    oldest, age = demand.observe(state, env.conn.simulation.getTime())
    done = False
    while not done:
        encoded = ql.encode_state(state, encoding)
        greedy = ql.choose_action(table, encoded, env.actions, epsilon=0, initial_q=initial_q)
        values = ql.get_q_values(table, encoded, env.actions, create=False, initial_q=initial_q)
        action, guard_edge, guard_age = greedy, None, 0
        if guard_seconds is not None and oldest is not None and age >= guard_seconds:
            guard_edge, guard_age = demand.groups[oldest]['edge'], age
            serving = [a for a in env.actions if any(env.green_states[a][i] == 'G' for i in demand.links[oldest])]
            action = max(serving, key=lambda a: values[env.actions.index(a)])
        next_state, base_reward, done, info = env.step(action)
        demand.record_green(info['samples'])
        oldest, age = demand.observe(next_state, info['end'])
        penalty, reward = ql.reward_components(-base_reward, age, starvation_lambda)
        yield {**info, 'state': state, 'encoded_state': encoded,
               'unseen': encoded not in table,
               'q_values': values, 'greedy_action': greedy,
               'guard_edge': guard_edge, 'guard_age': guard_age, 'next_incoming': next_state['incoming'],
               'mean_queue': -base_reward, 'oldest_age_end': age, 'starvation_penalty': penalty, 'reward': reward}
        state = next_state


def run_evaluation(map_name, algorithm, seed, output, model=None, *, horizon=None,
                   queue_bins=(0, 4), guard_seconds=None, expect_starvation_lambda=None):
    """Chay mot luot voi ghi nhan dung chung. horizon chi dung cho kiem thu tich hop ngan."""
    if algorithm not in ALGORITHMS:
        raise ValueError('algorithm must be FT, MP or QL')
    if (algorithm == 'QL') != (model is not None):
        raise ValueError('Supply model exactly when running QL')
    if guard_seconds is not None and (algorithm != 'QL' or type(guard_seconds) is not int or guard_seconds <= 0):
        raise ValueError('guard_seconds requires QL and a positive integer')
    output = Path(output).resolve()
    env = SumoEnv(map_name, output, seed, tripinfo=True)
    table = metadata = before_table = None
    model_before = None
    reward_spec = None
    if expect_starvation_lambda is not None and algorithm != 'QL':
        raise ValueError('A reward expectation requires QL')
    if model is not None:
        model = Path(model).resolve()
        model_before = sha256(model)
        table, metadata = ql.load_q_table(model, env, queue_bins)
        reward_spec = ql.model_reward(metadata)
        if expect_starvation_lambda is not None and reward_spec != ql.reward_config(expect_starvation_lambda):
            raise ValueError('Model reward differs from expected validation arm')
        before_table = copy.deepcopy(table)
    if horizon is not None:
        if type(horizon) is not int or not 0 < horizon <= env.end - env.begin:
            raise ValueError('Invalid test horizon')
        env.end = env.begin + horizon
    requests = requested_trips(env)
    sources_before = ql.source_hashes(env)
    output.mkdir(parents=True, exist_ok=False)
    code_hashes = {p.name: sha256(p) for p in sorted((ROOT / 'src').glob('*.py'))}
    totals = dict(green=0, yellow=0, all_red=0)
    unseen = 0
    max_observed_age = 0
    try:
        env.reset()
        version = env.conn.getVersion()
        run_metadata = dict(
            map=map_name, algorithm=algorithm, seed=seed, begin=env.begin, end=env.end,
            step_length=env.conn.simulation.getDeltaT(), sumo_version=version,
            environment=ql.environment_config(env), actions=env.actions,
            tls_id=env.tls_id, incoming_lanes=env.incoming_lanes,
            outgoing_lanes=env.outgoing_lanes, controlled_links=env.links,
            source_hashes=sources_before, code_hashes=code_hashes,
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            git_status=subprocess.check_output(['git', 'status', '--short', '--untracked-files=no'], cwd=ROOT, text=True),
            collision_check_junctions=True, collision_action='teleport',
            time_to_teleport='SUMO default (300 seconds in 1.28.0)',
            epsilon=0 if algorithm == 'QL' else None,
            model=str(model) if model else None, model_sha256_before=model_before,
            model_training=metadata['training'] if metadata else None,
            encoding=metadata['encoding'] if metadata else None,
            guard_seconds=guard_seconds,
            policy_variant='QL with observed incoming-edge guard' if guard_seconds else algorithm,
            requested=len(requests),
        )
        if reward_spec is not None:
            run_metadata['reward'] = reward_spec
        (output / 'metadata.json').write_text(json.dumps(run_metadata, indent=2), encoding='utf-8')
        if algorithm == 'FT':
            steps = fixed_time(env)
        elif algorithm == 'MP':
            steps = max_pressure(env)
        else:
            steps = greedy_steps(env, table, metadata['encoding'], guard_seconds,
                                 initial_q=metadata['training'].get('initial_q', 0.0),
                                 starvation_lambda=reward_spec['lambda'])
        with (output / 'signals.csv').open('x', newline='', encoding='utf-8') as sf, \
                (output / 'actions.csv').open('x', newline='', encoding='utf-8') as af:
            signals = csv.DictWriter(sf, fieldnames=SIGNAL_FIELDS)
            reward_fields = ('next_incoming', 'mean_queue', 'oldest_age_end', 'starvation_penalty', 'reward') if model else ()
            actions = csv.DictWriter(af, fieldnames=(
                'decision', 'start', 'end', 'previous_phase', 'action', 'elapsed',
                'green', 'yellow', 'all_red', 'queue_seconds',
                'state', 'pressures', 'encoded_state', 'q_values', 'unseen',
                'greedy_action', 'guard_edge', 'guard_age') + reward_fields)
            signals.writeheader()
            actions.writeheader()
            for decision, info in enumerate(steps):
                assert info['action'] in env.actions
                if info['end'] < env.end:
                    assert info['durations']['green'] == 30
                if algorithm == 'FT':
                    assert info['action'] == env.actions[decision % len(env.actions)]
                for stage, seconds in info['durations'].items():
                    totals[stage] += seconds
                unseen += info.get('unseen', False)
                if model:
                    max_observed_age = max(max_observed_age, info['oldest_age_end'])
                for sample in info['samples']:
                    write_signal_sample(signals, decision, info['action'], sample)
                actions.writerow(dict(
                    decision=decision,
                    **{k: info[k] for k in ('start', 'end', 'previous_phase', 'action', 'elapsed', 'queue_seconds')},
                    **info['durations'],
                    **{k: json.dumps(info.get(k)) for k in
                       ('state', 'pressures', 'encoded_state', 'q_values', 'unseen',
                        'greedy_action', 'guard_edge', 'guard_age') + reward_fields}))
        assert env.conn.simulation.getTime() == env.end
        assert sum(totals.values()) == env.end - env.begin
        process = env.conn._process
    finally:
        env.close()
    assert process.poll() == 0
    assert ql.source_hashes(env) == sources_before
    model_after = sha256(model) if model else None
    if model:
        assert model_before == model_after and table == before_table
    result = dict(map=map_name, algorithm=algorithm, seed=seed,
                  simulation_seconds=env.end - env.begin, begin=env.begin, end=env.end,
                  process_closed=True, total_seconds=totals, control_calls=decision+1,
                  unseen_decisions=unseen, model_sha256_before=model_before,
                  model_sha256_after=model_after,
                  **read_metrics(output, requests, env.begin, env.end))
    if reward_spec is not None:
        result.update(reward=reward_spec, max_observed_starvation_s=max_observed_age)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(f"{map_name} {algorithm} seed={seed}: valid={result['valid_completed']}/{len(requests)} "
          f"queue={result['average_queue_length']:.2f}, teleports={result['teleports']}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=ALGORITHMS)
    parser.add_argument('--map', choices=MAPS)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--queue-bins', type=int, nargs='+', default=[0, 4])
    parser.add_argument('--guard-seconds', type=int, help='Optional observed incoming-edge service constraint')
    parser.add_argument('--expect-starvation-lambda', type=float, help='Reject a Q model from another reward arm')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all', action='store_true', help='Run the predefined 30 runs into a NEW directory')
    args = parser.parse_args()
    if args.all:
        if (args.algorithm or args.map or args.model or args.guard_seconds is not None
                or args.expect_starvation_lambda is not None or args.queue_bins != [0, 4]):
            parser.error('--all requires original settings and cannot use algorithm/map/model/guard')
        args.output.mkdir(parents=True, exist_ok=False)
        for map_name in MAPS:
            for seed in SEEDS:
                for algorithm in ALGORITHMS:
                    model = ROOT / f'reports/stage5/train/{map_name}/{map_name}_q_table.json' if algorithm == 'QL' else None
                    run_evaluation(map_name, algorithm, seed,
                                   args.output / map_name / algorithm / f'seed_{seed}', model)
    else:
        if args.algorithm is None or args.map is None:
            parser.error('--algorithm and --map required without --all')
        run_evaluation(args.map, args.algorithm, args.seed, args.output, args.model,
                       queue_bins=args.queue_bins, guard_seconds=args.guard_seconds,
                       expect_starvation_lambda=args.expect_starvation_lambda)


if __name__ == '__main__':
    main()
