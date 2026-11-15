"""Stage 6.2: read saved evidence only; never import a SUMO runner or train."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
for _d in (ROOT / "src", ROOT / "tools"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

from evaluation_metrics import signal_metrics


MAPS = ('cologne1', 'ingolstadt1')
SEEDS = (42, 12345, 2026, 2027, 2028)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    return rows


def model_table(payload):
    table = {tuple(row['state']): row['values'] for row in payload['rows']}
    actions = payload['metadata']['actions']
    if not table or len(table) != len(payload['rows']):
        raise ValueError('Empty model or duplicate state')
    for values in table.values():
        if len(values) != len(actions) or not all(math.isfinite(q) for q in values):
            raise ValueError('Invalid Q row')
    return table


def greedy_evidence(state, table, actions, updates):
    values = table.get(state, [0.] * len(actions))
    maximum = max(values)
    winners = [a for a, q in zip(actions, values) if q == maximum]
    return dict(action=winners[0], winners=winners, unseen=state not in table,
                best_q=maximum, tied=len(winners) > 1,
                selected_updates=updates[state, winners[0]],
                untried_winners=[a for a in winners if updates[state, a] == 0],
                zero_beats_negative=maximum == 0 and min(values) < 0)


def observation_analysis(records, encoding):
    """Count two distinct losses: lane -> edge sums, then edge sums -> bins."""
    grouped = []
    raw_by_edge = defaultdict(set)
    edge_by_encoded = defaultdict(set)
    examples = defaultdict(list)
    for record in records:
        totals = []
        raw = []
        for side in ('incoming', 'outgoing'):
            lanes, queues = encoding[side + '_lanes'], record[side]
            if (len(lanes) != len(queues)
                    or any(type(q) is not int or q < 0 for q in queues)):
                raise ValueError('Invalid lane queues')
            by_lane = dict(zip(lanes, queues))
            raw.extend(queues)
            totals.extend(sum(by_lane[lane] for lane in group['lanes'])
                          for group in encoding[side + '_groups'])
        phase = record['phase']
        if phase not in encoding['phase_ids']:
            raise ValueError('Invalid observation phase')
        encoded = tuple(bisect_left(encoding['queue_bins'], q) for q in totals) + (phase,)
        if 'encoded' in record and encoded != tuple(record['encoded']):
            raise ValueError('Saved encoding differs from raw lane queues')
        edge = tuple(totals) + (phase,)
        raw = tuple(raw) + (phase,)
        raw_by_edge[edge].add(raw)
        edge_by_encoded[encoded].add(edge)
        grouped.append(totals)
        if edge not in [x['edge_queues_phase'] for x in examples[encoded]] and len(examples[encoded]) < 2:
            examples[encoded].append(dict(edge_queues_phase=edge, lane_queues_phase=raw))
    bins = []
    index = 0
    for side in ('incoming', 'outgoing'):
        for group in encoding[side + '_groups']:
            values = [row[index] for row in grouped]
            high = [v for v in values if bisect_left(encoding['queue_bins'], v) == 2]
            counts = [sum(bisect_left(encoding['queue_bins'], v) == b for v in values)
                      for b in range(len(encoding['queue_bins']) + 1)]
            bins.append(dict(side=side, edge=group['edge'], lanes=group['lanes'], samples=len(values),
                             bins=counts, proportions=[n / len(values) for n in counts],
                             min=min(values), max=max(values), mean=statistics.mean(values),
                             constant_raw=len(set(values)) == 1,
                             constant_bin=sum(n > 0 for n in counts) == 1,
                             bin2_min=min(high) if high else None,
                             bin2_max=max(high) if high else None,
                             above_5=sum(v > 5 for v in values),
                             distinct_bin2_queues=len(set(high))))
            index += 1
    aliases = dict(
        observations=len(records), raw_observations=len({r for rows in raw_by_edge.values() for r in rows}),
        edge_observations=len(raw_by_edge), encoded_states=len(edge_by_encoded),
        aggregation_collisions=sum(len(rows) > 1 for rows in raw_by_edge.values()),
        discretization_collisions=sum(len(rows) > 1 for rows in edge_by_encoded.values()),
        aggregation_examples=[dict(edge_queues_phase=edge, lane_queues_phase=sorted(rows)[:2])
                              for edge, rows in sorted(raw_by_edge.items()) if len(rows) > 1][:3],
        examples=[dict(state=state, distinct_edge_observations=len(rows), examples=examples[state])
                  for state, rows in sorted(edge_by_encoded.items()) if len(rows) > 1])
    return bins, aliases


def audit_signals(signals, metadata):
    """Reuse the existing offline metric; also verify each actual runtime green."""
    metrics = signal_metrics(signals, metadata['begin'], metadata['end'])
    runtime = dict(metadata['environment']['runtime_green_states'])
    count = len(signals[0]['signal'])
    if {link[0] for link in metadata['controlled_links']} != set(range(count)):
        raise ValueError('Incomplete controlled-link mapping')
    green = dict.fromkeys(metadata['actions'], 0)
    for row in signals:
        if row['stage'] == 'green':
            action = int(row['action'])
            if row['signal'] != runtime[action]:
                raise ValueError('Actual green differs from action mapping')
            green[action] += float(row['end']) - float(row['start'])
        elif 'G' in row['signal']:
            raise ValueError('Green outside green stage')
    return metrics, green


def replay_training(root, name, payload, mapping, *, training_dir=None):
    """Reconstruct visits, RNG branches and Bellman updates; compare every checkpoint."""
    settings = payload['metadata']['training']
    actions = payload['metadata']['actions']
    table, updates, visits = {}, Counter(), Counter()
    rng = random.Random(settings['exploration_seed'])
    episodes, observations = [], []
    training_dir = root / f'reports/stage5/train/{name}' if training_dir is None else Path(training_dir)
    for episode in range(1, settings['episodes_trained'] + 1):
        folder = training_dir / f'episode_{episode:03d}'
        rows = read_csv(folder / 'actions.csv')
        result = read_json(folder / 'result.json')
        checkpoint = read_json(folder / 'q_table.json')
        run_meta = read_json(folder / 'metadata.json')
        if (run_meta['encoding'] != payload['metadata']['encoding']
                or run_meta['environment'] != payload['metadata']['environment']):
            raise ValueError('Training encoding/timing mismatch')
        explored = 0
        greedy_ties = 0
        action_counts = Counter()
        rewards = []
        observed_states = set()
        for row in rows:
            state = tuple(json.loads(row['encoded_state']))
            observed_states.add(state)
            visits[state] += 1
            values = table.setdefault(state, [0.] * len(actions))
            assert values == json.loads(row['q_before'])
            epsilon = float(row['epsilon'])
            schedule = settings['epsilon_schedule']
            expected_epsilon = max(schedule['minimum'], schedule['start'] -
                                   (schedule['start'] - schedule['minimum']) *
                                   (episode - 1) / (schedule['episodes'] - 1))
            assert epsilon == expected_epsilon == run_meta['epsilon']
            exploring = epsilon > 0 and rng.random() < epsilon
            if exploring:
                action = rng.choice(actions)
                explored += 1
            else:
                evidence = greedy_evidence(state, table, actions, updates)
                action = evidence['action']
                greedy_ties += evidence['tied']
            assert action == int(row['action'])
            action_counts[action] += 1
            reward = float(row['reward'])
            assert reward == -float(row['queue_seconds']) / float(row['elapsed'])
            rewards.append(reward)
            target = reward
            done = row['done'] == 'True'
            assert done == (row is rows[-1])
            if not done:
                successor = tuple(json.loads(row['next_encoded_state']))
                successor_values = table.setdefault(successor, [0.] * len(actions))
                assert max(successor_values) == float(row['next_max'])
                target += settings['gamma'] * max(successor_values)
            else:
                assert row['next_encoded_state'] == 'null' and row['next_max'] == ''
            column = actions.index(action)
            values[column] += settings['alpha'] * (target - values[column])
            assert values == json.loads(row['q_after'])
            updates[state, action] += 1
            observations.append(dict(incoming=json.loads(row['incoming']), outgoing=json.loads(row['outgoing']),
                                     phase=int(row['previous_phase']), encoded=state))
        assert table == model_table(checkpoint)
        assert len(rows) == result['control_calls']
        assert {str(a): action_counts[a] for a in actions} == result['action_counts']
        assert math.isclose(sum(rewards), result['total_reward'], abs_tol=1e-9)
        metrics, green = audit_signals(read_csv(folder / 'signals.csv'),
                                       dict(mapping, begin=result['begin'], end=result['end']))
        assert metrics['queue_seconds'] == result['queue_seconds']
        vehicles = {k: int(v) for k, v in result['vehicles'].items()}
        # Stage 5 has no tripinfo: arrivals include teleported vehicles.
        arrived = vehicles['inserted'] - vehicles['running']
        episodes.append(dict(map=name, episode=episode, epsilon=run_meta['epsilon'], updates=len(rows),
                             random_branch=explored, random_fraction=explored / len(rows),
                             greedy_branch=len(rows)-explored, greedy_ties=greedy_ties,
                             total_reward=sum(rewards), queue_seconds=result['queue_seconds'],
                             average_queue_length=metrics['average_queue_length'],
                             q_states=len(table), decision_states=len(observed_states),
                             cumulative_updated_pairs=len(updates), action_counts=[action_counts[a] for a in actions],
                             green_seconds=[green[a] for a in actions],
                             movement_count=len(metrics['green_seconds_per_movement']),
                             unserved=metrics['unserved_movements'],
                             teleports=result['traci_teleport_starts'], inserted=vehicles['inserted'],
                             arrived_including_teleported=arrived, unfinished=vehicles['running'],
                             undeparted=vehicles['waiting'],
                             completion_valid=None, throughput_valid=None))
    assert table == model_table(payload)
    return updates, visits, episodes, observations


def evaluation_policy(rows, payload, updates, source):
    table = model_table(payload)
    actions = payload['metadata']['actions']
    details, observations = [], []
    for row in rows:
        state = tuple(json.loads(row['encoded_state']))
        evidence = greedy_evidence(state, table, actions, updates)
        assert evidence['action'] == int(row['action'])
        values = table.get(state, [0.] * len(actions))
        if 'q_before' in row:
            assert json.loads(row['q_before']) == json.loads(row['q_after']) == values
            assert float(row['epsilon']) == 0
            raw = dict(incoming=json.loads(row['incoming']), outgoing=json.loads(row['outgoing']),
                       phase=int(row['previous_phase']))
        else:
            assert json.loads(row['q_values']) == values
            assert json.loads(row['unseen']) == evidence['unseen']
            raw = json.loads(row['state'])
        details.append(dict(source=source, decision=int(row['decision']), start=float(row['start']),
                            state=state, values=values, **evidence))
        observations.append(dict(raw, encoded=state))
    return details, observations


def policy_summary(details):
    n = len(details)
    counts = {key: sum(bool(row[key]) for row in details) for key in ('unseen', 'tied', 'zero_beats_negative')}
    counts.update(decisions=n, unique_max=n-counts['tied'],
                  selected_untried=sum(row['selected_updates'] == 0 for row in details),
                  known_selected_untried=sum(not row['unseen'] and row['selected_updates'] == 0 for row in details),
                  known_ties=sum(not row['unseen'] and row['tied'] for row in details),
                  unseen_fraction=counts['unseen']/n, tie_fraction=counts['tied']/n,
                  unique_max_fraction=(n-counts['tied'])/n)
    return counts


def analyze_evaluation_run(folder, payload, updates, stored, stored_movements):
    """Audit one saved Stage 6 run and its controlled-link service, without SUMO."""
    result = read_json(folder / 'result.json')
    meta = read_json(folder / 'metadata.json')
    name, algorithm, seed = meta['map'], meta['algorithm'], meta['seed']
    actions = payload['metadata']['actions']
    assert (result['map'], result['algorithm'], result['seed']) == (name, algorithm, seed)
    assert meta['actions'] == actions
    assert meta['environment'] == payload['metadata']['environment']
    rows = read_csv(folder / 'actions.csv')
    metrics, green = audit_signals(read_csv(folder / 'signals.csv'), meta)
    for key, value in metrics.items():
        assert result[key] == value
    for key in ('queue_seconds', 'throughput', 'completion_rate', 'teleports', 'undeparted', 'unfinished'):
        assert math.isclose(float(stored[key]), result[key], abs_tol=1e-9)
    assert len(rows) == result['control_calls']
    action_counts = Counter(int(row['action']) for row in rows)
    for action in actions:
        assert sum(float(row['green']) for row in rows if int(row['action']) == action) == green[action]
    reward = -sum(float(row['queue_seconds']) / float(row['elapsed']) for row in rows)
    assert sum(float(row['queue_seconds']) for row in rows) == metrics['queue_seconds']
    details, observations, counts = [], [], {}
    if algorithm == 'QL':
        assert meta['epsilon'] == 0
        assert meta['model_sha256_before'] == result['model_sha256_before'] == result['model_sha256_after']
        details, observations = evaluation_policy(rows, payload, updates, f'stage6/QL/{name}/{seed}')
        counts = policy_summary(details)
        assert counts['unseen'] == result['unseen_decisions']
    run = dict(map=name, algorithm=algorithm, seed=seed, action_order=actions,
        action_counts=[action_counts[a] for a in actions], green_seconds=[green[a] for a in actions],
        decisions=len(rows), reconstructed_reward=reward,
        elapsed_min=min(float(r['elapsed']) for r in rows), elapsed_max=max(float(r['elapsed']) for r in rows),
        **{k: result[k] for k in ('queue_seconds', 'average_queue_length', 'completion_rate', 'throughput',
           'average_waiting_time', 'average_travel_time', 'average_time_loss', 'teleports',
           'requested', 'inserted', 'arrived', 'valid_completed', 'unfinished', 'undeparted',
           'unserved_movements', 'longest_movement_no_green')},
        served_movements=len(metrics['green_seconds_per_movement'])-len(metrics['unserved_movements']),
        policy=counts)
    assert len(stored_movements) == len(meta['controlled_links'])
    movements = []
    for index, incoming, outgoing, via in sorted(meta['controlled_links']):
        seconds = metrics['green_seconds_per_movement'][index]
        gap = metrics['longest_no_green_per_movement'][index]
        saved, = [r for r in stored_movements if int(r['movement']) == index]
        assert (saved['incoming'], saved['outgoing']) == (incoming, outgoing)
        assert (float(saved['green_seconds']), float(saved['longest_no_green'])) == (seconds, gap)
        movements.append(dict(map=name, algorithm=algorithm, seed=seed, movement=index,
            incoming_lane=incoming, incoming_edge=incoming.rsplit('_', 1)[0], outgoing_lane=outgoing, via=via,
            serving_actions=[a for a, s in meta['environment']['runtime_green_states'] if s[index] == 'G'],
            green_seconds=seconds, longest_no_green=gap))
    return run, movements, details, observations


def analyze(root):
    """Return all tables before writing anything; missing/inconsistent evidence fails."""
    stage5 = read_json(root / 'reports/stage5/analysis.json')
    summary5 = read_csv(root / 'reports/stage5/summary.csv')
    summary6 = read_csv(root / 'reports/stage6/runs.csv')
    service6 = read_csv(root / 'reports/stage6/movement_service.csv')
    expected_runs = {(name, alg, seed) for name in MAPS for alg in ('FT', 'MP', 'QL') for seed in SEEDS}
    assert len(summary6) == len(expected_runs) == 30
    assert {(r['map'], r['algorithm'], int(r['seed'])) for r in summary6} == expected_runs
    outputs = {key: [] for key in ('queue_bins', 'state_actions', 'states', 'policy_decisions', 'training', 'runs', 'movements')}
    analysis = dict(models={}, observations={}, policy={}, comparisons={})
    for name in MAPS:
        model = root / f'reports/stage5/train/{name}/{name}_q_table.json'
        payload = read_json(model)
        table = model_table(payload)
        actions = payload['metadata']['actions']
        encoding = payload['metadata']['encoding']
        model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
        assert model_sha == stage5['models'][name]['sha256']
        final_checkpoint = model.with_name('episode_003') / 'q_table.json'
        assert model.read_bytes() == final_checkpoint.read_bytes()
        mapping = read_json(root / f'reports/stage6/runs/{name}/QL/seed_42/metadata.json')
        assert mapping['environment'] == payload['metadata']['environment']
        updates, visits, episodes, train_obs = replay_training(root, name, payload, mapping)
        outputs['training'].extend(episodes)
        all_eval_visits = Counter()
        for source in ('stage5/train', 'stage5/evaluate', 'stage6/QL'):
            observations = train_obs if source == 'stage5/train' else []
            if source == 'stage5/evaluate':
                for seed in (42, 12345):
                    folder = root / f'reports/stage5/evaluate/{name}/seed_{seed}'
                    result = read_json(folder / 'result.json')
                    details, records = evaluation_policy(read_csv(folder / 'actions.csv'), payload, updates,
                                                          f'{source}/{name}/{seed}')
                    outputs['policy_decisions'].extend(dict(map=name, **r) for r in details)
                    observations.extend(records)
                    counts = policy_summary(details)
                    assert counts['unseen'] == result['unseen_decisions']
                    analysis['policy'][f'{source}/{name}/{seed}'] = counts
            if source == 'stage6/QL':
                for algorithm in ('FT', 'MP', 'QL'):
                    for seed in SEEDS:
                        folder = root / f'reports/stage6/runs/{name}/{algorithm}/seed_{seed}'
                        meta = read_json(folder / 'metadata.json')
                        assert (meta['map'], meta['algorithm'], meta['seed']) == (name, algorithm, seed)
                        assert meta['controlled_links'] == mapping['controlled_links']
                        stored = next(r for r in summary6 if (r['map'], r['algorithm'], int(r['seed'])) == (name, algorithm, seed))
                        stored_movements = [r for r in service6 if (r['map'], r['algorithm'], int(r['seed'])) == (name, algorithm, seed)]
                        run, movements, details, records = analyze_evaluation_run(folder, payload, updates, stored, stored_movements)
                        outputs['runs'].append(run)
                        outputs['movements'].extend(movements)
                        if algorithm == 'QL':
                            assert meta['model_sha256_before'] == model_sha
                            outputs['policy_decisions'].extend(dict(map=name, **r) for r in details)
                            observations.extend(records)
                            all_eval_visits.update(tuple(r['encoded']) for r in records)
                            analysis['policy'][f'{source}/{name}/{seed}'] = run['policy']
            bins, aliases = observation_analysis(observations, encoding)
            outputs['queue_bins'].extend(dict(map=name, source=source, **r) for r in bins)
            analysis['observations'][f'{name}/{source}'] = aliases
            if source.startswith('stage5/'):
                for row in bins:
                    old = stage5['queue_bins'][f"{name}/{source.split('/')[1]}"][f"{row['side']}/{row['edge']}"]
                    assert row['bins'] == old['bins'] and row['max'] == old['max']
        cells = [q for values in table.values() for q in values]
        for state, values in sorted(table.items()):
            for action, value in zip(actions, values):
                outputs['state_actions'].append(dict(map=name, state=state, action=action,
                                                      updates=updates[state, action], q_value=value))
        for state in sorted(set(table) | set(all_eval_visits)):
            values = table.get(state, [0.] * len(actions))
            evidence = greedy_evidence(state, table, actions, updates)
            learned = sum(updates[state, a] > 0 for a in actions)
            sorted_q = sorted(values, reverse=True)
            outputs['states'].append(dict(map=name, state=state, in_model=state in table,
                train_visits=visits[state], stage6_visits=all_eval_visits[state],
                updated_actions=learned, action_order=actions, updates=[updates[state, a] for a in actions],
                q_values=values, top_gap=sorted_q[0]-sorted_q[1], **evidence))
        winners = Counter(greedy_evidence(s, table, actions, updates)['action'] for s in table)
        analysis['models'][name] = dict(
            sha256=model_sha, actions=actions, encoding=encoding, table_states=len(table),
            decision_states=len(visits), successor_only_states=len(set(table)-set(visits)),
            theoretical_states=3**(len(encoding['incoming_groups'])+len(encoding['outgoing_groups']))*len(actions),
            incoming_groups=len(encoding['incoming_groups']), outgoing_groups=len(encoding['outgoing_groups']),
            updates=sum(updates.values()), updated_pairs=len(updates), q_cells=len(cells),
            observed_state_action_fraction=len(updates)/(len(visits)*len(actions)),
            single_update_pairs=sum(v == 1 for v in updates.values()),
            update_histogram=dict(sorted(Counter(updates.values()).items())),
            maximum_pair_updates=max(updates.values()),
            most_updated_pairs=[dict(state=s, action=a, updates=n) for (s, a), n in updates.items() if n == max(updates.values())],
            negative=sum(q < 0 for q in cells), zero=sum(q == 0 for q in cells), positive=sum(q > 0 for q in cells),
            updated_zero_pairs=sum(updates[s, a] > 0 and table[s][i] == 0 for s in table for i, a in enumerate(actions)),
            any_equal_value_states=sum(len(set(v)) < len(v) for v in table.values()),
            tied_max_states=sum(v.count(max(v)) > 1 for v in table.values()),
            partly_learned_states=sum(sum(updates[s, a] > 0 for a in actions) < len(actions) for s in table),
            greedy_actions_over_model={a: winners[a] for a in actions})
        ql_details = [r for r in outputs['policy_decisions']
                      if r['map'] == name and r['source'].startswith('stage6/')]
        eval_states = {r['state'] for r in ql_details}
        analysis['models'][name]['evaluation'] = dict(
            **policy_summary(ql_details), distinct_states=len(eval_states),
            distinct_unseen_states=len(eval_states-set(table)),
            known_states_not_evaluated=len(set(table)-eval_states),
            selection_reasons=[dict(action=action,
                unseen=sum(r['action'] == action and r['unseen'] for r in ql_details),
                known_tie=sum(r['action'] == action and not r['unseen'] and r['tied'] for r in ql_details),
                unique_max=sum(r['action'] == action and not r['tied'] for r in ql_details)) for action in actions])
        old = stage5['models'][name]
        assert len(updates) == old['unique_updated_state_actions'] and sum(updates.values()) == old['updates']
        for episode in episodes:
            old_row, = [r for r in summary5 if r['map'] == name and r['mode'] == 'train' and int(r['episode']) == episode['episode']]
            assert math.isclose(float(old_row['total_reward']), episode['total_reward'], abs_tol=1e-9)
        for algorithm in ('FT', 'MP', 'QL'):
            runs = [r for r in outputs['runs'] if r['map'] == name and r['algorithm'] == algorithm]
            analysis['comparisons'][f'{name}/{algorithm}'] = {
                key: statistics.mean(r[key] for r in runs) for key in (
                    'reconstructed_reward', 'average_queue_length', 'completion_rate', 'throughput', 'teleports',
                    'undeparted', 'unfinished', 'average_waiting_time', 'average_travel_time', 'average_time_loss')}
    source_paths = [root / f'reports/{stage}/{filename}' for stage, filename in (
        ('stage5', 'analysis.json'), ('stage5', 'summary.csv'),
        ('stage6', 'runs.csv'), ('stage6', 'movement_service.csv'))]
    for name in MAPS:
        source_paths.append(root / f'reports/stage5/train/{name}/{name}_q_table.json')
        for episode in (1, 2, 3):
            folder = root / f'reports/stage5/train/{name}/episode_{episode:03d}'
            source_paths.extend(folder / filename for filename in (
                'actions.csv', 'signals.csv', 'metadata.json', 'result.json', 'q_table.json'))
        for seed in (42, 12345):
            folder = root / f'reports/stage5/evaluate/{name}/seed_{seed}'
            source_paths.extend(folder / filename for filename in ('actions.csv', 'result.json'))
        for algorithm in ('FT', 'MP', 'QL'):
            for seed in SEEDS:
                folder = root / f'reports/stage6/runs/{name}/{algorithm}/seed_{seed}'
                source_paths.extend(folder / filename for filename in ('actions.csv', 'signals.csv', 'metadata.json', 'result.json'))
    analysis['input_sha256'] = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(source_paths)}
    return analysis, outputs


def write_outputs(output, analysis, tables):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'analysis.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    for name, rows in tables.items():
        with (output / f'{name}.csv').open('x', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/stage6_2')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    analysis, tables = analyze(ROOT)
    write_outputs(args.output, analysis, tables)
    print(json.dumps({name: {k: model[k] for k in ('table_states', 'updated_pairs', 'updates', 'sha256')}
                      for name, model in analysis['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
