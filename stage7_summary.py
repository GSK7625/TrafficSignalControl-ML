"""Stage 7 offline evidence; no simulator, training, or policy changes."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

import analyze_ql as audit


ROOT = Path(__file__).resolve().parent
STAGE = ROOT / 'reports/stage7'
VALIDATION_SEEDS = (42, 12345)
TEST_SEEDS = (31001, 31002, 31003)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coverage(rows, states, updates, actions):
    previous = len(states)
    for row in rows:
        state, action = tuple(json.loads(row['encoded_state'])), int(row['action'])
        if action not in actions:
            raise ValueError('Invalid action in training log')
        states.add(state)
        updates[state, action] += 1
    possible = len(states) * len(actions)
    return dict(new_states=len(states)-previous, encountered_states=len(states),
                updated_pairs=len(updates), possible_pairs=possible,
                coverage=len(updates)/possible)


def checkpoint_changes(previous, current):
    common = previous.keys() & current.keys()
    changed = sum(max(range(len(previous[s])), key=previous[s].__getitem__) !=
                  max(range(len(current[s])), key=current[s].__getitem__) for s in common)
    delta = max((abs(a-b) for s in common for a, b in zip(previous[s], current[s])), default=0)
    return dict(common_states=len(common), greedy_changes=changed,
                greedy_change_rate=changed/len(common) if common else 0, max_q_change=delta)


def training_evidence(folder, name, label):
    """Replay all Bellman/RNG evidence using Stage 6.2; count pairs per episode."""
    payload = audit.read_json(folder / f'{name}_q_table.json')
    settings = payload['metadata']['training']
    budget = 3 if label == 'QL-3ep' else 5 if label == 'pilot' else 15
    exploration = 17 if label == 'QL-15ep-e17' else 7
    assert settings['episodes_trained'] == settings['epsilon_schedule']['episodes'] == budget
    assert (settings['alpha'], settings['gamma'], settings['sumo_seed'], settings['exploration_seed']) == (.1, .95, 23423, exploration)
    assert settings['epsilon_schedule']['start'] == 1 and settings['epsilon_schedule']['minimum'] == .05
    mapping = audit.read_json(ROOT / f'reports/stage6/runs/{name}/FT/seed_42/metadata.json')
    updates, _, episodes, observations = audit.replay_training(ROOT, name, payload, mapping, training_dir=folder)
    audit.observation_analysis(observations, payload['metadata']['encoding'])
    states, counts, previous_table = set(), Counter(), {}
    pairs = []
    actions = payload['metadata']['actions']
    for episode in episodes:
        directory = folder / f"episode_{episode['episode']:03d}"
        rows = audit.read_csv(directory / 'actions.csv')
        episode.update(label=label, checkpoint_sha256=sha256(directory / 'q_table.json'),
                       **coverage(rows, states, counts, actions))
        checkpoint = audit.read_json(directory / 'q_table.json')
        current_table = audit.model_table(checkpoint)
        episode.update(checkpoint_changes(previous_table, current_table))
        previous_table = current_table
        for state, values in current_table.items():
            for action, value in zip(actions, values):
                pairs.append(dict(map=name, label=label, episode=episode['episode'], state=state,
                                  action=action, updates=counts[state, action], q_value=value))
        assert episode['q_states'] == len(states)
        assert episode['updates'] == len(rows)
        result = audit.read_json(directory / 'result.json')
        assert result['simulation_seconds'] == 3600 and result['process_closed']
        assert result['sumo_seed'] == 23423 and result['exploration_seed'] == exploration
    assert counts == updates
    return payload, updates, episodes, pairs


def run_evidence(folder, payload=None, updates=None):
    """Use the common metrics already saved; verify actual greens and greedy decisions."""
    result = audit.read_json(folder / 'result.json')
    metadata = audit.read_json(folder / 'metadata.json')
    signals, green = audit.audit_signals(audit.read_csv(folder / 'signals.csv'), metadata)
    for key, value in signals.items():
        assert result[key] == value
    assert result['simulation_seconds'] == 3600 and result['process_closed']
    result['movement_total'] = len(result['green_seconds_per_movement'])
    result['movement_served'] = result['movement_total'] - len(result['unserved_movements'])
    result['movement_coverage'] = result['movement_served'] / result['movement_total']
    result['green_seconds_per_action'] = green
    rows = audit.read_csv(folder / 'actions.csv')
    result['action_counts'] = dict(Counter(int(row['action']) for row in rows))
    details = []
    if payload is not None:
        details, _ = audit.evaluation_policy(rows, payload, updates, str(folder.relative_to(ROOT)))
        result.update(decisions=len(details), ties=sum(row['tied'] for row in details),
                      untried_selected=sum(row['selected_updates'] == 0 for row in details),
                      unseen_states=len({row['state'] for row in details if row['unseen']}))
        assert result['unseen_decisions'] == sum(row['unseen'] for row in details)
        for field, count in [('tie_rate', 'ties'), ('untried_rate', 'untried_selected'),
                             ('unseen_rate', 'unseen_decisions')]:
            result[field] = result[count] / result['decisions']
        model = Path(metadata['model'])
        assert sha256(model) == result['model_sha256_before'] == result['model_sha256_after']
        assert audit.model_table(audit.read_json(model)) == audit.model_table(payload)
    return result, details


def choose_model(rows):
    """Predeclared final-model rule; this function refuses any test seed."""
    labels = ('QL-15ep-e7', 'QL-15ep-e17')
    if any(row['seed'] not in VALIDATION_SEEDS for row in rows):
        raise ValueError('Only validation seeds may select a model')
    candidates = {label: [row for row in rows if row['label'] == label] for label in labels}
    for group in candidates.values():
        if sorted(row['seed'] for row in group) != sorted(VALIDATION_SEEDS):
            raise ValueError('Missing or duplicate validation runs')

    def rank(label):
        group = candidates[label]
        return (-statistics.mean(row['completion_rate'] for row in group),
                -min(row['movement_coverage'] for row in group),
                statistics.mean(row['teleports'] for row in group),
                max(row['longest_movement_no_green'] for row in group), labels.index(label))
    return min(labels, key=rank)


def verify_run_matrix(runs, selection):
    expected = set()
    for name in audit.MAPS:
        for label in ('FT', 'MP', 'QL-3ep'):
            expected.update(('stage6', name, label, seed, 1) for seed in audit.SEEDS)
        for label in ('QL-3ep', 'QL-15ep-e7', 'QL-15ep-e17'):
            expected.update(('validation', name, label, seed, 1) for seed in VALIDATION_SEEDS)
        chosen = selection['models'][name]
        for label in ('FT', 'MP', 'QL-3ep', chosen['label']):
            expected.update(('test', name, label, seed, repeat) for seed in TEST_SEEDS for repeat in (1, 2))
    actual = [(r['partition'], r['map'], r['label'], r['seed'], r['repeat']) for r in runs]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError(f'Incomplete or duplicate run matrix: {len(actual)}/{len(expected)}')
    for name in audit.MAPS:
        validation = [r for r in runs if r['partition'] == 'validation' and r['map'] == name]
        chosen = selection['models'][name]
        assert choose_model(validation) == chosen['label']
        for row in runs:
            if row['partition'] == 'test' and row['map'] == name and row['label'] == chosen['label']:
                assert row['model_sha256_before'] == row['model_sha256_after'] == chosen['sha256']


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def summarize(runs):
    """Unweighted run means with explicit counts; never merge maps or train/eval."""
    fields = ('valid_completed', 'average_waiting_time', 'average_travel_time', 'average_time_loss',
              'average_queue_length', 'completion_rate', 'throughput', 'insertion_rate',
              'unfinished', 'undeparted', 'teleports', 'teleported', 'collisions', 'emergency_braking',
              'movement_served', 'longest_movement_no_green', 'unseen_rate', 'tie_rate', 'untried_rate')
    groups = sorted({(row['partition'], row['map'], row['label']) for row in runs})
    summaries = []
    for partition, name, label in groups:
        group = [row for row in runs if (row['partition'], row['map'], row['label']) == (partition, name, label)]
        entry = dict(partition=partition, map=name, label=label, runs=len(group),
                     seeds=sorted({row['seed'] for row in group}), requested=group[0]['requested'],
                     completed_metric_count_total=sum(row['completed_metric_count'] for row in group))
        for field in fields:
            values = [row[field] for row in group if row.get(field) is not None]
            if values:
                entry[field] = statistics.mean(values)
                entry[field + '_min'] = min(values)
                entry[field + '_max'] = max(values)
        summaries.append(entry)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory only')
    args = parser.parse_args()
    # Fail before reading large logs if a previous report already exists.
    if args.output.exists():
        raise FileExistsError(args.output)
    models, episodes, pairs = {}, [], []
    for name in audit.MAPS:
        folders = [('QL-3ep', ROOT / f'reports/stage5/train/{name}'),
                   ('pilot', STAGE / 'pilot' / name)]
        folders += [(f'QL-15ep-e{seed}', STAGE / 'training' / name / f'exploration_{seed}') for seed in (7, 17)]
        for label, folder in folders:
            payload, updates, training, state_actions = training_evidence(folder, name, label)
            models[name, label] = payload, updates
            episodes.extend(training)
            pairs.extend(state_actions)
    runs, decisions = [], []
    for partition in ('validation', 'test', 'stage6'):
        directory = ROOT / 'reports/stage6/runs' if partition == 'stage6' else STAGE / partition
        for result_path in sorted(directory.glob('**/result.json')):
            relative = result_path.relative_to(directory).parts
            name, label = relative[:2]
            if partition == 'stage6' and label == 'QL':
                label = 'QL-3ep'
            key = name, label
            model = models.get(key)
            result, details = run_evidence(result_path.parent, *model) if model else run_evidence(result_path.parent)
            repeat = int(result_path.parent.name.removeprefix('repeat_')) if partition == 'test' else 1
            result.update(partition=partition, label=label, repeat=repeat,
                          source=str(result_path.parent.relative_to(ROOT)))
            runs.append(result)
            decisions.extend(dict(partition=partition, map=name, label=label, **row) for row in details)
    selection_path = STAGE / 'models/selection.json'
    selection = audit.read_json(selection_path)
    assert audit.read_json(STAGE / 'test_manifest.json')['selection_sha256'] == sha256(selection_path)
    verify_run_matrix(runs, selection)
    variations = []
    for name in audit.MAPS:
        for seed in TEST_SEEDS:
            for label in sorted({r['label'] for r in runs if r['partition'] == 'test' and r['map'] == name}):
                group = [r for r in runs if r['partition'] == 'test' and r['map'] == name and r['seed'] == seed and r['label'] == label]
                assert len(group) == 2
                row = dict(map=name, seed=seed, label=label, repeats=len(group))
                for field in ('valid_completed', 'completion_rate', 'average_queue_length', 'teleports', 'movement_served'):
                    values = [r[field] for r in group]
                    row[field + '_range'] = max(values) - min(values)
                variations.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('training.csv', episodes), ('state_actions.csv', pairs),
                           ('runs.csv', runs), ('policy_decisions.csv', decisions),
                           ('summary.csv', summarize(runs)), ('variation.csv', variations)]:
        write_csv(args.output / filename, rows)
    print(f'Verified {len(episodes)} episodes, {len(runs)} evaluations, {len(decisions)} greedy decisions')


if __name__ == '__main__':
    main()
