"""Bang chung offline cho Stage 8; dung lai tai dung training, do phu va metric chung."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent.parent
for _d in (ROOT / "src", ROOT / "tools"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import analyze_ql as audit
import stage7_summary as stage7


STAGE = ROOT / 'reports/stage8'


def repetition_evidence(rows):
    """Thoi gian xanh lien tuc khong tinh phan vang/do dau tien khi doi action."""
    if not rows:
        raise ValueError('Empty decisions')
    previous, count, seconds = None, 0, 0
    repeated, longest_count, longest_seconds = 0, 0, 0
    states = Counter()
    for row in rows:
        action = int(row['action'])
        if action == previous:
            count += 1
            seconds += float(row['green'])
            repeated += 1
            states[tuple(json.loads(row['encoded_state']))] += 1
        else:
            count, seconds = 1, float(row['green'])
        longest_count = max(longest_count, count)
        longest_seconds = max(longest_seconds, seconds)
        previous = action
    loops = []
    for state, repeats in states.most_common():
        matching = [r for r in rows if tuple(json.loads(r['encoded_state'])) == state]
        raw = [json.loads(r['state']) for r in matching]
        queues = {tuple(s['incoming']) + tuple(s['outgoing']) for s in raw}
        loops.append(dict(state=list(state), decisions=len(matching), repeated_choices=repeats,
                          actions=dict(Counter(int(r['action']) for r in matching)),
                          q_values=json.loads(matching[0]['q_values']),
                          distinct_lane_queues=len(queues)))
    return dict(repeated_choices=repeated, longest_same_action_decisions=longest_count,
                longest_same_action_green_seconds=longest_seconds), loops


def audit_guard_decisions(rows, signals, metadata, encoding):
    """Tai dung tuoi nhu cau va cac lan ghi de bat buoc tu hang doi va den G thuc te."""
    groups = encoding['incoming_groups']
    runtime = dict(metadata['environment']['runtime_green_states'])
    links = metadata['controlled_links']
    ages = audit.replay_edge_ages(rows, signals, metadata, encoding)
    for row, (index, age) in zip(rows, ages, strict=True):
        if age < metadata['guard_seconds']:
            index = None
        assert json.loads(row['guard_edge']) == (groups[index]['edge'] if index is not None else None)
        assert float(row['guard_age']) == (age if index is not None else 0)
        expected = int(row['greedy_action'])
        if index is not None:
            candidates = [a for a in metadata['actions'] if any(runtime[a][i] == 'G'
                          for i, incoming, _, _ in links if incoming in groups[index]['lanes'])]
            values = json.loads(row['q_values'])
            expected = max(candidates, key=lambda a: values[metadata['actions'].index(a)])
        assert int(row['action']) == expected


def run_evidence(folder, payload, updates):
    result = audit.read_json(folder / 'result.json')
    metadata = audit.read_json(folder / 'metadata.json')
    rows = audit.read_csv(folder / 'actions.csv')
    signal_rows = audit.read_csv(folder / 'signals.csv')
    signals, _ = audit.audit_signals(signal_rows, metadata)
    for key, value in signals.items():
        assert result[key] == value
    assert result['simulation_seconds'] == 3600 and result['process_closed']
    assert stage7.sha256(Path(metadata['model'])) == result['model_sha256_before'] == result['model_sha256_after']
    assert audit.read_json(Path(metadata['model'])) == payload
    assert result['seed'] == metadata['seed'] == int(folder.parent.name.removeprefix('seed_'))
    assert metadata['encoding'] == payload['metadata']['encoding']
    assert result['map'] == metadata['map'] == payload['metadata']['map']
    assert metadata['environment'] == payload['metadata']['environment']
    assert metadata['source_hashes'] == payload['metadata']['source_hashes']
    table = audit.model_table(payload)
    actions = payload['metadata']['actions']
    initial_q = payload['metadata']['training'].get('initial_q', 0.0)
    if 'overdue_seconds' in metadata['encoding']:
        audit.audit_overdue_states(rows, signal_rows, metadata, metadata['encoding'])
    if metadata.get('guard_seconds'):
        audit_guard_decisions(rows, signal_rows, metadata, payload['metadata']['encoding'])
    decisions, observations = [], []
    overrides = 0
    for row in rows:
        state = tuple(json.loads(row['encoded_state']))
        evidence = audit.greedy_evidence(state, table, actions, updates, initial_q)
        values = table.get(state, [initial_q] * len(actions))
        assert json.loads(row['q_values']) == values
        assert json.loads(row['unseen']) == evidence['unseen']
        selected = int(row['action'])
        assert selected in actions
        if not metadata.get('guard_seconds'):
            assert selected == evidence['action']
        else:
            assert int(row['greedy_action']) == evidence['action']
            overrides += selected != evidence['action']
        decisions.append(dict(evidence, selected_updates=updates[state, selected]))
        observations.append(dict(json.loads(row['state']), encoded=state))
    _, aliases = audit.observation_analysis(observations, payload['metadata']['encoding'])
    result.update(movement_served=len(result['green_seconds_per_movement'])-len(result['unserved_movements']),
                  movement_total=len(result['green_seconds_per_movement']),
                  action_counts=dict(Counter(int(r['action']) for r in rows)),
                  unseen_rate=sum(d['unseen'] for d in decisions)/len(rows),
                  tie_rate=sum(d['tied'] for d in decisions)/len(rows),
                  untried_rate=sum(d['selected_updates'] == 0 for d in decisions)/len(rows),
                  guard_overrides=overrides)
    repeats, loops = repetition_evidence(rows)
    result.update(repeats)
    return result, dict(loops=loops, observations=aliases,
                        longest_no_green_per_movement=result['longest_no_green_per_movement'])


def verify_runs(runs, state_only=False):
    """Bat buoc du ma tran da khai bao; khong bao gio coi so sanh thieu la hoan chinh."""
    labels = ['QL-Original', 'QL-NewState']
    if not state_only:
        labels.append('QL-NewState-Guard')
    expected = {('cologne1', 'validation', label, seed, repeat)
                for label in labels for seed in (42, 12345) for repeat in (1, 2)}
    expected.update(('cologne1', 'checkpoint_validation', label, seed, 1)
                    for label in ('QL-NewState-ep05', 'QL-NewState-ep10') for seed in (42, 12345))
    if not state_only:
        expected.update(('cologne1', 'test', label, seed, repeat)
                        for label in labels for seed in (41001, 41002, 41003) for repeat in (1, 2))
        expected.update(('cologne1', 'diagnostic', 'QL-NewState', 31001, repeat) for repeat in (1, 2))
    actual = [(r['map'], r['partition'], r['label'], r['seed'], r['repeat']) for r in runs]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError(f'Incomplete, duplicate or unexpected run matrix: {len(actual)}/{len(expected)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Directory may exist; the four output files must be new')
    parser.add_argument('--state-only', action='store_true', help='Only the saved 12-run pre-Guard comparison')
    args = parser.parse_args()
    names = ('summary.csv', 'runs.csv', 'training.csv', 'evidence.json')
    for name in names:
        if (args.output / name).exists():
            raise FileExistsError(args.output / name)
    folders = {
        'QL-Original': ROOT / 'reports/stage7/training/cologne1/exploration_7',
        'QL-NewState': STAGE / 'training/cologne1',
    }
    models, training, coverage = {}, [], {}
    selection = audit.read_json(STAGE / 'selection.json') if not args.state_only else None
    if selection:
        assert audit.read_json(STAGE / 'test_manifest.json')['selection_sha256'] == stage7.sha256(STAGE / 'selection.json')
    for label, folder in folders.items():
        payload, updates, episodes, _ = stage7.training_evidence(folder, 'cologne1', label)
        if selection:
            assert stage7.sha256(folder / 'cologne1_q_table.json') == selection['models'][label]['sha256']
        models[label] = payload, updates
        training.extend(episodes)
        actions = payload['metadata']['actions']
        states = audit.model_table(payload)
        coverage[label] = dict(states=len(states), updated_pairs=len(updates),
                               possible_pairs=len(states)*len(actions),
                               coverage=len(updates)/(len(states)*len(actions)),
                               unupdated_fraction=1-len(updates)/(len(states)*len(actions)))
    runs, evidence = [], {}
    for partition in ('validation', 'checkpoint_validation', 'test', 'diagnostic'):
        if args.state_only and partition in ('test', 'diagnostic'):
            continue
        for path in sorted((STAGE / partition).glob('cologne1/**/result.json')):
            label = path.relative_to(STAGE / partition).parts[1]
            if args.state_only and label == 'QL-NewState-Guard':
                continue
            if label not in ('QL-Original', 'QL-NewState', 'QL-NewState-Guard',
                             'QL-NewState-ep05', 'QL-NewState-ep10'):
                raise ValueError(f'Unexpected model label: {label}')
            metadata = audit.read_json(path.parent / 'metadata.json')
            assert metadata.get('guard_seconds') == (300 if label == 'QL-NewState-Guard' else None)
            payload, updates = models['QL-NewState' if label != 'QL-Original' else label]
            if label.startswith('QL-NewState-ep'):
                episode = int(label.removeprefix('QL-NewState-ep'))
                payload = audit.read_json(folders['QL-NewState'] / f'episode_{episode:03d}/q_table.json')
                updates = Counter()
                for e in range(1, episode+1):
                    for row in audit.read_csv(folders['QL-NewState'] / f'episode_{e:03d}/actions.csv'):
                        updates[tuple(json.loads(row['encoded_state'])), int(row['action'])] += 1
            result, details = run_evidence(path.parent, payload, updates)
            source = path.parent.relative_to(ROOT).as_posix()
            result.update(partition=partition, label=label, repeat=int(path.parent.name.removeprefix('repeat_')),
                          source=source)
            runs.append(result)
            evidence[source] = details
    verify_runs(runs, args.state_only)
    # Giu day du mien gia tri tung luot va phan bo action, khong nhan ban log tung giay.
    summary = stage7.summarize(runs)
    for entry in summary:
        group = [r for r in runs if (r['partition'], r['map'], r['label']) ==
                 (entry['partition'], entry['map'], entry['label'])]
        for field in ('longest_same_action_decisions', 'longest_same_action_green_seconds',
                      'repeated_choices', 'guard_overrides'):
            values = [r[field] for r in group]
            entry.update({field: statistics.mean(values), field+'_min': min(values), field+'_max': max(values)})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in [('summary.csv', summary), ('runs.csv', runs), ('training.csv', training)]:
        stage7.write_csv(args.output / name, rows)
    with (args.output / 'evidence.json').open('x', encoding='utf-8') as file:
        json.dump(dict(coverage=coverage, runs=evidence), file, indent=2, allow_nan=False)
    print(f'Verified {len(training)} episodes and {len(runs)} evaluations')


if __name__ == '__main__':
    main()
