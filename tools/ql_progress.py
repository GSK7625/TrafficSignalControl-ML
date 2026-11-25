"""Doc log train va cac luot validation greedy; in mot dong CSV cho moi checkpoint."""

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
from statistics import mean
import sys

import analyze_ql as audit
from stage7_summary import checkpoint_changes, coverage, sha256
from stage8_summary import repetition_evidence


def validation_metrics(folders, payload, updates):
    """Moi luot co cung trong so khi so completion, queue va teleport."""
    decisions = unseen_states = untried = 0
    overdue = service_tried = served = 0
    longest_chain = longest_green = tied = 0
    encoding = payload['metadata']['encoding']
    results = []
    maximum_ages = []
    for folder in folders:
        result = audit.read_json(folder / 'result.json')
        meta = audit.read_json(folder / 'metadata.json')
        settings = payload['metadata']['training']
        train_seeds = settings.get('sumo_seeds', [settings['sumo_seed']])
        environment = payload['metadata']['environment']
        if (result['simulation_seconds'] != environment['end'] - environment['begin']
                or meta['environment'] != environment):
            raise ValueError(f'Validation horizon or signal settings differ from training: {folder}')
        if (meta['algorithm'] != 'QL' or meta['map'] != payload['metadata']['map']
                or meta['seed'] in train_seeds or meta['epsilon'] != 0 or meta.get('guard_seconds') is not None):
            raise ValueError(f'Expected independent greedy validation: {folder}')
        rows = audit.read_csv(folder / 'actions.csv')
        if payload['metadata']['schema_version'] == 2 or 'oldest_age_end' in rows[0]:
            spec = payload['metadata'].get('reward', {'version': 'mean-queue-v1', 'lambda': 0.0})
            if meta.get('reward') != spec:
                raise ValueError(f'Validation reward differs from checkpoint: {folder}')
            maximum_ages.append(audit.audit_rewards(rows, audit.read_csv(folder / 'signals.csv'),
                                                   meta, payload['metadata']))
            if 'max_observed_starvation_s' in result:
                assert result['max_observed_starvation_s'] == maximum_ages[-1] and result['reward'] == spec
        details, _ = audit.evaluation_policy(rows, payload, updates, str(folder))
        counts = audit.policy_summary(details)
        repetition, _ = repetition_evidence(rows)
        longest_chain = max(longest_chain, repetition['longest_same_action_decisions'])
        longest_green = max(longest_green, repetition['longest_same_action_green_seconds'])
        tied += counts['tied']
        assert counts['decisions'] == result['control_calls']
        assert counts['unseen'] == result['unseen_decisions']
        decisions += counts['decisions']
        unseen_states += counts['unseen']
        untried += counts['selected_untried']
        if 'overdue_seconds' in encoding:
            greens = dict(meta['environment']['runtime_green_states'])
            for row, decision in zip(rows, details):
                state = decision['state']
                if state[-1] == 0:
                    continue
                lanes = encoding['incoming_groups'][state[-1] - 1]['lanes']
                links = [i for i, lane, _, _ in meta['controlled_links'] if lane in lanes]
                serving = [a for a, lights in greens.items() if any(lights[i] == 'G' for i in links)]
                overdue += 1
                service_tried += any(updates[state, a] > 0 for a in serving)
                served += decision['action'] in serving and float(row['green']) > 0
        results.append(result)
    metrics = dict(validation_runs=len(results), validation_decisions=decisions,
                validation_unseen_state_rate=unseen_states / decisions,
                validation_unseen_action_rate=untried / decisions,
                validation_greedy_tie_rate=tied / decisions,
                longest_same_action_decisions=longest_chain, longest_same_action_green_seconds=longest_green,
                overdue_decisions=overdue, overdue_serving_phase_tried=service_tried,
                overdue_served=served,
                completion=mean(r['completion_rate'] for r in results),
                queue=mean(r['average_queue_length'] for r in results),
                teleports=mean(r['teleports'] for r in results),
                max_gap=max(r['longest_movement_no_green'] for r in results))
    metrics['max_observed_starvation_s'] = max(maximum_ages) if len(maximum_ages) == len(results) else None
    for key in ('average_waiting_time', 'average_travel_time', 'average_time_loss', 'throughput',
                'unfinished', 'undeparted', 'valid_completed', 'requested'):
        values = [r[key] for r in results]
        metrics[key] = mean(values) if None not in values else None
    return metrics


def progress(training, validation=()):
    by_model = defaultdict(list)
    if len(set(p.resolve() for p in validation)) != len(validation):
        raise ValueError('Duplicate validation folder')
    for folder in validation:
        result = audit.read_json(folder / 'result.json')
        if result['model_sha256_before'] != result['model_sha256_after']:
            raise ValueError(f'Model changed during validation: {folder}')
        by_model[result['model_sha256_before']].append(folder)

    states, updates, previous = set(), Counter(), {}
    report = []
    for number, row in enumerate(audit.read_csv(training / 'training.csv'), 1):
        assert number == int(row['episode'])
        folder = training / f'episode_{number:03d}'
        model = folder / 'q_table.json'
        payload = audit.read_json(model)
        table = audit.model_table(payload)
        actions = audit.read_csv(folder / 'actions.csv')
        pairs_before = len(updates)
        visited = coverage(actions, states, updates, payload['metadata']['actions'])
        changes = checkpoint_changes(previous, table)
        assert len(table) == len(states) == int(row['q_states'])
        assert len(actions) == int(row['control_calls'])
        entry = dict(episode=number, sumo_seed=row['sumo_seed'], epsilon=row['epsilon'],
                     total_reward=row['total_reward'],
                     average_queue_length=float(row['queue_seconds']) / float(row['simulation_seconds']),
                     q_states=len(table), updated_pairs=len(updates), possible_pairs=visited['possible_pairs'],
                     coverage=visited['coverage'], train_unseen_action_rate=(len(updates)-pairs_before) / len(actions),
                     common_states=changes['common_states'], greedy_changes=changes['greedy_changes'],
                     greedy_change_rate=changes['greedy_change_rate'] if changes['common_states'] else None)
        runs = by_model.pop(sha256(model), [])
        if runs:
            entry.update(validation_metrics(runs, payload, updates))
        report.append(entry)
        previous = table
    if by_model:
        raise ValueError('A validation model SHA does not match any training checkpoint')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training', type=Path)
    parser.add_argument('--validation', type=Path, nargs='+', default=[])
    args = parser.parse_args()
    rows = progress(args.training, args.validation)
    fields = dict.fromkeys(key for row in rows for key in row)
    writer = csv.DictWriter(sys.stdout, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()
