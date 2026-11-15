"""Rebuild stage6 tables/paired differences/plots from every completed raw run."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from evaluate import MAPS, ALGORITHMS, SEEDS


METRICS = (
    'average_waiting_time', 'average_travel_time', 'average_queue_length', 'average_time_loss',
    'throughput', 'completion_rate', 'insertion_rate', 'undeparted_rate',
    'collisions', 'teleports', 'teleported', 'loaded', 'inserted', 'arrived',
    'valid_completed', 'arrived_teleported', 'unfinished', 'undeparted',
    'emergency_braking', 'emergency_stops', 'longest_movement_no_green',
)


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def experiment_coverage(rows):
    expected = {(m, a, s) for m in MAPS for a in ALGORITHMS for s in SEEDS}
    actual = {(r['map'], r['algorithm'], r['seed']) for r in rows}
    if len(actual) != len(rows):
        raise ValueError('Duplicate map/algorithm/seed')
    if actual - expected:
        raise ValueError('Non-held-out run in main experiment')
    return dict(expected=len(expected), completed=len(actual), complete=actual == expected,
                missing=[dict(map=m, algorithm=a, seed=s) for m, a, s in sorted(expected - actual)])


def aggregate_results(rows):
    summary = []
    for map_name in MAPS:
        for algorithm in ALGORITHMS:
            group = [r for r in rows if r['map'] == map_name and r['algorithm'] == algorithm]
            if not group:
                continue
            for metric in METRICS:
                values = [r[metric] for r in group if r.get(metric) is not None]
                summary.append(dict(
                    map=map_name, algorithm=algorithm, metric=metric,
                    n_seeds=len(group), n=len(values),
                    mean=statistics.mean(values) if values else None,
                    std=statistics.stdev(values) if len(values) > 1 else None,
                    minimum=min(values) if values else None, maximum=max(values) if values else None))
    return summary


def paired_differences(rows):
    lookup = {(r['map'], r['algorithm'], r['seed']): r for r in rows}
    pairs = []
    for map_name in MAPS:
        for a, b in combinations(ALGORITHMS, 2):
            for seed in SEEDS:
                ra, rb = lookup.get((map_name, a, seed)), lookup.get((map_name, b, seed))
                if ra is None or rb is None:
                    continue
                for metric in METRICS:
                    if ra.get(metric) is not None and rb.get(metric) is not None:
                        pairs.append(dict(map=map_name, algorithm_a=a, algorithm_b=b, seed=seed,
                                          metric=metric, difference_b_minus_a=rb[metric] - ra[metric]))
    return pairs


def verify_common_configuration(paths):
    """Fail on mixed demand/settings/code rather than silently average unfair runs."""
    reference = {}
    model_hashes = {}
    keys = ('source_hashes', 'code_hashes', 'environment', 'actions', 'tls_id',
            'controlled_links', 'incoming_lanes', 'outgoing_lanes', 'begin', 'end',
            'step_length', 'sumo_version', 'collision_check_junctions', 'collision_action',
            'time_to_teleport', 'requested')
    for path in paths:
        meta = json.loads(path.with_name('metadata.json').read_text(encoding='utf-8'))
        result = json.loads(path.read_text(encoding='utf-8'))
        common = {k: meta[k] for k in keys}
        map_name = meta['map']
        if map_name in reference and reference[map_name] != common:
            raise ValueError(f'Mixed evaluation configuration: {path}')
        reference[map_name] = common
        assert meta['seed'] == result['seed']
        assert meta['algorithm'] == result['algorithm']
        assert result['simulation_seconds'] == 3600
        assert result['queue_samples'] == 3600 and result['process_closed']
        if meta['algorithm'] == 'QL':
            assert meta['epsilon'] == 0
            assert result['model_sha256_before'] == result['model_sha256_after'] == meta['model_sha256_before']
            digest = meta['model_sha256_before']
            if map_name in model_hashes and model_hashes[map_name] != digest:
                raise ValueError(f'Mixed QL model across seeds: {path}')
            model_hashes[map_name] = digest
    return reference


def plot_metrics(rows, output):
    charts = [
        ('average_waiting_time', 'Average waiting time (valid completed cohort)', 'seconds/vehicle', 1),
        ('average_travel_time', 'Average travel time (valid completed cohort)', 'seconds/trip', 1),
        ('average_queue_length', 'Average observed incoming queue', 'vehicles', 1),
        ('average_time_loss', 'Average time loss (valid completed cohort)', 'seconds/vehicle', 1),
        ('throughput', 'Valid throughput (never teleported)', 'completed trips/hour', 1),
        ('completion_rate', 'Valid completion rate', '% of requested trips', 100),
        ('teleports', 'Teleport events (can repeat per vehicle)', 'events/hour', 1),
        ('undeparted_rate', 'Not inserted by horizon end', '% of requested trips', 100),
    ]
    output.mkdir(parents=True, exist_ok=True)
    for map_name in MAPS:
        for metric, title, unit, scale in charts:
            fig, ax = plt.subplots(figsize=(7.2, 4.5))
            labels = []
            for index, algorithm in enumerate(ALGORITHMS):
                group = [r for r in rows if r['map'] == map_name and r['algorithm'] == algorithm]
                values = [r[metric] * scale for r in group if r.get(metric) is not None]
                labels.append(f'{algorithm}\nn={len(values)}/{len(SEEDS)} seeds')
                if not values:
                    ax.text(index, .05, 'not measured', ha='center', transform=ax.get_xaxis_transform())
                    continue
                ax.bar(index, statistics.mean(values),
                       yerr=statistics.stdev(values) if len(values) > 1 else 0,
                       width=.62, color=('#517691', '#4b8b70', '#bd7956')[index], capsize=5)
                offsets = [0] if len(values) == 1 else [(.3 * i / (len(values)-1) - .15) for i in range(len(values))]
                ax.scatter([index + x for x in offsets], values, color='#252525', s=16, zorder=3)
            ax.set_xticks(range(3), labels)
            ax.set_ylabel(unit)
            ax.set_title(f'{map_name} — {title}', fontsize=11)
            ax.set_ylim(bottom=0)
            if scale == 100:
                ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=.25)
            ax.set_axisbelow(True)
            fig.text(.5, .015, 'Bars: mean; error bars: sample SD; dots: all retained seeds', ha='center', fontsize=8)
            fig.tight_layout(rect=(0, .045, 1, 1))
            fig.savefig(output / f'{map_name}_{metric}.png', dpi=160)
            plt.close(fig)


def summarize(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(runs).glob('*/*/seed_*/result.json'))
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in paths]
    coverage = experiment_coverage(rows)
    common = verify_common_configuration(paths)
    fields = ['map', 'algorithm', 'seed', *METRICS, 'requested', 'completed_metric_count',
              'throughput_all_arrivals', 'unfinished_teleported', 'queue_seconds', 'queue_samples',
              'model_sha256_before', 'model_sha256_after']
    write_csv(output / 'runs.csv', [{k: r[k] for k in fields} for r in rows], fields)
    summary = aggregate_results(rows)
    write_csv(output / 'summary.csv', summary)
    pairs = paired_differences(rows)
    write_csv(output / 'paired_differences.csv', pairs)
    pair_summary = []
    for m, a, b, metric in sorted({(r['map'], r['algorithm_a'], r['algorithm_b'], r['metric']) for r in pairs}):
        values = [r['difference_b_minus_a'] for r in pairs
                  if (r['map'], r['algorithm_a'], r['algorithm_b'], r['metric']) == (m, a, b, metric)]
        pair_summary.append(dict(map=m, algorithm_a=a, algorithm_b=b, metric=metric, n_pairs=len(values),
                                 mean_difference_b_minus_a=statistics.mean(values),
                                 std=statistics.stdev(values) if len(values) > 1 else None))
    write_csv(output / 'paired_summary.csv', pair_summary)
    movements = []
    for row, path in zip(rows, paths):
        meta = json.loads(path.with_name('metadata.json').read_text(encoding='utf-8'))
        for index, seconds in enumerate(row['green_seconds_per_movement']):
            links = [link for link in meta['controlled_links'] if link[0] == index]
            movements.append(dict(map=row['map'], algorithm=row['algorithm'], seed=row['seed'],
                                  movement=index, incoming=';'.join(sorted({x[1] for x in links})),
                                  outgoing=';'.join(sorted({x[2] for x in links})), green_seconds=seconds,
                                  longest_no_green=row['longest_no_green_per_movement'][index]))
    write_csv(output / 'movement_service.csv', movements)
    verification = dict(coverage=coverage, common_configuration=common,
                        signals_differ_across_seeds={
                            f'{m}/{a}': len({r['queue_seconds'] for r in rows if r['map'] == m and r['algorithm'] == a}) > 1
                            for m in MAPS for a in ALGORITHMS})
    (output / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    plot_metrics(rows, output / 'plots')
    print(json.dumps(coverage, indent=2))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('reports/stage6/runs'))
    parser.add_argument('--output', type=Path, default=Path('reports/stage6'))
    args = parser.parse_args()
    summarize(args.runs, args.output)


if __name__ == '__main__':
    main()
