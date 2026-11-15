"""Small numerical checks plus short real SUMO observer/read-only checks."""

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from evaluation_metrics import trip_metrics, signal_metrics, requested_trips


def test_raw_xml_csv_repeated_teleport_and_final_step_arrival(tmp_path):
    """Two teleport events for a single ID; unfinished/pending stay in demand."""
    from evaluation_metrics import read_metrics
    rows = [trip('a', 0, 3, 99, 99), trip('b', 1, 4, 2, 3),
            trip('c', 2, -1, 100, 100), trip('d', -1, -1)]
    root = ET.Element('tripinfos')
    for row in rows:
        ET.SubElement(root, 'tripinfo', row)
    ET.ElementTree(root).write(tmp_path / 'tripinfo.xml')
    (tmp_path / 'statistics.xml').write_text(
        '<statistics><vehicles loaded="4" inserted="3" running="1" waiting="1"/>'
        '<safety collisions="0" emergencyBraking="0" emergencyStops="0"/>'
        '<teleports total="2"/></statistics>')
    (tmp_path / 'collisions.xml').write_text('<collisions/>')
    (tmp_path / 'warnings.log').write_text('')
    with (tmp_path / 'signals.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=(
            'start', 'end', 'signal', 'queue', 'teleports', 'collisions'))
        writer.writeheader()
        for i, queue in enumerate((2, 3, 4, 5)):
            writer.writerow(dict(start=i, end=i+1, signal='G', queue=queue,
                                 teleports=json.dumps(['a'] if i in (1, 2) else []), collisions='[]'))
    result = read_metrics(tmp_path, {'a', 'b', 'c', 'd'}, 0, 4)
    assert result['teleports'] == 2 and result['teleported'] == 1
    assert result['teleport_ids'] == ['a']
    assert result['requested'] == 4 and result['inserted'] == 3
    assert result['arrived'] == 2 and result['valid_completed'] == 1
    assert result['unfinished'] == result['undeparted'] == 1
    assert result['average_travel_time'] == 3
    assert result['average_waiting_time'] == 2 and result['average_time_loss'] == 3
    assert result['queue_seconds'] == 14 and result['average_queue_length'] == 3.5
    assert result['completion_rate'] == .25 and result['throughput'] == 900


def trip(vehicle, depart, arrival, waiting=0, loss=0, **extra):
    return dict(id=vehicle, depart=str(depart), arrival=str(arrival),
                waitingTime=str(waiting), timeLoss=str(loss), **extra)


def test_trip_metrics_use_same_valid_cohort_and_unique_teleport_ids():
    rows = [trip('a', 10, 30, 4, 7, departDelay='100'),
            trip('b', 15, 45, 8, 11), trip('tele', 0, 50, 99, 100),
            trip('running', 20, -1, 50, 60), trip('pending', -1, -1)]
    result = trip_metrics(rows, ['tele', 'tele', 'running'], 5, 0, 100)
    assert result['average_waiting_time'] == 6
    assert result['average_travel_time'] == 25
    assert result['average_time_loss'] == 9
    assert result['valid_completed'] == result['completed_metric_count'] == 2
    assert result['arrived'] == 3 and result['arrived_teleported'] == 1
    assert result['teleported'] == 2 and result['unfinished'] == 1
    assert result['undeparted'] == 1 and result['inserted'] == 4
    assert result['completion_rate'] == .4 and result['insertion_rate'] == .8
    assert result['throughput'] == 72 and result['throughput_all_arrivals'] == 108


def test_unfinished_is_not_zero_duration_and_no_valid_trips_is_missing():
    result = trip_metrics([trip('p', -1, -1), trip('u', 0, -1)], [], 2, 0, 100)
    assert result['average_travel_time'] is None
    assert result['average_waiting_time'] is None
    assert result['average_time_loss'] is None
    assert result['throughput'] == result['completion_rate'] == 0
    with pytest.raises(ValueError, match='Duplicate'):
        trip_metrics([trip('a', 0, 1)] * 2, [], 2, 0, 100)


def test_signal_metrics_full_horizon_queue_and_movement_gaps():
    signals = [dict(start=i, end=i+1, queue=q, signal=s)
               for i, q, s in [(0, 2, 'Gr'), (1, 4, 'rr'), (2, 6, 'Gr'), (3, 0, 'rr')]]
    result = signal_metrics(signals, 0, 4)
    assert result['average_queue_length'] == 3
    assert result['green_seconds_per_movement'] == [2, 0]
    assert result['longest_no_green_per_movement'] == [1, 4]
    assert result['unserved_movements'] == [1]
    with pytest.raises(ValueError, match='coverage'):
        signal_metrics(signals[:-1], 0, 4)


@pytest.mark.parametrize('name,total', [('cologne1', 2015), ('ingolstadt1', 1716)])
def test_original_demand_and_fixed_models(name, total):
    from sumo_env import SumoEnv, ROOT
    import q_learning as ql
    env = SumoEnv(name)
    assert len(requested_trips(env)) == total
    model = ROOT / f'reports/stage5/train/{name}/{name}_q_table.json'
    table, meta = ql.load_q_table(model, env)
    assert len(table) == (57 if name == 'cologne1' else 15)
    assert meta['training']['episodes_trained'] == 3


def test_request_horizon_excludes_end_and_rejects_unsupported_flow(tmp_path):
    from types import SimpleNamespace
    config = tmp_path / 'test.sumocfg'
    config.write_text('<configuration><input><route-files value="r.xml"/></input></configuration>')
    route = tmp_path / 'r.xml'
    env = SimpleNamespace(config=config, begin=10, end=20)
    route.write_text('<routes><trip id="before" depart="9"/><trip id="yes" depart="10"/>'
                     '<trip id="end" depart="20"/></routes>')
    assert requested_trips(env) == {'yes'}
    route.write_text('<routes><flow id="f" begin="10" end="20" number="3"/></routes>')
    with pytest.raises(ValueError, match='flow'):
        requested_trips(env)


@pytest.mark.parametrize('algorithm', ['FT', 'MP', 'QL'])
def test_common_runner_short_real_sumo(tmp_path, algorithm):
    from evaluate import run_evaluation
    from sumo_env import ROOT
    model = ROOT / 'reports/stage5/train/ingolstadt1/ingolstadt1_q_table.json'
    before = model.read_bytes()
    result = run_evaluation('ingolstadt1', algorithm, 42, tmp_path / algorithm,
                            model if algorithm == 'QL' else None, horizon=47)
    assert result['simulation_seconds'] == 47
    assert result['process_closed']
    assert model.read_bytes() == before
    assert result['average_queue_length'] == result['queue_seconds'] / 47
    assert result['requested'] == result['inserted'] + result['undeparted']
    assert result['inserted'] == result['arrived'] + result['unfinished']
    meta = json.loads((tmp_path / algorithm / 'metadata.json').read_text())
    assert meta['seed'] == 42 and meta['environment']['end'] - meta['begin'] == 47
    assert meta['epsilon'] == (0 if algorithm == 'QL' else None)
    if algorithm == 'QL':
        assert result['model_sha256_before'] == result['model_sha256_after'] == hashlib.sha256(before).hexdigest()
    with pytest.raises(FileExistsError):
        run_evaluation('ingolstadt1', algorithm, 42, tmp_path / algorithm,
                       model if algorithm == 'QL' else None, horizon=47)


def test_tripinfo_output_matches_short_85_second_control(tmp_path):
    from sumo_env import SumoEnv
    from ft import fixed_time
    recorded = []
    for enabled in (False, True):
        env = SumoEnv('ingolstadt1', tmp_path / str(enabled), seed=42, tripinfo=enabled)
        env.end = env.begin + 85
        try:
            env.reset()
            recorded.append([sample for info in fixed_time(env) for sample in info['samples']])
        finally:
            env.close()
    assert recorded[0] == recorded[1]


def test_summary_missing_seeds_and_paired_differences():
    from plot_results import aggregate_results, paired_differences, experiment_coverage
    rows = [dict(map='cologne1', algorithm=a, seed=s, average_queue_length=q)
            for a, s, q in [('FT', 42, 10), ('FT', 12345, 14), ('MP', 42, 6)]]
    summary = aggregate_results(rows)
    ft_queue = next(r for r in summary if r['algorithm'] == 'FT' and r['metric'] == 'average_queue_length')
    assert ft_queue['mean'] == 12 and ft_queue['std'] == pytest.approx(8**.5)
    assert ft_queue['n'] == 2 and ft_queue['minimum'] == 10 and ft_queue['maximum'] == 14
    pairs = paired_differences(rows)
    queue_pair = [r for r in pairs if r['metric'] == 'average_queue_length']
    assert len(queue_pair) == 1 and queue_pair[0]['difference_b_minus_a'] == -4
    coverage = experiment_coverage(rows)
    assert coverage['completed'] == 3 and coverage['expected'] == 30
    assert not coverage['complete'] and len(coverage['missing']) == 27
    with pytest.raises(ValueError, match='Duplicate'):
        experiment_coverage(rows + rows[:1])


def test_summary_rejects_changed_model_between_seeds(tmp_path):
    from plot_results import verify_common_configuration
    from sumo_env import ROOT
    source = ROOT / 'reports/stage6/pilot/cologne1/QL/seed_42'
    meta = json.loads((source / 'metadata.json').read_text())
    result = json.loads((source / 'result.json').read_text())
    paths = []
    for seed, model_hash in [(42, meta['model_sha256_before']), (12345, '0' * 64)]:
        folder = tmp_path / str(seed)
        folder.mkdir()
        changed_meta = dict(meta, seed=seed, model_sha256_before=model_hash)
        changed_result = dict(result, seed=seed, model_sha256_before=model_hash, model_sha256_after=model_hash)
        (folder / 'metadata.json').write_text(json.dumps(changed_meta))
        path = folder / 'result.json'
        path.write_text(json.dumps(changed_result))
        paths.append(path)
    with pytest.raises(ValueError, match='Mixed QL model'):
        verify_common_configuration(paths)


RUNS = Path(__file__).parent / 'reports/stage6/runs'


@pytest.mark.parametrize('path', sorted(RUNS.glob('*/*/seed_*/result.json')))
def test_recorded_metrics_and_signals_independent_reference(path):
    result = json.loads(path.read_text())
    meta = json.loads(path.with_name('metadata.json').read_text())
    with path.with_name('signals.csv').open() as f:
        signals = list(csv.DictReader(f))
    with path.with_name('actions.csv').open() as f:
        actions = list(csv.DictReader(f))
    trips = ET.parse(path.with_name('tripinfo.xml')).getroot().findall('tripinfo')
    teleported = {v for s in signals for v in json.loads(s['teleports'])}
    arrived = [r for r in trips if float(r.get('arrival')) >= 0 and r.get('vaporized') == '']
    valid = [r for r in arrived if r.get('id') not in teleported]
    assert len(trips) == result['requested'] == result['loaded']
    assert len(arrived) == result['arrived']
    assert len(valid) == result['valid_completed']
    assert len(teleported) == result['teleported']
    assert result['teleports'] == sum(len(json.loads(s['teleports'])) for s in signals)
    assert result['undeparted'] == sum(float(r.get('depart')) < 0 for r in trips)
    assert result['unfinished'] == sum(float(r.get('depart')) >= 0 and float(r.get('arrival')) < 0 for r in trips)
    assert result['loaded'] == result['undeparted'] + result['unfinished'] + result['arrived']
    assert result['average_waiting_time'] == pytest.approx(sum(float(r.get('waitingTime')) for r in valid) / len(valid))
    assert result['average_time_loss'] == pytest.approx(sum(float(r.get('timeLoss')) for r in valid) / len(valid))
    assert result['average_travel_time'] == pytest.approx(sum(float(r.get('arrival')) - float(r.get('depart')) for r in valid) / len(valid))
    assert all(float(r.get('duration')) == float(r.get('arrival')) - float(r.get('depart')) for r in valid)
    assert result['average_queue_length'] == sum(float(s['queue']) for s in signals) / 3600
    assert result['completion_rate'] == len(valid) / len(trips)
    assert result['throughput'] == len(valid)
    assert len(signals) == 3600
    assert float(signals[0]['start']) == meta['begin']
    assert float(signals[-1]['end']) == meta['end']
    assert all(a['end'] == b['start'] for a, b in zip(signals, signals[1:]))
    runtime = dict(meta['environment']['runtime_green_states'])
    for index, row in enumerate(actions):
        samples = [s for s in signals if int(s['decision']) == index]
        assert len(samples) == float(row['elapsed'])
        if index < len(actions) - 1:
            assert float(row['green']) == 30
        assert all(s['signal'] == runtime[int(row['action'])] for s in samples if s['stage'] == 'green')
        if int(row['previous_phase']) != int(row['action']) and float(row['green']) > 0:
            assert float(row['yellow']) == meta['environment']['yellow_seconds']
            assert float(row['all_red']) >= 2
            first_green = next(s for s in samples if s['stage'] == 'green')
            assert json.loads(first_green['junction_before']) == []
        if result['algorithm'] == 'QL':
            values = json.loads(row['q_values'])
            best = max(range(len(values)), key=values.__getitem__)
            assert int(row['action']) == meta['actions'][best]


def test_delivered_experiment_complete_and_same_configuration():
    from plot_results import experiment_coverage, verify_common_configuration
    paths = sorted(RUNS.glob('*/*/seed_*/result.json'))
    rows = [json.loads(p.read_text()) for p in paths]
    assert experiment_coverage(rows)['complete']
    assert len(rows) == 30
    assert set(verify_common_configuration(paths)) == {'cologne1', 'ingolstadt1'}
    for row in rows:
        if row['algorithm'] == 'QL':
            model = RUNS.parents[1] / f"stage5/train/{row['map']}/{row['map']}_q_table.json"
            assert hashlib.sha256(model.read_bytes()).hexdigest() == row['model_sha256_before'] == row['model_sha256_after']
