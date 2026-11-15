"""Rebuild Stage 6.1 evidence from saved files; never starts SUMO."""

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stage6_1_probe import first_difference

REPORT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def raw_metric_check(folder):
    result = load(folder / 'result.json')
    metrics = result.get('metrics', result)
    metadata = load(folder / 'metadata.json')
    map_name = metadata['map']
    begin, end = result['begin'], result['end']
    requests = {v.get('id') for v in ET.parse(ROOT / f'data/{map_name}/{map_name}.rou.xml').getroot()
                if v.tag in ('trip', 'vehicle') and begin <= float(v.get('depart')) < end}
    trips = [v for v in ET.parse(folder / 'tripinfo.xml').getroot().findall('tripinfo')
             if v.get('id') in requests]
    signals = list(csv.DictReader((folder / 'signals.csv').open(encoding='utf-8')))
    events = [v for row in signals for v in json.loads(row['teleports'])]
    teleported = set(events)
    inserted = [v for v in trips if float(v.get('depart')) >= 0]
    arrived = [v for v in inserted if begin <= float(v.get('arrival')) <= end
               and v.get('vaporized', '') in ('', 'false', '0')]
    valid = [v for v in arrived if v.get('id') not in teleported]
    expected = dict(requested=len(requests), inserted=len(inserted), arrived=len(arrived),
                    valid_completed=len(valid), unfinished=len(inserted)-len(arrived),
                    undeparted=len(requests)-len(inserted), teleported=len(teleported),
                    teleports=len(events), queue_seconds=sum(float(v['queue']) for v in signals),
                    completion_rate=len(valid)/len(requests), throughput=len(valid)*3600/(end-begin))
    expected['average_queue_length'] = expected['queue_seconds']/(end-begin)
    for metric, field in [('average_waiting_time', 'waitingTime'), ('average_time_loss', 'timeLoss')]:
        expected[metric] = sum(float(v.get(field)) for v in valid)/len(valid) if valid else None
    expected['average_travel_time'] = sum(float(v.get('arrival'))-float(v.get('depart'))
                                          for v in valid)/len(valid) if valid else None
    assert len(trips) == len(requests)
    for key, value in expected.items():
        if value is None:
            assert metrics[key] is None
        else:
            assert math.isclose(metrics[key], value, abs_tol=1e-9)
    vehicles = ET.parse(folder / 'statistics.xml').getroot().find('vehicles').attrib
    assert expected['inserted'] == int(vehicles['inserted'])
    assert expected['unfinished'] == int(vehicles['running'])
    # A fractional departure in the last second is requested but not yet attempted.
    scheduled = {v.get('id'): float(v.get('depart')) for v in
                 ET.parse(ROOT / f'data/{map_name}/{map_name}.rou.xml').getroot()
                 if v.get('id') in requests}
    not_yet_due = [v.get('id') for v in trips if float(v.get('depart')) < 0
                   and scheduled[v.get('id')] > end - 1]
    assert expected['undeparted'] == int(vehicles['waiting']) + len(not_yet_due)
    if (folder / 'trace.json').exists():
        trace = load(folder / 'trace.json')
        if trace:
            assert [sum(v['queue']) for v in trace] == [float(v['queue']) for v in signals]
            assert len(metadata['incoming_lanes']) == len(set(metadata['incoming_lanes']))
    return dict(path=str(folder.relative_to(ROOT)), checked=list(expected), passed=True,
                requested_not_yet_eligible_for_insertion=not_yet_due)


def main():
    protected = load(REPORT / 'protected_hashes_before.json')
    changed = [name for name, sha in protected.items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha]
    assert not changed
    probe = REPORT / 'reproducibility'
    baseline = load(probe / '0_A/trace.json')
    runs, differences = [], {}
    for folder in sorted(probe.iterdir()):
        if not (folder / 'result.json').exists():
            continue
        result = load(folder / 'result.json')
        meta = load(folder / 'metadata.json')
        trace = load(folder / 'trace.json')
        runs.append(dict(run=folder.name, horizon=result['end']-result['begin'],
                         mode=meta['mode'], observer=meta['observe'], reverse_reads=meta['reverse_reads'],
                         queue_seconds=result['queue_seconds'], samples_sha256=result['samples_sha256'],
                         trace_sha256=result['trace_sha256'] if trace else None))
        if trace:
            differences[folder.name] = first_difference(baseline[:len(trace)], trace)
    a, b = baseline, load(probe / '2_A/trace.json')
    component_first = {k: next((x['time'] for x, y in zip(a, b) if x[k] != y[k]), None)
                       for k in ('time', 'signal', 'departed', 'active', 'vehicles', 'queue', 'teleports', 'collisions')}
    native = {}
    for folder in sorted(probe.glob('native_*')):
        steps = ET.parse(gzip.open(folder / 'fcd.xml.gz')).getroot()
        native[folder.name] = {float(step.get('time')):
                              {v.get('id'): v.attrib for v in step} for step in steps}
    native_base = native['native_0']
    native_first = {name: next((t for t in native_base if native_base[t] != steps[t]), None)
                    for name, steps in native.items()}
    states, rng_speeds = [], {}
    for folder in sorted(probe.glob('rng_*')):
        states.append(ET.tostring(ET.parse(gzip.open(folder / 'state.xml.gz')).getroot()))
        fcd = ET.parse(gzip.open(folder / 'fcd.xml.gz')).getroot()
        step = next(s for s in fcd if float(s.get('time')) == 25853)
        rng_speeds[folder.name] = float(step.find("vehicle[@id='142128_415_0']").get('speed'))
    assert len(states) == 4 and all(s == states[0] for s in states)
    assert len(set(rng_speeds.values())) == 2
    assert set(t for t in native_first.values() if t is not None) == {25853}
    initial = [load(folder / 'metadata.json')['internal_foes'] for folder in sorted(probe.glob('initial_[0-9]'))]
    assert len(initial) == 5 and any(x != initial[0] for x in initial)
    safety_prefix = {}
    for group, sub, name, seed in [('collision_analysis', 'observed', 'ingolstadt1', 2026),
                                  ('emergency_braking_analysis', 'repeat_0', 'cologne1', 2028)]:
        samples = load(REPORT / group / sub / 'samples.json')
        old = list(csv.DictReader((ROOT / f'reports/stage6/runs/{name}/FT/seed_{seed}/signals.csv').open()))
        normalized = [{k: json.loads(v) if k in ('junction_before', 'teleports', 'collisions')
                       else float(v) if k in ('start', 'end', 'queue')
                       else int(v) if k in ('decision', 'action') else v for k, v in row.items()}
                      for row in old[:len(samples)]]
        assert samples == normalized
        safety_prefix[group] = dict(matching_samples=len(samples), exact=True)
    checks = [raw_metric_check(folder) for folder in sorted((ROOT / 'reports/stage6/runs').glob('*/*/seed_*'))]
    for path in sorted(REPORT.rglob('result.json')):
        if 'metrics' in load(path):
            checks.append(raw_metric_check(path.parent))
    delivery = load(ROOT / 'reports/stage6/provenance.json')['code_at_delivery']
    production = ('sumo_env.py', 'evaluate.py', 'evaluation_metrics.py', 'ft.py', 'mp.py', 'q_learning.py')
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == delivery[p] for p in production)
    tests = re.search(r'(\d+) passed in ([\d.]+)s', (REPORT / 'tests_final.log').read_text())
    assert tests is not None
    result = dict(protected_files=len(protected), protected_changed=changed,
                  production_unchanged=list(production), runs=runs,
                  first_differences=differences, first_by_component=component_first,
                  native_first_difference_fcd_start=native_first,
                  saved_state_time=25853, saved_state_including_rng_identical=True,
                  saved_state_sha256=hashlib.sha256(states[0]).hexdigest(),
                  rng_replay_speed_at_fcd_25853=rng_speeds,
                  initial_internal_foe_order_varies=True, safety_matches_stage6_prefix=safety_prefix,
                  independent_raw_metric_checks=checks,
                  tests_before=dict(passed=142, seconds=12.80, interpreter='.venv/Scripts/python.exe'),
                  tests_final=dict(passed=int(tests[1]), seconds=float(tests[2]), log='tests_final.log'),
                  exact_upstream_root_cause='not identified; nondeterminism isolated to SUMO execution',
                  decision='metrics reconciled; simulator repeatability unresolved; do not claim QL improvements yet')
    (REPORT / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Protected {len(protected)} files unchanged; {len(checks)} raw metric checks passed.')
    print('First difference by component:', component_first)


if __name__ == '__main__':
    main()
