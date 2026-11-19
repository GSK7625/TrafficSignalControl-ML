"""Dinh dang trace tin hieu va cac metric dung chung tu SUMO."""

import csv
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


SIGNAL_FIELDS = (
    'decision', 'action', 'start', 'end', 'stage', 'signal', 'queue',
    'junction_before', 'collisions', 'teleports',
)


def write_signal_sample(writer, decision, action, sample):
    csv_sample = {key: json.dumps(value) if isinstance(value, (tuple, list)) else value
                  for key, value in sample.items()}
    writer.writerow(dict(decision=decision, action=action, **csv_sample))


def requested_trips(env):
    """Dem cac yeu cau chuyen di ro rang trong [begin,end); khong dung inserted lam nhu cau."""
    config = ET.parse(env.config).getroot()
    ids = set()
    for name in config.find('input/route-files').get('value').split(','):
        root = ET.parse(env.config.parent / name.strip()).getroot()
        if any(e.tag in ('flow', 'person', 'personFlow') for e in root):
            raise ValueError('Only explicit trip/vehicle requests supported, not flow/person')
        for element in root:
            if element.tag not in ('trip', 'vehicle'):
                continue
            depart = float(element.get('depart'))
            if not math.isfinite(depart):
                raise ValueError('Request depart must be finite')
            if env.begin <= depart < env.end:
                vehicle = element.get('id')
                if not vehicle or vehicle in ids:
                    raise ValueError('Missing or duplicate request ID')
                ids.add(vehicle)
    if not ids:
        raise ValueError('No requests in horizon')
    return ids


def trip_metrics(trips, teleport_ids, requested, begin, end):
    """Gia tri trung binh CHI tinh xe da den dich, khong bi xoa va chua tung teleport.

    Xe den dich dung moc end thuoc ve buoc SUMO cuoi cung duoc quan sat.
    Thoi luong chua hoan thanh trong XML chi la mot phan, khong dung lam thoi luong hoan thanh.
    """
    if end <= begin or requested <= 0:
        raise ValueError('Positive horizon and requested count required')
    rows = {row['id']: row for row in trips}
    if len(rows) != len(trips):
        raise ValueError('Duplicate tripinfo ID')
    if len(rows) != requested:
        raise ValueError('Tripinfo does not cover every requested vehicle')
    teleported = set(teleport_ids)
    if not teleported <= rows.keys():
        raise ValueError('Teleport vehicle missing from tripinfo')
    inserted = {v for v, row in rows.items() if float(row['depart']) >= 0}
    finished = {v for v in inserted if begin <= float(rows[v]['arrival']) <= end}
    arrived = {v for v in finished if rows[v].get('vaporized', '') in ('', 'false', '0')}
    removed = finished - arrived
    valid = arrived - teleported
    for v in valid:
        row = rows[v]
        if (float(row['arrival']) < float(row['depart'])
                or any(not math.isfinite(float(row[k])) or float(row[k]) < 0
                       for k in ('waitingTime', 'timeLoss'))):
            raise ValueError('Invalid completed trip measurements')
    def mean(field):
        return statistics.mean(float(rows[v][field]) for v in sorted(valid)) if valid else None

    travel = statistics.mean(float(rows[v]['arrival']) - float(rows[v]['depart'])
                             for v in sorted(valid)) if valid else None
    return {
        'requested': requested, 'inserted': len(inserted), 'arrived': len(arrived),
        'valid_completed': len(valid), 'completed_metric_count': len(valid),
        'arrived_teleported': len(arrived & teleported), 'teleported': len(teleported),
        'unfinished': len(inserted - arrived), 'undeparted': requested - len(inserted),
        'removed_without_arrival': len(removed),
        'unfinished_teleported': len((inserted - arrived) & teleported),
        'average_waiting_time': mean('waitingTime'), 'average_travel_time': travel,
        'average_time_loss': mean('timeLoss'),
        'throughput': len(valid) * 3600 / (end - begin),
        'throughput_all_arrivals': len(arrived) * 3600 / (end - begin),
        'completion_rate': len(valid) / requested, 'insertion_rate': len(inserted) / requested,
        'undeparted_rate': (requested - len(inserted)) / requested,
    }


def signal_metrics(signals, begin, end):
    """Hang doi tinh tren TOAN BO mau 1s; khoang cho movement tinh ca hai dau khung gio."""
    if not signals or len(signals) != end - begin:
        raise ValueError('Incomplete signal coverage')
    count = len(signals[0]['signal'])
    green = [0] * count
    gaps = [0] * count
    longest = [0] * count
    queue_seconds = 0
    expected = begin
    for row in signals:
        start, finish = float(row['start']), float(row['end'])
        if start != expected or finish - start != 1 or len(row['signal']) != count:
            raise ValueError('Invalid signal coverage or step length')
        queue = float(row['queue'])
        if not math.isfinite(queue) or queue < 0 or not queue.is_integer():
            raise ValueError('Queue must be a nonnegative vehicle count')
        queue_seconds += queue
        for index, light in enumerate(row['signal']):
            if light == 'G':
                green[index] += 1
                gaps[index] = 0
            else:
                gaps[index] += 1
                longest[index] = max(longest[index], gaps[index])
        expected = finish
    if expected != end:
        raise ValueError('Incomplete signal coverage')
    return {
        'queue_seconds': queue_seconds, 'average_queue_length': queue_seconds / (end - begin),
        'queue_samples': len(signals), 'green_seconds_per_movement': green,
        'longest_no_green_per_movement': longest,
        'unserved_movements': [i for i, seconds in enumerate(green) if seconds == 0],
        'longest_movement_no_green': max(longest),
    }


def read_metrics(output, request_ids, begin, end):
    """Doc ket qua SUMO da dong va doi chieu cheo XML voi du lieu ghi tu TraCI."""
    output = Path(output)
    with (output / 'signals.csv').open(newline='', encoding='utf-8') as file:
        signals = list(csv.DictReader(file))
    trips = [e.attrib for e in ET.parse(output / 'tripinfo.xml').getroot().findall('tripinfo')
             if e.get('id') in request_ids]
    teleport_events = [v for row in signals for v in json.loads(row['teleports'])]
    result = trip_metrics(trips, teleport_events, len(request_ids), begin, end)
    result.update(signal_metrics(signals, begin, end))
    root = ET.parse(output / 'statistics.xml').getroot()
    vehicles = {k: int(v) for k, v in root.find('vehicles').attrib.items()}
    safety = {k: int(v) for k, v in root.find('safety').attrib.items()}
    collisions = sum(len(json.loads(row['collisions'])) for row in signals)
    teleports = int(root.find('teleports').get('total'))
    assert teleports == len(teleport_events)
    assert safety['collisions'] == collisions
    assert len(ET.parse(output / 'collisions.xml').getroot().findall('collision')) == collisions
    warnings = (output / 'warnings.log').read_text(encoding='utf-8')
    result.update(
        loaded=vehicles['loaded'], sumo_running=vehicles['running'], sumo_waiting=vehicles['waiting'],
        teleports=teleports, teleport_ids=sorted(set(teleport_events)), collisions=collisions,
        emergency_braking=safety['emergencyBraking'], emergency_stops=safety['emergencyStops'],
        emergency_braking_warnings=warnings.count('performs emergency braking'),
        emergency_stop_warnings=warnings.count('performs emergency stop'),
    )
    # Cac luot chay du 1 gio phai doi khop nhu cau, tripinfo va tong so luong xe.
    if len(request_ids) == vehicles['loaded']:
        assert result['inserted'] == vehicles['inserted']
        assert result['undeparted'] == vehicles['waiting']
        # Xe bi loai truoc horizon khong con trong bo dem running cua SUMO.
        assert result['unfinished'] - result['removed_without_arrival'] == vehicles['running']
    return result
