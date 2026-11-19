"""Dieu tra FT cho Stage 6.1. Khong train hay thay doi cau hinh mo phong."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import sumolib
import traci
import traci.constants as tc

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation_metrics import SIGNAL_FIELDS, read_metrics, requested_trips, write_signal_sample
from ft import fixed_time
from q_learning import source_hashes
from sumo_env import ROOT, SumoEnv


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def first_difference(left, right):
    """So sanh gia tri chinh xac theo thu tu nhan qua, khong lam tron so thuc."""
    fields = ('time', 'signal', 'departed', 'active', 'vehicles', 'queue',
              'teleports', 'collisions')
    for index, (a, b) in enumerate(zip(left, right)):
        for field in fields:
            if a.get(field) != b.get(field):
                return dict(index=index, time=a['time'], field=field,
                            left=a.get(field), right=b.get(field))
    if len(left) != len(right):
        return dict(index=min(len(left), len(right)), field='length',
                    left=len(left), right=len(right))
    return None


def signal_commands(samples):
    """Khoi phuc cac lenh dat den cua reset + _run_signal, gom ca giai toa keo dai tung giay."""
    commands = [(samples[0]['start'], samples[0]['signal'])]
    previous = None
    red_start = None
    for row in samples:
        key = (row['decision'], row['stage'])
        if key != previous:
            commands.append((row['start'], row['signal']))
            red_start = row['start'] if row['stage'] == 'all_red' else None
        elif row['stage'] == 'all_red' and row['start'] >= red_start + 2:
            commands.append((row['start'], row['signal']))
        previous = key
    return commands


def replay_signals(source, output, *, fcd=True, save_rng=False):
    """Phat lai dung cac lenh dat den/buoc mo phong ma khong qua SumoEnv, doc hang doi hay observer.

    Cach nay co lap khoang thoi gian truoc khi tin hieu lech do phan hoi hang doi.
    Moc thoi gian FCD danh dau luc BAT DAU mot buoc; anh chup TraCI danh dau luc KET THUC buoc do.
    """
    source, output = Path(source), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source / 'metadata.json').read_text())
    actual_binary_hash = hashlib.sha256(Path(metadata['binary']).read_bytes()).hexdigest()
    assert actual_binary_hash == metadata['binary_sha256']
    assert source_hashes(SumoEnv(metadata['map'])) == metadata['input_hashes']
    samples = json.loads((source / 'samples.json').read_text())
    commands = signal_commands(samples)
    argv = (source / 'command.txt').read_text().splitlines()
    for option in ('--log', '--error-log', '--statistic-output', '--collision-output', '--tripinfo-output'):
        if option in argv:
            i = argv.index(option) + 1
            argv[i] = str(output / Path(argv[i]).name)
    if fcd:
        # shortcut: cua so chan doan Cologne1; chinh lai neu dieu tra replay khac.
        argv += ['--fcd-output', str(output / 'fcd.xml.gz'),
                 '--device.fcd.begin', '25845', '--precision', '12',
                 '--fcd-output.acceleration', 'true']
    if save_rng:
        argv += ['--save-state.times', '25853', '--save-state.files', str(output / 'state.xml.gz'),
                 '--save-state.rng', 'true', '--save-state.precision', '17']
    label = f'replay-{output.name}'
    conn = None
    try:
        traci.start(argv, label=label, doSwitch=False)
        conn = traci.getConnection(label)
        process = conn._process
        (output / 'metadata.json').write_text(json.dumps(dict(
            argv=process.args, source=str(source), commands=commands,
            binary_sha256=actual_binary_hash, input_hashes=metadata['input_hashes'],
            fcd=fcd, save_rng=save_rng,
            note='fixed recorded schedule, no clearance/metric/vehicle getters'), indent=2))
        index = 0
        for timestamp in range(int(samples[0]['start']), int(samples[-1]['end'])):
            while index < len(commands) and commands[index][0] == timestamp:
                conn.trafficlight.setRedYellowGreenState(metadata['tls_id'], commands[index][1])
                index += 1
            conn.simulationStep()
        assert conn.simulation.getTime() == samples[-1]['end']
    finally:
        if conn is not None:
            conn.close()
    assert process.poll() == 0 and index == len(commands)


def run_probe(map_name, seed, horizon, output, mode, *, observe=False,
              detail_window=None, reverse_reads=False, watched=()):
    """A=khong tripinfo; B=co tripinfo; C=co tripinfo va cac metric CSV hien tai.

    Moi che do deu giu nguyen cac lenh doc giai toa bat buoc va lay mau cua SumoEnv.
    Observer chi them getter/subscription, khong bao gio step hay dat den/xe.
    """
    if mode not in ('A', 'B', 'C'):
        raise ValueError('mode must be A, B or C')
    output = Path(output).resolve()
    env = SumoEnv(map_name, output, seed, tripinfo=mode != 'A')
    if type(horizon) is not int or not 0 < horizon <= env.end - env.begin:
        raise ValueError('Invalid horizon')
    env.end = env.begin + horizon
    output.mkdir(parents=True, exist_ok=False)
    binary = Path(sumolib.checkBinary('sumo')).resolve()
    metadata = dict(map=map_name, seed=seed, mode=mode, observe=observe,
                    reverse_reads=reverse_reads, detail_window=detail_window,
                    watched=list(watched),
                    begin=env.begin, end=env.end, python=sys.version,
                    python_executable=sys.executable,
                    packages={p: importlib.metadata.version(p) for p in
                              ('traci', 'sumolib', 'pytest', 'numpy', 'matplotlib')},
                    traci_module=traci.__file__, sumolib_module=sumolib.__file__,
                    binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                    code_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / 'src').glob('*.py'))},
                    input_hashes=source_hashes(env), actions=env.actions,
                    tls_id=env.tls_id, incoming_lanes=env.incoming_lanes)
    samples, trace, details, departures, watched_properties = [], [], [], [], {}
    variables = (tc.VAR_LANE_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED,
                 tc.VAR_ACCELERATION, tc.VAR_POSITION, tc.VAR_ANGLE)
    try:
        env.reset()
        conn = env.conn
        process = conn._process
        metadata.update(argv=process.args, sumo_version=conn.getVersion(),
                        step_length=conn.simulation.getDeltaT(),
                        internal_foes={lane: conn.lane.getInternalFoes(lane)
                                       for lane in env.internal_lanes})
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        original_step = conn.simulationStep

        def observed_step():
            original_step()
            now = conn.simulation.getTime()
            active = sorted(conn.vehicle.getIDList())
            departed = list(conn.simulation.getDepartedIDList())
            for vehicle in reversed(departed) if reverse_reads else departed:
                conn.vehicle.subscribe(vehicle, variables)
                departures.append(dict(time=now, id=vehicle,
                                       route=conn.vehicle.getRoute(vehicle),
                                       speed_factor=conn.vehicle.getSpeedFactor(vehicle)))
            subscriptions = conn.vehicle.getAllSubscriptionResults()
            states = {v: [subscriptions[v][key] for key in variables] for v in active}
            lanes = reversed(env.incoming_lanes) if reverse_reads else env.incoming_lanes
            queues = {lane: conn.lane.getLastStepHaltingNumber(lane) for lane in lanes}
            trace.append(dict(time=now, signal=conn.trafficlight.getRedYellowGreenState(env.tls_id),
                              departed=departed, active=digest(active), vehicles=digest(states),
                              queue=[queues[lane] for lane in env.incoming_lanes],
                              teleports=list(conn.simulation.getStartingTeleportIDList()),
                              collisions=[vars(c) for c in conn.simulation.getCollisions()]))
            if detail_window and detail_window[0] <= now <= detail_window[1]:
                extra = {}
                for vehicle in watched:
                    if vehicle not in active:
                        continue
                    if vehicle not in watched_properties:
                        watched_properties[vehicle] = dict(
                            route=conn.vehicle.getRoute(vehicle), length=conn.vehicle.getLength(vehicle),
                            min_gap=conn.vehicle.getMinGap(vehicle), tau=conn.vehicle.getTau(vehicle),
                            decel=conn.vehicle.getDecel(vehicle),
                            emergency_decel=conn.vehicle.getEmergencyDecel(vehicle))
                    extra[vehicle] = dict(leader=conn.vehicle.getLeader(vehicle, 100),
                                          next_tls=conn.vehicle.getNextTLS(vehicle),
                                          junction_foes=conn.vehicle.getJunctionFoes(vehicle, 100))
                details.append(dict(time=now, states=states, watched=extra))

        if observe:
            conn.simulationStep = observed_step
        with (output / 'signals.csv').open('x', newline='', encoding='utf-8') as sf:
            writer = csv.DictWriter(sf, fieldnames=SIGNAL_FIELDS)
            writer.writeheader()
            for decision, info in enumerate(fixed_time(env)):
                for row in info['samples']:
                    samples.append(dict(decision=decision, action=info['action'], **row))
                    if mode == 'C':
                        write_signal_sample(writer, decision, info['action'], row)
        assert conn.simulation.getTime() == env.end
    finally:
        env.close()
    assert process.poll() == 0
    assert source_hashes(env) == metadata['input_hashes']
    statistics = ET.parse(output / 'statistics.xml').getroot()
    result = dict(begin=env.begin, end=env.end, queue_seconds=sum(s['queue'] for s in samples),
                  samples_sha256=digest(samples), trace_sha256=digest(trace),
                  vehicles=statistics.find('vehicles').attrib,
                  safety=statistics.find('safety').attrib,
                  teleports=statistics.find('teleports').attrib, process_exit=process.returncode)
    if mode == 'C':
        result['metrics'] = read_metrics(output, requested_trips(env), env.begin, env.end)
        assert result['metrics']['queue_seconds'] == result['queue_seconds']
    for name, value in [('samples', samples), ('trace', trace), ('details', details),
                        ('departures', departures), ('watched_properties', watched_properties),
                        ('result', result)]:
        (output / f'{name}.json').write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    print(f'{output.name}: queue={result["queue_seconds"]}, safety={result["safety"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', choices=('cologne1', 'ingolstadt1'), default='cologne1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--horizon', type=int, default=700)
    parser.add_argument('--mode', choices=('A', 'B', 'C'), default='A')
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--reverse-reads', action='store_true')
    parser.add_argument('--detail-window', type=float, nargs=2)
    parser.add_argument('--watched', nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run_probe(args.map, args.seed, args.horizon, args.output, args.mode,
              observe=args.observe, detail_window=args.detail_window,
              reverse_reads=args.reverse_reads, watched=args.watched)


if __name__ == '__main__':
    main()
