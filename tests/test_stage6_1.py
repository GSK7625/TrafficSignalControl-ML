"""Small checks for the investigation tools, not a determinism guarantee."""

def test_first_difference_keeps_time_and_component_order():
    from stage6_1_probe import first_difference
    a = [dict(time=1, signal='G', departed=['a'], active='x', vehicles='y', queue=[0]),
         dict(time=2, signal='r', departed=[], active='x', vehicles='y', queue=[1])]
    b = [dict(a[0]), dict(a[1], vehicles='z', queue=[2])]
    assert first_difference(a, b) == dict(index=1, time=2, field='vehicles', left='y', right='z')
    assert first_difference(a, a) is None
    assert first_difference(a, a[:1])['field'] == 'length'


def test_signal_commands_preserve_repeated_clearance_and_reset():
    from stage6_1_probe import signal_commands
    rows = [dict(start=i, decision=d, stage=s, signal=v) for i, d, s, v in
            [(0, 0, 'green', 'G'), (1, 1, 'yellow', 'y'),
             (2, 1, 'all_red', 'r'), (3, 1, 'all_red', 'r'),
             (4, 1, 'all_red', 'r'), (5, 1, 'green', 'G')]]
    assert signal_commands(rows) == [(0, 'G'), (0, 'G'), (1, 'y'),
                                     (2, 'r'), (4, 'r'), (5, 'G')]


def test_replay_matches_actual_environment_setters_with_clearance_and_cutoff():
    from ft import fixed_time
    from stage6_1_probe import signal_commands
    from test_sumo_env import clock_env
    for horizon in (37, 39, 44, 70):
        env = clock_env()
        env.end = env.begin + horizon
        env._junction_vehicles = lambda: ('tail',) if env.conn.time < env.begin + 43 else ()
        calls = []
        original_setter = env.conn.trafficlight.setRedYellowGreenState

        def record_setter(tls_id, signal):
            calls.append((env.conn.time, signal))
            original_setter(tls_id, signal)

        env.conn.trafficlight.setRedYellowGreenState = record_setter
        env.conn.trafficlight.setRedYellowGreenState(env.tls_id, env.green_states[0])
        samples = [dict(decision=i, **sample) for i, info in enumerate(fixed_time(env))
                   for sample in info['samples']]
        assert signal_commands(samples) == calls


def test_probe_records_real_clock_and_does_not_overwrite(tmp_path):
    import pytest
    from stage6_1_probe import run_probe, replay_signals
    output = tmp_path / 'probe'
    result = run_probe('ingolstadt1', 42, 47, output, 'C', observe=True, watched=['absent'])
    assert result['end'] == 57647
    assert result['metrics']['queue_samples'] == 47
    assert result['metrics']['requested'] == result['metrics']['inserted'] + result['metrics']['undeparted']
    import json
    assert json.loads((output / 'metadata.json').read_text())['watched'] == ['absent']
    replay = tmp_path / 'replay'
    replay_signals(output, replay, fcd=False)
    import xml.etree.ElementTree as ET
    original = ET.parse(output / 'statistics.xml').getroot()
    repeated = ET.parse(replay / 'statistics.xml').getroot()
    for tag in ('vehicles', 'safety', 'teleports'):
        assert original.find(tag).attrib == repeated.find(tag).attrib
    with pytest.raises(FileExistsError):
        run_probe('ingolstadt1', 42, 47, output, 'C', observe=True)
