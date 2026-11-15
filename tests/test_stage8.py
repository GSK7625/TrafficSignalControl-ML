"""Small checks for Stage 8's offline repetition evidence."""

import json

import pytest


@pytest.mark.parametrize('pattern,first_override', [('positive', 2), ('zero', None), ('reset', 4)])
def test_guard_serves_overdue_observed_edges_through_step_and_keeps_q_frozen(pattern, first_override):
    from types import SimpleNamespace
    from copy import deepcopy
    from evaluate import greedy_steps
    from sumo_env import SumoEnv
    import q_learning as ql
    env = SumoEnv('cologne1')
    encoding = ql.build_encoding(env, (0, 4, 14))
    now, chosen = [0], []
    env.current_phase = 0
    env.conn = SimpleNamespace(simulation=SimpleNamespace(getTime=lambda: now[0]))
    def get_state():
        zero = pattern == 'zero' or (pattern == 'reset' and 30 <= now[0] < 60)
        return dict(incoming=((0 if zero else 5),)*8, outgoing=(0,)*8, phase=env.current_phase)

    env.get_state = get_state

    def step(action):
        chosen.append(action)
        start = now[0]
        now[0] += 30
        env.current_phase = action
        return env.get_state(), -10, now[0] >= 180, dict(
            action=action, start=start, end=now[0],
            samples=[dict(start=start, end=now[0], stage='green', signal=env.green_states[action])])

    env.step = step
    table = {(2, 2, 2, 2, 0, 0, 0, 0, a): [-1, -2, -3, -4, -5, -6, -7, -8]
             for a in env.actions}
    before = deepcopy(table)
    rows = list(greedy_steps(env, table, encoding, guard_seconds=60))
    if first_override is None:
        assert chosen == [0]*6 and all(row['guard_edge'] is None for row in rows)
    else:
        assert chosen[:first_override] == [0]*first_override
        assert chosen[first_override] == 1  # Oldest edge -32038056#3, best eligible Q.
        assert rows[first_override]['guard_age'] == 60 and rows[first_override]['guard_edge'] is not None
    assert all(a in env.actions for a in chosen)
    assert table == before
    assert [r['action'] for r in rows] == chosen


@pytest.mark.parametrize('limit', [0, -1, float('nan'), True])
def test_invalid_guard_limit_rejected(limit):
    from evaluate import greedy_steps
    with pytest.raises(ValueError, match='guard_seconds'):
        next(greedy_steps(None, {}, {}, guard_seconds=limit))


def test_new_encoding_guard_real_sumo_uses_common_metrics_and_never_changes_model(tmp_path):
    import q_learning as ql
    from evaluate import run_evaluation
    from sumo_env import SumoEnv
    metadata = ql.make_metadata(SumoEnv('cologne1'), episodes=15, alpha=.1, gamma=.95,
                               epsilon_start=1, epsilon_min=.05, exploration_seed=7,
                               queue_bins=(0, 4, 14))
    model = tmp_path / 'new.json'
    ql.save_q_table(model, {}, metadata)
    before = model.read_bytes()
    result = run_evaluation('cologne1', 'QL', 42, tmp_path/'guard', model,
                            horizon=150, queue_bins=(0, 4, 14), guard_seconds=1)
    assert model.read_bytes() == before
    assert result['model_sha256_before'] == result['model_sha256_after']
    assert result['queue_samples'] == 150 and result['process_closed']
    assert result['average_queue_length'] == result['queue_seconds']/150
    import analyze_ql as audit
    rows = audit.read_csv(tmp_path/'guard/actions.csv')
    assert any(json.loads(row['guard_edge']) is not None for row in rows)


def test_repetition_counts_green_time_and_keeps_loop_q_evidence():
    from stage8_summary import repetition_evidence
    rows = []
    for i, (action, green, elapsed) in enumerate([(7, 30, 40), (7, 30, 30), (2, 20, 30)]):
        rows.append(dict(action=str(action), green=str(green), elapsed=str(elapsed),
                         start=str(i*40), end=str(i*40+elapsed), encoded_state='[2, 7]',
                         q_values='[-2, -1]', state=json.dumps(dict(incoming=[5+i], outgoing=[0], phase=7))))
    result, loops = repetition_evidence(rows)
    assert result == dict(repeated_choices=1, longest_same_action_decisions=2,
                          longest_same_action_green_seconds=60)
    assert loops[0]['state'] == [2, 7] and loops[0]['q_values'] == [-2, -1]
    assert loops[0]['distinct_lane_queues'] == 3
    with pytest.raises(ValueError, match='Empty'):
        repetition_evidence([])


def test_stage8_matrix_rejects_missing_duplicates_unknown_labels_and_old_test_seeds():
    from stage8_summary import verify_runs
    rows = []
    for label in ('QL-Original', 'QL-NewState', 'QL-NewState-Guard'):
        for partition, seeds in [('validation', (42, 12345)), ('test', (41001, 41002, 41003))]:
            for seed in seeds:
                for repeat in (1, 2):
                    rows.append(dict(map='cologne1', label=label, partition=partition, seed=seed, repeat=repeat))
    for label in ('QL-NewState-ep05', 'QL-NewState-ep10'):
        rows.extend(dict(map='cologne1', label=label, partition='checkpoint_validation', seed=s, repeat=1)
                    for s in (42, 12345))
    rows.extend(dict(map='cologne1', label='QL-NewState', partition='diagnostic', seed=31001, repeat=r)
                for r in (1, 2))
    verify_runs(rows)
    with pytest.raises(ValueError, match='matrix'):
        verify_runs(rows[:-1])
    with pytest.raises(ValueError, match='matrix'):
        verify_runs(rows + [rows[0]])
    rows[0] = dict(rows[0], label='typo')
    with pytest.raises(ValueError, match='matrix'):
        verify_runs(rows)
    rows[0] = dict(rows[0], label='QL-Original')
    rows[4] = dict(rows[4], seed=31001)
    with pytest.raises(ValueError, match='matrix'):
        verify_runs(rows)


def test_offline_guard_replay_rejects_omitted_override_and_false_demand():
    from copy import deepcopy
    from stage8_summary import audit_guard_decisions
    encoding = dict(incoming_lanes=['a_0', 'b_0'], incoming_groups=[
        dict(edge='a', lanes=['a_0']), dict(edge='b', lanes=['b_0'])])
    metadata = dict(guard_seconds=30, actions=[0, 2], controlled_links=[
        [0, 'a_0', 'out_0', 'via0'], [1, 'b_0', 'out_1', 'via1']],
        environment=dict(runtime_green_states=[[0, 'Gr'], [2, 'rG']]))
    rows = [dict(decision=str(i), start=str(30*i), action=str(a), greedy_action='0',
                 q_values='[-1,-2]', guard_edge=json.dumps(edge), guard_age=str(age),
                 state=json.dumps(dict(incoming=[5, 5])))
            for i, a, edge, age in [(0, 0, None, 0), (1, 2, 'b', 30)]]
    signals = [dict(decision='0', signal='Gr'), dict(decision='1', signal='rG')]
    audit_guard_decisions(rows, signals, metadata, encoding)
    bad = deepcopy(rows)
    bad[1]['guard_edge'] = 'null'
    with pytest.raises(AssertionError):
        audit_guard_decisions(bad, signals, metadata, encoding)
    bad = deepcopy(rows)
    bad[1]['state'] = json.dumps(dict(incoming=[5, 0]))
    with pytest.raises(AssertionError):
        audit_guard_decisions(bad, signals, metadata, encoding)
