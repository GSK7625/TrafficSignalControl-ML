"""Offline checks: zero is not a visit count; green gaps include horizon edges."""

from collections import Counter

import pytest

import analyze_ql as analysis


def test_greedy_evidence_separates_untried_zero_from_updated_zero():
    actions = [0, 2, 4]
    state = (2, 0)
    counts = Counter({(state, 0): 3, (state, 2): 1})
    evidence = analysis.greedy_evidence(state, {state: [-2, 0, 0]}, actions, counts)
    assert evidence['action'] == 2
    assert evidence['winners'] == [2, 4]
    assert evidence['selected_updates'] == 1
    assert evidence['untried_winners'] == [4]
    assert evidence['zero_beats_negative'] is True
    assert evidence['unseen'] is False
    unknown = analysis.greedy_evidence((1, 0), {state: [-2, 0, 0]}, actions, counts)
    assert unknown['action'] == 0 and unknown['unseen'] is True
    assert unknown['winners'] == [0, 2, 4]


def test_queue_aliasing_distinguishes_edge_aggregation_and_discretization():
    encoding = {'incoming_lanes': ['a_0', 'a_1'], 'outgoing_lanes': ['b_0'],
                'incoming_groups': [{'edge': 'a', 'lanes': ['a_0', 'a_1']}],
                'outgoing_groups': [{'edge': 'b', 'lanes': ['b_0']}], 'phase_ids': [0]}
    records = [{'incoming': [x, y], 'outgoing': [0], 'phase': 0}
               for x, y in [(5, 0), (0, 5), (113, 0)]]
    bins, aliases = analysis.observation_analysis(records, encoding)
    assert bins[0]['bins'] == [0, 0, 3]
    assert bins[0]['bin2_min'] == 5 and bins[0]['bin2_max'] == 113
    assert bins[1]['constant_raw'] is True
    assert aliases['raw_observations'] == 3
    assert aliases['edge_observations'] == 2
    assert aliases['encoded_states'] == 1
    assert aliases['aggregation_collisions'] == 1
    assert aliases['discretization_collisions'] == 1


def test_signal_audit_uses_actual_green_and_checks_edges_of_horizon():
    meta = {'begin': 10, 'end': 15, 'actions': [0, 2],
            'environment': {'runtime_green_states': [[0, 'Gr'], [2, 'rG']]},
            'controlled_links': [[0, 'a_0', 'b_0', ':v0'], [1, 'c_0', 'd_0', ':v1']]}
    rows = [dict(start=10+i, end=11+i, signal=signal, stage=stage, action=str(action),
                 queue='2', decision=str(decision))
            for i, (signal, stage, action, decision) in enumerate([
                ('Gr', 'green', 0, 0), ('Gr', 'green', 0, 0),
                ('yr', 'yellow', 2, 1), ('rr', 'all_red', 2, 1),
                ('rG', 'green', 2, 1)])]
    metrics, green = analysis.audit_signals(rows, meta)
    assert green == {0: 2, 2: 1}
    assert metrics['green_seconds_per_movement'] == [2, 1]
    assert metrics['longest_no_green_per_movement'] == [3, 4]
    assert metrics['queue_seconds'] == 10
    rows[-1]['signal'] = 'Gr'
    with pytest.raises(ValueError, match='green'):
        analysis.audit_signals(rows, meta)


def test_missing_or_empty_input_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        analysis.read_csv(tmp_path / 'missing.csv')
    empty = tmp_path / 'empty.csv'
    empty.write_text('state,action\n')
    with pytest.raises(ValueError, match='Empty'):
        analysis.read_csv(empty)
