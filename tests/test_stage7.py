"""Cac kiem tra offline nho cho bang chung Stage 7 va quy tac chon model khai bao truoc."""

from collections import Counter
from pathlib import Path

import pytest

import analyze_ql as audit
import q_learning as ql
import stage7_summary as stage7


def test_replay_accepts_explicit_training_folder_without_changing_stage5():
    folder = Path('reports/stage5/train/ingolstadt1')
    payload = audit.read_json(folder / 'ingolstadt1_q_table.json')
    mapping = audit.read_json(Path('reports/stage6/runs/ingolstadt1/FT/seed_42/metadata.json'))
    old = audit.replay_training(Path('.'), 'ingolstadt1', payload, mapping)
    explicit = audit.replay_training(Path('.'), 'ingolstadt1', payload, mapping, training_dir=folder)
    assert old == explicit


def test_episode_coverage_counts_updated_zero_and_new_states():
    rows = [dict(encoded_state='[0]', action='0'), dict(encoded_state='[1]', action='2')]
    visits, updates = set(), Counter()
    first = stage7.coverage(rows, visits, updates, [0, 2, 4])
    assert first == dict(new_states=2, encountered_states=2, updated_pairs=2,
                         possible_pairs=6, coverage=2/6)
    second = stage7.coverage(rows[:1], visits, updates, [0, 2, 4])
    assert second['new_states'] == 0 and second['updated_pairs'] == 2
    assert updates[(0,), 0] == 2
    with pytest.raises(ValueError, match='action'):
        stage7.coverage([dict(encoded_state='[0]', action='1')], visits, updates, [0, 2, 4])


def test_selection_uses_validation_service_before_reward_and_refuses_test():
    rows = [dict(label=label, seed=seed, completion_rate=completion,
                 movement_coverage=1, teleports=teleports, longest_movement_no_green=40)
            for label, completion, teleports in [('QL-15ep-e7', .8, 0), ('QL-15ep-e17', .9, 50)]
            for seed in (42, 12345)]
    assert stage7.choose_model(rows) == 'QL-15ep-e17'
    for row in rows:
        row['completion_rate'] = .9
        row['teleports'] = 0
    assert stage7.choose_model(rows) == 'QL-15ep-e7'
    rows[-1]['seed'] = 31001
    with pytest.raises(ValueError, match='validation'):
        stage7.choose_model(rows)


def test_fifteen_episode_epsilon_exact_schedule():
    values = [ql.episode_epsilon(i, 15, 1, .05) for i in range(1, 16)]
    assert values == pytest.approx([1 - .95*i/14 for i in range(15)])
    assert values[0] == 1 and values[-1] == pytest.approx(.05)
    assert all(a > b for a, b in zip(values, values[1:]))


def test_checkpoint_changes_use_common_states_and_first_max():
    result = stage7.checkpoint_changes({(0,): [-1, -1], (1,): [-2, -1]},
                                      {(0,): [-2, -1], (1,): [-3, -2], (2,): [0, 0]})
    assert result == dict(common_states=2, greedy_changes=1, greedy_change_rate=.5,
                          max_q_change=1)


def test_run_matrix_rejects_empty_missing_or_duplicate_evidence():
    selection = {'models': {name: dict(label='QL-15ep-e7', sha256='abc') for name in audit.MAPS}}
    with pytest.raises(ValueError, match='matrix'):
        stage7.verify_run_matrix([], selection)
    rows = []
    for name in audit.MAPS:
        for partition, labels, seeds, repeats in (
            ('stage6', ['FT', 'MP', 'QL-3ep'], audit.SEEDS, [1]),
            ('validation', ['QL-3ep', 'QL-15ep-e7', 'QL-15ep-e17'], [42, 12345], [1]),
            ('test', ['FT', 'MP', 'QL-3ep', 'QL-15ep-e7'], [31001, 31002, 31003], [1, 2])):
            for label in labels:
                for seed in seeds:
                    for repeat in repeats:
                        rows.append(dict(partition=partition, map=name, label=label, seed=seed,
                                         repeat=repeat, completion_rate=.9 if label.endswith('e7') else .8,
                                         movement_coverage=1, teleports=0, longest_movement_no_green=40,
                                         model_sha256_before='abc', model_sha256_after='abc'))
    stage7.verify_run_matrix(rows, selection)
    with pytest.raises(ValueError, match='matrix'):
        stage7.verify_run_matrix(rows[:-1], selection)
    with pytest.raises(ValueError, match='matrix'):
        stage7.verify_run_matrix(rows + rows[:1], selection)
    rows[-1]['model_sha256_after'] = 'changed'
    with pytest.raises(AssertionError):
        stage7.verify_run_matrix(rows, selection)
