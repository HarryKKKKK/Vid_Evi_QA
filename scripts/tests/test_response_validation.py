"""Regression cases for strict output validation and full-denominator reporting."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from prompt_v2 import (PROMPT_VERSION, PROMPT_SHA256, RESPONSE_VALIDATION_VERSION,
                       TEMPLATE, response_errors, reusable)
from test_evaluate_hierarchical_prompt import load_conflict_evaluator


def example(index):
    return [json.loads(line) for line in TEMPLATE.splitlines()
            if line.startswith('{"label":')][index]


@pytest.mark.parametrize('group_id', [[], {}, ['G1'], 1, None])
def test_malformed_group_ids_report_errors_instead_of_crashing(group_id):
    value = example(0)
    value['evidence'][0]['group_id'] = group_id
    assert 'invalid or duplicate group_id' in response_errors(value)


def test_two_groups_for_the_same_option_are_not_a_valid_conflict():
    value = example(2)
    value['evidence'][1]['description'] = 'Option A: another frame of the same answer.'
    errors = response_errors(value)
    assert 'duplicate supported option' in errors
    assert 'conflict requires separate supported-answer groups' in errors


@pytest.mark.parametrize('change', ['empty', 'mismatched', 'multiple', 'unhashable'])
def test_answerable_decision_agrees_with_the_supported_option(change):
    value = example(0)
    if change == 'empty':
        value['evidence'] = []
    elif change == 'mismatched':
        value['answer'] = 'A'  # Evidence still supports C.
    elif change == 'multiple':
        value['evidence'].append(example(2)['evidence'][1])
    else:
        value['answer'] = []
    assert response_errors(value)


def test_insufficient_decision_cannot_claim_a_supported_option():
    value = example(1)
    value['evidence'] = example(0)['evidence']
    assert 'insufficient rejection contains supported-option evidence' in response_errors(value)


def test_huge_numeric_output_is_invalid_without_overflow():
    value = example(0)
    value['evidence'][0]['evidence_seconds'] = [[0, 10**400]]
    assert 'invalid evidence interval' in response_errors(value)


def test_empty_description_after_option_prefix_is_invalid():
    value = example(0)
    value['evidence'][0]['description'] = 'Option C: '
    assert 'empty evidence description' in response_errors(value)


def test_conflict_score_does_not_reward_invalid_output():
    evaluator = load_conflict_evaluator()
    value = example(2)
    assert evaluator.score(value, ['A', 'D'])['label_correct'] is True
    value['evidence'][1]['description'] = 'Option A: duplicate support.'
    scored = evaluator.score(value, ['A', 'D'])
    assert scored['decision_matches_gold'] is True
    assert scored['label_correct'] is False
    assert scored['schema_errors']
    assert scored['response_validation_version'] == RESPONSE_VALIDATION_VERSION
    assert evaluator.score([], ['A', 'D'])['label_correct'] is False


def test_resume_revalidates_old_output_without_invalidating_all_valid_records():
    row = {'prompt_version': PROMPT_VERSION, 'prompt_sha256': PROMPT_SHA256,
           'input_fingerprint': 'same-input', 'model_parsed': example(2)}
    assert reusable(row, 'same-input')  # Valid older records need no version stamp.
    row['model_parsed']['evidence'][1]['description'] = 'Option A: duplicate support.'
    assert not reusable(row, 'same-input')


@pytest.mark.parametrize('valid', [True, False])
def test_merge_status_includes_schema_validity_and_keeps_invalid_rows(tmp_path, valid):
    release = tmp_path / 'release'
    shard = release / 'results/model/shard_000'
    shard.mkdir(parents=True)
    manifest = {'record_id': 'record-1', 'evidence_condition': 'sufficient',
                'choices': list('abcde'), 'expected_behavior': 'answer'}
    (release / 'smoke.jsonl').write_text(json.dumps(manifest) + '\n', encoding='utf-8')
    parsed = example(0)
    if not valid:
        parsed['evidence'] = []
    row = {**manifest, 'prompt_version': PROMPT_VERSION, 'prompt_sha256': PROMPT_SHA256,
           'model_parsed': parsed}
    (shard / 'merged.classify.json').write_text(json.dumps([row]), encoding='utf-8')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/eval/open_v2/merge_full.py'),
                             str(release), 'model'], capture_output=True, text=True, timeout=30)
    output = release / 'results/model'
    report = json.loads((output / 'merged.classify.validation.json').read_text())
    assert (result.returncode == 0) is valid, result.stderr
    assert (report['status'] == 'complete') is valid
    assert report['invalid_records'] == (0 if valid else 1)
    assert report['result_records'] == report['expected_records'] == 1
    assert report['response_validation_version'] == RESPONSE_VALIDATION_VERSION
    assert json.loads((output / 'merged.classify.json').read_text()) == [row]
