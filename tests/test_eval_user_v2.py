import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_evaluate_hierarchical_prompt import load_evaluator, load_conflict_evaluator
ev = load_evaluator()
from prompt_v2 import build_prompt, fingerprint, response_errors, PROMPT_VERSION, PROMPT_SHA256, TEMPLATE
from check_v2_run import read_manifest


def valid(conflict=False):
    return {'label': 'UNANSWERABLE' if conflict else 'ANSWERABLE', 'answer': 'R' if conflict else 'A',
            'decision_probabilities': {'ANSWERABLE': 0.1 if conflict else 0.9, 'UNANSWERABLE': 0.9 if conflict else 0.1},
            'reject_reason': 'CONFLICTING_EVIDENCE' if conflict else None,
            'evidence': [{'group_id': 'G'+str(i), 'evidence_seconds': [[1, 1]], 'description': 'object '+str(i)}
                         for i in range(2 if conflict else 1)], 'reason': 'Two supported choices.' if conflict else 'One supported choice.'}


class V2Tests(unittest.TestCase):
    def test_no_pipe_and_all_three_examples_are_valid_json(self):
        self.assertEqual(PROMPT_VERSION, 'v2.1')
        self.assertNotIn('|', TEMPLATE)
        self.assertNotIn('start_second', TEMPLATE)
        examples = [json.loads(line) for line in TEMPLATE.splitlines() if line.startswith('{"label":')]
        self.assertEqual(len(examples), 3)
        for example in examples:
            self.assertEqual(response_errors(example), [])
        self.assertEqual([x['reject_reason'] for x in examples], [None, 'INSUFFICIENT_EVIDENCE', 'CONFLICTING_EVIDENCE'])
        self.assertEqual(examples[2]['evidence'][0]['evidence_seconds'], examples[2]['evidence'][1]['evidence_seconds'])

    def test_exact_template_and_shared_entry_points(self):
        choices = ['carrot', 'chair', 'cup', 'hat', 'fork']
        actual = build_prompt('Which {object}?', choices)
        expected = TEMPLATE.replace('{question}', 'Which {object}?')
        for letter, choice in zip('ABCDE', choices):
            expected = expected.replace('{option_'+letter+'}', choice)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, ev.build_instruction('Which {object}?', choices, task='classify'))
        self.assertEqual(actual, load_conflict_evaluator().build_prompt('Which {object}?',
            [{'option_id': k, 'answer': v} for k, v in zip('ABCDE', choices)]))
        for forbidden in ['option_probabilities', 'reject_reason_probabilities', 'quality', 'overall_resolvability']:
            self.assertNotIn(forbidden, actual)
        with self.assertRaises(ValueError):
            build_prompt('q', ['x'])

    def test_overlap_conflict_and_empty_insufficient(self):
        self.assertEqual(response_errors(valid(True)), [])
        item = valid(True)
        item.update(reject_reason='INSUFFICIENT_EVIDENCE', evidence=[])
        self.assertEqual(response_errors(item), [])
        self.assertEqual(response_errors(valid()), [])

    def test_conflict_key_survives_metadata_normalization(self):
        original = {'id': 'scene1', 'source_dataset': 'etbench', 'evidence_condition': 'conflicting'}
        result = {**original, 'evidence_condition': 'conflict', 'video_id': 'v', 'qid': 'scene1'}
        self.assertEqual(ev._record_key(original), ev._record_key(result))

    def test_invalid_outputs_not_silently_accepted(self):
        for mutate in [lambda x: x.update(answer='R'), lambda x: x.pop('reject_reason'),
                       lambda x: x.update(decision_probabilities={'ANSWERABLE': 1, 'UNANSWERABLE': 1}),
                       lambda x: x.update(decision_probabilities={'ANSWERABLE': float('nan'), 'UNANSWERABLE': 0}),
                       lambda x: x['evidence'][0].update(evidence_seconds=[[3, 1]]),
                       lambda x: x.update(reason='')]:
            item = valid(); mutate(item)
            self.assertTrue(response_errors(item))
        item = valid(True); item['evidence'].pop()
        self.assertTrue(response_errors(item))

    def test_removed_metrics_are_unavailable_not_zero(self):
        scored = load_conflict_evaluator().score(valid(True), ['A', 'B'])
        self.assertTrue(scored['label_correct'])
        for key in ['option_f1', 'option_set_exact', 'expected_option_probability_mass']:
            self.assertIsNone(scored[key])

    def test_real_resume_flow_and_active_jsonl_cleanup(self):
        entry = {'id': 'qa1', 'record_id': 'record1', 'question': 'q', 'choices': list('abcde')}
        calls = []
        def fake(idx, item, *args, **kwargs):
            calls.append(idx)
            return {'status': 'ok', 'idx': idx, 'video_id': 'v', 'qid': 'qa1', 'short_status': 'ANSWERABLE',
                    'result': {**item, 'prompt_version': PROMPT_VERSION, 'model_parsed': valid()}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(ev, 'process_entry', fake):
            output = Path(tmp)/'merged.classify.json'
            def run(items, **extra):
                ev.run_evaluation(items, Path(tmp), output, model_tag='test', task='classify', workers=1, **extra)
            run([entry]); self.assertEqual(len(calls), 1)
            run([entry]); self.assertEqual(len(calls), 1)
            entry['choices'][0] = 'changed'; run([entry]); self.assertEqual(len(calls), 2)
            run([entry], num_frames=16); self.assertEqual(len(calls), 3)
            data = json.loads(output.with_suffix('.jsonl').read_text())
            data['model_parsed']['decision_probabilities'] = {}
            output.with_suffix('.jsonl').write_text(json.dumps(data)+'\n'+json.dumps({'record_id':'obsolete'})+'\n')
            run([entry], num_frames=16); self.assertEqual(len(calls), 4)
            rows = [json.loads(x) for x in output.with_suffix('.jsonl').read_text().splitlines()]
            self.assertEqual([x['record_id'] for x in rows], ['record1'])
            self.assertTrue(list(Path(tmp).glob('*.history.*')))

    def test_manifest_count_is_dynamic_and_ids_unique(self):
        row = {'record_id': 'one', 'choices': list('abcde'), 'evidence_condition': 'conflicting', 'expected_behavior': 'reject'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'manifest.jsonl'
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(len(read_manifest(path)), 1)
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError): read_manifest(path)


if __name__ == '__main__':
    unittest.main()
