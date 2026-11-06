"""Read-only post-publication checks on either host."""
import argparse
from pathlib import Path
from collections import Counter
import apply_personal_scene_repairs as core

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', required=True, type=Path)
args = p.parse_args()
root = args.root.resolve()
base = root/'reports/final_benchmark_aligned/scene_manual_v3'
release = base/'manual_scene_v3_20260902_spatial01'
revised = core.read_rows(release/'scene_all_revised_with_types.jsonl')
expected = {r['id']:r for r in revised}
assert len(expected) == 15
transaction = core.read_rows(release/'transaction.json')
assert transaction['status'] == 'committed'
for entry in transaction['files']:
    path = root/entry['path']
    assert core.digest(path.read_bytes()) == entry['after_sha256'], str(path)
    if entry['before_sha256'] is not None:
        assert core.digest((release/'backup'/entry['path']).read_bytes()) == entry['before_sha256']
for rel in ['final/conflict_qa_five_options.jsonl','final/all_unique_qa.jsonl',
            'final/all_condition_records.jsonl','scene_atomic_v2/conflict_qa_five_options.current.jsonl']:
    path = root/'reports/final_benchmark_aligned'/rel
    rows = core.read_rows(path)
    old = core.read_rows(release/'backup'/path.relative_to(root))
    assert {r['id']:r for r in rows if core.is_scene(r)} == expected
    assert sum(core.is_scene(r) for r in rows) == 15
    assert [r for r in rows if not core.is_scene(r)] == [r for r in old if not core.is_scene(r)]
for r in revised:
    assert len(r['options']) == 5
    assert 2 <= len(r['valid_option_ids']) <= 5
    assert r['valid_option_ids'] == [o['option_id'] for o in r['options'] if o['is_valid']]
    assert r['expected_behavior'] == 'reject' and r['expected_reject_reason'] == 'CONFLICTING_EVIDENCE'
    assert r['qa_fingerprint'] == core.fingerprint(r)
old = core.read_rows(base/'manual_scene_v3_20260902_batch01/scene_repaired.jsonl')
for r in old:
    assert {k:v for k,v in expected[r['id']].items() if k != 'answer_multiplicity'} == r
queue = core.read_rows(release/'scene_review_queue.jsonl')
assert len(queue) == 360 and len({r['id'] for r in queue+revised}) == 375
for rel in ['scripts/eval/evaluate.py','etbench_pipeline/conflict/evaluate_conflict_v2.py']:
    text = (root/rel).read_text(encoding='utf-8')
    compile(text,rel,'exec')
    assert 'different intervals support at least two incompatible' not in text
    assert 'simultaneously in the same frame' in text
    assert 'get("prompt_version") == "hierarchical_classify_multi_answer_v2"' in text
print('PASS: 4 active inputs match; 15 scene / 360 pending; non-scene unchanged; all backups and hashes valid.')
print(dict(Counter(r['answer_multiplicity']['type'] for r in revised)))
