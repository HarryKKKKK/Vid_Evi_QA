"""Read-only verification of the text-only scene publication on either host."""
import argparse
from pathlib import Path
import apply_personal_scene_repairs as core
import publish_scene_text_split360 as task

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
args = p.parse_args()
root = args.root.resolve()
base = root/task.BASE
out = base/'scene_manual_v3'/task.RELEASE
new,held,audit = task.build(root)
assert (len(new),len(held)) == (324,36)
all_rows = core.read_rows(out/'scene_all_current.jsonl')
previous = core.read_rows(base/'scene_manual_v3/manual_scene_v3_20260902_spatial01/scene_all_revised_with_types.jsonl')
assert all_rows == previous+new
assert core.read_rows(out/'exceptions.jsonl') == held
assert len({r['id'] for r in all_rows+held}) == 375
manifest = core.read_rows(out/'transaction.json')
assert manifest['status'] == 'committed'
for entry in manifest['files']:
    path = root/entry['path']
    assert core.digest(path.read_bytes()) == entry['after_sha256'], path
    if entry['before_sha256']:
        assert core.digest((out/'backup'/entry['path']).read_bytes()) == entry['before_sha256']
for rel in ['final/conflict_qa_five_options.jsonl','final/all_unique_qa.jsonl',
            'final/all_condition_records.jsonl','scene_atomic_v2/conflict_qa_five_options.current.jsonl']:
    path = base/rel
    rows = core.read_rows(path)
    old = core.read_rows(out/'backup'/path.relative_to(root))
    assert [r for r in rows if not core.is_scene(r)] == [r for r in old if not core.is_scene(r)]
    assert [r for r in rows if core.is_scene(r)] == all_rows
for r in all_rows:
    assert len(r['options']) == 5
    assert r['valid_option_ids'] == [o['option_id'] for o in r['options'] if o['is_valid']]
    assert 2 <= len(r['valid_option_ids']) <= 5
    assert r['expected_behavior'] == 'reject'
    assert r['expected_reject_reason'] == 'CONFLICTING_EVIDENCE'
    assert core.fingerprint(r) == r['qa_fingerprint']
print('PASS: all 360 dispositions, 324 newly published, 36 withheld, 15 previous unchanged; 4 inputs match.')
print('PASS: five options, inherited gold, rejection labels, backup hashes, and non-scene preservation.')
print('Scene-only SHA256:', core.digest(core.encode(Path('scene.jsonl'),all_rows)))
