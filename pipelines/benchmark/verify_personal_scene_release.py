"""Read-only post-publication verification against recoverable backups."""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/verify_personal_scene_release.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/verify_personal_scene_release.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
release = root/'reports/final_benchmark_aligned/scene_manual_v3/manual_scene_v3_20260902_batch01'

def rows(path):
    text = path.read_text(encoding='utf-8')
    return [json.loads(x) for x in text.splitlines() if x.strip()] if path.suffix == '.jsonl' else json.loads(text)

def scene(r):
    return r.get('conflict_type', r.get('question_type')) == 'scene_context' or '__scene_context' in str(r.get('id',''))

transaction = rows(release/'transaction.json')
assert transaction['status'] == 'committed'
for f in transaction['files']:
    path = root/f['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == f['after_sha256'], f['path']
    if f['before_sha256'] is not None:
        backup = release/'backup'/f['path']
        assert hashlib.sha256(backup.read_bytes()).hexdigest() == f['before_sha256']
        if path.suffix in {'.jsonl', '.json'} and path.name not in {'summary.json'}:
            before, after = rows(backup), rows(path)
            if isinstance(before, list) and isinstance(after, list):
                assert [r for r in before if not scene(r)] == [r for r in after if not scene(r)], f['path']
new = rows(release/'scene_repaired.jsonl')
assert len(new) == 12
for name in ['conflict_qa_five_options.jsonl', 'all_unique_qa.jsonl', 'all_condition_records.jsonl']:
    current = rows(root/'reports/final_benchmark_aligned/final'/name)
    assert [r for r in current if scene(r)] == new
    # Condition records intentionally repeat a NextGQA ID across conditions.
    if name != 'all_condition_records.jsonl':
        assert len({r['id'] for r in current}) == len(current), name
    assert len({r['id'] for r in current if scene(r)}) == 12
current = rows(root/'reports/final_benchmark_aligned/scene_atomic_v2/conflict_qa_five_options.current.jsonl')
assert len(current) == 1308 and [r for r in current if scene(r)] == new
for r in new:
    assert len(r['options']) == 5
    assert r['valid_option_ids'] == [o['option_id'] for o in r['options'] if o['is_valid']]
    assert all(bool(o['supported_by']) == o['is_valid'] for o in r['options'])
    assert r['expected_behavior'] == 'reject'
for directory in (root/'reports/conflict_final_audit').iterdir():
    if directory.is_dir():
        for name in ['conflict_eval.json', 'conflict_eval.jsonl']:
            path = directory/name
            if path.is_file():
                assert not any(scene(r) for r in rows(path)), str(path)
assert len(rows(release/'scene_retired.jsonl')) == 9
assert len(rows(release/'scene_pending.jsonl')) == 354
print('PASS: 12 revised scene QA in all four active entrances; 9 retired; 354 pending; all five-option labels consistent; all non-scene rows preserved; stale scene caches absent; all backups and publication hashes verified.')
