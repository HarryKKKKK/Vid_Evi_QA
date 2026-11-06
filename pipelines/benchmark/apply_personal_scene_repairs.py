#!/usr/bin/env python3
"""Publish explicitly authored, visually reviewed scene edits with recoverable backups.

No semantic splitting or model inference. All answers and supporting IDs are authored.
Default is a dry run. Preserves every non-scene record and frozen source files.
"""
from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/apply_personal_scene_repairs.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/apply_personal_scene_repairs.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

RELEASE = 'manual_scene_v3_20260902_batch01'


def read_rows(path):
    if path.suffix == '.jsonl':
        return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    return json.loads(path.read_text(encoding='utf-8'))


def encode(path, value):
    text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in value) if path.suffix == '.jsonl' else json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    return text.encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(r):
    value = {k: r.get(k) for k in ['id', 'question', 'valid_option_ids']}
    value['options'] = [{k: o[k] for k in ['option_id', 'answer']} for o in r['options']]
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


def is_scene(r):
    return r.get('conflict_type', r.get('question_type')) == 'scene_context' or '__scene_context' in str(r.get('id', ''))


def key(r):
    return r['id'].removeprefix('etbench_perception_test_').removesuffix('__scene_context')


def replace_scene(rows, new):
    result = [r for r in rows if not is_scene(r)] + copy.deepcopy(new)
    assert [r for r in result if not is_scene(r)] == [r for r in rows if not is_scene(r)]
    return result


def validate_multiplicity(row, annotation, sheets):
    """Validate authored witnesses, not infer scene meaning from interval counts."""
    valid = set(row['valid_option_ids'])
    options = {o['option_id']: o for o in row['options']}
    def witness(point, positive, negative=()):
        assert positive and set(positive) <= valid
        assert set(negative) <= valid and not set(positive) & set(negative)
        iid, sec = point['instance_id'], point['seconds']
        assert any(f['instance_id'] == iid and abs(f['requested_seconds']-sec) < 0.00001
                   for s in sheets for f in s['frames']), 'Witness is not a reviewed frame'
        for oid in positive:
            assert any(e['instance_id'] == iid for e in options[oid]['supported_by'])
    spatial = annotation.get('spatial_witnesses', [])
    temporal = annotation.get('temporal_witnesses', [])
    for point in spatial:
        assert len(set(point['option_ids'])) >= 2 and point['judgment_zh'].strip()
        witness(point, point['option_ids'])
    for change in temporal:
        a, b = change['earlier'], change['later']
        assert a['seconds'] < b['seconds'] and change['judgment_zh'].strip()
        witness(a, a['valid_option_ids'], a['invalid_option_ids'])
        witness(b, b['valid_option_ids'], b['invalid_option_ids'])
        assert set(a['valid_option_ids']) & set(b['invalid_option_ids'])
        assert set(b['valid_option_ids']) & set(a['invalid_option_ids'])
    derived = 'mixed' if spatial and temporal else 'spatial' if spatial else 'temporal' if temporal else 'unclassified'
    assert annotation['type'] == derived, 'Classification must match authored witnesses'
    assert annotation['scope'] == 'reviewed_sampled_frames_not_exhaustive_video'


def build_repairs(source, authored, frames):
    new_policy = authored.get('acceptance_policy') == 'time_or_space_v1'
    release = authored.get('release', RELEASE)
    originals = {key(r): r for r in source if is_scene(r)}
    assert len(originals) == 375
    by_key = {}
    for sheet in frames:
        by_key.setdefault(sheet['key'], []).append(sheet)
    repaired, retired = [], []
    seen = set()
    for note in authored['repairs']:
        k = note['key']
        assert k not in seen
        seen.add(k)
        original = originals[k]
        sheets = by_key[k]
        assert all(s['video'] == original['video'] for s in sheets)
        intervals = {iid: span for sheet in sheets for iid, span in sheet['intervals']}
        assert len(note['options']) == 5
        assert len({x[0].strip().casefold() for x in note['options']}) == 5
        row = copy.deepcopy(original)
        row['question'] = note['question']
        options = []
        for index, (answer, supports, judgment) in enumerate(note['options']):
            assert answer.strip() and judgment.strip() and set(supports) <= intervals.keys()
            evidence = []
            for iid in supports:
                times = [f['requested_seconds'] for s in sheets for f in s['frames'] if f['instance_id'] == iid]
                assert len(times) >= 5 and all(intervals[iid][0] <= t <= intervals[iid][1] for t in times)
                evidence.append({'instance_id': iid, 'interval': intervals[iid], 'sampled_seconds': times,
                                 'visual_basis': judgment, 'verification_method': 'primary_assistant_frame_review'})
            options.append({'option_id': 'ABCDE'[index], 'answer': answer, 'is_valid': bool(supports),
                            'supported_by': evidence, 'verification_status': 'personally_verified_sampled_frames',
                            'visual_judgment_zh': judgment, 'semantic_group_id': f'{k}_atomic_{index}',
                            'reviewed_instance_ids': sorted(intervals)})
        row['options'] = options
        row['choices'] = [o['answer'] for o in options]
        row['valid_option_ids'] = [o['option_id'] for o in options if o['is_valid']]
        assert 2 <= len(row['valid_option_ids']) <= (5 if new_policy else 4)
        if not new_policy:
            assert len({e['instance_id'] for o in options for e in o['supported_by']}) >= 2
        row['alignment_version'] = release
        row['status'] = 'accepted'
        row['answer_mode'] = 'multiple_valid_answers'
        # Retain server evaluation contract: these alternatives imply rejection,
        # not a newly invented ordinary multi-select evaluation task.
        assert original['expected_behavior'] == 'reject'
        row['qa_fingerprint'] = fingerprint(row)
        row['historical_provenance'] = original.get('provenance', {})
        row['provenance'] = {'qa_author': 'primary_assistant', 'official_intervals_locked': True,
                             'grounding_method': 'personal_sampled_frame_review', 'source_qa_fingerprint': fingerprint(original)}
        row['manual_review'] = {'release': release, 'method': authored['verification_method'],
                                'change_zh': note['change_zh'], 'contact_sheets': [s['sheet'] for s in sheets],
                                'intervals_reviewed': sorted(intervals), 'continuous_video_viewed': False}
        row.pop('expected_reject_reason', None)
        row['expected_reject_reason'] = 'CONFLICTING_EVIDENCE'
        if new_policy:
            validate_multiplicity(row, note['answer_multiplicity'], sheets)
            row['answer_multiplicity'] = copy.deepcopy(note['answer_multiplicity'])
            row['scene_acceptance_policy'] = authored['acceptance_policy']
            if authored.get('evaluation_contract') == 'reject_if_multiple_valid_options':
                row['evaluation_contract_status'] = 'user_confirmed_multiple_valid_options_trigger_rejection'
            else:
                row['status'] = 'revised_awaiting_evaluation_contract'
                row['expected_behavior'] = None
                row['expected_reject_reason'] = None
                row['evaluation_contract_status'] = 'pending_user_choice'
        assert row['official_action'] == original['official_action'] and row['video'] == original['video']
        repaired.append(row)
    for note in authored['retired_after_visual_review']:
        assert note['key'] not in seen and note['key'] in by_key
        seen.add(note['key'])
        retired.append({'id': originals[note['key']]['id'], **note, 'status': 'retired_from_conflict',
                        'source_record': originals[note['key']]})
    pending = [{'id': r['id'], 'key': k, 'status': 'withheld_pending_personal_visual_repair', 'source_record': r}
               for k, r in originals.items() if k not in seen]
    assert len(repaired) + len(retired) + len(pending) == 375
    return repaired, retired, pending


def plan(root, authored_path, frames_paths):
    base = root / 'reports/final_benchmark_aligned'
    final = base / 'final'
    scene_base = base / 'scene_manual_v3'
    source = read_rows(final/'conflict_qa_five_options.jsonl')
    authored = read_rows(authored_path)
    frames = [sheet for path in frames_paths for sheet in read_rows(path)]
    authoritative = read_rows(scene_base/'remote_snapshot/conflict_qa_five_options.jsonl')
    repaired, retired, pending = build_repairs(authoritative, authored, frames)
    changes, report = {}, {}
    # Validate content against the reviewed snapshot, ignoring unrelated server
    # evaluation metadata that must never be overwritten by the local copy.
    snapshot = {r['id']: r for r in authoritative}
    for r in source:
        assert r['id'] in snapshot
        for field in ['question', 'options', 'valid_option_ids', 'video']:
            assert r.get(field) == snapshot[r['id']].get(field), (r['id'], field, 'source drift')
    for path in [final/'conflict_qa_five_options.jsonl', final/'all_unique_qa.jsonl', final/'all_condition_records.jsonl',
                 base/'scene_atomic_v2/conflict_qa_five_options.current.jsonl']:
        old = read_rows(path)
        if path.parent == final:
            assert {r['id'] for r in old if is_scene(r)} == {r['id'] for r in source if is_scene(r)}
        else:
            assert not any(is_scene(r) for r in old), 'Unexpected scene publication since review'
            assert [r for r in old if not is_scene(r)] == [r for r in source if not is_scene(r)]
        new = replace_scene(old, repaired)
        changes[path] = encode(path, new)
        report[str(path.relative_to(root))] = {'before': len(old), 'after': len(new),
                                             'old_scene_removed': sum(is_scene(r) for r in old), 'new_scene': len(repaired)}
    conflict = replace_scene(source, repaired)
    summary_path = final/'summary.json'
    summary = read_rows(summary_path)
    summary.update({'status': 'partial_scene_manual_repair', 'scene_manual_release': RELEASE,
                    'conflict_qa': len(conflict), 'unresolved_conflict_qa': len(pending),
                    'retired_scene_qa': len(retired), 'pending_scene_qa': len(pending),
                    'all_unique_qa': len(read_rows(final/'all_unique_qa.jsonl'))-375+len(repaired),
                    'all_condition_records': len(read_rows(final/'all_condition_records.jsonl'))-375+len(repaired),
                    'conflict_type_counts': dict(Counter(r['conflict_type'] for r in conflict)),
                    'conflict_valid_answer_count_distribution': dict(Counter(str(len(r['valid_option_ids'])) for r in conflict)),
                    'conflict_valid_position_counts': dict(Counter(o for r in conflict for o in r['valid_option_ids'])),
                    'scene_verification_note': authored['verification_method']})
    changes[summary_path] = encode(summary_path, summary)
    v2_summary = base/'scene_atomic_v2/summary.json'
    old_v2 = read_rows(v2_summary)
    old_v2.update({'status': 'manual_scene_partial_published', 'manual_release': RELEASE,
                   'published_scene_records': len(repaired), 'pending_scene_records': len(pending),
                   'retired_scene_records': len(retired), 'automatic_scene_publishing_disabled': True})
    changes[v2_summary] = encode(v2_summary, old_v2)

    # Purge stale predictions from BOTH cache formats. Backups keep all old
    # answers recoverable; no evaluation is launched here.
    eval_report = {}
    for directory in sorted((root/'reports/conflict_final_audit').iterdir()):
        if not directory.is_dir():
            continue
        for filename in ['conflict_eval.json', 'conflict_eval.jsonl']:
            path = directory/filename
            if not path.is_file():
                continue
            old = read_rows(path)
            new = [r for r in old if not is_scene(r)]
            eval_report[str(path.relative_to(root))] = {'before': len(old), 'after': len(new), 'scene_archived': len(old)-len(new)}
            if len(old) != len(new):
                changes[path] = encode(path, new)
        # Old aggregates contain invalidated scene rows, even when JSON was
        # previously filtered. Do not keep their old scores marked complete.
        path = directory/'summary.json'
        if path.is_file() and (directory/'conflict_eval.json').is_file():
            old = read_rows(path)
            remaining = [r for r in read_rows(directory/'conflict_eval.json') if not is_scene(r)]
            new = {'status': 'scene_v1_invalidated_scene_v3_needs_eval',
                   'model_tag': old.get('model_tag'), 'source_task': old.get('source_task'),
                   'active_non_scene_cases': len(remaining), 'scene_v3_cases_evaluated': 0,
                   'scene_v3_cases_pending': len(repaired),
                   'metrics_status': 'old_mixed_scene_scores_archived_not_reused', 'release': RELEASE}
            changes[path] = encode(path, new)
    lock = base/'scene_atomic_v2/MANUAL_PUBLICATION_LOCK.json'
    changes[lock] = encode(lock, {'mode': 'manual_only', 'release': RELEASE, 'cancelled_scene_jobs': [374849, 374850],
                                 'reason': 'User requested personal scene editing, not Slurm semantic rewriting.'})
    stats = {'release': RELEASE, 'status': 'partial_published', 'repaired_scene': len(repaired),
             'retired_after_visual_review': len(retired), 'pending_scene': len(pending),
             'personally_frame_reviewed_qa': len(repaired)+len(retired),
             'contact_sheet_frames_inspected': sum(len(s['frames']) for s in frames),
             'active_conflict_records': len(conflict), 'qa_files': report, 'eval_files': eval_report,
             'new_eval_launched': False, 'all_375_formally_repaired': False,
             'non_scene_records_unchanged': True, 'verification_method': authored['verification_method']}
    return changes, stats, repaired, retired, pending


def commit(root, changes, release_dir):
    release_dir.mkdir(parents=True, exist_ok=False)
    original = {path: path.read_bytes() if path.exists() else None for path in changes}
    manifest = []
    for path, payload in changes.items():
        assert path.resolve().is_relative_to(root.resolve())
        rel = path.relative_to(root)
        staged = release_dir/'staged'/rel
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(payload)
        if original[path] is not None:
            backup = release_dir/'backup'/rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            backup.write_bytes(original[path])
        manifest.append({'path': str(rel), 'before_sha256': digest(original[path]) if original[path] is not None else None,
                         'after_sha256': digest(payload)})
    (release_dir/'transaction.json').write_bytes(encode(Path('transaction.json'), {'status': 'prepared', 'files': manifest}))
    # Optimistic concurrency check before changing ANY active file.
    assert all((p.read_bytes() if p.exists() else None) == old for p, old in original.items()), 'Concurrent write detected'
    for path in changes:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(release_dir/'staged'/path.relative_to(root), path)
    assert all(path.read_bytes() == value for path, value in changes.items())
    (release_dir/'transaction.json').write_bytes(encode(Path('transaction.json'), {'status': 'committed', 'files': manifest}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    base = root/'reports/final_benchmark_aligned/scene_manual_v3'
    authored_path = base/'authored_repairs.json'
    frames_paths = [base/f'frames_batch{i:02d}/manifest.json' for i in [1, 2]]
    changes, stats, repaired, retired, pending = plan(root, authored_path, frames_paths)
    if args.apply:
        release_dir = base/RELEASE
        commit(root, changes, release_dir)
        for name, rows in [('scene_repaired.jsonl', repaired), ('scene_retired.jsonl', retired), ('scene_pending.jsonl', pending)]:
            (release_dir/name).write_bytes(encode(Path(name), rows))
        (release_dir/'summary.json').write_bytes(encode(Path('summary.json'), stats))
        lines = ['# 正式 scene 修订：第一批', '',
                 f"已发布 {len(repaired)} 道，退出conflict {len(retired)} 道，仍隔离待修 {len(pending)} 道。不是375道全部完成。", '',
                 '每题五项；逐官方区间抽帧本人核验，非连续全视频播放。旧题库及旧预测均有backup。', '']
        for r in repaired:
            lines += [f"## {key(r)} · {r['video']}", '', r['question'], '']
            lines += [f"- {o['option_id']}. {o['answer']}（{'支持' if o['is_valid'] else '不支持'}）" for o in r['options']]
            lines += ['', '修改：'+r['manual_review']['change_zh'], '',
                      '新候选金标：'+', '.join(r['valid_option_ids'])+'；最终评测行为仍为冲突拒答。', '']
        lines += ['## 退出conflict', '']+[f"- {r['key']}：{r['reason_zh']}" for r in retired]
        (release_dir/'APPLIED.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    else:
        stats['status'] = 'dry_run_no_active_writes'
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
