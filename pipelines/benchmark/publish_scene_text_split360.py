"""Publish human-authored text-only projections of existing positive scene options.

No images, model inference, new objects or new negative options. Default: dry-run.
"""
from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/publish_scene_text_split360.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/publish_scene_text_split360.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import apply_personal_scene_repairs as core

RELEASE = 'manual_scene_v3_20260902_text360'
BASE = Path('results/reports/final_benchmark_aligned')


def build(root):
    base = root/BASE/'scene_manual_v3'
    queue = core.read_rows(base/'manual_scene_v3_20260902_spatial01/scene_review_queue.jsonl')
    originals = {r['key']:r['source_record'] for r in queue}
    decisions = {}
    for i in range(1,5):
        part = core.read_rows(base/f'text_split_360/part_{i:02d}.json')
        assert not decisions.keys() & part.keys()
        decisions.update(part)
    assert decisions.keys() == originals.keys() and len(decisions) == 360
    revised, held, audit = [], [], []
    for k, note in decisions.items():
        original = originals[k]
        if '_hold' in note:
            held.append({'key':k,'id':original['id'],'reason_zh':note['_hold'],
                'status':'text_review_complete_requires_question_or_gold_repair','source_record':original})
            continue
        row = copy.deepcopy(original)
        source_options = {o['option_id']:o for o in original['options']}
        options = []
        if '_keep' in note:
            options = copy.deepcopy(original['options'])
            for o in options:
                o['source_option_ids'] = [o['option_id']]
                o['text_revision_kind'] = 'existing_atomic_answer_retained'
        else:
            for oid, answers in note.items():
                if oid.startswith('_'):
                    continue
                parent = source_options[oid]
                assert parent['is_valid'], (k,oid,'Cannot derive a positive answer from a negative source')
                for answer in answers:
                    assert answer.strip()
                    options.append({'answer':answer,'is_valid':True,'source_option_ids':[oid],
                        'supported_by':[{'instance_id':e['instance_id'],'interval':e['interval'],
                            'evidence_basis':'inherited_positive_option_not_new_visual_verification',
                            'source_option_id':oid} for e in parent.get('supported_by',[])],
                        'verification_status':'text_projection_of_existing_gold_only',
                        'text_revision_kind':'atomic_positive_projection'})
            assert 2 <= len(options) <= 5, (k,len(options))
            assert len({o['answer'].casefold().strip() for o in options}) == len(options), k
            # Only old negatives can fill the remaining slots. They are not rewritten.
            for parent in original['options']:
                if len(options) == 5:
                    break
                if parent['is_valid'] or parent['answer'].casefold().strip() in {o['answer'].casefold().strip() for o in options}:
                    continue
                o = copy.deepcopy(parent)
                o['source_option_ids'] = [parent['option_id']]
                o['verification_status'] = 'unchanged_existing_negative_not_reverified'
                o['text_revision_kind'] = 'original_negative_retained'
                options.append(o)
            if len(options) != 5:
                held.append({'key':k,'id':original['id'],
                    'reason_zh':'拆分去重后的合题正确项加原有错误项不足五个；不能把无关旧金标当错误项，也不擅自新增干扰物。',
                    'status':'text_review_complete_requires_additional_option_authority','source_record':original})
                continue
            # Spread positive positions reproducibly; never change truth to balance labels.
            options.sort(key=lambda o:core.digest((k+'|'+o['answer']).encode()))
        for i,o in enumerate(options):
            o['option_id'] = 'ABCDE'[i]
        row['options'] = options
        row['choices'] = [o['answer'] for o in options]
        row['valid_option_ids'] = [o['option_id'] for o in options if o['is_valid']]
        assert len(options) == 5 and 2 <= len(row['valid_option_ids']) <= 5
        row['alignment_version'] = RELEASE
        row['qa_fingerprint'] = core.fingerprint(row)
        row['expected_behavior'] = 'reject'
        row['expected_reject_reason'] = 'CONFLICTING_EVIDENCE'
        row['status'] = 'accepted_text_only_inherited_gold'
        row['scene_acceptance_policy'] = 'time_or_space_v1'
        row['historical_provenance'] = original.get('provenance',{})
        row['provenance'] = {'qa_author':'primary_assistant_text_review',
            'truth_label_source':'semantic_projection_of_existing_valid_options',
            'source_qa_fingerprint':core.fingerprint(original), 'official_intervals_locked':True,
            'new_visual_verification':False}
        row['manual_review'] = {'release':RELEASE,'method':'text_only_per_user_instruction',
            'question_changed':False, 'images_used_for_this_revision':False,
            'original_negative_options_reverified':False,
            'note_zh':note.get('_keep','只拆/筛选旧正确项，去同义及包含描述，剩余槽沿用旧错误项。'),
            'caveat_zh':note.get('_flag','原金标的视觉真实性沿用，不构成本轮重新核验。')}
        positives = [o for o in options if o['is_valid']]
        groups = {}
        for o in positives:
            for parent_id in o['source_option_ids']:
                groups.setdefault(parent_id,[]).append(o['option_id'])
        spatial_candidates = [{'source_option_id':oid,'option_ids':ids}
            for oid,ids in groups.items() if len(ids) >= 2]
        row['answer_multiplicity'] = {'type':'unclassified','annotation_basis':'source_text_only',
            'spatial_witnesses':[],'temporal_witnesses':[],
            'text_hint':'spatial_candidate' if spatial_candidates else 'undetermined',
            'compound_source_groups':spatial_candidates,
            'note_zh':'同一旧正确项拆出多个答案提示空间共存，但不是同帧证据；不同来源区间也不等于已证时间变化。'}
        assert row['question'] == original['question'] and row['video'] == original['video']
        revised.append(row)
        audit.append({'key':k,'id':row['id'],'video':row['video'],'question':row['question'],
            'operation':'retain_atomic' if '_keep' in note else 'split_select_deduplicate',
            'old_valid_options':[o for o in original['options'] if o['is_valid']],
            'new_valid_options':[{'option_id':o['option_id'],'answer':o['answer'],
                'source_option_ids':o['source_option_ids']} for o in positives],
            'retained_negative_options':sum(not o['is_valid'] for o in options),
            'text_spatial_candidate':bool(spatial_candidates),'caveat_zh':note.get('_flag')})
    assert len(revised)+len(held) == 360
    return revised, held, audit


def plan(root):
    revised,held,audit = build(root)
    base = root/BASE
    existing = core.read_rows(base/'scene_manual_v3/manual_scene_v3_20260902_spatial01/scene_all_revised_with_types.jsonl')
    assert len(existing) == 15
    combined = existing+revised
    assert len({r['id'] for r in combined+held}) == 375
    changes, observed = {}, {}
    def put(path,data):
        observed[path] = path.read_bytes() if path.exists() else None
        changes[path] = core.encode(path,data)
    counts = {}
    for rel in ['final/conflict_qa_five_options.jsonl','final/all_unique_qa.jsonl',
                'final/all_condition_records.jsonl','scene_atomic_v2/conflict_qa_five_options.current.jsonl']:
        path = base/rel
        old = core.read_rows(path)
        assert {r['id']:r for r in old if core.is_scene(r)} == {r['id']:r for r in existing}, 'Active scene changed; rebase needed'
        updated = core.replace_scene(old,combined)
        counts[rel] = len(updated)
        put(path,updated)
    scene_counts = dict(Counter(r['answer_multiplicity']['type'] for r in combined))
    stats = {'release':RELEASE,'input_remaining':360,'text_decisions_completed':360,
        'newly_published_inherited_gold_qa':len(revised),'split_selection_qa':sum(r['operation']=='split_select_deduplicate' for r in audit),
        'existing_atomic_qa_retained':sum(r['operation']=='retain_atomic' for r in audit),
        'previous_published_qa_unchanged':15,'active_scene_total':len(combined),
        'requires_question_or_gold_repair':len(held),'all_360_published':not held,
        'new_positive_count_distribution':dict(Counter(str(len(r['valid_option_ids'])) for r in revised)),
        'text_spatial_candidates':sum(r['text_spatial_candidate'] for r in audit),
        'answer_multiplicity_counts':scene_counts,'new_images_or_models_used':False,'new_eval_launched':False,
        'question_changes':0,'original_negatives_reverified':False,'qa_file_counts':counts}
    path = base/'final/summary.json'
    data = core.read_rows(path)
    conflict = core.replace_scene(core.read_rows(base/'final/conflict_qa_five_options.jsonl'),combined)
    data.update({'status':'scene_text_split_complete_with_exceptions','scene_manual_release':RELEASE,
        'conflict_qa':len(conflict),'all_unique_qa':counts['final/all_unique_qa.jsonl'],
        'all_condition_records':counts['final/all_condition_records.jsonl'],
        'pending_scene_qa':len(held),'unresolved_conflict_qa':len(held),'retired_scene_qa':0,
        'conflict_type_counts':dict(Counter(r['conflict_type'] for r in conflict)),
        'conflict_valid_answer_count_distribution':dict(Counter(str(len(r['valid_option_ids'])) for r in conflict)),
        'conflict_valid_position_counts':dict(Counter(x for r in conflict for x in r['valid_option_ids'])),
        'scene_answer_multiplicity_counts':scene_counts,
        'scene_verification_note':'15 prior frame-reviewed QA unchanged; new text-only QA inherit old truth labels, no new image verification.',
        'scene_text_split_stats':stats})
    put(path,data)
    for name in ['summary.json','manual_review_progress.json','MANUAL_PUBLICATION_LOCK.json']:
        path = base/'scene_atomic_v2'/name
        data = core.read_rows(path)
        data.update({'manual_release':RELEASE,'published_scene_records':len(combined),
            'pending_scene_records':len(held),'retired_scene_records':0,'scene_text_split_stats':stats})
        if name == 'manual_review_progress.json':
            data.update({'published_new_scene_qa':len(combined),'pending_formal_scene_repair':len(held),
                'formal_release_directory':'../scene_manual_v3/'+RELEASE,
                'formal_repair_status':'text_review_all360_complete_with_exceptions'})
        put(path,data)
    fingerprints = {r['id']:core.fingerprint(r) for r in combined}
    stats['stale_scene_results_archived'] = {}
    for directory in (root/'results/reports/conflict_final_audit').iterdir():
        if not directory.is_dir():
            continue
        for name in ['conflict_eval.json','conflict_eval.jsonl']:
            path = directory/name
            if not path.is_file():
                continue
            old = core.read_rows(path)
            updated = [r for r in old if not core.is_scene(r) or
                (r['id'] in fingerprints and r.get('qa_fingerprint') == fingerprints[r['id']])]
            stats['stale_scene_results_archived'][str(path.relative_to(root))] = len(old)-len(updated)
            if updated != old:
                put(path,updated)
        path = directory/'summary.json'
        if path.is_file() and (directory/'conflict_eval.json').is_file():
            data = core.read_rows(path)
            data.update({'release':RELEASE,'scene_v3_cases_pending':len(combined),
                'scene_v3_cases_evaluated':0,'status':'updated_scene_requires_new_evaluation'})
            put(path,data)
    notice = (f'当前：360道均已完成文字处理；新增发布{len(revised)}道（继承旧金标，未新看图），'
        f'原15道保持不变，正式scene共{len(combined)}道。{len(held)}道因题干不匹配/去重后不足两个答案仍隔离，见异常清单。'
        '每道发布题5项、至少2正确，可5项全对。时间/空间不凭文本冒充已核验；仅记录空间候选。未启动eval。')
    for path in [base/'scene_manual_v3/README.md',base/'scene_atomic_v2/manual_review_all_375/INDEX.md',
                 base/'scene_atomic_v2/MANUAL_WORKFLOW.md']:
        old = path.read_text(encoding='utf-8') if path.exists() else ''
        observed[path] = path.read_bytes() if path.exists() else None
        changes[path] = ('> '+notice+'\n> 最新记录：../scene_manual_v3/'+RELEASE+'/REVISIONS.md\n\n以下为历史记录。\n\n'+old).encode('utf-8')
    return changes,observed,stats,combined,revised,held,audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--apply',action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    changes,observed,stats,combined,revised,held,audit = plan(root)
    if args.apply:
        assert all((p.read_bytes() if p.exists() else None)==v for p,v in observed.items()), 'Concurrent change'
        out = root/BASE/'scene_manual_v3'/RELEASE
        core.commit(root,changes,out)
        for name,value in [('scene_all_current.jsonl',combined),('scene_text_revised.jsonl',revised),
            ('exceptions.jsonl',held),('source_to_new_options.jsonl',audit),('summary.json',stats)]:
            (out/name).write_bytes(core.encode(Path(name),value))
        lines = ['# 剩余360题：纯文本拆分实际记录','',
            f"360道均完成处理：新增发布{len(revised)}道，原15道不变，当前{len(combined)}道scene；{len(held)}道仍隔离。",'',
            '仅依据旧正确选项的文本拆分/去重/筛选五项，不进行新视觉核验。补位错误项保持原文、继承旧标签。',
            '同一旧选项拆出多个物件标为空间候选，不假称已找到同帧时间戳；不自动推断时间型。',
            '问题、视频和官方区间不改；全部发布题期望拒答。被舍弃的正确物件不会变为错误项。',
            '非scene及原15道逐条保留，覆盖文件有backup与SHA256；未运行新eval。','']
        for row in revised:
            lines += [f"## {core.key(row)} · {row['video']}",'',row['question'],'']
            lines += [f"- {o['option_id']}. {o['answer']} — {'正确' if o['is_valid'] else '原错误项'}；来源 {','.join(o['source_option_ids'])}" for o in row['options']]
            lines += ['', '正确项：'+','.join(row['valid_option_ids']), '']
        lines += ['## 无法仅靠拆分完成的题','']
        lines += [f"- {r['key']}：{r['reason_zh']}" for r in held]
        (out/'REVISIONS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
