"""Publish explicit human scene edits under the user-confirmed time/space policy."""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/package_scene_multiplicity_revision.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/package_scene_multiplicity_revision.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import argparse
import copy
import json
from pathlib import Path
from collections import Counter
import apply_personal_scene_repairs as core


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    base = root/'reports/final_benchmark_aligned/scene_manual_v3'
    authored = core.read_rows(base/'authored_spatial_repairs.json')
    source = core.read_rows(base/'remote_snapshot/conflict_qa_five_options.jsonl')
    frames = [s for i in [1, 2] for s in core.read_rows(base/f'frames_batch{i:02d}/manifest.json')]
    revised, retired, _ = core.build_repairs(source, authored, frames)
    assert not retired
    existing = core.read_rows(base/'manual_scene_v3_20260902_batch01/scene_repaired.jsonl')
    active_paths = [root/'reports/final_benchmark_aligned'/rel for rel in [
        'final/conflict_qa_five_options.jsonl', 'final/all_unique_qa.jsonl',
        'final/all_condition_records.jsonl', 'scene_atomic_v2/conflict_qa_five_options.current.jsonl']]
    before = {p: core.digest(p.read_bytes()) for p in active_paths}
    for path in active_paths:
        assert {r['id']: r for r in core.read_rows(path) if core.is_scene(r)} == {r['id']: r for r in existing}, 'Active scene drift'
    classifications = {n['key']: n['answer_multiplicity'] for n in authored['existing_qa_classifications']}
    annotated_existing = copy.deepcopy(existing)
    for row in annotated_existing:
        k = core.key(row)
        annotation = classifications.get(k, {'type': 'unclassified', 'spatial_witnesses': [],
            'temporal_witnesses': [], 'scope': 'reviewed_sampled_frames_not_exhaustive_video',
            'note_zh': '旧题正确项已核验，时间/空间来源尚未按新标准复核，不自动推断。'})
        core.validate_multiplicity(row, annotation, [s for s in frames if s['key'] == k])
        row['answer_multiplicity'] = annotation
    all_revised = annotated_existing + revised
    done = {r['id'] for r in all_revised}
    old_retired = {r['key'] for r in core.read_rows(base/'manual_scene_v3_20260902_batch01/scene_retired.jsonl')}
    queue = [{'id':r['id'], 'key':core.key(r), 'status':'pending_review_under_time_or_space_policy',
              'previous_retirement_requires_reconsideration':core.key(r) in old_retired,
              'source_record':r} for r in source if core.is_scene(r) and r['id'] not in done]
    assert len(all_revised) == 15 and len(queue) == 360
    assert len({r['id'] for r in all_revised + queue}) == 375
    type_index = [{'id':r['id'], 'key':core.key(r), 'video':r['video'],
                   'valid_option_ids':r['valid_option_ids'], 'answer_multiplicity':r['answer_multiplicity']}
                  for r in all_revised]
    stats = {'release':authored['release'], 'newly_revised_spatial_qa':len(revised),
        'existing_active_scene_qa':len(existing), 'revised_qa_total':len(all_revised),
        'classification_counts':dict(Counter(r['answer_multiplicity']['type'] for r in all_revised)),
        'pending_revision_under_new_policy':len(queue), 'previous_retirements_reopened':len(old_retired),
        'previous_retirements_revised':sum(core.key(r) in old_retired for r in revised),
        'new_active_publication':bool(args.apply), 'new_eval_launched':False,
        'evaluation_contract_status':'reject_if_multiple_valid_options',
        'active_file_sha256_before':{str(p.relative_to(root)):h for p,h in before.items()}}
    assert authored['evaluation_contract'] == 'reject_if_multiple_valid_options'
    changes = {p:core.encode(p,core.replace_scene(core.read_rows(p),all_revised)) for p in active_paths}
    final = root/'reports/final_benchmark_aligned/final'
    conflict = core.replace_scene(core.read_rows(active_paths[0]), all_revised)
    summary_path = final/'summary.json'
    summary = core.read_rows(summary_path)
    summary.update({'scene_manual_release':authored['release'], 'conflict_qa':len(conflict),
        'retired_scene_qa':0, 'pending_scene_qa':len(queue), 'unresolved_conflict_qa':len(queue),
        'all_unique_qa':len(core.read_rows(active_paths[1]))-len(existing)+len(all_revised),
        'all_condition_records':len(core.read_rows(active_paths[2]))-len(existing)+len(all_revised),
        'conflict_type_counts':dict(Counter(r['conflict_type'] for r in conflict)),
        'conflict_valid_answer_count_distribution':dict(Counter(str(len(r['valid_option_ids'])) for r in conflict)),
        'conflict_valid_position_counts':dict(Counter(x for r in conflict for x in r['valid_option_ids'])),
        'scene_answer_multiplicity_counts':stats['classification_counts'],
        'scene_acceptance_policy':'time_or_space_v1', 'scene_verification_note':authored['verification_method']})
    changes[summary_path] = core.encode(summary_path,summary)
    v2 = root/'reports/final_benchmark_aligned/scene_atomic_v2'
    for filename in ['summary.json','manual_review_progress.json','MANUAL_PUBLICATION_LOCK.json']:
        path = v2/filename
        data = core.read_rows(path)
        data.update({'manual_release':authored['release'], 'scene_acceptance_policy':'time_or_space_v1',
            'published_scene_records':15, 'pending_scene_records':360, 'retired_scene_records':0,
            'scene_answer_multiplicity_counts':stats['classification_counts']})
        if filename == 'manual_review_progress.json':
            data.update({'published_new_scene_qa':15, 'retired_scene_after_visual_review':0,
                'pending_formal_scene_repair':360, 'former_retirement_decisions_reopened':9,
                'formal_release_directory':'../scene_manual_v3/'+authored['release'],
                'historical_decision_counts_superseded_by_time_or_space_policy':True})
        changes[path] = core.encode(path,data)
    # Apply exact targeted prompt edits on each host, preserving other edits.
    replacements = [
        ('Use CONFLICTING_EVIDENCE when different intervals support at least two incompatible answers and the conflict cannot be resolved.',
         'Use CONFLICTING_EVIDENCE when at least two distinct options are correct for the question, so no unique option can be selected. The correct options may be supported at different times or simultaneously in the same frame; factual contradiction and separate intervals are not required.'),
        ('For CONFLICTING_EVIDENCE, provide at least two evidence groups describing the incompatible information.',
         'For CONFLICTING_EVIDENCE, provide at least two evidence groups explaining why distinct options satisfy the question. The groups may share the same timestamp when both answers are supported in one frame.')]
    for rel in ['scripts/eval/evaluate.py','etbench_pipeline/conflict/evaluate_conflict_v2.py']:
        path = root/rel
        content = path.read_text(encoding='utf-8')
        for old,new in replacements:
            assert content.count(old) == 1 or (old not in content and content.count(new) == 1), rel
            content = content.replace(old,new)
        if rel.endswith('evaluate_conflict_v2.py'):
            content = content.replace('"prompt_version": "hierarchical_classify_v1"', '"prompt_version": "hierarchical_classify_multi_answer_v2"')
            old = 'if x.get("qa_fingerprint") == selected_fingerprints.get(x.get("id"))'
            new = old+'\n        and x.get("prompt_version") == "hierarchical_classify_multi_answer_v2"'
            if new not in content:
                assert content.count(old) == 1
                content = content.replace(old,new)
            old = '"question": record["question"],'
            new = '"answer_multiplicity_type": record.get("answer_multiplicity", {}).get("type", "unclassified"),\n            '+old
            if '"answer_multiplicity_type": record.get(' not in content:
                assert content.count(old) == 2
                content = content.replace(old,new)
        else:
            content = content.replace('"hierarchical_classify_v1" if task == "classify"', '"hierarchical_classify_multi_answer_v2" if task == "classify"')
            old = 'existing_by_key = load_existing_keys(jsonl_path)'
            new = old+'\n    if task == "classify":\n        existing_by_key = {k: v for k, v in existing_by_key.items()\n                           if v.get("prompt_version") == "hierarchical_classify_multi_answer_v2"}'
            if new not in content:
                assert content.count(old) == 1
                content = content.replace(old,new)
        compile(content,str(path),'exec')
        changes[path] = content.encode('utf-8')
    for directory in (root/'reports/conflict_final_audit').iterdir():
        path = directory/'summary.json'
        if path.is_file() and (directory/'conflict_eval.json').is_file():
            data = core.read_rows(path)
            data.update({'release':authored['release'], 'scene_v3_cases_pending':15,
                'current_prompt_version':'hierarchical_classify_multi_answer_v2',
                'existing_predictions_are_not_current_prompt_results':True})
            changes[path] = core.encode(path,data)
    prompt_doc = root/'etbench_pipeline/conflict/CONFLICT_EVAL_PROMPT.md'
    changes[prompt_doc] = ('# Conflict multi-answer evaluation contract\n\n'
        'User-confirmed 2026-09-02: at least two distinct correct options trigger UNANSWERABLE / R / CONFLICTING_EVIDENCE. '
        'The evidence may occur at different times or simultaneously in one frame; factual contradiction is not required. '
        'Each option must independently satisfy the question. Two evidence groups may share timestamps.\n\n'
        'Prompt version: hierarchical_classify_multi_answer_v2. Previous-version caches are not reused. '
        'Temporal / spatial / mixed / unclassified labels are analysis metadata only; they are not supplied to the model. '
        'No new evaluation is launched by the manual data publisher.\n\n'
        'The output retains decision probabilities, conditional option probabilities, evidence groups, and resolvability. '
        'Option-set diagnostics are separate from the required rejection decision. Both TAL and repaired TEM records may be evaluated; '
        'the existing null-official-action guard remains enabled.\n').encode('utf-8')
    announcement = ('更新：按用户澄清，至少两个选项正确即构成conflict，时间型与同帧空间型均拒答。'
        '目前正式15道scene；已标空间3道、时间1道，其余11道来源待细分；360道待修。'
        '旧9道退出结论已撤回，3道拆项后恢复，6道重新待审。没有启动新eval。')
    for path in [base/'README.md',v2/'manual_review_all_375/INDEX.md',v2/'MANUAL_WORKFLOW.md']:
        content = path.read_text(encoding='utf-8') if path.is_file() else ''
        history = '\n\n以下先前日期内容保留为历史，旧准入/退出结论不代表当前标准。\n\n'+content if content else '\n'
        changes[path] = ('> '+announcement+'\n> 正式修改记录：reports/final_benchmark_aligned/scene_manual_v3/'+authored['release']+'/REVISIONS.md'+history).encode('utf-8')
    out = base/authored['release']
    if args.apply:
        assert before == {p:core.digest(p.read_bytes()) for p in active_paths}
        core.commit(root,changes,out)
        for filename, data in [('scene_spatial_revised.jsonl',revised), ('scene_all_revised_with_types.jsonl',all_revised),
                ('scene_answer_basis.jsonl',type_index), ('scene_review_queue.jsonl',queue), ('summary.json',stats)]:
            (out/filename).write_bytes(core.encode(Path(filename),data))
        lines = ['# Scene 时间/空间多答案修订', '',
            '3道空间题已实际重写题干/选项/金标，另1道现有题补时间型证据。',
            '新3题和原12题已写入四个正式入口，共15题；时间型、空间型均拒答。其余360题按新标准待修。旧9道退出结论均重开，其中3道已修，6道回待审。',
            '评测提示已更新并使用新版本号，旧提示缓存不复用；未启动新eval。所有覆盖文件先备份，见transaction.json和backup。', '',
            '同帧证据时间为抽帧请求时间（秒），不是声称连续播放完整视频。', '']
        for row in all_revised:
            if row['answer_multiplicity']['type'] == 'unclassified':
                continue
            lines += [f"## {core.key(row)} · {row['video']} · {row['answer_multiplicity']['type']}", '', row['question'], '']
            lines += [f"- {o['option_id']}. {o['answer']} — {'正确' if o['is_valid'] else '不支持'}" for o in row['options']]
            for w in row['answer_multiplicity']['spatial_witnesses']:
                lines += ['', f"同帧：{w['seconds']:.4f}s，{w['instance_id']}，{', '.join(w['option_ids'])}。{w['judgment_zh']}"]
            for w in row['answer_multiplicity']['temporal_witnesses']:
                lines += ['', f"时间变化：{w['earlier']['seconds']:.4f}s → {w['later']['seconds']:.4f}s。{w['judgment_zh']}"]
            lines += ['']
        (out/'REVISIONS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if args.apply:
        assert all(p.read_bytes() == value for p,value in changes.items())
    else:
        assert before == {p:core.digest(p.read_bytes()) for p in active_paths}
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
