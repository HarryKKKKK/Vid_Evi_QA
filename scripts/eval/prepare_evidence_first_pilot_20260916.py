"""Select three diagnostic pairs without looking at model correctness."""
import json
from pathlib import Path
import random
import sys

root = Path(sys.argv[1])
out = Path(sys.argv[2])
source = root / 'results/final_filtered_hierarchical_v2/release_20260902_v21/all_condition_records.jsonl'
rows = [json.loads(x) for x in source.read_text().splitlines()]
mapping = json.loads((root / 'source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json').read_text())
pairs = {}
for row in rows:
    if row['source_dataset'] == 'nextgqa' and row['evidence_condition'] in ('sufficient', 'insufficient'):
        pairs.setdefault(row['id'], {})[row['evidence_condition']] = row
baseline_path = root / 'results/open_v22_qwen_small_full_8192_20260907/results/qwen3_8b/merged.classify.json'
baseline = json.loads(baseline_path.read_text())
if isinstance(baseline, dict):
    baseline = baseline.get('results', baseline.get('records', []))
print('BASELINE', len(baseline), 'KEYS', list(baseline[0]) if baseline else [])
lookup = {}
for b in baseline:
    identity = b.get('record_id')
    if identity:
        lookup[identity] = b
    lookup[(str(b.get('video_id')), str(b.get('qid')), b.get('evidence_condition', b.get('mode')))] = b
eligible = []
for key, pair in sorted(pairs.items()):
    if set(pair) != {'sufficient', 'insufficient'}:
        continue
    s, i = pair['sufficient'], pair['insufficient']
    vid = str(s['video_id'])
    if vid not in mapping:
        continue
    sp = root / 'source_datasets/next_gqa/videos' / (str(mapping[vid]) + '.mp4')
    ip = root / i['video_path']
    if sp.is_file() and ip.is_file():
        eligible.append((key, pair, sp, ip))
rng = random.Random(20260916)
rng.shuffle(eligible)
selected, used = [], set()
for kind, keyword in [('before', 'before'), ('after', 'after'), ('cause', 'why')]:
    group = [x for x in eligible if x[1]['sufficient']['video_id'] not in used
             and keyword in x[1]['sufficient']['question'].lower().split()]
    if not group:
        group = [x for x in eligible if x[1]['sufficient']['video_id'] not in used]
    chosen = group[0]
    selected.append((kind, chosen))
    used.add(chosen[1]['sufficient']['video_id'])
manifest, review = [], []
for kind, (key, pair, sp, ip) in selected:
    for condition, video in [('sufficient', sp), ('insufficient', ip)]:
        row = pair[condition]
        b = lookup.get(row['record_id'], lookup.get((str(row['video_id']), str(row['qid']), condition), {}))
        item = {'id': row['record_id'], 'video_path': str(video), 'question': row['question'],
                'options': {chr(65+n): value for n, value in enumerate(row['choices'])}}
        if b.get('sampled_seconds'):
            item['sampled_seconds'] = b['sampled_seconds']
        manifest.append(item)
        review.append({'id': item['id'], 'pair_id': key, 'kind': kind, 'condition': condition,
                       'gold_options': row['valid_option_ids'], 'question': row['question'],
                       'options': item['options'], 'evidence_intervals': row['evidence_intervals'],
                       'historical_baseline': b})
        print(kind, condition, row['question'], 'baseline_match', bool(b), 'timestamps', len(item.get('sampled_seconds', [])))
out.mkdir(parents=True, exist_ok=False)
(out / 'pilot.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in manifest))
(out / 'review_only.json').write_text(json.dumps(review, indent=2))
(out / 'selection.json').write_text(json.dumps({'seed':20260916,'selection':'first shuffled before/after/why from distinct videos; no outcome selection','source':str(source)}, indent=2))
