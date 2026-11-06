"""Freeze small development and source-video-disjoint validation sets before tuning."""
import json
from pathlib import Path
import random
import sys
from collections import Counter

root, out = map(Path, sys.argv[1:3])
old = root/'results/evidence_first/pilot_20260916_v4'
old_inputs = [json.loads(x) for x in (old/'pilot.jsonl').read_text().splitlines()]
old_review = json.loads((old/'review_only.json').read_text())
source = root/'results/open_v22_qwen_small_full_8192_20260907/results/qwen3_8b/merged.classify.json'
rows = json.loads(source.read_text())
print('conditions', Counter(r['evidence_condition'] for r in rows))
mapping = json.loads((root/'source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json').read_text())
rng = random.Random(2026091602)
def cluster(r):
    return str(r['source_dataset']) + '::' + str(r['video_id'])
def path(r):
    if r.get('video_path'):
        p = Path(r['video_path'])
        return p if p.is_absolute() else root/p
    if r['source_dataset'] == 'nextgqa':
        return root/'source_datasets/next_gqa/videos'/(str(mapping[str(r['video_id'])])+'.mp4')
    if r['source_dataset'] == 'etbench' and r.get('video'):
        return root/'source_datasets/et_bench/ETBench/videos'/r['video']
    raise ValueError('No explicit video: '+json.dumps({k:v for k,v in r.items() if k not in ['api_logprobs','model_response','model_parsed']})[:9000])
def convert(r, split):
    options = {chr(65+i):x for i,x in enumerate(r['choices'])}
    item = dict(id=r.get('record_id',r['id']),video_path=str(path(r)),question=r['question'],options=options)
    if r.get('sampled_seconds'):
        item['sampled_seconds'] = r['sampled_seconds']
    rev = dict(id=item['id'], pair_id=r['id'], kind=r.get('question_type','conflict'),
               condition=r['evidence_condition'], gold_options=r['valid_option_ids'],
               question=r['question'], options=options, evidence_intervals=r.get('evidence_intervals',[]),
               split=split, source_cluster=cluster(r), historical_baseline=r)
    return item, rev
dev, dev_review = list(old_inputs), list(old_review)
used = {cluster(r['historical_baseline']) for r in old_review}
for r in dev_review:
    r['split']='development'
    r['source_cluster']=cluster(r['historical_baseline'])
conflicts = [r for r in rows if r['evidence_condition']=='conflict' and len(r['valid_option_ids'])>=2]
rng.shuffle(conflicts)
selected_conflicts=[]
for r in conflicts:
    if cluster(r) not in used and path(r).is_file():
        selected_conflicts.append(r)
        used.add(cluster(r))
        if len(selected_conflicts)==6:
            break
assert len(selected_conflicts)==6
for r in selected_conflicts[:3]:
    item, rev=convert(r,'development'); dev.append(item); dev_review.append(rev)
validation, val_review=[],[]
for r in selected_conflicts[3:]:
    item, rev=convert(r,'validation'); validation.append(item); val_review.append(rev)
pairs={}
for r in rows:
    if r['source_dataset']=='nextgqa' and r['evidence_condition'] in ['sufficient','insufficient']:
        pairs.setdefault(r['id'],{})[r['evidence_condition']]=r
candidates=list(pairs.values()); rng.shuffle(candidates)
selected_pairs=[]
for pair in candidates:
    if len(pair)==2 and cluster(pair['sufficient']) not in used and all(path(r).is_file() for r in pair.values()):
        selected_pairs.append(pair); used.add(cluster(pair['sufficient']))
        if len(selected_pairs)==3: break
for pair in selected_pairs:
    for condition in ['sufficient','insufficient']:
        item, rev=convert(pair[condition],'validation'); validation.append(item); val_review.append(rev)
assert not ({r['source_cluster'] for r in dev_review} & {r['source_cluster'] for r in val_review})
out.mkdir(parents=True,exist_ok=False)
for name, inputs, review in [('dev',dev,dev_review),('validation',validation,val_review)]:
    (out/(name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in inputs))
    (out/(name+'_review.json')).write_text(json.dumps(review,indent=2))
    print(name, len(inputs), 'conditions', Counter(r['condition'] for r in review))
(out/'selection.json').write_text(json.dumps({'seed':2026091602,'source':str(source),
    'development':'previous three pairs plus three shuffled conflict examples',
    'validation':'three unseen pairs plus three unseen conflicts; source-video disjoint',
    'policy':'no selection by model outcome; validation labels not inspected during development'},indent=2))
for r in dev_review:
    print(r['condition'],r['question'])
