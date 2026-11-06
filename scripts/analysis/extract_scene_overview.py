"""One midpoint per official interval; visual triage only, never auto-labels QA."""
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
base = args.root/'results/reports/final_benchmark_aligned/scene_manual_v3'
pending = [json.loads(x) for x in (base/'manual_scene_v3_20260902_batch01/scene_pending.jsonl').read_text().splitlines()]
groups = {r['job_id']: r for r in map(json.loads, (args.root/'results/build/etbench/conflict/visual_contrast_jobs.jsonl').read_text().splitlines())}
args.out.mkdir(parents=True, exist_ok=True)
pages = []
for start in range(0, len(pending), 4):
    chunk = pending[start:start+4]
    height = sum(((len(groups[r['source_record']['parent_job_id']]['instances'])+3)//4)*204 for r in chunk)
    sheet = Image.new('RGB', (1280, height), 'white')
    draw = ImageDraw.Draw(sheet)
    y, entries = 0, []
    for entry in chunk:
        r, key = entry['source_record'], entry['key']
        group = groups[r['parent_job_id']]
        cap = cv2.VideoCapture(str(args.root/'source_datasets/et_bench/ETBench/videos'/r['video']))
        frames = []
        for i, ins in enumerate(group['instances']):
            a, b = ins['core']
            t = (a+b)/2
            cap.set(cv2.CAP_PROP_POS_MSEC, t*1000)
            ok, frame = cap.read()
            assert ok, (key, t)
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((320, 180))
            xx, yy = (i%4)*320, y+(i//4)*204
            sheet.paste(im, (xx, yy+24))
            draw.text((xx+3, yy+3), f'{key} {ins["instance_id"]} {t:.2f}s', fill='black')
            frames.append({'instance_id':ins['instance_id'], 'interval':ins['core'], 'second':t})
        cap.release()
        entries.append({'key':key,'video':r['video'],'frames':frames,'question':r['question'],
                        'options':[[o['option_id'],o['answer'],o['is_valid']] for o in r['options']]})
        y += ((len(group['instances'])+3)//4)*204
    name = f'overview_{start//4:03d}.jpg'
    sheet.save(args.out/name, quality=90)
    page = {'sheet':name,'entries':entries,'method':'one_midpoint_per_official_interval_triage_not_full_verification'}
    (args.out/name.replace('.jpg','.json')).write_text(json.dumps(page,ensure_ascii=False,indent=2))
    pages.append(page)
    print(name, ','.join(x['key'] for x in entries), flush=True)
(args.out/'manifest.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2))
