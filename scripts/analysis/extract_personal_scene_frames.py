"""Extract timestamped contact sheets for human review, without semantic models."""
import argparse
import json
from pathlib import Path
import cv2
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--keys', required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--samples', type=int, default=5)
args = p.parse_args()
scene = [json.loads(x) for x in (args.root/'reports/final_benchmark_aligned/scene_atomic_v2/scene_v1_removed.jsonl').read_text().splitlines()]
groups = {r['job_id']: r for r in map(json.loads, (args.root/'etbench_pipeline/conflict/visual_contrast_jobs.jsonl').read_text().splitlines())}
keys = set(args.keys.split(','))
args.out.mkdir(parents=True, exist_ok=True)
manifest = []
for r in scene:
    key = r['parent_job_id'].removeprefix('etbench_perception_test_')
    if key not in keys:
        continue
    group = groups[r['parent_job_id']]
    intervals = [(x['instance_id'], x['core']) for x in group['instances']]
    cap = cv2.VideoCapture(str(args.root/'source_datasets/et_bench/ETBench/videos'/r['video']))
    assert cap.isOpened(), r['video']
    fps = cap.get(cv2.CAP_PROP_FPS)
    for page in range(0, len(intervals), 4):
        chunk = intervals[page:page+4]
        sheet = Image.new('RGB', (360*args.samples, 234*len(chunk)), 'white')
        draw = ImageDraw.Draw(sheet)
        frame_meta = []
        for row, (iid, interval) in enumerate(chunk):
            a, b = interval
            for col in range(args.samples):
                sec = a+(b-a)*(col+.5)/args.samples
                cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
                ok, frame = cap.read()
                assert ok, (key, sec)
                im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im.thumbnail((360, 202))
                sheet.paste(im, (col*360, row*234+30))
                draw.text((col*360+3, row*234+4), f'{key} {iid} {sec:.3f}s', fill='black')
                frame_meta.append({'instance_id': iid, 'requested_seconds': sec, 'fps': fps})
        name = f'{key}_{page//4:02d}.jpg'
        sheet.save(args.out/name, quality=88)
        manifest.append({'key': key, 'video': r['video'], 'sheet': name, 'frames': frame_meta,
                         'intervals': chunk, 'question': r['question'], 'options': r['options']})
    cap.release()
    print(key, len(intervals)*args.samples, flush=True)
(args.out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
