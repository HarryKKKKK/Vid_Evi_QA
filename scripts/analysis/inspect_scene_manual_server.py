"""Read-only inventory for the personal scene repair (no model calls)."""
import hashlib
import json
from pathlib import Path

root = Path('/aifs4su/hansirui_2nd/harry/Vid_Evi_QA')
for rel in ['results/reports/final_benchmark_aligned/final/conflict_qa_five_options.jsonl',
            'results/reports/final_benchmark_aligned/scene_atomic_v2/conflict_qa_five_options.current.jsonl',
            'results/reports/final_benchmark_aligned/scene_atomic_v2/scene_v1_removed.jsonl']:
    path = root / rel
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    print(json.dumps({'path': rel, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'rows': len(rows), 'scene': sum(r.get('conflict_type') == 'scene_context' for r in rows)}))
scene = [json.loads(x) for x in (root/'results/reports/final_benchmark_aligned/scene_atomic_v2/scene_v1_removed.jsonl').read_text().splitlines()]
video_root = root / 'source_datasets/et_bench/ETBench/videos'
print('VIDEOS', len(scene), sum((video_root/r['video']).is_file() for r in scene))
for path in sorted((root/'results/reports/conflict_final_audit').glob('*/conflict_eval.json')):
    rows = json.loads(path.read_text())
    print(json.dumps({'eval': str(path.relative_to(root)), 'rows': len(rows),
                      'scene': sum(r.get('conflict_type', r.get('question_type')) == 'scene_context' for r in rows),
                      'keys': list(rows[0]) if rows else []}))
for path in sorted(root.glob('*MODEL*')):
    print('LEDGER', str(path), 'scene_lines', [x for x in path.read_text().splitlines() if 'scene' in x.lower() or 'conflict' in x.lower()][:30])
try:
    import cv2
    from PIL import Image
    print('CV2', cv2.__version__, 'PIL', Image.__version__)
except ImportError as exc:
    print('IMPORT', str(exc))
