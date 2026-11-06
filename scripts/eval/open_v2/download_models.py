"""Download explicitly selected pinned public checkpoints; no environment changes."""
import argparse
import json
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download
from models import MODELS

ROOT = Path('/aifs4su/hansirui_2nd/harry/Vid_Evi_QA')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', choices=list(MODELS), required=True)
    args = parser.parse_args()
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise SystemExit('Less than 100 GiB free; refusing model downloads')
    for key in args.models:
        spec = MODELS[key]
        if 'repo' not in spec:
            continue
        dest = ROOT/'models'/spec['name']
        print('DOWNLOADING', key, spec['revision'], flush=True)
        snapshot_download(spec['repo'], revision=spec['revision'], local_dir=dest, max_workers=4)
        config = json.loads((dest/'config.json').read_text())
        for index in dest.glob('*.index.json'):
            for shard in set(json.loads(index.read_text())['weight_map'].values()):
                if not (dest/shard).is_file():
                    raise RuntimeError(f'Missing {dest/shard}')
        print('READY', key, config.get('architectures'), flush=True)

if __name__ == '__main__':
    main()
