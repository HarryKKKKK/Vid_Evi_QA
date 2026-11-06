"""Freeze reproducible inputs/code, verify them, and submit four v2 jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from script_export import copy_script
import subprocess
from datetime import datetime, timezone
from check_v2_run import read_manifest
from prompt_v2 import PROMPT_VERSION, PROMPT_SHA256

ROOT = Path('/aifs4su/hansirui_2nd/harry/Vid_Evi_QA')
FILES = ['evaluate.py', 'prompt_v2.py', 'check_v2_run.py', 'run_merged_model.sh',
         'eval_merged_v2.sbatch', 'manage_v2_release.py', 'script_export.py',
         'run_eval.sh', 'vllm_models.sh', 'vllm_service.sh']
RUNNER_FILES = {'run_eval.sh', 'vllm_models.sh', 'vllm_service.sh'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(release):
    config = json.loads((release/'release.json').read_text())
    for name, expected in config['sha256'].items():
        if sha(release/name) != expected:
            raise SystemExit(f'release modified: {name}')
    rows = read_manifest(release/'all_condition_records.jsonl')
    if len(rows) != config['records']:
        raise SystemExit('manifest size mismatch')
    print(f'RELEASE_VERIFIED records={len(rows)} release={release}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'verify', 'submit'])
    parser.add_argument('release', type=Path)
    args = parser.parse_args()
    release = args.release.resolve()
    if args.action == 'prepare':
        if release.exists():
            raise SystemExit('release already exists; verify/reuse or choose a new name')
        source = ROOT/'reports/final_benchmark_aligned/final/all_condition_records.jsonl'
        rows = read_manifest(source)
        (release/'code').mkdir(parents=True)
        shutil.copy2(source, release/'all_condition_records.jsonl')
        for name in FILES:
            source_dir = ROOT/'configs' if name == 'vllm_models.sh' else ROOT/'jobs/lib' if name in RUNNER_FILES else ROOT/'scripts/eval'
            copy_script(source_dir/name, release/'code'/name)
        config = {'created_utc': datetime.now(timezone.utc).isoformat(), 'prompt_version': PROMPT_VERSION,
                  'prompt_sha256': PROMPT_SHA256,
                  'records': len(rows), 'source_manifest': str(source), 'media_mode': 'frames',
                  'notes': 'All four models use 32 timestamp-labeled images. Old results are not imported.',
                  'sha256': {name: sha(release/name) for name in ['all_condition_records.jsonl'] + ['code/'+x for x in FILES]}}
        (release/'release.json').write_text(json.dumps(config, indent=2)+'\n')
    verify(release)
    if args.action == 'submit':
        # Exclusive lock prevents duplicate dispatch. If dispatch is interrupted,
        # inspect Slurm and the receipt before manually resolving the lock.
        lock = release/'submission.lock'
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        receipt_path = release/'submission.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        for kind, gpus in [('qwen25', 2), ('qwen3', 2), ('internvl', 1), ('llava', 1)]:
            if kind in receipt:
                continue
            cmd = ['sbatch', '--parsable', f'--job-name=ev2-{kind}', f'--gres=gpu:{gpus}',
                   str(release/'code/eval_merged_v2.sbatch'), str(release), kind]
            output = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
            jobid = output.split(';')[0]
            if not jobid.isdigit():
                raise RuntimeError(f'Unexpected sbatch output; inspect queue before retry: {output}')
            receipt[kind] = {'job_id': jobid, 'command': cmd, 'submitted_utc': datetime.now(timezone.utc).isoformat()}
            temp = receipt_path.with_suffix('.tmp')
            temp.write_text(json.dumps(receipt, indent=2)+'\n')
            temp.replace(receipt_path)
            print(f'SUBMITTED {kind} {jobid}', flush=True)
        lock.unlink()


if __name__ == '__main__':
    main()
