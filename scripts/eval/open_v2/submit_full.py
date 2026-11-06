"""Submit exactly one resumable array and dependent merge job per small model."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


KINDS = ('minicpm', 'internvl_4b', 'molmo2_o_7b')


def write_atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('release', type=Path)
    parser.add_argument('--max-concurrent', type=int, default=4)
    args = parser.parse_args()
    release = args.release.resolve()
    metadata = json.loads((release/'deployment.json').read_text())
    shards = int(metadata.get('num_shards') or 0)
    if metadata.get('sample_profile') != 'full' or shards < 1:
        raise SystemExit('Release is not a sharded full evaluation')
    if args.max_concurrent < 1:
        raise SystemExit('--max-concurrent must be positive')
    receipt_path = release/'submission.json'
    if (release/'submission.lock').exists():
        raise SystemExit('Submission lock exists; inspect Slurm and receipt before retrying')
    lock_fd = os.open(release/'submission.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(lock_fd)
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    try:
        for kind in KINDS:
            if kind in receipt:
                print(f'ALREADY_SUBMITTED {kind}', flush=True)
                continue
            array_cmd = ['sbatch', '--parsable', f'--job-name=v22-{kind}',
                         f'--array=0-{shards-1}%{args.max_concurrent}',
                         str(release/'adapters/full_small_array.sbatch'), str(release), kind]
            array_id = subprocess.check_output(array_cmd, text=True).strip().split(';')[0]
            if not array_id.isdigit():
                raise RuntimeError(f'Unexpected array submission response for {kind}: {array_id}')
            merge_cmd = ['sbatch', '--parsable', f'--job-name=v22m-{kind}',
                         f'--dependency=afterany:{array_id}',
                         str(release/'adapters/merge_full.sbatch'), str(release), kind]
            merge_id = subprocess.check_output(merge_cmd, text=True).strip().split(';')[0]
            if not merge_id.isdigit():
                raise RuntimeError(f'Unexpected merge submission response for {kind}: {merge_id}')
            receipt[kind] = {'array_job_id':array_id, 'merge_job_id':merge_id,
                             'array_command':array_cmd, 'merge_command':merge_cmd,
                             'submitted_utc':datetime.now(timezone.utc).isoformat()}
            write_atomic(receipt_path, receipt)
            print(f'SUBMITTED {kind} array={array_id} merge={merge_id}', flush=True)
    finally:
        (release/'submission.lock').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
