"""Submit audited full arrays for frontier models that passed format20 smoke."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


KINDS = ("videollama3_7b", "qwen36_35b_a3b_fp8")


def atomic_write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--max-concurrent", type=int, default=4)
    args = parser.parse_args()
    release = args.release.resolve()
    metadata = json.loads((release / "deployment.json").read_text())
    shards = int(metadata.get("num_shards") or 0)
    if metadata.get("sample_profile") != "full" or shards < 1:
        raise SystemExit("Release is not a sharded full evaluation")
    expected = {"VideoLLaMA3-7B", "Qwen3.6-35B-A3B-FP8"}
    if expected - set(metadata.get("models", {})):
        raise SystemExit("Frozen release is missing a requested model")
    lock = release / "submission.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    receipt_path = release / "submission.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    try:
        for kind in KINDS:
            if kind in receipt:
                continue
            script = (release / "adapters/videollama3_full_array.sbatch" if kind == "videollama3_7b"
                      else release / "adapters/frontier_full_array.sbatch")
            extra = [] if kind == "videollama3_7b" else [kind]
            array_command = ["sbatch", "--parsable", f"--job-name=v22f-{kind}",
                             "--exclude=dgx-100",
                             f"--array=0-{shards - 1}%{args.max_concurrent}",
                             str(script), str(release), *extra]
            array_id = subprocess.check_output(array_command, text=True).strip().split(";")[0]
            merge_command = ["sbatch", "--parsable", f"--job-name=v22fm-{kind}",
                             f"--dependency=afterany:{array_id}",
                             str(release / "adapters/merge_full.sbatch"), str(release), kind]
            merge_id = subprocess.check_output(merge_command, text=True).strip().split(";")[0]
            if not array_id.isdigit() or not merge_id.isdigit():
                raise RuntimeError(f"Unexpected Slurm response: {array_id}, {merge_id}")
            receipt[kind] = {"array_job_id": array_id, "merge_job_id": merge_id,
                             "array_command": array_command, "merge_command": merge_command,
                             "submitted_utc": datetime.now(timezone.utc).isoformat()}
            atomic_write(receipt_path, receipt)
            print(f"SUBMITTED {kind} array={array_id} merge={merge_id}", flush=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
