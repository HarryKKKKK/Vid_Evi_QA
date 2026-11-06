"""Run VideoLLaMA3 native engineering smoke under a Slurm GPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from frontier_models import MODELS

ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
QENV = Path("/aifs4su/hansirui_2nd/harry/envs_qwen3vl")
VIDEO_ENV = ROOT / ".runtime/videollama3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("This smoke runner requires a Slurm GPU allocation")
    release = args.release.resolve()
    manifest = Path(os.environ.get("V2_MANIFEST", release / "smoke.jsonl")).resolve()
    if not manifest.is_relative_to(release) or not manifest.is_file():
        raise RuntimeError("Manifest must be a frozen file inside the release")
    spec = MODELS["videollama3_7b"]
    deployment = json.loads((release / "deployment.json").read_text())
    for relative, expected in deployment["sha256"].items():
        if hashlib.sha256((release / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen file changed: {relative}")
    configured_result = os.environ.get("V2_RESULT_DIR")
    result_dir = (Path(configured_result).resolve() if configured_result else
                  release / "results" / f"videollama3_7b_{os.environ['SLURM_JOB_ID']}")
    if not result_dir.is_relative_to((release / "results").resolve()):
        raise RuntimeError("Result directory must stay inside the frozen release")
    result_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "testing", "model": spec["name"], "revision": spec["revision"],
        "job_id": os.environ["SLURM_JOB_ID"], "backend": "transformers_native",
        "sampling": "the same 32 timestamp-labeled JPEG frames, width560, uniform",
        "json_constraint": False, "logprobs_available": False,
        "evaluation_scope": "engineering_smoke",
        "max_output_tokens": spec.get("max_output_tokens", 8192),
        "note": "Prompt and selected JPEG bytes are unchanged, but native generation has no JSON grammar or token logprobs; do not treat this as a formal protocol-equivalent score.",
    }
    status_file = result_dir / "deployment_status.json"
    env = dict(os.environ)
    env.update(
        PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        VIDEOLLAMA3_MODEL_PATH=str(ROOT / "models" / spec["name"]),
        VLLM_MODEL="videollama3_7b",
        V2_MAX_TOKENS=str(spec.get("max_output_tokens", 8192)),
        PATH=str(QENV / "bin") + ":" + env["PATH"],
    )
    try:
        probe = subprocess.check_output([
            str(VIDEO_ENV / "bin/python"), "-c",
            "import json,torch;print(json.dumps({'count':torch.cuda.device_count(),'free_gib':torch.cuda.mem_get_info(0)[0]/1024**3 if torch.cuda.device_count() else 0,'name':torch.cuda.get_device_name(0) if torch.cuda.device_count() else None}))",
        ], env=env, text=True)
        status["gpu_allocation"] = json.loads(probe.strip().splitlines()[-1])
        if status["gpu_allocation"]["count"] != 1 or status["gpu_allocation"]["free_gib"] < 20:
            raise RuntimeError(f"Allocated GPU is unavailable or occupied: {status['gpu_allocation']}")
        with (result_dir / "evaluation.log").open("w") as log:
            subprocess.run([
                str(VIDEO_ENV / "bin/python"), "-u", str(release / "adapters/evaluate_videollama3.py"),
                "--manifest-jsonl", str(manifest), "--task", "classify",
                "--video-dir", str(ROOT / "source_datasets/next_gqa/videos"),
                "--results-dir", str(result_dir), "--model-tag", os.environ.get("V2_MODEL_TAG", "videollama3_7b_native_smoke_v2_2"),
                "--num-frames", "32", "--frame-width", "560", "--sampling", "uniform",
                "--media-mode", "video", "--workers", "1",
            ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        validation_process = subprocess.run([
            str(QENV / "bin/python"), str(release / "eval/check_v2_run.py"), str(manifest),
            "--result", str(result_dir / "merged.classify.json"),
        ], cwd=ROOT, env=env)
        validation = json.loads((result_dir / "merged.classify.validation.json").read_text())
        status["validation"] = validation
        if validation["missing_ids"] or validation["result_records"] != validation["expected_records"]:
            raise RuntimeError("Inference/transport incomplete")
        status["status"] = "passed" if validation_process.returncode == 0 else "ready_with_schema_warnings"
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
