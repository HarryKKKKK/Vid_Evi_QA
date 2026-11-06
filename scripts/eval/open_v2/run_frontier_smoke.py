"""Run one pinned frontier model on the frozen v2.2 format20 smoke set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from frontier_models import MODELS

ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
QENV = Path("/aifs4su/hansirui_2nd/harry/envs_qwen3vl")
NEW_ENV = ROOT / ".runtime/vllm0191"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("kind", choices=MODELS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("This smoke runner requires a Slurm GPU allocation")
    spec = MODELS[args.kind]
    if spec["backend"] == "transformers_native":
        raise SystemExit("VideoLLaMA3 uses run_videollama3_smoke.py")
    release = args.release.resolve()
    manifest = Path(os.environ.get("V2_MANIFEST", release / "smoke.jsonl")).resolve()
    if not manifest.is_relative_to(release) or not manifest.is_file():
        raise RuntimeError("Manifest must be a frozen file inside the release")
    deployment = json.loads((release / "deployment.json").read_text())
    for relative, expected in deployment["sha256"].items():
        actual = hashlib.sha256((release / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen file changed: {relative}")
    if spec["name"] not in deployment["models"]:
        raise RuntimeError("Model absent from frozen deployment")
    runtime = NEW_ENV if spec["backend"].startswith("vllm_0191") else QENV
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    served = args.kind
    url = f"http://127.0.0.1:{port}/v1"
    configured_result = os.environ.get("V2_RESULT_DIR")
    result_dir = (Path(configured_result).resolve() if configured_result else
                  release / "results" / f"{args.kind}_{os.environ['SLURM_JOB_ID']}")
    if not result_dir.is_relative_to((release / "results").resolve()):
        raise RuntimeError("Result directory must stay inside the frozen release")
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(runtime / "bin/vllm"), "serve", str(ROOT / "models" / spec["name"]),
        "--served-model-name", served, "--trust-remote-code",
        "--tensor-parallel-size", str(spec["gpus"]),
        "--max-model-len", str(spec["length"]),
        "--limit-mm-per-prompt", '{"image":0,"video":1}',
        "--media-io-kwargs", '{"video":{"num_frames":32}}',
        "--mm-processor-cache-gb", "0", "--no-enable-prefix-caching",
        "--gpu-memory-utilization", str(spec.get("gpu_memory_utilization", 0.92)), "--enforce-eager",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    if spec.get("reasoning_parser"):
        command += ["--reasoning-parser", spec["reasoning_parser"]]
    if spec.get("cpu_offload_gb"):
        command += ["--cpu-offload-gb", str(spec["cpu_offload_gb"])]
    if spec.get("mm_encoder_tp_mode"):
        command += ["--mm-encoder-tp-mode", spec["mm_encoder_tp_mode"]]
    if args.kind == "kimi_k25_int4":
        command += ["--kv-cache-dtype", "fp8", "--max-num-seqs", "1"]
    env = dict(os.environ)
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
        env.pop(key, None)
    env.update(
        NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1",
        PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        CUDA_HOME="/usr/local/cuda", PATH=str(runtime / "bin") + ":" + str(QENV / "bin") + ":" + env["PATH"],
        VLLM_BASE_URL=url, VLLM_MODEL=served,
        V2_REQUEST_EXTRA=json.dumps(spec.get("request_extra", {})),
        V2_MAX_TOKENS=str(spec.get("max_output_tokens", 768)),
    )
    status = {
        "status": "starting", "model": spec["name"], "model_key": args.kind,
        "revision": spec["revision"], "job_id": os.environ["SLURM_JOB_ID"],
        "runtime": str(runtime), "command": command, "manifest": str(manifest),
        "sampling": "32 timestamp-labeled frames, width560, uniform",
        "json_constraint": True, "logprobs_available": True,
        "max_output_tokens": spec.get("max_output_tokens", 768),
    }
    status_file = result_dir / "deployment_status.json"
    process = None
    try:
        probe = subprocess.check_output([
            str(runtime / "bin/python"), "-c",
            "import json,os,torch;print(json.dumps({'visible':os.getenv('CUDA_VISIBLE_DEVICES'),'count':torch.cuda.device_count(),'devices':[{'name':torch.cuda.get_device_name(i),'free_gib':torch.cuda.mem_get_info(i)[0]/1024**3,'total_gib':torch.cuda.mem_get_info(i)[1]/1024**3} for i in range(torch.cuda.device_count())]}))",
        ], env=env, text=True)
        status["gpu_allocation"] = json.loads(probe.strip().splitlines()[-1])
        if status["gpu_allocation"]["count"] != spec["gpus"]:
            raise RuntimeError(f"Expected {spec['gpus']} allocated GPUs")
        low = [device for device in status["gpu_allocation"]["devices"]
               if device["free_gib"] < spec.get("min_free_gib", 40)]
        if low:
            raise RuntimeError(f"Allocated GPU(s) below required free memory: {low}")
        with (result_dir / "server.log").open("w") as server_log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=server_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(360):
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}; inspect server.log")
                try:
                    with urllib.request.urlopen(url + "/models", timeout=3) as response:
                        if served in [row["id"] for row in json.load(response)["data"]]:
                            break
                except Exception:
                    time.sleep(5)
            else:
                raise RuntimeError("Server startup timed out after 30 minutes")
        status["status"] = "testing"
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        with (result_dir / "evaluation.log").open("w") as evaluation_log:
            subprocess.run([
                str(QENV / "bin/python"), "-u", str(release / "adapters/evaluate_open.py"),
                "--manifest-jsonl", str(manifest), "--task", "classify",
                "--video-dir", str(ROOT / "source_datasets/next_gqa/videos"),
                "--results-dir", str(result_dir), "--model-tag", os.environ.get("V2_MODEL_TAG", f"{args.kind}_smoke_v2_2"),
                "--num-frames", "32", "--frame-width", "560", "--sampling", "uniform",
                "--media-mode", "video", "--workers", os.environ.get("V2_WORKERS", "1"),
            ], cwd=ROOT, env=env, stdout=evaluation_log, stderr=subprocess.STDOUT, check=True)
        validation_process = subprocess.run([
            str(QENV / "bin/python"), str(release / "eval/check_v2_run.py"), str(manifest),
            "--result", str(result_dir / "merged.classify.json"),
        ], cwd=ROOT, env=env)
        validation = json.loads((result_dir / "merged.classify.validation.json").read_text())
        status["validation"] = validation
        if validation["missing_ids"] or validation["obsolete_ids"] or validation["result_records"] != validation["expected_records"]:
            raise RuntimeError("Inference/transport incomplete")
        status["status"] = "passed" if validation_process.returncode == 0 else "ready_with_schema_warnings"
    except Exception as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
