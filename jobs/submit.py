"""Plan/submit configured jobs. Uses only Python's standard library; no GPU imports."""
from __future__ import annotations

import argparse
import hashlib
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs"
OVERRIDES = {"partition", "qos", "gres", "cpus-per-task", "mem", "time", "array",
             "dependency", "account", "nodelist", "exclude", "job-name"}


def read_config(name):
    return json.loads((CONFIG / name).read_text(encoding="utf-8"))


def assignments(values, allowed=None):
    result = {}
    for value in values:
        key, sep, val = value.partition("=")
        if not sep or not val or "\n" in val or "\0" in val:
            raise ValueError("Expected a nonempty KEY=VALUE without newlines")
        if allowed is not None and key not in allowed:
            raise ValueError(f"Unsupported resource override: {key}")
        if allowed is None and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
            raise ValueError(f"Invalid environment name: {key}")
        result[key] = val
    return result


def runtime_path(value):
    return PurePosixPath(value) if str(value).startswith("/") else Path(value).resolve()


def at_root(root, value):
    path = PurePosixPath(value) if type(root) is PurePosixPath else Path(value)
    return path if path.is_absolute() else root / path


def partial_preflight(root, env):
    candidates = at_root(root, env.get("PARTIAL_CANDIDATES", "nextgqa_pipeline/partial_candidates_n500_seed20260819.json"))
    video_dir = at_root(root, env.get("PARTIAL_VIDEO_DIR", "source_datasets/next_gqa/partial_videos_no_evidence_leak"))
    rows = json.loads(candidates.read_text(encoding="utf-8"))
    alphas = [float(v) for v in env.get("PARTIAL_ALPHAS", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9").split(",")]
    anchors = int(env.get("PARTIAL_EXPECTED_ANCHORS", 500))
    tasks = int(env.get("PARTIAL_EXPECTED_TASKS", 4500))
    if len(rows) != anchors or len(rows) * len(alphas) != tasks:
        raise ValueError("Partial candidate/alpha counts do not match the preset")
    missing = sum(not (video_dir / f"{row['video_id']}_q{row['qid']}_c2_noise_a{round(alpha * 100):03d}.mp4").is_file()
                  for row in rows for alpha in alphas)
    if missing:
        raise ValueError(f"Missing {missing} pre-generated Partial videos. On-demand noise is not implemented; no job submitted.")


def plan(args):
    cluster = read_config("cluster.json")
    root = runtime_path(args.root or os.environ.get("VID_EVI_ROOT") or cluster["root"])
    env_dir = args.python_env or os.environ.get("VID_EVI_ENV") or cluster["python_env"]
    env = assignments(args.env)
    env.update(VID_EVI_ROOT=str(root), VID_EVI_ENV=env_dir)
    warnings = []
    body = None
    if args.action == "eval":
        key = f"{args.name}.{args.model}"
        presets = read_config("evaluations.json")
        if key not in presets:
            raise ValueError(f"Unknown evaluation preset {key}; run 'list'")
        preset = copy.deepcopy(presets[key])
        model = read_config("models.json")[args.model]
        resources = dict(cluster["profiles"][model["resource"]], **preset["slurm"])
        # Explicit config wins over ambient shell state; --env wins over the preset.
        runtime_keys = {"EVAL_TASK", "QWEN_USE_FAST", "FILTERED_JSON", "VIDEO_DIR", "FREEZE_DIR", "MERGED_MANIFEST", "EVAL_CODE_DIR", "VLLM_PORT", "EVAL_WORKERS", "PARTIAL_FILTERED_JSON", "PARTIAL_CANDIDATES", "PARTIAL_VIDEO_DIR", "PARTIAL_ALPHAS", "PARTIAL_EXPECTED_ANCHORS", "PARTIAL_EXPECTED_TASKS", "CONFLICT_QA_PATH", "CONFLICT_FRAMES_PER_OCCURRENCE", "CONFLICT_MAX_FRAMES", "CONFLICT_FRAME_WIDTH", "CONFLICT_ALLOW_NULL_ACTION", "CONFLICT_SOURCE_TASK"}
        ambient = {k: os.environ[k] for k in runtime_keys if k in os.environ}
        env = {**ambient, **preset["env"], **env}
        if args.manifest:
            env["MERGED_MANIFEST" if preset["mode"] == "merged" else "CONFLICT_QA_PATH" if preset["mode"] == "conflict_eval" else "FILTERED_JSON"] = str(at_root(root, args.manifest))
            if preset["mode"] == "partial":
                env["PARTIAL_FILTERED_JSON"] = str(at_root(root, args.manifest))
        output = str(at_root(root, args.results_dir or preset["results_dir"]))
        tag = args.model_tag or preset["model_tag"]
        runner = "etbench_pipeline/conflict/run_conflict_eval_model.sh" if preset["mode"] == "conflict_eval" else "jobs/lib/run_eval.sh"
        command = ["bash", str(root / runner)]
        if preset["mode"] != "conflict_eval":
            command.append(preset["mode"])
        command += [model["kind"], str(at_root(root, args.model_path or model["path"])), model["served_name"], tag, output]
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd -- " + shlex.quote(str(root))]
        lines += [f"export {k}={shlex.quote(v)}" for k, v in sorted(env.items())]
        if preset.get("seed_source"):
            result = str(type(root)(output) / "partial.classify.jsonl")
            seed = [str(runtime_path(env_dir) / "bin/python"), str(root / "scripts/eval/seed_partial_results.py"),
                    "--source", str(at_root(root, preset["seed_source"])),
                    "--candidates", str(at_root(root, env["PARTIAL_CANDIDATES"])),
                    "--alphas", env["PARTIAL_ALPHAS"], "--model-tag", tag, "--output-dir", output]
            lines += [f"if [[ ! -s {shlex.quote(result)} ]]; then", "  " + shlex.join(seed), "fi"]
        lines.append("exec " + shlex.join(command))
        body = "\n".join(lines) + "\n"
        execution = ["bash", "-s"]
        if preset.get("requires_partial_videos"):
            warnings.append("Requires pre-generated Partial videos removed in storage cleanup; submission is blocked if any are missing.")
    else:
        jobs = read_config("jobs.json")
        if args.name not in jobs:
            raise ValueError(f"Unknown job {args.name}; run 'list'")
        job = jobs[args.name]
        # Dedicated/frozen scripts retain host-specific paths and resource defaults.
        if root != runtime_path(cluster["root"]) or env_dir != cluster["python_env"]:
            raise ValueError("Dedicated jobs retain HKUST paths; root/runtime overrides currently apply only to 'eval'")
        resources = dict(job["resources"])
        execution = ["bash", str(root / job["script"]), *args.script_args]
        command = execution
        script_text = (REPO / job["script"]).read_text(encoding="utf-8")
        min_args = job.get("min_args", 0)
        if len(args.script_args) < min_args:
            raise ValueError(f"{args.name} needs {min_args} positional arguments: {job.get('usage', '')}")
        if "SLURM_ARRAY_TASK_ID" in script_text and not resources.get("array"):
            warnings.append("Array index required: provide --array for Slurm or SLURM_ARRAY_TASK_ID for local execution.")
        key = args.name
    resources.update(assignments(args.resource, OVERRIDES))
    if args.array:
        resources["array"] = args.array
    if args.dependency:
        resources["dependency"] = args.dependency
    if args.action == "eval" and resources.get("array"):
        raise ValueError("Evaluation presets do not shard by array index; use a release array job to avoid duplicate writers")
    resources.setdefault("job-name", key.replace(".", "-"))
    # CLI output/error override #SBATCH directives, avoiding log opens before mkdir.
    resources.setdefault("output", f"logs/jobs/{key}/%A_%a.out")
    resources.setdefault("error", f"logs/jobs/{key}/%A_%a.err")
    for name in ("output", "error"):
        resources[name] = str(at_root(root, resources[name]))
    sbatch = ["sbatch", "--parsable", f"--chdir={root}"]
    sbatch += [f"--{k}={v}" for k, v in resources.items()]
    if body is None:
        sbatch += execution[1:]
    return {"name": key, "root": str(root), "environment": env, "resources": resources,
            "command": command, "local_command": execution, "sbatch": sbatch, "script": body,
            "warnings": warnings, "partial": args.action == "eval" and preset.get("requires_partial_videos", False)}


def execute(p, local=False):
    root = Path(p["root"])
    if not root.is_dir():
        raise ValueError(f"Project root does not exist: {root}")
    if local and p["resources"].get("dependency"):
        raise ValueError("--dependency requires Slurm; local execution cannot enforce it")
    if p["partial"]:
        partial_preflight(root, p["environment"])
    if any("Array index required" in w for w in p["warnings"]):
        if local and not (p["environment"].get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_ARRAY_TASK_ID")):
            raise ValueError("Local array task requires SLURM_ARRAY_TASK_ID")
        if not local and not p["resources"].get("array"):
            raise ValueError("This job requires --array")
    if local and p["resources"].get("array") and not (p["environment"].get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_ARRAY_TASK_ID")):
        raise ValueError("Local execution runs one array task; set SLURM_ARRAY_TASK_ID explicitly")
    if not local and not shutil.which("sbatch"):
        raise ValueError("sbatch is unavailable on this host; use a Slurm submission host or explicit --run-local")
    for name in ("output", "error"):
        Path(p["resources"][name]).parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **p["environment"])
    if local:
        return subprocess.run(p["local_command"], input=p["script"], text=True, cwd=root, env=env).returncode
    receipts = root / "reports/job_submissions"
    receipts.mkdir(parents=True, exist_ok=True)
    receipt = receipts / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8] + ".json")
    # Write an intent first: interrupted submission must not be blindly retried.
    record = {"status": "submitting", "name": p["name"], "command": p["sbatch"], "created_utc": datetime.now(timezone.utc).isoformat(),
              "runner_command": p["command"],
              "script_sha256": hashlib.sha256((p["script"] or "").encode()).hexdigest()}
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(p["sbatch"], input=p["script"], text=True, cwd=root, env=env, capture_output=True)
    record.update(stdout=result.stdout.strip(), stderr=result.stderr.strip(), returncode=result.returncode)
    job_id = result.stdout.strip().split(";")[0]
    record["status"] = "submitted" if result.returncode == 0 and job_id.isdigit() else "inspect-before-retry"
    if record["status"] == "submitted":
        record["job_id"] = job_id
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"Receipt: {receipt}")
    return 0 if record["status"] == "submitted" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="List evaluation presets and specialized jobs")
    for action in ("eval", "job"):
        child = sub.add_parser(action)
        child.add_argument("name", help="Experiment (eval) or registered job name (job)")
        child.add_argument("--root")
        child.add_argument("--python-env")
        child.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
        child.add_argument("--resource", action="append", default=[], metavar="KEY=VALUE")
        child.add_argument("--array")
        child.add_argument("--dependency")
        mode = child.add_mutually_exclusive_group()
        mode.add_argument("--submit", action="store_true")
        mode.add_argument("--run-local", action="store_true")
        mode.add_argument("--dry-run", action="store_true", help="Default: print plan without side effects")
        if action == "eval":
            child.add_argument("--model", required=True, choices=list(read_config("models.json")))
            child.add_argument("--model-path")
            child.add_argument("--manifest")
            child.add_argument("--results-dir")
            child.add_argument("--model-tag")
    # Script arguments are deliberately separated; unknown options never silently pass through.
    raw = list(sys.argv[1:] if argv is None else argv)
    split = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:split])
    args.script_args = raw[split + 1:]
    if args.action != "job" and args.script_args:
        parser.error("Only 'job' accepts positional script arguments after --")
    if args.action == "list":
        print("Evaluation presets (eval EXPERIMENT --model MODEL):")
        for key, value in sorted(read_config("evaluations.json").items()):
            print(f"  {key}" + (" [requires Partial videos]" if value.get("requires_partial_videos") else ""))
        print("Specialized jobs (job NAME -- ARGUMENTS):")
        for key, value in sorted(read_config("jobs.json").items()):
            print(f"  {key}: {value['script']}")
        return 0
    try:
        p = plan(args)
        if args.submit or args.run_local:
            return execute(p, local=args.run_local)
        print(json.dumps(p, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
