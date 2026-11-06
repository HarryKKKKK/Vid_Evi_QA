"""Freeze a format20 deployment for pinned frontier checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from script_export import copy_script

from frontier_models import MODELS
from prepare import BASE, SOURCE, select_records

ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--base-release", type=Path, default=BASE)
    parser.add_argument("--template-release", type=Path,
                        help="Reuse an already frozen eval directory and format20 manifest")
    parser.add_argument("--sample-profile", choices=["format20", "full"], default="format20")
    parser.add_argument("--models", nargs="+", choices=MODELS, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit("Destination already exists; frozen deployments are immutable")
    base = args.base_release.resolve()
    if args.template_release:
        template = args.template_release.resolve()
        selected = [json.loads(line) for line in (template / "smoke.jsonl").read_text().splitlines() if line.strip()]
        template_info = json.loads((template / "deployment.json").read_text())
        if args.sample_profile == "format20" and len(selected) != 20:
            raise RuntimeError("Template smoke manifest is not the expected format20 set")
        if args.sample_profile == "full" and template_info.get("sample_profile") != "full":
            raise RuntimeError("Template is not a frozen full release")
        eval_source = template / "eval"
    else:
        rows = [json.loads(line) for line in (base / "all_condition_records.jsonl").read_text().splitlines() if line.strip()]
        selected = select_records(rows, "format20")
        eval_source = base / "code"
    destination.mkdir(parents=True)
    shutil.copytree(eval_source, destination / "eval", copy_function=copy_script, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(SOURCE, destination / "adapters", copy_function=copy_script, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (destination / "smoke.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
    num_shards = 0
    if args.sample_profile == "full":
        shutil.copytree(template / "shards", destination / "shards")
        num_shards = len(list((destination / "shards").glob("*.jsonl")))
        if num_shards != int(template_info.get("num_shards") or 0):
            raise RuntimeError("Frozen shard count does not match template metadata")
    info = {"source_release": str(args.template_release.resolve() if args.template_release else base),
            "records": len(selected), "sample_profile": args.sample_profile,
            "num_shards": num_shards, "models": {}}
    for key in args.models:
        spec = MODELS[key]
        model_path = ROOT / "models" / spec["name"]
        indexes = list(model_path.glob("*.safetensors.index.json"))
        if not indexes and not (model_path / "model.safetensors").is_file():
            raise RuntimeError(f"No checkpoint found in {model_path}")
        for index in indexes:
            for shard in set(json.loads(index.read_text()).get("weight_map", {}).values()):
                if not (index.parent / shard).is_file():
                    raise RuntimeError(f"Missing weight shard {index.parent / shard}")
        info["models"][spec["name"]] = {
            "key": key,
            "path": str(model_path),
            "revision": spec["revision"],
            "config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
            "backend": spec["backend"],
            "gpus": spec["gpus"],
        }
    info["sha256"] = {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.rglob("*") if path.is_file()
    }
    (destination / "deployment.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
