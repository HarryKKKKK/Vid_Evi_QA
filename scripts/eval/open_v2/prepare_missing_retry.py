"""Build deterministic manifests containing only records missing from a merge."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load_prompt(path: Path):
    spec = importlib.util.spec_from_file_location("missing_retry_prompt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--prompt-module", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards < 1 or args.start_index < 0:
        raise SystemExit("start-index must be nonnegative and num-shards positive")
    prompt = load_prompt(args.prompt_module.resolve())
    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted_order = [prompt.record_identity(row) for row in rows]
    if any(key is None for key in wanted_order) or len(wanted_order) != len(set(wanted_order)):
        raise SystemExit("Manifest identities are missing or duplicated")
    wanted = dict(zip(wanted_order, rows))
    completed_rows = [json.loads(line) for line in args.baseline.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {prompt.record_identity(row) for row in completed_rows}
    missing = [key for key in wanted_order if key not in completed]
    shards = [[] for _ in range(args.num_shards)]
    for offset, key in enumerate(missing):
        shards[offset % args.num_shards].append(wanted[key])
    metadata = []
    for offset, shard_rows in enumerate(shards):
        index = args.start_index + offset
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard_rows)
        path = args.output_dir / f"{index:03d}.jsonl"
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Existing retry manifest differs from the deterministic plan: {path}")
        else:
            atomic_write(path, content)
        metadata.append({"index": index, "records": len(shard_rows), "path": str(path),
                         "sha256": hashlib.sha256(content.encode()).hexdigest()})
    plan = {"policy": "missing_only", "manifest": str(args.manifest.resolve()),
            "baseline": str(args.baseline.resolve()), "missing_records": len(missing),
            "missing_ids": missing, "shards": metadata}
    atomic_write(args.output_dir / "plan.json", json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"missing_records": len(missing), "shards": metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
