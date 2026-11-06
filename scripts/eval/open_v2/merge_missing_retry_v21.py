"""Add missing-only retry shards to a legacy v2.1 monolithic merge."""
import argparse
import importlib.util
import json
from pathlib import Path


def load_prompt(path):
    spec = importlib.util.spec_from_file_location("merge_missing_prompt", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path, required=True)
    parser.add_argument("--prompt-module", type=Path, required=True)
    args = parser.parse_args()
    prompt = load_prompt(args.prompt_module)
    manifest = [json.loads(x) for x in args.manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    wanted = [prompt.record_identity(x) for x in manifest]
    rows = [json.loads(x) for x in args.baseline.read_text(encoding="utf-8").splitlines() if x.strip()]
    found = {prompt.record_identity(x): x for x in rows}
    retry_rows = []
    for path in sorted(args.retry_root.glob("shard_*/merged.classify.jsonl")):
        retry_rows.extend(json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
    duplicate_retry = [prompt.record_identity(x) for x in retry_rows if prompt.record_identity(x) in found]
    for row in retry_rows:
        found.setdefault(prompt.record_identity(row), row)
    missing = [key for key in wanted if key not in found]
    ordered = [found[key] for key in wanted if key in found]
    output_jsonl = args.baseline.with_name("merged.with_missing_retry.classify.jsonl")
    output_json = args.baseline.with_name("merged.with_missing_retry.classify.json")
    output_jsonl.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ordered), encoding="utf-8")
    output_json.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"status": "complete" if not missing and not duplicate_retry else "incomplete",
              "expected_records": len(wanted), "result_records": len(ordered),
              "baseline_records": len(rows), "retry_records": len(retry_rows),
              "missing_ids": missing, "duplicate_retry_ids": duplicate_retry,
              "output_jsonl": str(output_jsonl)}
    output_jsonl.with_suffix(".validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "complete":
        raise SystemExit("Missing retry merge is incomplete")


if __name__ == "__main__":
    main()
