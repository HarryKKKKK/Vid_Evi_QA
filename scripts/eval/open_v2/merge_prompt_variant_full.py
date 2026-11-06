"""Merge full shards while validating against a selected prompt version."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("kind")
    parser.add_argument("--prompt-version", choices=("v2.1", "v2.2"), required=True)
    args = parser.parse_args()
    release = args.release.resolve()
    old = load("merge_prompt_v21", ROOT / "results/final_filtered_hierarchical_v2/release_20260902_v21/code/prompt_v2.py")
    current = load("merge_prompt_v22", ROOT / "scripts/eval/prompt_v2.py")
    chosen = old if args.prompt_version == "v2.1" else current

    manifest = [json.loads(line) for line in (release / "smoke.jsonl").read_text(encoding="utf-8").splitlines() if line]
    wanted = [current.record_identity(row) for row in manifest]
    if not wanted or any(key is None for key in wanted) or len(set(wanted)) != len(wanted):
        raise SystemExit("Full manifest has missing or duplicate identities")

    found = {}
    duplicates = []
    shard_files = sorted((release / "results" / args.kind).glob("shard_*/merged.classify.json"))
    for path in shard_files:
        for row in json.loads(path.read_text(encoding="utf-8")):
            key = current.record_identity(row)
            if key in found:
                duplicates.append(key)
            else:
                found[key] = row
    ordered = [found[key] for key in wanted if key in found]
    output_dir = release / "results" / args.kind
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "merged.classify.json").write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "merged.classify.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8")

    invalid = [current.record_identity(row) for row in ordered if
               row.get("prompt_version") != chosen.PROMPT_VERSION or
               row.get("prompt_sha256") != chosen.PROMPT_SHA256 or
               current.response_errors(row.get("model_parsed"))]
    missing = [key for key in wanted if key not in found]
    obsolete = sorted(str(key) for key in set(found) - set(wanted))
    report = {
        "status": "complete" if not missing and not duplicates and not obsolete else "incomplete",
        "kind": args.kind,
        "expected_records": len(wanted),
        "result_records": len(ordered),
        "shard_files": len(shard_files),
        "missing_ids": missing,
        "duplicate_ids": duplicates,
        "obsolete_ids": obsolete,
        "invalid_records": len(invalid),
        "invalid_ids": invalid,
        "prompt_version": chosen.PROMPT_VERSION,
        "prompt_sha256": chosen.PROMPT_SHA256,
        "conditions": dict(Counter(row.get("evidence_condition") for row in ordered)),
    }
    (output_dir / "merged.classify.validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "complete":
        raise SystemExit("One or more full-evaluation shards are incomplete")


if __name__ == "__main__":
    main()
