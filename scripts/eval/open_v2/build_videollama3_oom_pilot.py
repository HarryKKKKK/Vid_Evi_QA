"""Build a small pilot manifest from distinct cases missing due to OOM."""
import argparse
import json
from pathlib import Path


def case_key(record_id):
    parts = record_id.split("::")
    # A VideoLLaMA3 OOM is dominated by the visual input.  Select distinct
    # videos rather than multiple questions that reuse the same 32 frames.
    return tuple(parts[:2]) if len(parts) >= 2 else (record_id,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validation_path = (
        args.release / "results" / "videollama3_7b" /
        "merged.classify.validation.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    wanted = []
    seen = set()
    for record_id in validation.get("missing_ids", []):
        key = case_key(record_id)
        if key in seen:
            continue
        seen.add(key)
        wanted.append(record_id)
        if len(wanted) == args.count:
            break
    if len(wanted) < args.count:
        raise RuntimeError(f"Only found {len(wanted)} distinct missing cases")

    wanted_set = set(wanted)
    selected = {}
    for shard in sorted((args.release / "shards").glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            record_id = row.get("record_id")
            if record_id in wanted_set:
                selected[record_id] = row
    missing = [record_id for record_id in wanted if record_id not in selected]
    if missing:
        raise RuntimeError(f"Pilot records absent from frozen shards: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record_id in wanted:
            handle.write(json.dumps(selected[record_id], ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "records": len(wanted),
        "distinct_cases": len({case_key(x) for x in wanted}),
        "record_ids": wanted,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
