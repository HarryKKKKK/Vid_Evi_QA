#!/usr/bin/env bash
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
RELEASE=$ROOT/results/frontier_v22_full_20260907_r2
OUT=$ROOT/results/gpt4o_v2_full_20260914
PYTHON=$ROOT/../envs_qwen3vl/bin/python
ADAPTER=$ROOT/scripts/eval/run_gpt_v2_api.py
MAX_PARALLEL=${MAX_PARALLEL:-8}
INNER_WORKERS=${INNER_WORKERS:-2}
export PATH="$ROOT/../envs_qwen3vl/bin:$PATH"

mkdir -p "$OUT/results/gpt4o" "$OUT/logs"
cp -n "$RELEASE/eval/prompt_v2.py" "$OUT/source_prompt_v22.py" || true
cp -n "$ROOT/results/open_v2_deployment_20260902_r4/eval/prompt_v2.py" "$OUT/prompt_v2.py" || true

run_shard() {
  local shard=$1
  local stem
  stem=$(basename "$shard" .jsonl)
  local result_dir="$OUT/results/gpt4o/shard_$stem"
  mkdir -p "$result_dir"
  "$PYTHON" -u "$ADAPTER" \
    --manifest-jsonl "$shard" --task classify \
    --video-dir "$ROOT/source_datasets/next_gqa/videos" \
    --results-dir "$result_dir" --model-tag gpt4o_full_v2 \
    --num-frames 32 --frame-width 560 --sampling uniform \
    --media-mode frames --workers "$INNER_WORKERS" \
    >"$OUT/logs/shard_$stem.log" 2>&1
}

export -f run_shard
export ROOT RELEASE OUT PYTHON ADAPTER INNER_WORKERS
find "$RELEASE/shards" -maxdepth 1 -type f -name '*.jsonl' -print0 \
  | sort -z \
  | xargs -0 -n1 -P "$MAX_PARALLEL" bash -c 'run_shard "$1"' _

"$PYTHON" - "$OUT" "$RELEASE" <<'PY'
import json, sys
from pathlib import Path

out, release = map(Path, sys.argv[1:])
rows = []
for path in sorted((out / "results/gpt4o").glob("shard_*/merged.classify.json")):
    rows.extend(json.loads(path.read_text(encoding="utf-8")))
(out / "results/gpt4o/merged.classify.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
summary = {
    "status": "complete",
    "records": len(rows),
    "expected_records": sum(1 for p in (release / "shards").glob("*.jsonl") for _ in p.open()),
    "model": "gpt-4o",
    "prompt_version": "v2",
}
(out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
PY
