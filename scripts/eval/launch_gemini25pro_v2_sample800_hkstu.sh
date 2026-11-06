#!/usr/bin/env bash
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=$ROOT/../envs_qwen3vl/bin/python
MANIFEST=$ROOT/results/closed_v2_sample800_20260919/sample800.jsonl
OUT=$ROOT/results/gemini25pro_v2_sample800_20260919

test "$(wc -l < "$MANIFEST")" -eq 800
mkdir -p "$OUT"
export PATH="$ROOT/../envs_qwen3vl/bin:$PATH"
export XHUB_KEY_FILE="$HOME/.config/vid_evi_qa/xhub_api_key_colleague"
export OPENAI_BASE_URL=https://api3.xhub.chat/v1
export OPENAI_MODEL=gemini-2.5-pro

exec "$PYTHON" -u "$ROOT/scripts/eval/run_gpt_v2_api.py" \
  --manifest-jsonl "$MANIFEST" --task classify \
  --video-dir "$ROOT/source_datasets/next_gqa/videos" \
  --results-dir "$OUT" --model-tag gemini25pro_sample800_v2 \
  --num-frames 32 --frame-width 560 --sampling uniform \
  --media-mode frames --workers 4
