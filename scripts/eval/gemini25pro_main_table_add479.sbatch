#!/bin/bash
#SBATCH --job-name=gem-main479
#SBATCH --partition=debug
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/gemini25pro_main_table_add479_20260921/logs/%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/gemini25pro_main_table_add479_20260921/logs/%A_%a.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=$ROOT/../envs_qwen3vl/bin/python
RUN=$ROOT/results/gemini25pro_main_table_add479_20260921
SHARD=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
MANIFEST=$RUN/manifests/$SHARD.jsonl
OUT=$RUN/results/shard_$SHARD

test -s "$MANIFEST"
test -s "$HOME/.config/vid_evi_qa/xhub_api_key_colleague"
mkdir -p "$OUT"
export PATH="$ROOT/../envs_qwen3vl/bin:$PATH"
export XHUB_KEY_FILE="$HOME/.config/vid_evi_qa/xhub_api_key_colleague"
export OPENAI_BASE_URL=https://api3.xhub.chat/v1
export OPENAI_MODEL=gemini-2.5-pro
export GPT_MAX_TOKENS=4096

cd "$ROOT"
exec "$PYTHON" -u scripts/eval/run_gpt_v2_api.py \
  --manifest-jsonl "$MANIFEST" --task classify \
  --video-dir "$ROOT/source_datasets/next_gqa/videos" \
  --results-dir "$OUT" --model-tag gemini25pro_main_table_v2 \
  --num-frames 32 --frame-width 560 --sampling uniform \
  --media-mode frames --workers 1
