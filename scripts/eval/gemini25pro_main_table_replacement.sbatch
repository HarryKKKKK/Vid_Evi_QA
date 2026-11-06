#!/bin/bash
#SBATCH --job-name=gem-main-repl
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/gemini25pro_main_table_add479_20260921/logs/replacement_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/gemini25pro_main_table_add479_20260921/logs/replacement_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
RUN=$ROOT/results/gemini25pro_main_table_add479_20260921
export PATH="$ROOT/../envs_qwen3vl/bin:$PATH"
export XHUB_KEY_FILE="$HOME/.config/vid_evi_qa/xhub_api_key_colleague"
export OPENAI_BASE_URL=https://api3.xhub.chat/v1
export OPENAI_MODEL=gemini-2.5-pro
export GPT_MAX_TOKENS=4096
cd "$ROOT"
exec "$ROOT/../envs_qwen3vl/bin/python" -u scripts/eval/run_gpt_v2_api.py \
  --manifest-jsonl "$RUN/replacement_retry.jsonl" --task classify \
  --video-dir "$ROOT/source_datasets/next_gqa/videos" \
  --results-dir "$RUN/results/replacement" --model-tag gemini25pro_main_table_v2 \
  --num-frames 32 --frame-width 560 --sampling uniform --media-mode frames --workers 1
