#!/bin/bash
#SBATCH --job-name=conflict_slot_review
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/conflict_slot_review_%j.out
#SBATCH --error=logs/conflict_slot_review_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
KEY_FILE=$HOME/.config/vid_evi_qa/xhub_api_key
cd "$ROOT"
if [[ ! -s "$KEY_FILE" ]]; then
  echo "Missing protected API key file" >&2
  exit 1
fi
IFS= read -r OPENAI_API_KEY < "$KEY_FILE" || true
export OPENAI_API_KEY
export OPENAI_BASE_URL=https://api3.xhub.chat/v1
export OPENAI_MODEL=gpt-4o

exec "$PYTHON" -u scripts/analysis/review_conflict_distractor_slots_gpt.py \
  --unresolved reports/final_benchmark_aligned/final/unresolved_alignment.json \
  --verification reports/final_benchmark_aligned/work/conflict_distractor_verification.jsonl \
  --verification reports/final_benchmark_aligned/work/conflict_distractor_verification_retry1.jsonl \
  --output-review reports/final_benchmark_aligned/work/conflict_distractor_slot_review.jsonl \
  --output-verification reports/final_benchmark_aligned/work/conflict_distractor_verification_text_override.jsonl \
  --workers 8
