#!/bin/bash
#SBATCH --job-name=conflict_distractors_gpt
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/conflict_distractors_gpt_%j.out
#SBATCH --error=logs/conflict_distractors_gpt_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
KEY_FILE=${XHUB_KEY_FILE:-${HOME}/.config/vid_evi_qa/xhub_api_key}
cd "${ROOT}"
if [[ ! -s "${KEY_FILE}" ]]; then
  echo "Missing protected API key file" >&2
  exit 1
fi
IFS= read -r OPENAI_API_KEY < "${KEY_FILE}" || true
export OPENAI_API_KEY
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api3.xhub.chat/v1}
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
exec "${PYTHON}" -u scripts/analysis/generate_conflict_distractors_gpt.py \
  --input "${DISTRACTOR_REQUESTS:-reports/final_benchmark_aligned/work/conflict_distractor_requests.jsonl}" \
  --output "${DISTRACTOR_CANDIDATES:-reports/final_benchmark_aligned/work/conflict_distractor_candidates.jsonl}" \
  --workers 8
