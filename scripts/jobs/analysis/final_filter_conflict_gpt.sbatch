#!/bin/bash
#SBATCH --job-name=conflict_gpt_final
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=results/logs/conflict_final_audit/gpt_%j.out
#SBATCH --error=results/logs/conflict_final_audit/gpt_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
OUTPUT=${CONFLICT_FINAL_OUTPUT:-results/reports/conflict_final_audit}
KEY_FILE=${XHUB_KEY_FILE:-${HOME}/.config/vid_evi_qa/xhub_api_key}
cd "${ROOT}"
if [[ ! -s "${KEY_FILE}" ]]; then
  echo "Missing protected API key file: ${KEY_FILE}" >&2
  exit 1
fi
IFS= read -r OPENAI_API_KEY < "${KEY_FILE}" || true
export OPENAI_API_KEY
export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api3.xhub.chat/v1}
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
exec "${PYTHON}" scripts/analysis/final_filter_conflict_gpt.py \
  --candidates "${OUTPUT}/gated/gpt_candidates.json" \
  --pre-gpt-rejected "${OUTPUT}/gated/pre_gpt_rejected.json" \
  --output-dir "${OUTPUT}/final" \
  --workers 8 \
  --permutations 1 \
  --keep-confidence 0.65
