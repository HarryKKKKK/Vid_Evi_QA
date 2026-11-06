#!/bin/bash
#SBATCH --job-name=conflict_gate
#SBATCH -p debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/conflict_final_audit/gate_%j.out
#SBATCH --error=logs/conflict_final_audit/gate_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
OUTPUT=${CONFLICT_FINAL_OUTPUT:-reports/conflict_final_audit}
cd "${ROOT}"
exec "${PYTHON}" scripts/analysis/build_conflict_gpt_candidates.py \
  --qa etbench_pipeline/conflict/full_v2_qwen3vl_mistral24b/merged/conflict_qa_v2.jsonl \
  --text-audit reports/conflict_qa_audit/full_2145/summary/merged_judgments.json \
  --visual "${OUTPUT}/internvl35/conflict_eval.json" \
  --visual "${OUTPUT}/llava_onevision/conflict_eval.json" \
  --output-dir "${OUTPUT}/gated"
