#!/bin/bash
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
OUTPUT=${1:-reports/conflict_final_audit}
cd "${ROOT}"

# Supersede the dependency chain attached to the earlier failed LLaVA task.
scancel 373197 373198 373355 373356 2>/dev/null || true

RETRY_JOB=$(sbatch --parsable --array=1 \
  --export="ALL,CONFLICT_FINAL_OUTPUT=${OUTPUT}" \
  scripts/analysis/audit_conflict_visual_all.sbatch)
RETRY_ID=${RETRY_JOB%%;*}
GATE_JOB=$(sbatch --parsable --dependency="afterok:${RETRY_ID}" \
  --export="ALL,CONFLICT_FINAL_OUTPUT=${OUTPUT}" \
  scripts/analysis/build_conflict_gpt_candidates.sbatch)
GATE_ID=${GATE_JOB%%;*}
GPT_JOB=$(sbatch --parsable --dependency="afterok:${GATE_ID}" \
  --export="ALL,CONFLICT_FINAL_OUTPUT=${OUTPUT}" \
  scripts/analysis/final_filter_conflict_gpt.sbatch)

echo "LLAVA_RETRY=${RETRY_JOB}"
echo "GATE_JOB=${GATE_JOB}"
echo "GPT_JOB=${GPT_JOB}"
squeue -j "${RETRY_ID},${GATE_ID},${GPT_JOB%%;*}"
