#!/bin/bash
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
OUTPUT=${1:-results/reports/conflict_final_audit}
cd "${ROOT}"
mkdir -p "${OUTPUT}" results/logs/conflict_final_audit
"/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python" scripts/analysis/prepare_conflict_visual_review.py \
  --input results/build/etbench/conflict/full_v2_qwen3vl_mistral24b/merged/conflict_qa_v2.jsonl \
  --output-dir "${OUTPUT}/prefilter"
VLM_JOB=$(sbatch --parsable --export="ALL,CONFLICT_FINAL_OUTPUT=${OUTPUT}" scripts/jobs/analysis/audit_conflict_visual_all.sbatch)
VLM_ID=${VLM_JOB%%;*}
GATE_JOB=$(sbatch --parsable --dependency="afterok:${VLM_ID}" --export="ALL,CONFLICT_FINAL_OUTPUT=${OUTPUT}" scripts/jobs/analysis/build_conflict_gpt_candidates.sbatch)
GATE_ID=${GATE_JOB%%;*}
GPT_JOB=$(sbatch --parsable --dependency="afterok:${GATE_ID}" --export="ALL,CONFLICT_FINAL_OUTPUT=${OUTPUT}" scripts/jobs/analysis/final_filter_conflict_gpt.sbatch)
echo "VLM_JOB=${VLM_JOB}"
echo "GATE_JOB=${GATE_JOB}"
echo "GPT_JOB=${GPT_JOB}"
squeue -j "${VLM_ID},${GATE_ID},${GPT_JOB%%;*}"
