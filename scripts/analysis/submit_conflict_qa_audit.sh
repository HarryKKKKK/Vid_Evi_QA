#!/bin/bash
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
OUTPUT=${1:-results/reports/conflict_qa_audit/full_2145}
PER_TYPE=${2:-0}
MAX_NEW_TOKENS=${3:-160}

cd "${ROOT}"
mkdir -p "${OUTPUT}" results/logs/conflict_qa_audit

AUDIT_JOB=$(sbatch --parsable \
    --export="ALL,CONFLICT_AUDIT_OUTPUT=${OUTPUT},CONFLICT_AUDIT_PER_TYPE=${PER_TYPE},CONFLICT_AUDIT_MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" \
    scripts/jobs/analysis/run_conflict_qa_audit.sbatch)
AUDIT_ID=${AUDIT_JOB%%;*}
MERGE_JOB=$(sbatch --parsable \
    --dependency="afterok:${AUDIT_ID}" \
    --export="ALL,CONFLICT_AUDIT_OUTPUT=${OUTPUT}" \
    scripts/jobs/analysis/summarize_conflict_qa_audit.sbatch)

echo "AUDIT_JOB=${AUDIT_JOB}"
echo "MERGE_JOB=${MERGE_JOB}"
squeue -j "${AUDIT_ID},${MERGE_JOB%%;*}"
