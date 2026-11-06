#!/bin/bash
#SBATCH --job-name=conflict_audit_merge
#SBATCH -p debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/conflict_qa_audit/merge_%j.out
#SBATCH --error=logs/conflict_qa_audit/merge_%j.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
OUTPUT=${CONFLICT_AUDIT_OUTPUT:-reports/conflict_qa_audit/full_2145}

cd "${ROOT}"
exec "${PYTHON}" scripts/analysis/summarize_conflict_qa_audit.py \
    --sample "${OUTPUT}/sample.json" \
    --judgment "${OUTPUT}/qwen_judgments.jsonl" \
    --judgment "${OUTPUT}/llama_judgments.jsonl" \
    --output-dir "${OUTPUT}/summary"
