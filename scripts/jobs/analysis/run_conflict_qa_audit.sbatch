#!/bin/bash
#SBATCH --job-name=conflict_qa_audit
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/conflict_qa_audit/model_%A_%a.out
#SBATCH --error=results/logs/conflict_qa_audit/model_%A_%a.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
OUTPUT=${CONFLICT_AUDIT_OUTPUT:-results/reports/conflict_qa_audit/full_2145}
PER_TYPE=${CONFLICT_AUDIT_PER_TYPE:-0}
MAX_NEW_TOKENS=${CONFLICT_AUDIT_MAX_NEW_TOKENS:-160}

cd "${ROOT}"
mkdir -p "${OUTPUT}" results/logs/conflict_qa_audit

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
    BACKEND=qwen
    MODEL_PATH=/aifs4su/hansirui_2nd/harry/models/Qwen3-8B
    MODEL_NAME=Qwen3-8B
else
    BACKEND=llama
    MODEL_PATH=/aifs4su/hansirui_2nd/harry/models/Meta-Llama-3.1-8B-Instruct-Nous
    MODEL_NAME=Llama-3.1-8B-Instruct
fi

exec "${PYTHON}" scripts/analysis/audit_conflict_qa_open_model.py \
    --backend "${BACKEND}" \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT}" \
    --per-type "${PER_TYPE}" \
    --batch-size 8 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --device cuda
