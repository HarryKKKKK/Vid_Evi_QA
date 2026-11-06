#!/bin/bash
#SBATCH --job-name=internvl_a010_full
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/internvl_a010_full_%j.out
#SBATCH --error=logs/internvl_a010_full_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
OUT_DIR=nextgqa_result_internvl35_32f_patch3_partial_a010_full_n5279
MODEL_TAG=internvl35_32f_patch3_partial_a010_full_n5279
cd "${ROOT}"

if [[ ! -s "${OUT_DIR}/partial.classify.jsonl" ]]; then
    "${ENV_PYTHON}" scripts/eval/seed_partial_results.py \
        --source nextgqa_result_internvl35_32f_patch3_partial_a010_a090_n500/partial.classify.jsonl \
        --candidates nextgqa_pipeline/partial_candidates_full_n5279.json \
        --alphas 0.1 \
        --model-tag "${MODEL_TAG}" \
        --output-dir "${OUT_DIR}"
fi

export PARTIAL_CANDIDATES=nextgqa_pipeline/partial_candidates_full_n5279.json
export PARTIAL_ALPHAS=0.1
export PARTIAL_EXPECTED_ANCHORS=5279
export PARTIAL_EXPECTED_TASKS=5279
EVAL_WORKERS=8 bash scripts/eval/run_partial_model.sh \
    internvl \
    "${ROOT}/models/InternVL3_5-8B" \
    internvl3.5-8b \
    "${MODEL_TAG}" \
    "${OUT_DIR}"
