#!/bin/bash
#SBATCH --job-name=internvl_partial4_full
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/internvl_partial4_full_%j.out
#SBATCH --error=logs/internvl_partial4_full_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
export PARTIAL_CANDIDATES=nextgqa_pipeline/partial_candidates_full_n5279.json
export PARTIAL_ALPHAS=0.2,0.4,0.5,0.6
export PARTIAL_EXPECTED_ANCHORS=5279
export PARTIAL_EXPECTED_TASKS=21116
EVAL_WORKERS=8 bash scripts/eval/run_partial_model.sh \
    internvl \
    "${ROOT}/models/InternVL3_5-8B" \
    internvl3.5-8b \
    internvl35_32f_patch3_partial_selected4_full_n5279 \
    nextgqa_result_internvl35_32f_patch3_partial_selected4_full_n5279
