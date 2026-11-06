#!/bin/bash
#SBATCH --job-name=internvl_partial_n500
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/internvl_partial_n500_%j.out
#SBATCH --error=logs/internvl_partial_n500_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
EVAL_WORKERS=8 bash scripts/eval/run_partial_model.sh \
    internvl \
    "${ROOT}/models/InternVL3_5-8B" \
    internvl3.5-8b \
    internvl35_32f_patch3_partial_n500 \
    nextgqa_result_internvl35_32f_patch3_partial_a010_a090_n500

