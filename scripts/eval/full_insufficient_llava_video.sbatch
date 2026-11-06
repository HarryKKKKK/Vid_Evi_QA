#!/bin/bash
#SBATCH --job-name=llava_insuff_32f
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/llava_insuff_32f_%j.out
#SBATCH --error=logs/llava_insuff_32f_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
EVAL_WORKERS=8 bash scripts/eval/run_insufficient_model.sh \
    llava \
    "${ROOT}/models/LLaVA-OneVision-Qwen2-7B-OV" \
    llava-onevision-qwen2-7b-ov \
    llava_onevision_32f_video \
    nextgqa_result_llava_onevision_32f_video_insufficient

