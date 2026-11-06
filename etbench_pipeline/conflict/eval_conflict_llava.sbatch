#!/bin/bash
#SBATCH --job-name=conflict_llava
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/conflict_llava_%j.out
#SBATCH --error=logs/conflict_llava_%j.err
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
EVAL_WORKERS=8 bash etbench_pipeline/conflict/run_conflict_eval_model.sh llava \
  "${ROOT}/models/LLaVA-OneVision-Qwen2-7B-OV" llava-onevision-qwen2-7b-ov llava_onevision_conflict_tal \
  etbench_pipeline/conflict/eval_results/llava_onevision_conflict_tal
