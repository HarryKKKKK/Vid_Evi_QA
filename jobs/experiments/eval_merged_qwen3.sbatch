#!/bin/bash
#SBATCH --job-name=merged_qwen3
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/merged_qwen3_%j.out
#SBATCH --error=logs/merged_qwen3_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
EVAL_WORKERS=8 bash scripts/eval/run_merged_model.sh \
  qwen3 "${ROOT}/models/Qwen3-VL" qwen3-vl qwen3vl_merged_v1 \
  results/final_filtered_hierarchical_v1/qwen3vl_merged_v1
