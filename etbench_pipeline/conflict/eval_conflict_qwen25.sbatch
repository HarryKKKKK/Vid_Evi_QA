#!/bin/bash
#SBATCH --job-name=conflict_qwen25
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/conflict_qwen25_%j.out
#SBATCH --error=logs/conflict_qwen25_%j.err
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
EVAL_WORKERS=8 bash etbench_pipeline/conflict/run_conflict_eval_model.sh qwen25 \
  "${ROOT}/models/Qwen2.5-VL-32B-Instruct" qwen2.5-vl-32b qwen25vl32b_conflict_tal \
  etbench_pipeline/conflict/eval_results/qwen25vl32b_conflict_tal
