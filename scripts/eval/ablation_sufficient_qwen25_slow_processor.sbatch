#!/bin/bash
#SBATCH --job-name=qwen25_suff_slow
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/qwen25_suff_slow_%j.out
#SBATCH --error=logs/qwen25_suff_slow_%j.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"

# Processor-only ablation: all other sufficient-run settings remain unchanged.
QWEN_USE_FAST=false EVAL_WORKERS=8 bash scripts/eval/run_sufficient_model.sh \
    qwen25 \
    "${ROOT}/models/Qwen2.5-VL-32B-Instruct" \
    qwen2.5-vl-32b \
    qwen25vl32b_32f_slow_processor \
    ablation_results/qwen25_32f_sufficient_slow_processor
