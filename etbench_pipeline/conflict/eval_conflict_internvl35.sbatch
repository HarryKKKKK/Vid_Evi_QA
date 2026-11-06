#!/bin/bash
#SBATCH --job-name=conflict_internvl
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/conflict_internvl_%j.out
#SBATCH --error=logs/conflict_internvl_%j.err
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
EVAL_WORKERS=8 bash etbench_pipeline/conflict/run_conflict_eval_model.sh internvl \
  "${ROOT}/models/InternVL3_5-8B" internvl3.5-8b internvl35_conflict_tal \
  etbench_pipeline/conflict/eval_results/internvl35_conflict_tal
