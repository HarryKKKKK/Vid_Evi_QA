#!/bin/bash
#SBATCH --job-name=nextgqa_calibrate
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/nextgqa_calibrate_%j.out
#SBATCH --error=logs/nextgqa_calibrate_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl
cd "${ROOT}"
mkdir -p logs

python -u nextgqa_pipeline/insufficient/calibrate.py \
  --measurements 'nextgqa_pipeline/insufficient/measurements.*.jsonl' \
  --target-recall 0.95 \
  --minimum-controls 20 \
  --residual-ratio-threshold "${RESIDUAL_THRESHOLD:-0.10}"
