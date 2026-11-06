#!/bin/bash
#SBATCH --job-name=finalize_aligned_benchmark
#SBATCH -p debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/finalize_aligned_benchmark_%j.out
#SBATCH --error=logs/finalize_aligned_benchmark_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
cd "${ROOT}"
exec "${PYTHON}" scripts/analysis/finalize_aligned_benchmark.py \
  --work-dir reports/final_benchmark_aligned/work \
  --groups etbench_pipeline/conflict/visual_contrast_jobs.jsonl \
  --output-dir reports/final_benchmark_aligned/final \
  --min-confidence 0.8
