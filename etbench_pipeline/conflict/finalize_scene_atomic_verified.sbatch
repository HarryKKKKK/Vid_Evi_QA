#!/bin/bash
#SBATCH --job-name=scene-v2-final
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/scene_atomic_v2_finalize_%j.out
#SBATCH --error=logs/scene_atomic_v2_finalize_%j.err
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u \
  etbench_pipeline/conflict/finalize_scene_atomic_verified.py \
  --requests reports/final_benchmark_aligned/scene_atomic_v2/scene_rebuild_requests.jsonl \
  --verification-dir reports/final_benchmark_aligned/scene_atomic_v2/verification \
  --groups etbench_pipeline/conflict/visual_contrast_jobs.jsonl \
  --current reports/final_benchmark_aligned/scene_atomic_v2/conflict_qa_five_options.current.jsonl \
  --output-dir reports/final_benchmark_aligned/scene_atomic_v2/finalized
