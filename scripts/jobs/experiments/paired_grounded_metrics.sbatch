#!/bin/bash
#SBATCH --job-name=paired-grounded
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/paired_grounded_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/paired_grounded_%j.err
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PY=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
OUT=${ROOT}/results/paired_grounded_metrics/v22_20260915_complete
cd "${ROOT}"
mkdir -p results/logs "${OUT}"
"${PY}" -u scripts/analysis/compute_paired_grounded_metrics.py \
  --bootstrap 10000 --seed 20260915 --output-dir "${OUT}" \
  --model 'Qwen3-VL-8B=results/open_v22_qwen_small_full_8192_20260907/results/qwen3_8b/merged.classify.json' \
  --model 'Qwen2.5-VL-7B=results/open_v22_qwen_small_full_8192_20260907/results/qwen25_7b/merged.classify.json' \
  --model 'Qwen3.6-35B-A3B=results/frontier_v22_full_20260907_r2/results/qwen36_35b_a3b_fp8/merged.classify.json' \
  --model 'GLM-4.5V=results/frontier_v22_full_glm45v_8192_20260907/results/glm45v_fp8_tp2_vllm0191/merged.classify.json' \
  --model 'VideoLLaMA3-7B=results/frontier_v22_full_20260907_r2/results/videollama3_7b_vt4096/merged.classify.json' \
  --model 'LLaVA-NeXT-Video-7B=results/llava_v22_full_20260906/results/llava_next_video_7b/merged.classify.json' \
  --model 'LLaVA-OneVision-7B=results/llava_v22_full_20260906/results/llava_ov_7b/merged.classify.json'
