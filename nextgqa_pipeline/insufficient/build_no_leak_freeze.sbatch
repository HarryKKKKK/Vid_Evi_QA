#!/bin/bash
#SBATCH --job-name=nextgqa_no_leak_freeze
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/nextgqa_no_leak_freeze_%A_%a.out
#SBATCH --error=logs/nextgqa_no_leak_freeze_%A_%a.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
NUM_SHARDS=${NUM_SHARDS:-16}
JOBS=${JOBS:-4}
THREADS=${THREADS:-4}
OFFICIAL_PADDING=${OFFICIAL_PADDING:-1}
SAMPLING_FPS=${SAMPLING_FPS:-1}
BLUR_SIGMA=${BLUR_SIGMA:-30}

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl
cd "${ROOT}"
mkdir -p logs

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then EXTRA_ARGS+=(--limit "${LIMIT}"); fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then EXTRA_ARGS+=(--overwrite); fi

python -u nextgqa_pipeline/insufficient/build_no_leak_freeze.py \
  --filtered-json nextgqa_pipeline/nextgqa_filtered_no_evidence_leak.json \
  --output-dir source_datasets/next_gqa/freeze_videos_no_evidence_leak \
  --audit nextgqa_pipeline/insufficient/no_leak_freeze_build.jsonl \
  --official-padding "${OFFICIAL_PADDING}" \
  --sampling-fps "${SAMPLING_FPS}" \
  --blur-sigma "${BLUR_SIGMA}" \
  --jobs "${JOBS}" \
  --threads "${THREADS}" \
  --num-shards "${NUM_SHARDS}" \
  "${EXTRA_ARGS[@]}"
