#!/bin/bash
#SBATCH --job-name=videollama3-vt4096
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/videollama3_vt4096_%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/videollama3_vt4096_%A_%a.err
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
RELEASE=${1:?usage: videollama3_4096_full_array.sbatch RELEASE}
SHARD=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
MANIFEST="${RELEASE}/shards/${SHARD}.jsonl"
RESULT_DIR="${RELEASE}/results/videollama3_7b_vt4096/shard_${SHARD}"

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${ENV_DIR}/bin:${PATH}"
export VIDEOLLAMA3_MODEL_PATH="${ROOT}/models/VideoLLaMA3-7B"
export VIDEOLLAMA3_VISUAL_MAX_TOKENS=4096
export VIDEOLLAMA3_ATTN_IMPLEMENTATION=sdpa
export V2_MAX_TOKENS=8192

cd "${ROOT}"
mkdir -p "${RESULT_DIR}"
"${ROOT}/.runtime/videollama3/bin/python" -u \
  scripts/eval/open_v2/evaluate_videollama3_oomfix.py \
  --manifest-jsonl "${MANIFEST}" --task classify \
  --video-dir source_datasets/next_gqa/videos --results-dir "${RESULT_DIR}" \
  --model-tag videollama3_7b_vt4096_full_v2_2 --num-frames 32 \
  --frame-width 560 --sampling uniform --media-mode video --workers 1

"${ENV_DIR}/bin/python" scripts/eval/check_v2_run.py "${MANIFEST}" \
  --result "${RESULT_DIR}/merged.classify.json" || true
