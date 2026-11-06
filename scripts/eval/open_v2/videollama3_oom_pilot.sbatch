#!/bin/bash
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/videollama3_oom_pilot_%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/videollama3_oom_pilot_%A_%a.err
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
RELEASE=${1:?usage: videollama3_oom_pilot.sbatch RELEASE MANIFEST}
MANIFEST=${2:?usage: videollama3_oom_pilot.sbatch RELEASE MANIFEST}
case "${SLURM_ARRAY_TASK_ID}" in
  0) VISUAL_TOKENS=16384 ;;
  1) VISUAL_TOKENS=8192 ;;
  2) VISUAL_TOKENS=4096 ;;
  *) echo "unexpected task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin:${PATH}"
export VIDEOLLAMA3_MODEL_PATH="${ROOT}/models/VideoLLaMA3-7B"
export VIDEOLLAMA3_VISUAL_MAX_TOKENS="${VISUAL_TOKENS}"
export VIDEOLLAMA3_ATTN_IMPLEMENTATION=sdpa
export V2_MAX_TOKENS=8192
export V2_MANIFEST="${MANIFEST}"
export V2_MODEL_TAG="videollama3_7b_oom_pilot_vt${VISUAL_TOKENS}"
export V2_RESULT_DIR="${RELEASE}/results/videollama3_oom_pilot/vt${VISUAL_TOKENS}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${V2_RESULT_DIR}"
{
  echo "visual_max_tokens=${VISUAL_TOKENS}"
  echo "manifest=${MANIFEST}"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
} > "${V2_RESULT_DIR}/pilot_config.txt"

set +e
"${ROOT}/.runtime/videollama3/bin/python" -u \
  "${ROOT}/scripts/eval/open_v2/evaluate_videollama3_oomfix.py" \
  --manifest-jsonl "${MANIFEST}" --task classify \
  --video-dir "${ROOT}/source_datasets/next_gqa/videos" \
  --results-dir "${V2_RESULT_DIR}" --model-tag "${V2_MODEL_TAG}" \
  --num-frames 32 --frame-width 560 --sampling uniform \
  --media-mode video --workers 1
EVAL_RC=$?
set -e

"${ROOT}/.runtime/videollama3/bin/python" \
  "${ROOT}/scripts/eval/check_v2_run.py" "${MANIFEST}" \
  --result "${V2_RESULT_DIR}/merged.classify.json" || true
exit "${EVAL_RC}"
