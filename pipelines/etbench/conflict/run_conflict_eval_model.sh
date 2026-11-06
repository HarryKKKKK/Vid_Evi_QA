#!/bin/bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 MODEL_KIND MODEL_PATH SERVED_NAME MODEL_TAG RESULTS_DIR" >&2
  exit 2
fi

MODEL_KIND=$1
MODEL_PATH=$2
SERVED_NAME=$3
MODEL_TAG=$4
RESULTS_DIR=$5
ROOT=${VID_EVI_ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
ENV_DIR=${VID_EVI_ENV:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl}
RUN_ID=${SLURM_JOB_ID:-$$}
PORT=$((26000 + RANDOM % 1000))
EVAL_WORKERS=${EVAL_WORKERS:-8}
FRAMES_PER_OCCURRENCE=${CONFLICT_FRAMES_PER_OCCURRENCE:-4}
MAX_FRAMES=${CONFLICT_MAX_FRAMES:-32}
FRAME_WIDTH=${CONFLICT_FRAME_WIDTH:-560}
# Default to the manually maintained active five-option release. An explicit
# CONFLICT_QA_PATH still selects historical/prefilter inputs when requested.
QA_PATH=${CONFLICT_QA_PATH:-results/reports/final_benchmark_aligned/scene_atomic_v2/conflict_qa_five_options.current.jsonl}
EXTRA_EVAL_ARGS=()
if [[ "${CONFLICT_ALLOW_NULL_ACTION:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--allow-null-official-action)
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PATH="${ENV_DIR}/bin:${PATH}"

case "${MODEL_KIND}" in
  qwen25)
    MAX_MODEL_LEN=32768; TP=2; MEDIA_MODE=frames
    MM_LIMIT='{"image": 32, "video": 0}'
    EXTRA_SERVER_ARGS=(--mm-processor-kwargs '{"max_pixels":75264,"use_fast":true}')
    ;;
  llava)
    MAX_MODEL_LEN=32768; TP=1; MEDIA_MODE=frames
    MM_LIMIT='{"image": 32, "video": 0}'
    EXTRA_SERVER_ARGS=()
    ;;
  internvl)
    MAX_MODEL_LEN=40960; TP=1; MEDIA_MODE=frames
    MM_LIMIT='{"image": 32, "video": 0}'
    EXTRA_SERVER_ARGS=(--trust-remote-code --mm-processor-kwargs '{"max_dynamic_patch": 3}')
    ;;
  *) echo "Unknown MODEL_KIND=${MODEL_KIND}" >&2; exit 2 ;;
esac

cd "${ROOT}"
mkdir -p results/logs "${RESULTS_DIR}"
LOG_FILE="results/logs/conflict_${MODEL_TAG}_${RUN_ID}.vllm.log"
setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt "${MM_LIMIT}" \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port "${PORT}" --host 127.0.0.1 \
  "${EXTRA_SERVER_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill -- "-${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM exited during startup" >&2; tail -160 "${LOG_FILE}" >&2; exit 1
  fi
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}" && break
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${SERVED_NAME}"

"${ENV_DIR}/bin/python" -u scripts/src/vid_evi_qa/evaluation/evaluate_conflict_v2.py \
  --qa "${QA_PATH}" \
  --groups results/build/etbench/conflict/visual_contrast_jobs.jsonl \
  --core-grounding results/build/etbench/conflict/full_v2_qwen3vl_mistral24b/merged/core_grounding.jsonl \
  --temporal-grounding results/build/etbench/conflict/full_v2_qwen3vl_mistral24b/merged/temporal_grounding.jsonl \
  --video-root source_datasets/et_bench/ETBench/videos \
  --output-dir "${RESULTS_DIR}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED_NAME}" \
  --model-tag "${MODEL_TAG}" \
  --source-task "${CONFLICT_SOURCE_TASK:-all}" \
  --media-mode "${MEDIA_MODE}" \
  --frames-per-occurrence "${FRAMES_PER_OCCURRENCE}" \
  --max-frames "${MAX_FRAMES}" \
  --frame-width "${FRAME_WIDTH}" \
  --workers "${EVAL_WORKERS}" \
  "${EXTRA_EVAL_ARGS[@]}"
