#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
ENV_DIR=${ENV_DIR:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl}
OUTPUT_DIR=${OUTPUT_DIR:-results/build/etbench/conflict/action_content_recovery_smoke}
GROUNDING_GPU=${GROUNDING_GPU:-0}
QA_GPU=${QA_GPU:-1}
GROUNDING_PORT=${GROUNDING_PORT:-18480}
QA_PORT=${QA_PORT:-18481}
MAX_CASES=${MAX_CASES:-2}
GROUNDING_MAX_MODEL_LEN=${GROUNDING_MAX_MODEL_LEN:-7936}

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}/logs"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

timeout 30s "${ENV_DIR}/bin/python" -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())' || {
    echo "CUDA preflight failed; check GPU allocation and nvidia-fabricmanager before launching vLLM." >&2
    exit 1
  }

CUDA_VISIBLE_DEVICES="${GROUNDING_GPU}" setsid "${ENV_DIR}/bin/vllm" serve models/Qwen3-VL \
  --served-model-name qwen3-vl-evidence-grounder \
  --tensor-parallel-size 1 \
  --max-model-len "${GROUNDING_MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"image": 16, "video": 0}' \
  --mm-processor-kwargs '{"max_pixels": 75264}' \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port "${GROUNDING_PORT}" --host 127.0.0.1 >"${OUTPUT_DIR}/logs/qwen3vl_vllm.log" 2>&1 &
GROUNDING_PID=$!

CUDA_VISIBLE_DEVICES="${QA_GPU}" setsid "${ENV_DIR}/bin/vllm" serve models/Mistral-Small-24B-Instruct-2501 \
  --served-model-name mistral-small-24b-instruct-2501 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port "${QA_PORT}" --host 127.0.0.1 >"${OUTPUT_DIR}/logs/mistral_vllm.log" 2>&1 &
QA_PID=$!

cleanup() {
  kill -- "-${GROUNDING_PID}" 2>/dev/null || true
  kill -- "-${QA_PID}" 2>/dev/null || true
  wait "${GROUNDING_PID}" 2>/dev/null || true
  wait "${QA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_model() {
  local pid=$1 port=$2 name=$3 log=$4
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited during startup" >&2
      tail -120 "${log}" >&2
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q "${name}"; then
      echo "READY ${name}"
      return 0
    fi
    sleep 5
  done
  echo "Readiness timeout: ${name}" >&2
  return 1
}

wait_for_model "${GROUNDING_PID}" "${GROUNDING_PORT}" qwen3-vl-evidence-grounder "${OUTPUT_DIR}/logs/qwen3vl_vllm.log" &
WAIT_GROUNDING=$!
wait_for_model "${QA_PID}" "${QA_PORT}" mistral-small-24b-instruct-2501 "${OUTPUT_DIR}/logs/mistral_vllm.log" &
WAIT_QA=$!
wait "${WAIT_GROUNDING}"
wait "${WAIT_QA}"

"${ENV_DIR}/bin/python" -u pipelines/etbench/conflict/run_action_content_recovery.py \
  --groups results/build/etbench/conflict/visual_contrast_jobs.jsonl \
  --previous-output-dir results/build/etbench/conflict/full_v2_qwen3vl_mistral24b/merged \
  --video-root source_datasets/et_bench/ETBench/videos \
  --output-dir "${OUTPUT_DIR}" \
  --grounding-base-url "http://127.0.0.1:${GROUNDING_PORT}/v1" \
  --grounding-model qwen3-vl-evidence-grounder \
  --qa-base-url "http://127.0.0.1:${QA_PORT}/v1" \
  --qa-model mistral-small-24b-instruct-2501 \
  --max-cases "${MAX_CASES}"
