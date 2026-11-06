#!/bin/bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: $0 MODEL_PATH SERVED_NAME CUDA_DEVICES PORT TP [TRUST_REMOTE_CODE]" >&2
    exit 2
fi

MODEL_PATH=$1
SERVED_NAME=$2
CUDA_DEVICES=$3
PORT=$4
TP=$5
TRUST_REMOTE_CODE=${6:-false}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
LOG_FILE="${ROOT}/results/logs/smoke_${SERVED_NAME}.log"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

cd "${ROOT}"
mkdir -p results/logs

extra_args=()
if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
    extra_args+=(--trust-remote-code)
fi

setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt '{"image": 8, "video": 0}' \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --enforce-eager \
    --port "${PORT}" \
    --host 127.0.0.1 \
    "${extra_args[@]}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM exited during startup; tail of ${LOG_FILE}:" >&2
        tail -100 "${LOG_FILE}" >&2
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${SERVED_NAME}"; then
        "${ENV_DIR}/bin/python" scripts/eval/smoke_test_vlm.py \
            --base-url "http://127.0.0.1:${PORT}/v1" \
            --model "${SERVED_NAME}" \
            --video-dir source_datasets/next_gqa/videos \
            --ffmpeg "${ENV_DIR}/bin/ffmpeg"
        echo "SMOKE_SERVE_OK model=${SERVED_NAME} gpus=${CUDA_DEVICES}"
        exit 0
    fi
    sleep 5
done

echo "vLLM did not become ready; tail of ${LOG_FILE}:" >&2
tail -100 "${LOG_FILE}" >&2
exit 1
