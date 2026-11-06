#!/usr/bin/env bash
# Sourced after preflight; terminates only this invocation's process group.
setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "${MM_LIMIT}" \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --port "${PORT}" \
    --host 127.0.0.1 \
    "${EXTRA_SERVER_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

READY=0
for _ in $(seq 1 240); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM exited during startup; tail of ${LOG_FILE}:" >&2
        tail -160 "${LOG_FILE}" >&2
        exit 1
    fi
    if curl --max-time 5 -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
        | grep -Fq "${SERVED_NAME}"; then
        READY=1
        break
    fi
    sleep 5
done
if [[ "${READY}" -ne 1 ]]; then
    echo "vLLM readiness timeout: ${SERVED_NAME}" >&2
    exit 1
fi

export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_MODEL="${SERVED_NAME}"
