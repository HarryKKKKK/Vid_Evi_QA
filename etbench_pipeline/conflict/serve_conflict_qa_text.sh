#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
VLLM=${VLLM:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/vllm}
QA_MODEL_PATH=${QA_MODEL_PATH:-${ROOT}/models/Mistral-Small-24B-Instruct-2501}

GPU=${GPU:-1}
PORT=${PORT:-18009}
SERVED_NAME=${SERVED_NAME:-mistral-small-24b-instruct-2501}

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA

exec "${VLLM}" serve "${QA_MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port "${PORT}" \
  --host 127.0.0.1
