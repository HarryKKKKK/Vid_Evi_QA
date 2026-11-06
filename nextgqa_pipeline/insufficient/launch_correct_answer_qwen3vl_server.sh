#!/usr/bin/env bash

set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
VLLM=${VLLM:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/vllm}
GPU=${GPU:-2}
PORT=${PORT:-18008}

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export PYTHONNOUSERSITE=1

exec "${VLLM}" serve models/Qwen3-VL \
    --served-model-name qwen3-vl-evidence-grounder \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 64, "video": 0}' \
    --mm-processor-kwargs '{"max_pixels": 75264}' \
    --mm-processor-cache-gb 0 \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --port "${PORT}" \
    --host 127.0.0.1
