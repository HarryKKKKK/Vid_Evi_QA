#!/usr/bin/env bash
# Run only after two GPUs are explicitly assigned to this experiment.
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
ENV_DIR=${ENV_DIR:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl}
CUDA_DEVICES=${CUDA_DEVICES:?Set CUDA_DEVICES to two assigned GPUs, e.g. 0,2}
PORT=${PORT:-18025}
MODEL_PATH=${MODEL_PATH:-${ROOT}/models/Qwen2.5-VL-32B-Instruct}

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

exec "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name qwen2.5-vl-32b-conservatism \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":32,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":75264,"use_fast":true}' \
  --mm-processor-cache-gb 0 --gpu-memory-utilization 0.90 \
  --enforce-eager --host 127.0.0.1 --port "${PORT}"
