#!/usr/bin/env bash

set -euo pipefail

PYTHON=${PYTHON:-/aifs4su/hansirui_3rd/zxiebk/workspace/workspace7/venvs/sglang/bin/python}
SGLANG_SITE=/aifs4su/hansirui_3rd/zxiebk/workspace/workspace7/venvs/sglang/lib/python3.10/site-packages
MODEL=${MODEL:-/home/hansirui_2nd/models/Qwen3.5-9B}
GPU=${GPU:-2}
PORT=${PORT:-18007}

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH="/home/hansirui_2nd/sql_runtime_patch:${SGLANG_SITE}"
export SGLANG_ENABLE_JIT_DEEPGEMM=0

exec "${PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL}" \
    --served-model-name Qwen3.5-9B \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --context-length 65536 \
    --mem-fraction-static 0.55 \
    --max-running-requests 8 \
    --random-seed 3407 \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --sampling-backend pytorch
