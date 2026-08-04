#!/bin/bash
#SBATCH --job-name=llama4_scout_serve
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/llama4_serve_%j.out
#SBATCH --error=logs/llama4_serve_%j.err

# Serves Llama-4-Scout-17B-16E-Instruct via vLLM as a second model, alongside
# the existing Qwen3-VL server (eval.sh). Mirrors eval.sh's SLURM/vLLM launch
# pattern but on its own port/served-model-name so both can run concurrently
# on separate job allocations.
#
# Why Llama 4 Scout and not Llama-3.2-Vision (11B/90B): evaluate.py sends
# NUM_FRAMES=32 sampled frames as separate image_url blocks in one prompt
# (see build_messages()/`--num-frames`). Llama-3.2-Vision (mllama) is an
# adapter/cross-attention design trained on single-image inputs -- vLLM lets
# you pass multiple images but Meta's own model card and vLLM issue #10983
# note quality collapses (effectively only the first image gets attended to).
# Llama 4 Scout is natively multimodal (early-fusion, image tokens go through
# the same attention stack as text) and was built/day-0-supported in vLLM
# (>=0.8.3) for exactly this multi-image case.
#
# Prereqs (run once, not part of this job):
#   1. Accept the Llama 4 license on HF (meta-llama/Llama-4-Scout-17B-16E-Instruct)
#      and export HF_TOKEN, then download weights, e.g.:
#        huggingface-cli download meta-llama/Llama-4-Scout-17B-16E-Instruct \
#            --local-dir /aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/Llama-4-Scout-17B-16E-Instruct
#   2. Confirm vLLM in envs_qwen3vl is new enough for Llama 4 (>=0.8.3) --
#      it already must postdate that to support Qwen3-VL, so no separate
#      env should be needed. Sanity check:
#        conda run -p /aifs4su/hansirui_2nd/harry/envs_qwen3vl python -c \
#            "import vllm; print(vllm.__version__)"

set -uo pipefail -e

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

echo "CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES:-<unset>}]"
echo "SLURM_JOB_ID=[${SLURM_JOB_ID:-<unset>}]"
echo "SLURM_JOB_GPUS=[${SLURM_JOB_GPUS:-<unset>}]"
echo "SLURM_STEP_GPUS=[${SLURM_STEP_GPUS:-<unset>}]"

MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/Llama-4-Scout-17B-16E-Instruct"
SERVED_NAME="llama4-scout"
PORT=8001   # different from Qwen3-VL's 8000 so both servers can run at once
TP=2

# H100/H800 (Hopper) has native FP8 tensor cores, so we quantize the bf16
# checkpoint on load rather than needing a separate pre-quantized checkpoint.
# 109B total params @ fp8 (~1 byte/param) is ~109GB across 2x80G=160GB,
# leaving headroom for KV cache at --max-model-len 32768.
# (If this ever moves to A100 GPUs: A100 has no native fp8 compute, swap this
# for a pre-quantized AWQ/INT4 checkpoint instead, e.g. drop --quantization
# fp8 --kv-cache-dtype fp8 and point MODEL_PATH at an AWQ build of Scout.)
QUANTIZATION="fp8"
KV_CACHE_DTYPE="fp8"

export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_MODEL="${SERVED_NAME}"

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

mkdir -p logs

echo "Job on node: $(hostname)"
nvidia-smi

echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"

echo "Starting vLLM server (Llama 4 Scout)..."

vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" \
    --quantization "${QUANTIZATION}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 130}' \
    --enforce-eager \
    --port "${PORT}" \
    --host 0.0.0.0 &

SERVER_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM to be ready..."
READY=0

for i in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM process died during startup. See err log above."
        exit 1
    fi

    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "${SERVED_NAME}"; then
        echo "vLLM ready after ~$((i * 10))s"
        READY=1
        break
    fi

    echo "Still waiting... $((i * 10))s"
    sleep 10
done

if [ "${READY}" -ne 1 ]; then
    echo "ERROR: vLLM not ready after 20 min, aborting."
    echo "Last /v1/models response:"
    curl -s "http://localhost:${PORT}/v1/models" || true
    exit 1
fi

echo "vLLM server up at ${VLLM_BASE_URL} (served-model-name=${SERVED_NAME})."
echo "Run evaluate.py against it, e.g.:"
echo "  VLLM_BASE_URL=${VLLM_BASE_URL} VLLM_MODEL=${SERVED_NAME} \\"
echo "      python scripts/eval/evaluate.py --mode sufficient --task qa \\"
echo "      --filtered-json nextgqa_pipeline/nextgqa_filtered.json \\"
echo "      --video-dir source_datasets/next_gqa/videos \\"
echo "      --results-dir nextgqa_result --workers 8 --limit 5"

# Keep the server alive for the job's duration so evaluate.py can be run
# separately (either in another srun/exec into this allocation, or by
# uncommenting a direct evaluate.py call below like eval.sh does).
wait "${SERVER_PID}"
