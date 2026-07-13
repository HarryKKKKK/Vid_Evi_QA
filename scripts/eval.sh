#!/bin/bash
#SBATCH --job-name=cgbench_eval
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

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

MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/Qwen3-VL"
PORT=8000
TP=2

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_MODEL="qwen3-vl"

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

mkdir -p logs

echo "Job on node: $(hostname)"
nvidia-smi

echo "Starting vLLM server..."

vllm serve "${MODEL_PATH}" \
    --served-model-name qwen3-vl \
    --tensor-parallel-size "${TP}" \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 130}' \
    --mm-processor-kwargs '{"max_pixels": 75264}' \
    --mm-processor-cache-gb 0 \
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

    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "qwen3-vl"; then
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

echo "Starting evaluation..."

# Set to "--limit 5" (or similar) for a quick sanity-check run before
# committing this 2-GPU, 10-hour job to the full sweep. Leave empty for a
# full run over every entry in --filtered-json.
LIMIT_ARGS=""

# Temporarily disable errexit so a non-zero exit from evaluate.py can be
# captured and reported here, instead of -e killing the script before this
# point is ever reached.
set +e
python scripts/evaluate.py \
    --mode sufficient --task qa \
    --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
    --video-dir source_datasets/next_gqa/videos \
    --results-dir nextgqa_result \
    --workers 8 \
    ${LIMIT_ARGS}
EVAL_EXIT=$?
set -e

if [[ "${EVAL_EXIT}" -ne 0 ]]; then
    echo "ERROR: evaluate.py exited with code ${EVAL_EXIT}" >&2
    exit "${EVAL_EXIT}"
fi

echo "Eval finished (exit code ${EVAL_EXIT})."