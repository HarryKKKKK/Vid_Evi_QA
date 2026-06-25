#!/bin/bash
#SBATCH --job-name=cgbench_eval
#SBATCH -p debug
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -uo pipefail

MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/Qwen3-VL"
PORT=8000
TP=2

export HF_HUB_OFFLINE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_MODEL="qwen3-vl"

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

echo "Job on node: $(hostname)"
nvidia-smi

vllm serve "${MODEL_PATH}" \
    --served-model-name qwen3-vl \
    --tensor-parallel-size ${TP} \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 100}' \
    --mm-processor-kwargs '{"max_pixels": 75264}' \
    --enforce-eager \
    --port ${PORT} --host 0.0.0.0 &
SERVER_PID=$!

cleanup() { kill $SERVER_PID 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for vLLM to be ready..."
READY=0
for i in $(seq 1 120); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: vLLM process died during startup. See log above."; exit 1
    fi
    if curl -s "http://localhost:${PORT}/health" > /dev/null; then
        echo "vLLM ready after ~$((i*10))s"; READY=1; break
    fi
    sleep 10
done
if [ "$READY" -ne 1 ]; then
    echo "ERROR: vLLM not ready after 20 min, aborting."; exit 1
fi

python cgbench_pipeline/evaluate.py --mode sufficient --limit 5 --num-frames 96

echo "Eval finished."