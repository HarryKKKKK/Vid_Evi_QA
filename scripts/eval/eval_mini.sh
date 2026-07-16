#!/bin/bash
#SBATCH --job-name=eval_minicpmv45
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_minicpmv45_%j.out
#SBATCH --error=logs/eval_minicpmv45_%j.err

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

# ==============================================================================
# TP / MAX_MODEL_LEN / LIMIT_IMAGES / 是否需要 --trust-remote-code 这几项
# 尚未经过 probe_minicpmv45.sbatch.sh 的实测确认,不能沿用 InternVL3.5 的
# 40960 或任何别的模型的数值直接套用。脚本在这里中止,逼你先跑探测。
# ==============================================================================
echo "[FATAL] TP / MAX_MODEL_LEN / LIMIT_IMAGES 尚未实测确认,脚本中止。" >&2
echo "        请先运行 probe_minicpmv45.sbatch.sh,确认好数值后删除本行及上面这段 exit。" >&2
exit 1

MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/MiniCPM-V-4_5"
SERVED_NAME="minicpm-v-4_5"
PORT=8000

TP=<TODO: 需实测确认>
MAX_MODEL_LEN=<TODO: 需实测确认>
LIMIT_IMAGES=<TODO: 需实测确认>

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_MODEL="${SERVED_NAME}"

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

mkdir -p logs

echo "Job on node: $(hostname)"
nvidia-smi

echo "Starting vLLM server (MiniCPM-V-4_5)..."

# --trust-remote-code 是否必要未经验证(vLLM 有原生实现,可能不需要),
# 若探测阶段报 unexpected argument 之类的错,删掉这一行。
vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "{\"image\": ${LIMIT_IMAGES}, \"video\": 0}" \
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

echo "Starting evaluation (MiniCPM-V-4_5)..."

LIMIT_ARGS=""

set +e

python scripts/eval/evaluate.py \
    --mode sufficient --task qa \
    --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
    --video-dir source_datasets/next_gqa/videos \
    --results-dir nextgqa_result_minicpmv45 \
    --model-tag minicpmv45 \
    --workers 8 \
    ${LIMIT_ARGS}
EVAL_EXIT=$?
set -e

if [[ "${EVAL_EXIT}" -ne 0 ]]; then
    echo "ERROR: evaluate.py exited with code ${EVAL_EXIT}" >&2
    exit "${EVAL_EXIT}"
fi

echo "Eval finished (exit code ${EVAL_EXIT})."