#!/bin/bash
#SBATCH --job-name=eval_minicpmv45
#SBATCH -p llm
#SBATCH --qos=llm
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
# 以下几项已在探测阶段(job 206709)验证过能正常启动 server:
#   - conda 环境 envs_qwen3vl(vLLM 0.11.0 原生支持 MiniCPMV4_5)
#   - VLLM_ATTENTION_BACKEND=TORCH_SDPA(避开 PTX 工具链报错)
#   - --trust-remote-code 不报错(是否必需未确认,但无害)
#   - --limit-mm-per-prompt 的 JSON 格式被正常解析
#
# 注意: MAX_MODEL_LEN=16384 在 32 帧真实抽帧场景下是否够用,
# 目前【尚未验证过】—— job 206729 那次因为 GPU 显存被其他进程占用
# (Free memory 24.92/79.11 GiB)在启动阶段就直接失败了,根本没跑到
# evaluate.py 那一步。这次去掉 --limit 直接跑全量,如果中途报
# "prompt 太长" 之类的 400 错误,说明 MAX_MODEL_LEN 需要调大;
# 个别样本报错会被记录到 .skipped.jsonl,不会中断整体 job,
# 跑完后请检查该文件确认跳过数量是否在可接受范围。
# ==============================================================================

MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/MiniCPM-V-4_5"
SERVED_NAME="minicpm-v-4_5"
PORT=8000

TP=1
MAX_MODEL_LEN=16384
LIMIT_IMAGES=130

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

# python scripts/eval/evaluate.py \
#     --mode sufficient --task qa \
#     --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
#     --video-dir source_datasets/next_gqa/videos \
#     --results-dir nextgqa_result_minicpmv45 \
#     --model-tag minicpmv45 \
#     --workers 8 \
#     ${LIMIT_ARGS}

# python scripts/eval/evaluate.py \
#     --mode sufficient --task classify \
#     --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
#     --video-dir source_datasets/next_gqa/videos \
#     --results-dir nextgqa_result_minicpmv45 \
#     --model-tag minicpmv45 \
#     --workers 8 \
#     ${LIMIT_ARGS}

python scripts/eval/evaluate.py \
    --mode insufficient --task classify \
    --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
    --insufficient-video-dir source_datasets/next_gqa/freeze_videos \
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