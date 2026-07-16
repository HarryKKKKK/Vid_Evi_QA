#!/bin/bash
#SBATCH --job-name=eval_performace
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_internvl3_5_%j.out
#SBATCH --error=logs/eval_internvl3_5_%j.err

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

MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/InternVL3_5-8B"
SERVED_NAME="internvl3_5-8b"
PORT=8000

# ---- 以下三个值: ----
# TP=1:            单卡 A100/H800 80GB 跑得动,权重占用 15.9GiB
# MAX_MODEL_LEN=131072: 探测阶段用 130 张 336px 占位图测出的 34886 tokens
#                  被证实不能代表真实场景 -- 正式跑批(job 206034)用
#                  evaluate.py 默认的 32 帧 x 560px 真实抽帧,实测 prompt
#                  长度稳定在 107,060~107,088 tokens 附近(来自多条不同
#                  视频的 400 报错信息)。131072 在此基础上留了约 24000
#                  tokens 余量,覆盖尚未跑到的更长样本 + max_tokens=512
#                  输出预算 + chat template 开销。注意: 调大这个值会降低
#                  并发数(KV cache 总 token 数 / MAX_MODEL_LEN),预期会
#                  变慢,不代表出错。
# LIMIT_IMAGES=130: 与 evaluate.py 里 NUM_FRAMES 上限匹配(实际默认跑
#                  32 帧,130 是留给未来提高帧数的上限,不是当前用满的值)
TP=1
MAX_MODEL_LEN=131072
LIMIT_IMAGES=130

export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_MODEL="${SERVED_NAME}"

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
# envs_qwen3vl: 已通过实测确认这个环境的 vLLM 0.11.0 支持 InternVL3.5 架构
# 且 TORCH_SDPA 后端能正常跑通(envs 里的 vLLM 0.7.0 版本连 CLI 参数格式
# 都对不上,大概率不支持这个架构,已放弃)。
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

mkdir -p logs

echo "Job on node: $(hostname)"
nvidia-smi

echo "Starting vLLM server (InternVL3_5-8B)..."

# --limit-mm-per-prompt 用 JSON 格式(vLLM 0.11.0 要求),显式把 video
# 配额设为 0 -- 避免 dummy video profiling 触发的类型错误(见探测阶段
# job 205907 的报错)。
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

echo "Starting evaluation (InternVL3_5-8B)..."

LIMIT_ARGS=""

set +e

# --model-tag 写死为 internvl3_5,--results-dir 也写死到对应目录,
# 不依赖任何外部变量传入。
python scripts/eval/evaluate.py \
    --mode sufficient --task qa \
    --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
    --video-dir source_datasets/next_gqa/videos \
    --results-dir nextgqa_result_internvl3_5 \
    --model-tag internvl3_5 \
    --workers 8 \
    ${LIMIT_ARGS}
EVAL_EXIT=$?
set -e

if [[ "${EVAL_EXIT}" -ne 0 ]]; then
    echo "ERROR: evaluate.py exited with code ${EVAL_EXIT}" >&2
    exit "${EVAL_EXIT}"
fi

echo "Eval finished (exit code ${EVAL_EXIT})."