#!/bin/bash
#SBATCH --job-name=llavaov-v21-full
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/llavaov_v21_native_%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/llavaov_v21_native_%A_%a.err
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
RELEASE=${1:?usage: llava_onevision_v21_native_full_array.sbatch RELEASE}
SHARD=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
MANIFEST="${RELEASE}/shards/${SHARD}.jsonl"
RESULT_DIR="${RELEASE}/results/llava_onevision_v21_native/shard_${SHARD}"
PORT=$((28000 + RANDOM % 1000))

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PATH="${ENV_DIR}/bin:${PATH}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_MODEL=llava-onevision-qwen2-7b-ov
export V2_MAX_TOKENS=8192
export PROMPT_ABLATION_VARIANT=v2.1

cd "${ROOT}"
mkdir -p "${RESULT_DIR}"
SERVER_LOG="${RESULT_DIR}/server.log"
setsid "${ENV_DIR}/bin/vllm" serve "${ROOT}/models/LLaVA-OneVision-Qwen2-7B-OV" \
  --served-model-name "${VLLM_MODEL}" --tensor-parallel-size 1 \
  --max-model-len 32768 --limit-mm-per-prompt '{"image":0,"video":1}' \
  --media-io-kwargs '{"video":{"num_frames":32}}' \
  --mm-processor-cache-gb 0 --gpu-memory-utilization 0.90 --enforce-eager \
  --port "${PORT}" --host 127.0.0.1 >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill -- "-${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then tail -160 "${SERVER_LOG}" >&2; exit 1; fi
  if curl -fsS "${VLLM_BASE_URL}/models" 2>/dev/null | grep -q "${VLLM_MODEL}"; then READY=1; break; fi
  sleep 5
done
[[ "${READY}" -eq 1 ]] || { echo "vLLM readiness timeout" >&2; exit 1; }

"${ENV_DIR}/bin/python" -u scripts/eval/open_v2/evaluate_prompt_equivalence.py \
  --manifest-jsonl "${MANIFEST}" --task classify \
  --video-dir source_datasets/next_gqa/videos --results-dir "${RESULT_DIR}" \
  --model-tag llava_onevision_v21_native_full --num-frames 32 --media-mode video \
  --frame-width 560 --sampling uniform --workers 8

"${ENV_DIR}/bin/python" scripts/eval/check_v2_run.py "${MANIFEST}" \
  --result "${RESULT_DIR}/merged.classify.json" || true
