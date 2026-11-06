#!/bin/bash
#SBATCH --job-name=v21v22-eq
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/prompt_eq_%x_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/prompt_eq_%x_%j.err
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
KIND=${1:?usage: prompt_equivalence_model.sbatch MODEL_KIND MANIFEST OUTPUT_ROOT}
MANIFEST=${2:?usage: prompt_equivalence_model.sbatch MODEL_KIND MANIFEST OUTPUT_ROOT}
OUTPUT_ROOT=${3:?usage: prompt_equivalence_model.sbatch MODEL_KIND MANIFEST OUTPUT_ROOT}
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
PORT=$((27000 + RANDOM % 1000))

case "${KIND}" in
  qwen25) MODEL=Qwen2.5-VL-32B-Instruct; SERVED=qwen2.5-vl-32b; TP=2; MAX_LEN=32768; MEDIA_MODE=frames; MM_LIMIT='{"image":32,"video":0}'; EXTRA=(--mm-processor-kwargs '{"max_pixels":75264}') ;;
  qwen3) MODEL=Qwen3-VL; SERVED=qwen3-vl; TP=2; MAX_LEN=32768; MEDIA_MODE=frames; MM_LIMIT='{"image":32,"video":0}'; EXTRA=(--mm-processor-kwargs '{"max_pixels":75264}') ;;
  internvl) MODEL=InternVL3_5-8B; SERVED=internvl3.5-8b; TP=1; MAX_LEN=40960; MEDIA_MODE=frames; MM_LIMIT='{"image":32,"video":0}'; EXTRA=(--trust-remote-code --mm-processor-kwargs '{"max_dynamic_patch":3}') ;;
  # OneVision's native video pooling uses about 196 tokens/frame. The same 32
  # JPEGs therefore fit in 32k, unlike presenting them as 32 any-resolution images.
  llava) MODEL=LLaVA-OneVision-Qwen2-7B-OV; SERVED=llava-onevision-qwen2-7b-ov; TP=1; MAX_LEN=32768; MEDIA_MODE=video; MM_LIMIT='{"image":0,"video":1}'; EXTRA=(--media-io-kwargs '{"video":{"num_frames":32}}') ;;
  *) echo "unknown model kind: ${KIND}" >&2; exit 2 ;;
esac

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
export VLLM_MODEL="${SERVED}"
export V2_MAX_TOKENS=8192

cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}/${KIND}"
SERVER_LOG="${OUTPUT_ROOT}/${KIND}/server_${SLURM_JOB_ID}.log"
setsid "${ENV_DIR}/bin/vllm" serve "${ROOT}/models/${MODEL}" \
  --served-model-name "${SERVED}" --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" --limit-mm-per-prompt "${MM_LIMIT}" \
  --mm-processor-cache-gb 0 --gpu-memory-utilization 0.90 --enforce-eager \
  --port "${PORT}" --host 127.0.0.1 "${EXTRA[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill -- "-${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then tail -160 "${SERVER_LOG}" >&2; exit 1; fi
  if curl -fsS "${VLLM_BASE_URL}/models" 2>/dev/null | grep -q "${SERVED}"; then READY=1; break; fi
  sleep 5
done
[[ "${READY}" -eq 1 ]] || { echo "vLLM readiness timeout" >&2; exit 1; }

for VARIANT in v2.1 v2.2; do
  export PROMPT_ABLATION_VARIANT="${VARIANT}"
  RESULT_DIR="${OUTPUT_ROOT}/${KIND}/${VARIANT}"
  mkdir -p "${RESULT_DIR}"
  "${ENV_DIR}/bin/python" -u scripts/eval/open_v2/evaluate_prompt_equivalence.py \
    --manifest-jsonl "${MANIFEST}" --task classify \
    --video-dir source_datasets/next_gqa/videos --results-dir "${RESULT_DIR}" \
    --model-tag "${KIND}_prompt_${VARIANT}" --num-frames 32 --media-mode "${MEDIA_MODE}" \
    --frame-width 560 --sampling uniform --workers 8
done
