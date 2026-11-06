#!/bin/bash
#SBATCH --job-name=v22-missing-retry
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/missing_retry_%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/missing_retry_%A_%a.err
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
QENV=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
NEW_ENV=${ROOT}/.runtime/vllm0191
KIND=${1:?usage: missing_retry_array.sbatch KIND MANIFEST_DIR RESULT_ROOT}
MANIFEST_DIR=${2:?}
RESULT_ROOT=${3:?}
SHARD=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
MANIFEST=${MANIFEST_DIR}/${SHARD}.jsonl
RESULT_DIR=${RESULT_ROOT}/shard_${SHARD}
PORT=$((29000 + RANDOM % 1000))

case "${KIND}" in
  qwen25_7b)
    ENV_DIR=${QENV}; MODEL=Qwen2.5-VL-7B-Instruct; SERVED=qwen25_7b
    TP=1; MAX_LEN=32768; MEDIA_MODE=frames; VARIANT=v2.2; MAX_TOKENS=8192; GPU_UTIL=0.85
    CACHE_ARGS=(--no-enable-prefix-caching)
    MM_LIMIT='{"image":32,"video":0}'; EXTRA=(--trust-remote-code --mm-processor-kwargs '{"max_pixels":75264}') ;;
  qwen3_8b)
    ENV_DIR=${QENV}; MODEL=Qwen3-VL-8B-Instruct; SERVED=qwen3_8b
    TP=1; MAX_LEN=32768; MEDIA_MODE=frames; VARIANT=v2.2; MAX_TOKENS=8192; GPU_UTIL=0.85
    CACHE_ARGS=(--no-enable-prefix-caching)
    MM_LIMIT='{"image":32,"video":0}'; EXTRA=(--trust-remote-code --mm-processor-kwargs '{"max_pixels":75264}') ;;
  internvl35_8b)
    ENV_DIR=${QENV}; MODEL=InternVL3_5-8B; SERVED=internvl3.5-8b
    TP=1; MAX_LEN=40960; MEDIA_MODE=frames; VARIANT=v2.1; MAX_TOKENS=768; GPU_UTIL=0.90
    CACHE_ARGS=(--no-enable-prefix-caching)
    MM_LIMIT='{"image":32,"video":0}'; EXTRA=(--trust-remote-code --mm-processor-kwargs '{"max_dynamic_patch":3}') ;;
  llava_onevision_v21_native)
    ENV_DIR=${QENV}; MODEL=LLaVA-OneVision-Qwen2-7B-OV; SERVED=llava-onevision-qwen2-7b-ov
    TP=1; MAX_LEN=32768; MEDIA_MODE=video; VARIANT=v2.1; MAX_TOKENS=8192; GPU_UTIL=0.90
    CACHE_ARGS=()
    MM_LIMIT='{"image":0,"video":1}'; EXTRA=(--media-io-kwargs '{"video":{"num_frames":32}}') ;;
  glm45v_fp8_tp2_vllm0191)
    ENV_DIR=${NEW_ENV}; MODEL=GLM-4.5V-FP8; SERVED=glm45v_fp8_tp2_vllm0191
    TP=2; MAX_LEN=32768; MEDIA_MODE=video; VARIANT=v2.2; MAX_TOKENS=8192; GPU_UTIL=0.92
    CACHE_ARGS=(--no-enable-prefix-caching)
    MM_LIMIT='{"image":0,"video":1}'; EXTRA=(--trust-remote-code --media-io-kwargs '{"video":{"num_frames":32}}' --reasoning-parser glm45) ;;
  *) echo "Unknown model kind: ${KIND}" >&2; exit 2 ;;
esac

[[ -s "${MANIFEST}" ]] || { echo "Missing or empty manifest: ${MANIFEST}" >&2; exit 2; }
[[ ! -e "${RESULT_DIR}" ]] || { echo "Refusing to overwrite ${RESULT_DIR}" >&2; exit 2; }
mkdir -p "${RESULT_DIR}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda VLLM_ATTENTION_BACKEND=TORCH_SDPA VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 PATH="${ENV_DIR}/bin:${QENV}/bin:${PATH}"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1" VLLM_MODEL="${SERVED}"
export V2_MAX_TOKENS="${MAX_TOKENS}" V2_REQUEST_TIMEOUT=600 PROMPT_ABLATION_VARIANT="${VARIANT}"

cd "${ROOT}"
setsid "${ENV_DIR}/bin/vllm" serve "${ROOT}/models/${MODEL}" \
  --served-model-name "${SERVED}" --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" --limit-mm-per-prompt "${MM_LIMIT}" \
  --mm-processor-cache-gb 0 "${CACHE_ARGS[@]}" \
  --gpu-memory-utilization "${GPU_UTIL}" --enforce-eager --host 127.0.0.1 --port "${PORT}" \
  "${EXTRA[@]}" >"${RESULT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill -- "-${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 360); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then tail -160 "${RESULT_DIR}/server.log" >&2; exit 1; fi
  if curl -fsS "${VLLM_BASE_URL}/models" 2>/dev/null | grep -q "${SERVED}"; then READY=1; break; fi
  sleep 5
done
[[ "${READY:-0}" -eq 1 ]] || { echo "vLLM readiness timeout" >&2; exit 1; }

if [[ "${KIND}" == internvl35_8b ]]; then
  EVALUATOR=results/final_filtered_hierarchical_v2/release_20260902_v21/code/evaluate.py
elif [[ "${VARIANT}" == v2.1 ]]; then
  EVALUATOR=scripts/eval/open_v2/evaluate_prompt_equivalence.py
else
  EVALUATOR=scripts/eval/evaluate.py
fi
"${QENV}/bin/python" -u "${EVALUATOR}" \
  --manifest-jsonl "${MANIFEST}" --task classify \
  --video-dir source_datasets/next_gqa/videos --results-dir "${RESULT_DIR}" \
  --model-tag "${KIND}_missing_retry_20260914" --num-frames 32 \
  --frame-width 560 --sampling uniform --media-mode "${MEDIA_MODE}" --workers 1

test -s "${RESULT_DIR}/merged.classify.jsonl"
