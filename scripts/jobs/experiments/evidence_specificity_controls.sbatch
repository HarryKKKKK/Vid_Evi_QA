#!/bin/bash
#SBATCH --job-name=evi-specificity
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/evi_specificity_%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/evi_specificity_%A_%a.err
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
QENV=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
NEW_ENV=${ROOT}/.runtime/vllm0191
RUN_TAG=${RUN_TAG:-representative_n750_v1}
PROTOCOL=${PROTOCOL:?set PROTOCOL to selective or forced}
SHARD_COUNT=${SHARD_COUNT:-5}
INDEX=${SLURM_ARRAY_TASK_ID:?array task required}
MODEL_INDEX=$((INDEX / SHARD_COUNT))
SHARD_INDEX=$((INDEX % SHARD_COUNT))
SHARD=$(printf '%03d' "${SHARD_INDEX}")

KINDS=(qwen3_8b llava_ov_7b qwen36_35b_a3b_fp8)
PATHS=(Qwen3-VL-8B-Instruct LLaVA-OneVision-Qwen2-7B-OV Qwen3.6-35B-A3B-FP8)
SERVED=(qwen3_8b llava-onevision-qwen2-7b-ov qwen36_35b_a3b_fp8)
TAGS=(qwen3vl_8b llava_onevision_7b qwen36_35b_a3b)
MAX_LENS=(32768 32768 32768)
WORKER_COUNTS=(3 2 2)
if (( MODEL_INDEX < 0 || MODEL_INDEX >= ${#KINDS[@]} )); then
  echo "array index must be 0-$(( ${#KINDS[@]} * SHARD_COUNT - 1 ))" >&2
  exit 2
fi

KIND=${KINDS[$MODEL_INDEX]}
MODEL_PATH=${ROOT}/models/${PATHS[$MODEL_INDEX]}
SERVED_NAME=${SERVED[$MODEL_INDEX]}
MODEL_TAG=${TAGS[$MODEL_INDEX]}
MAX_MODEL_LEN=${MAX_LENS[$MODEL_INDEX]}
WORKERS=${WORKER_COUNTS[$MODEL_INDEX]}
ENV_DIR=${QENV}
EXTRA_SERVER=()
EXTRA_EVAL=()
if [[ "${KIND}" == "qwen3_8b" ]]; then
  EXTRA_SERVER+=(--mm-processor-kwargs '{"max_pixels":75264}')
elif [[ "${KIND}" == "qwen36_35b_a3b_fp8" ]]; then
  ENV_DIR=${NEW_ENV}
  EXTRA_SERVER+=(--trust-remote-code --reasoning-parser qwen3 --no-enable-prefix-caching)
  EXTRA_EVAL+=(--extra-body-json '{"chat_template_kwargs":{"enable_thinking":false}}')
fi

PORT=$((22000 + SLURM_JOB_ID % 3000 + INDEX))
SAMPLE=${ROOT}/results/evidence_specificity_controls/${RUN_TAG}/shards/${SHARD}.json
RESULT_DIR=${ROOT}/results/evidence_specificity_controls/${RUN_TAG}/${PROTOCOL}/${MODEL_TAG}
OUTPUT=${RESULT_DIR}/shard_${SHARD}.jsonl
SERVER_LOG=${RESULT_DIR}/server_${SLURM_JOB_ID}_${SHARD}.log

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PATH="${ENV_DIR}/bin:${QENV}/bin:${PATH}"

cd "${ROOT}"
mkdir -p results/logs "${RESULT_DIR}"
test -s "${SAMPLE}"
setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" --limit-mm-per-prompt '{"image":32,"video":0}' \
  --mm-processor-cache-gb 0 --gpu-memory-utilization 0.90 --enforce-eager \
  --host 127.0.0.1 --port "${PORT}" "${EXTRA_SERVER[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() { kill -- "-${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
READY=0
for _ in $(seq 1 360); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then tail -160 "${SERVER_LOG}" >&2; exit 1; fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}"; then READY=1; break; fi
  sleep 5
done
[[ "${READY}" -eq 1 ]] || { echo "vLLM readiness timeout" >&2; exit 1; }

ARGS=(--sample "${SAMPLE}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED_NAME}" --model-tag "${MODEL_TAG}" --output "${OUTPUT}" \
  --protocol "${PROTOCOL}" --workers "${WORKERS}" --limit 150 \
  --ffmpeg "${QENV}/bin/ffmpeg" "${EXTRA_EVAL[@]}")
if [[ "${PROTOCOL}" == "selective" ]]; then
  ARGS+=(--alphas 0.2,0.6 --seeds 17,29,43)
else
  ARGS+=(--forced-alphas 0.2,0.6 --forced-seed 17)
fi
"${QENV}/bin/python" -u scripts/analysis/partial_seeded_control_experiment.py "${ARGS[@]}"
