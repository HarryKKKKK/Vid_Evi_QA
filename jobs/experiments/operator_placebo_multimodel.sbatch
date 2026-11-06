#!/bin/bash
#SBATCH --job-name=operator_placebo
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/operator_placebo_%A_%a.out
#SBATCH --error=logs/operator_placebo_%A_%a.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
INDEX=${SLURM_ARRAY_TASK_ID:-0}
KINDS=(qwen qwen internvl)
PATHS=("${ROOT}/models/Qwen3-VL" "${ROOT}/models/Qwen2.5-VL-32B-Instruct" "${ROOT}/models/InternVL3_5-8B")
NAMES=(qwen3-operator qwen25-operator internvl35-operator)
TAGS=(qwen3vl qwen25vl32b internvl35_8b)
TPS=(2 2 1)
MAX_LENS=(32768 32768 40960)
if (( INDEX < 0 || INDEX >= ${#KINDS[@]} )); then exit 2; fi
KIND=${KINDS[$INDEX]}; MODEL_PATH=${PATHS[$INDEX]}; SERVED_NAME=${NAMES[$INDEX]}
MODEL_TAG=${TAGS[$INDEX]}; TP=${TPS[$INDEX]}; MAX_MODEL_LEN=${MAX_LENS[$INDEX]}
PORT=$((18900 + SLURM_JOB_ID % 700 + INDEX))
SAMPLE=${SAMPLE:-ablation_results/qwen25_conservatism/sample_n300.json}
OUTPUT=${OUTPUT_PREFIX:-ablation_results/operator_placebo_v2}/operator_${MODEL_TAG}.jsonl
WORKERS=${WORKERS:-3}
LIMIT=${LIMIT:-100}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1; export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1 VLLM_ATTENTION_BACKEND=TORCH_SDPA VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA VLLM_ALLREDUCE_USE_SYMM_MEM=0
cd "${ROOT}"; mkdir -p logs "$(dirname "${OUTPUT}")"

EXTRA_ARGS=()
if [[ "${KIND}" == qwen ]]; then EXTRA_ARGS+=(--mm-processor-kwargs '{"max_pixels":75264}'); fi
if [[ "${KIND}" == internvl ]]; then EXTRA_ARGS+=(--trust-remote-code --mm-processor-kwargs '{"max_dynamic_patch":3}'); fi
setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP}" --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"image":32,"video":0}' --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.90 --enforce-eager --host 127.0.0.1 --port "${PORT}" \
  "${EXTRA_ARGS[@]}" >"logs/operator_placebo_${SLURM_JOB_ID}_${INDEX}.vllm.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill -- "-${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then tail -120 "logs/operator_placebo_${SLURM_JOB_ID}_${INDEX}.vllm.log" >&2; exit 1; fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}"; then break; fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${SERVED_NAME}"
"${ENV_DIR}/bin/python" -u scripts/analysis/operator_placebo_experiment.py \
  --sample "${SAMPLE}" --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED_NAME}" \
  --model-tag "${MODEL_TAG}" --output "${OUTPUT}" --workers "${WORKERS}" \
  --limit "${LIMIT}" --ffmpeg "${ENV_DIR}/bin/ffmpeg"
