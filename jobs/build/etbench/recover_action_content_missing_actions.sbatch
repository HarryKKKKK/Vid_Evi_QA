#!/bin/bash
#SBATCH --job-name=etb_action_recover
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/etb_action_recover_%A_%a.out
#SBATCH --error=logs/etb_action_recover_%A_%a.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
NUM_SHARDS=8
GROUNDING_MAX_MODEL_LEN=${GROUNDING_MAX_MODEL_LEN:-7936}
SHARD_NAME=$(printf "shard_%02d" "${SLURM_ARRAY_TASK_ID}")
OUTPUT_DIR="etbench_pipeline/conflict/action_content_recovery/${SHARD_NAME}"
GROUNDING_PORT=$((18400 + SLURM_ARRAY_TASK_ID * 2))
QA_PORT=$((GROUNDING_PORT + 1))

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

cd "${ROOT}"
mkdir -p logs "${OUTPUT_DIR}/logs"

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#ALLOCATED_GPUS[@]}" -lt 2 ]]; then
  echo "Need two allocated GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" setsid "${ENV_DIR}/bin/vllm" serve models/Qwen3-VL \
  --served-model-name qwen3-vl-evidence-grounder \
  --tensor-parallel-size 1 \
  --max-model-len "${GROUNDING_MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"image": 16, "video": 0}' \
  --mm-processor-kwargs '{"max_pixels": 75264}' \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port "${GROUNDING_PORT}" --host 127.0.0.1 >"${OUTPUT_DIR}/logs/qwen3vl_vllm.log" 2>&1 &
GROUNDING_PID=$!

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[1]}" setsid "${ENV_DIR}/bin/vllm" serve models/Mistral-Small-24B-Instruct-2501 \
  --served-model-name mistral-small-24b-instruct-2501 \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port "${QA_PORT}" --host 127.0.0.1 >"${OUTPUT_DIR}/logs/mistral_vllm.log" 2>&1 &
QA_PID=$!

cleanup() {
  kill -- "-${GROUNDING_PID}" 2>/dev/null || true
  kill -- "-${QA_PID}" 2>/dev/null || true
  wait "${GROUNDING_PID}" 2>/dev/null || true
  wait "${QA_PID}" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_model() {
  local pid=$1 port=$2 name=$3 log=$4
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited during startup" >&2
      tail -120 "${log}" >&2
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q "${name}"; then
      echo "READY ${name}"
      return 0
    fi
    sleep 5
  done
  echo "Readiness timeout: ${name}" >&2
  return 1
}

wait_for_model "${GROUNDING_PID}" "${GROUNDING_PORT}" qwen3-vl-evidence-grounder "${OUTPUT_DIR}/logs/qwen3vl_vllm.log" &
WAIT_GROUNDING=$!
wait_for_model "${QA_PID}" "${QA_PORT}" mistral-small-24b-instruct-2501 "${OUTPUT_DIR}/logs/mistral_vllm.log" &
WAIT_QA=$!
wait "${WAIT_GROUNDING}"
wait "${WAIT_QA}"

EXTRA_ARGS=()
if [[ -n "${MAX_CASES:-}" ]]; then
  EXTRA_ARGS+=(--max-cases "${MAX_CASES}")
fi
if [[ "${RETRY_SKIPPED:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--retry-skipped)
fi

"${ENV_DIR}/bin/python" -u etbench_pipeline/conflict/run_action_content_recovery.py \
  --groups etbench_pipeline/conflict/visual_contrast_jobs.jsonl \
  --previous-output-dir etbench_pipeline/conflict/full_v2_qwen3vl_mistral24b/merged \
  --video-root source_datasets/et_bench/ETBench/videos \
  --output-dir "${OUTPUT_DIR}" \
  --grounding-base-url "http://127.0.0.1:${GROUNDING_PORT}/v1" \
  --grounding-model qwen3-vl-evidence-grounder \
  --qa-base-url "http://127.0.0.1:${QA_PORT}/v1" \
  --qa-model mistral-small-24b-instruct-2501 \
  --action-frames-per-occurrence 3 \
  --max-action-images 16 \
  --core-frames 12 \
  --min-action-confidence 0.85 \
  --min-grounding-confidence 0.75 \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/logs/recovery.log"
