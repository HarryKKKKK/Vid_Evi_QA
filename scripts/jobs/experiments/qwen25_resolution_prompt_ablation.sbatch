#!/bin/bash
#SBATCH --job-name=qwen25_res_prompt
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/qwen25_res_prompt_%A_%a.out
#SBATCH --error=results/logs/qwen25_res_prompt_%A_%a.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
PIXELS=(75264 150528 301056)
INDEX=${SLURM_ARRAY_TASK_ID:-0}
if (( INDEX < 0 || INDEX >= ${#PIXELS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must be 0-$(( ${#PIXELS[@]} - 1 ))" >&2
  exit 2
fi
MAX_PIXELS=${PIXELS[$INDEX]}
TAG="p${MAX_PIXELS}"
MODEL_PATH="${ROOT}/models/Qwen2.5-VL-32B-Instruct"
SERVED_NAME="qwen25-res-${MAX_PIXELS}"
PORT=$((18100 + SLURM_JOB_ID % 700 + INDEX))
SAMPLE=${SAMPLE:-results/ablation_results/qwen25_conservatism/sample_n300.json}
OUTPUT=${OUTPUT_PREFIX:-results/ablation_results/qwen25_resolution_prompt_v2}/results_${TAG}.jsonl
WORKERS=${WORKERS:-4}
CONDITIONS=${CONDITIONS:-original,calibrated,forced_choice}
LIMIT=${LIMIT:-}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

cd "${ROOT}"
mkdir -p results/logs "$(dirname "${OUTPUT}")"

setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":32,"video":0}' \
  --mm-processor-kwargs "{\"max_pixels\":${MAX_PIXELS},\"use_fast\":true}" \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager --host 127.0.0.1 --port "${PORT}" \
  >"results/logs/qwen25_res_prompt_${SLURM_JOB_ID}_${INDEX}.vllm.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -- "-${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -120 "results/logs/qwen25_res_prompt_${SLURM_JOB_ID}_${INDEX}.vllm.log" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}"; then
    break
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${SERVED_NAME}"

EXTRA_EVAL_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EXTRA_EVAL_ARGS+=(--limit "${LIMIT}")
fi

"${ENV_DIR}/bin/python" -u scripts/analysis/qwen25_conservatism_experiment.py \
  --sample "${SAMPLE}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED_NAME}" \
  --output "${OUTPUT}" \
  --conditions "${CONDITIONS}" \
  --experiment-tag "resolution_prompt_v2_${TAG}" \
  --max-pixels "${MAX_PIXELS}" \
  --workers "${WORKERS}" \
  --ffmpeg "${ENV_DIR}/bin/ffmpeg" \
  "${EXTRA_EVAL_ARGS[@]}"
