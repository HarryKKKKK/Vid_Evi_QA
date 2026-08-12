#!/bin/bash
#SBATCH --job-name=nextgqa_ground
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nextgqa_ground_%A_%a.out
#SBATCH --error=logs/nextgqa_ground_%A_%a.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
MODEL_PATH="${GROUNDING_MODEL_PATH:-${ROOT}/models/Qwen3-VL}"
MODEL_TAG="${GROUNDING_MODEL_TAG:-qwen3vl_grounder}"
SERVED_NAME="${GROUNDING_SERVED_NAME:-qwen3-vl-grounder}"
PORT=$((19000 + SLURM_JOB_ID % 1000))
NUM_SHARDS=${NUM_SHARDS:-8}
FPS=${FPS:-2}
MAX_WINDOWS_PER_ITEM=${MAX_WINDOWS_PER_ITEM:-5}
MAX_IMAGES=${MAX_IMAGES:-64}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
TP=${TP:-2}

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl
cd "${ROOT}"
mkdir -p logs

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}, \"video\": 0}" \
  --mm-processor-kwargs '{"max_pixels": 75264}' \
  --mm-processor-cache-gb 0 \
  --enforce-eager \
  --port "${PORT}" \
  --host 127.0.0.1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Grounding server exited during startup" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}"; then
    break
  fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${SERVED_NAME}"

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--overwrite)
fi

python -u nextgqa_pipeline/insufficient/ground_windows.py \
  --endpoint "${MODEL_TAG},http://127.0.0.1:${PORT}/v1,${SERVED_NAME}" \
  --fps "${FPS}" \
  --max-windows-per-item "${MAX_WINDOWS_PER_ITEM}" \
  --num-shards "${NUM_SHARDS}" \
  "${EXTRA_ARGS[@]}"
