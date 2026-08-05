#!/bin/bash
#SBATCH --job-name=nextgqa_finalize
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nextgqa_finalize_%A_%a.out
#SBATCH --error=logs/nextgqa_finalize_%A_%a.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
MODEL_PATH="${ROOT}/models/InternVL3_5-8B"
MODEL_TAG=internvl3_5_8b_construct
SERVED_NAME=internvl3_5-8b
PORT=$((19000 + SLURM_JOB_ID % 1000))
NUM_SHARDS=${NUM_SHARDS:-8}
MAX_IMAGES=${MAX_IMAGES:-192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl
cd "${ROOT}"
mkdir -p logs

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt "{\"image\": ${MAX_IMAGES}, \"video\": 0}" \
  --enforce-eager \
  --port "${PORT}" \
  --host 127.0.0.1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "InternVL server exited during startup" >&2
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

python -u nextgqa_pipeline/insufficient/finalize.py \
  --endpoint "${MODEL_TAG},http://127.0.0.1:${PORT}/v1,${SERVED_NAME}" \
  --measurements 'nextgqa_pipeline/insufficient/measurements.*.jsonl' \
  --thresholds nextgqa_pipeline/insufficient/thresholds.json \
  --output-dir source_datasets/next_gqa/insufficient_videos_calibrated \
  --maximum-coverage 0.60 \
  --minimum-visible-seconds 10 \
  --num-shards "${NUM_SHARDS}" \
  "${EXTRA_ARGS[@]}"
