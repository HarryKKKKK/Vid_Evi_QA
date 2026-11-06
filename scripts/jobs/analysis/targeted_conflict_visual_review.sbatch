#!/bin/bash
#SBATCH --job-name=conflict_targeted
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/conflict_targeted_%j.out
#SBATCH --error=results/logs/conflict_targeted_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
MODEL_PATH=${TARGETED_MODEL_PATH:-${ROOT}/models/Qwen3-VL}
SERVED_NAME=${TARGETED_SERVED_NAME:-qwen3-vl-targeted-conflict-review}
OUTPUT=${TARGETED_OUTPUT:-results/reports/conflict_targeted_review/qwen3vl}
PHASE=${TARGETED_PHASE:-all}
PORT=$((27000 + RANDOM % 1000))

cd "${ROOT}"
mkdir -p "${OUTPUT}/actions" "${OUTPUT}/options" logs
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PATH="${ENV_DIR}/bin:${PATH}"

SERVER_LOG="results/logs/conflict_targeted_qwen3vl_${SLURM_JOB_ID}.vllm.log"
setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size 1 \
  --max-model-len 12288 \
  --limit-mm-per-prompt '{"image": 40, "video": 0}' \
  --mm-processor-kwargs '{"max_pixels":75264,"use_fast":true}' \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager \
  --port "${PORT}" --host 127.0.0.1 >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill -- "-${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vLLM exited during startup" >&2
    tail -160 "${SERVER_LOG}" >&2
    exit 1
  fi
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}" && break
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${SERVED_NAME}"

COMMON=(
  --groups results/build/etbench/conflict/visual_contrast_jobs.jsonl
  --video-root source_datasets/et_bench/ETBench/videos
  --base-url "http://127.0.0.1:${PORT}/v1"
  --model "${SERVED_NAME}"
  --workers "${TARGETED_WORKERS:-4}"
  --frame-width 448
)
QA=${TARGETED_QA:-results/build/etbench/conflict/full_v2_qwen3vl_mistral24b/merged/conflict_qa_v2.jsonl}

if [[ "${PHASE}" != "options" ]]; then
  "${ENV_DIR}/bin/python" -u scripts/analysis/targeted_conflict_visual_review.py actions \
    --qa "${QA}" --output-dir "${OUTPUT}/actions" \
    --frames-per-interval 4 --min-action-confidence 0.70 \
    "${COMMON[@]}" ${TARGETED_ACTION_EXTRA_ARGS:-}
fi

if [[ "${PHASE}" != "actions" ]]; then
  OPTION_QA="${OUTPUT}/actions/qa_with_repaired_actions.jsonl"
  if [[ "${PHASE}" == "options" ]]; then
    OPTION_QA="${QA}"
  fi
  "${ENV_DIR}/bin/python" -u scripts/analysis/targeted_conflict_visual_review.py options \
    --qa "${OPTION_QA}" --output-dir "${OUTPUT}/options" \
    --frames-per-interval "${TARGETED_OPTION_FRAMES:-8}" --padding 0.75 --min-support-confidence 0.65 \
    "${COMMON[@]}" ${TARGETED_OPTION_EXTRA_ARGS:-}
fi
