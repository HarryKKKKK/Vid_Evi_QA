#!/bin/bash
#SBATCH --job-name=nextgqa_eval_no_leak
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/nextgqa_eval_no_leak_%j.out
#SBATCH --error=logs/nextgqa_eval_no_leak_%j.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
FILTERED_JSON=${FILTERED_JSON:-nextgqa_pipeline/nextgqa_filtered_no_evidence_leak.json}
FREEZE_DIR=${FREEZE_DIR:-source_datasets/next_gqa/freeze_videos_no_evidence_leak}
RESULTS_DIR=${RESULTS_DIR:-nextgqa_result_qwen3vl_no_evidence_leak}
MODEL_PATH=${MODEL_PATH:-${ROOT}/models/Qwen3-VL}
SERVED_NAME=${SERVED_NAME:-qwen3-vl-no-leak-eval}
WORKERS=${WORKERS:-8}
PORT=$((23000 + SLURM_JOB_ID % 1000))

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl
cd "${ROOT}"
mkdir -p logs "${RESULTS_DIR}"

# Refuse to run a partial evaluation when any expected freeze file is absent.
python - "${FILTERED_JSON}" "${FREEZE_DIR}" <<'PY'
import json, sys
from pathlib import Path
metadata, directory = Path(sys.argv[1]), Path(sys.argv[2])
rows = json.loads(metadata.read_text(encoding="utf-8"))
missing = [
    f"{row['video_id']}_q{row['qid']}"
    for row in rows
    if not (directory / f"{row['video_id']}_q{row['qid']}_freeze.mp4").is_file()
]
print(f"[PREFLIGHT] metadata={len(rows)} freeze_present={len(rows)-len(missing)} missing={len(missing)}")
if missing:
    raise SystemExit(f"missing freeze videos; first keys: {missing[:10]}")
PY

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 130, "video": 0}' \
  --mm-processor-kwargs '{"max_pixels": 75264}' \
  --mm-processor-cache-gb 0 \
  --enforce-eager \
  --port "${PORT}" \
  --host 127.0.0.1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Qwen3-VL evaluation server exited during startup" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}"; then
    break
  fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${SERVED_NAME}"

export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_MODEL="${SERVED_NAME}"
EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then EXTRA_ARGS+=(--limit "${LIMIT}"); fi

python -u scripts/eval/evaluate.py \
  --mode insufficient \
  --task classify \
  --filtered-json "${FILTERED_JSON}" \
  --insufficient-video-dir "${FREEZE_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --model-tag qwen3vl_no_evidence_leak \
  --workers "${WORKERS}" \
  "${EXTRA_ARGS[@]}"

python scripts/eval/unanswerable_metrics.py "${RESULTS_DIR}/insufficient.classify.json"
