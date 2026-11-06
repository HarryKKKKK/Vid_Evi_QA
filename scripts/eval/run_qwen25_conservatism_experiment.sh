#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
PYTHON=${PYTHON:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python}
BASE_URL=${BASE_URL:-http://127.0.0.1:18025/v1}
MODEL=${MODEL:-qwen2.5-vl-32b-conservatism}
WORKERS=${WORKERS:-4}
WAIT_SECONDS=${WAIT_SECONDS:-60}
SAMPLE=${SAMPLE:-ablation_results/qwen25_conservatism/sample_n300.json}
OUTPUT=${OUTPUT:-ablation_results/qwen25_conservatism/results.jsonl}

cd "${ROOT}"
mkdir -p logs ablation_results/qwen25_conservatism

"${PYTHON}" scripts/analysis/qwen25_conservatism_experiment.py \
  --sample "${SAMPLE}" --prepare-only

until curl -fsS "${BASE_URL}/models" | grep -q "${MODEL}"; do
  echo "[WAIT] Qwen2.5 endpoint not ready: ${BASE_URL} model=${MODEL}" >&2
  sleep "${WAIT_SECONDS}"
done

exec "${PYTHON}" -u scripts/analysis/qwen25_conservatism_experiment.py \
  --sample "${SAMPLE}" --base-url "${BASE_URL}" --model "${MODEL}" \
  --output "${OUTPUT}" --workers "${WORKERS}" \
  --ffmpeg /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/ffmpeg
