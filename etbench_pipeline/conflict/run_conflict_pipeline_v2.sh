#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
PYTHON=${PYTHON:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python}
GROUPS=${GROUPS:-etbench_pipeline/conflict/visual_contrast_jobs.jsonl}
VIDEO_ROOT=${VIDEO_ROOT:-source_datasets/et_bench/ETBench/videos}
OUTPUT_DIR=${OUTPUT_DIR:-etbench_pipeline/conflict/results_v2}

: "${GROUNDING_BASE_URL:?Set GROUNDING_BASE_URL, e.g. http://grounding-node:18008/v1}"
: "${GROUNDING_MODEL:?Set GROUNDING_MODEL, e.g. qwen3-vl-evidence-grounder}"
: "${QA_BASE_URL:?Set QA_BASE_URL for the text-only OpenAI-compatible endpoint}"
: "${QA_MODEL:?Set QA_MODEL, e.g. qwen3-32b}"

cd "${ROOT}"
EXTRA_ARGS=()
if [[ "${QA_THINKING:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--qa-thinking)
fi
exec "${PYTHON}" -u etbench_pipeline/conflict/run_conflict_pipeline_v2.py \
  --groups "${GROUPS}" \
  --video-root "${VIDEO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --grounding-base-url "${GROUNDING_BASE_URL}" \
  --grounding-model "${GROUNDING_MODEL}" \
  --qa-base-url "${QA_BASE_URL}" \
  --qa-model "${QA_MODEL}" \
  --core-frames "${CORE_FRAMES:-12}" \
  --context-frames "${CONTEXT_FRAMES:-4}" \
  --context-seconds "${CONTEXT_SECONDS:-5}" \
  --frame-width "${FRAME_WIDTH:-560}" \
  --min-confidence "${MIN_CONFIDENCE:-0.75}" \
  "${EXTRA_ARGS[@]}"
