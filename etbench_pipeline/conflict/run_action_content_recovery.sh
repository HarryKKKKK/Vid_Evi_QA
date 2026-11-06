#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
PYTHON=${PYTHON:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python}
GROUPS=${GROUPS:-etbench_pipeline/conflict/visual_contrast_jobs.jsonl}
PREVIOUS_OUTPUT_DIR=${PREVIOUS_OUTPUT_DIR:-etbench_pipeline/conflict/full_v2_qwen3vl_mistral24b/merged}
VIDEO_ROOT=${VIDEO_ROOT:-source_datasets/et_bench/ETBench/videos}
OUTPUT_DIR=${OUTPUT_DIR:-etbench_pipeline/conflict/action_content_recovery}

cd "${ROOT}"

EXTRA_ARGS=()
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--dry-run)
else
  : "${GROUNDING_BASE_URL:?Set GROUNDING_BASE_URL for the recovery VLM endpoint}"
  : "${GROUNDING_MODEL:?Set GROUNDING_MODEL, e.g. qwen3-vl-evidence-grounder}"
  : "${QA_BASE_URL:?Set QA_BASE_URL for the text-only endpoint}"
  : "${QA_MODEL:?Set QA_MODEL, e.g. mistral-small-24b-instruct-2501}"
  EXTRA_ARGS+=(
    --grounding-base-url "${GROUNDING_BASE_URL}"
    --grounding-model "${GROUNDING_MODEL}"
    --qa-base-url "${QA_BASE_URL}"
    --qa-model "${QA_MODEL}"
  )
fi
if [[ "${QA_THINKING:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--qa-thinking)
fi
if [[ "${RETRY_SKIPPED:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--retry-skipped)
fi
if [[ -n "${MAX_CASES:-}" ]]; then
  EXTRA_ARGS+=(--max-cases "${MAX_CASES}")
fi

exec "${PYTHON}" -u etbench_pipeline/conflict/run_action_content_recovery.py \
  --groups "${GROUPS}" \
  --previous-output-dir "${PREVIOUS_OUTPUT_DIR}" \
  --video-root "${VIDEO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --action-frames-per-occurrence "${ACTION_FRAMES_PER_OCCURRENCE:-3}" \
  --max-action-images "${MAX_ACTION_IMAGES:-16}" \
  --core-frames "${CORE_FRAMES:-12}" \
  --frame-width "${FRAME_WIDTH:-560}" \
  --min-action-confidence "${MIN_ACTION_CONFIDENCE:-0.85}" \
  --min-grounding-confidence "${MIN_GROUNDING_CONFIDENCE:-0.75}" \
  --num-shards "${NUM_SHARDS:-1}" \
  --shard-index "${SHARD_INDEX:-0}" \
  "${EXTRA_ARGS[@]}"
