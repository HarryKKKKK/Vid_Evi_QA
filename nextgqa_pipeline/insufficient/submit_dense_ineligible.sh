#!/bin/bash
# Submit dense fallback grounding and its dependent Qwen3 text judgment.
# The shell may exit immediately after this script returns; Slurm owns both arrays.

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
NUM_SHARDS=${NUM_SHARDS:-16}
MAX_CONCURRENT_WORKERS=${MAX_CONCURRENT_WORKERS:-8}
QWEN3_MODEL_PATH=${QWEN3_MODEL_PATH:-${ROOT}/models/Qwen3-VL}
CORE_SECONDS=${CORE_SECONDS:-10}
CONTEXT_PADDING=${CONTEXT_PADDING:-2}
FPS=${FPS:-2}

if (( NUM_SHARDS < 1 || MAX_CONCURRENT_WORKERS < 1 )); then
  echo "NUM_SHARDS and MAX_CONCURRENT_WORKERS must be at least 1" >&2
  exit 2
fi
if [[ ! -d "${QWEN3_MODEL_PATH}" ]]; then
  echo "Qwen3 model directory does not exist: ${QWEN3_MODEL_PATH}" >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p logs
ARRAY_SPEC="0-$((NUM_SHARDS - 1))%${MAX_CONCURRENT_WORKERS}"

GROUND_JOB=$(
  NUM_SHARDS="${NUM_SHARDS}" FPS="${FPS}" CORE_SECONDS="${CORE_SECONDS}" \
  CONTEXT_PADDING="${CONTEXT_PADDING}" GROUNDING_MODEL_PATH="${QWEN3_MODEL_PATH}" \
  sbatch --parsable --array="${ARRAY_SPEC}" \
    nextgqa_pipeline/insufficient/dense_ground_ineligible.sbatch
)

CLASSIFIER_JOB=$(
  NUM_SHARDS="${NUM_SHARDS}" CLASSIFIER_MODEL_PATH="${QWEN3_MODEL_PATH}" \
  sbatch --parsable --dependency="afterok:${GROUND_JOB}" --array="${ARRAY_SPEC}" \
    nextgqa_pipeline/insufficient/classify_dense_ineligible.sbatch
)

cat <<EOF
Submitted dense ineligible audit successfully.

Dense grounding array: ${GROUND_JOB}
Classification array:  ${CLASSIFIER_JOB}
Dependency:             afterok:${GROUND_JOB}
Array specification:    ${ARRAY_SPEC}
Window policy:          ${CORE_SECONDS}s core + ${CONTEXT_PADDING}s context on each side
Sampling:               ${FPS} FPS, official evidence frozen
Model for both stages:  ${QWEN3_MODEL_PATH}

You may exit SSH now. Monitor later with:
  squeue -j ${GROUND_JOB},${CLASSIFIER_JOB}
  sacct -j ${GROUND_JOB},${CLASSIFIER_JOB} --format=JobID,JobName,State,Elapsed,ExitCode
EOF
