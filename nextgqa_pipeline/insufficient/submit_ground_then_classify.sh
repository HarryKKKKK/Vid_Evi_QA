#!/bin/bash
# Submit full NExT-GQA grounding followed by Qwen3 text classification.
# This wrapper exits after printing both job IDs; Slurm keeps the arrays alive
# after the SSH session is closed.

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
NUM_SHARDS=${NUM_SHARDS:-16}
MAX_CONCURRENT_WORKERS=${MAX_CONCURRENT_WORKERS:-8}
MAX_WINDOWS_PER_ITEM=${MAX_WINDOWS_PER_ITEM:-5}
QWEN3_MODEL_PATH=${QWEN3_MODEL_PATH:-${ROOT}/models/Qwen3-VL}

if (( NUM_SHARDS < 1 )); then
  echo "NUM_SHARDS must be at least 1" >&2
  exit 2
fi
if (( MAX_CONCURRENT_WORKERS < 1 )); then
  echo "MAX_CONCURRENT_WORKERS must be at least 1" >&2
  exit 2
fi
if (( MAX_WINDOWS_PER_ITEM < 1 )); then
  echo "MAX_WINDOWS_PER_ITEM must be at least 1 for this submission wrapper" >&2
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
  NUM_SHARDS="${NUM_SHARDS}" \
  MAX_WINDOWS_PER_ITEM="${MAX_WINDOWS_PER_ITEM}" \
  GROUNDING_MODEL_PATH="${QWEN3_MODEL_PATH}" \
  GROUNDING_MODEL_TAG=qwen3vl_grounder \
  GROUNDING_SERVED_NAME=qwen3-vl-grounder \
  sbatch --parsable \
    --array="${ARRAY_SPEC}" \
    nextgqa_pipeline/insufficient/ground_windows.sbatch
)

CLASSIFIER_JOB=$(
  CLASSIFIER_MODEL_PATH="${QWEN3_MODEL_PATH}" \
  CLASSIFIER_MODEL_TAG=qwen3vl_text_judge \
  CLASSIFIER_SERVED_NAME=qwen3-vl-text-judge \
  NUM_SHARDS="${NUM_SHARDS}" \
  sbatch --parsable \
    --dependency="afterok:${GROUND_JOB}" \
    --array="${ARRAY_SPEC}" \
    nextgqa_pipeline/insufficient/classify_groundings.sbatch
)

cat <<EOF
Submitted successfully.

Grounding array:      ${GROUND_JOB}
Classification array: ${CLASSIFIER_JOB}
Dependency:            afterok:${GROUND_JOB}
Array specification:   ${ARRAY_SPEC}
Maximum windows/item:  ${MAX_WINDOWS_PER_ITEM}
Model for both stages: ${QWEN3_MODEL_PATH}

You may exit the SSH session now. Slurm will keep the jobs running.

Monitor later with:
  squeue -j ${GROUND_JOB},${CLASSIFIER_JOB}
  sacct -j ${GROUND_JOB},${CLASSIFIER_JOB} --format=JobID,JobName,State,Elapsed,ExitCode
EOF
