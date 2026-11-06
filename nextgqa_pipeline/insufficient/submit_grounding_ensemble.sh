#!/bin/bash
# Submit one independent grounding array per model and optionally a dependent
# ensemble text-judge array.
#
# GROUNDING_MODEL_SPECS uses semicolon-separated records:
#   tag|model_kind|model_path|served_name|tensor_parallel_size
#
# Example:
#   export GROUNDING_MODEL_SPECS='qwen3|qwen|/models/Qwen3-VL|qwen3-ground|2;internvl35|internvl|/models/InternVL3_5|internvl35-ground|2;llava|llava|/models/LLaVA-OneVision|llava-ground|1'
#   export CLASSIFIER_MODEL_PATH=/models/Qwen3-32B
#   bash nextgqa_pipeline/insufficient/submit_grounding_ensemble.sh

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"

: "${GROUNDING_MODEL_SPECS:?Set semicolon-separated tag|path|served_name|tp records}"
NUM_SHARDS=${NUM_SHARDS:-8}
MIN_GROUNDER_SUPPORT=${MIN_GROUNDER_SUPPORT:-2}
GROUNDING_PREFIX=${GROUNDING_PREFIX:-nextgqa_pipeline/insufficient/ensemble_window_groundings}
CLASSIFICATION_OUTPUT=${CLASSIFICATION_OUTPUT:-nextgqa_pipeline/insufficient/ensemble_leak_classifications.jsonl}

IFS=';' read -r -a specs <<< "${GROUNDING_MODEL_SPECS}"
if (( ${#specs[@]} < 2 )); then
  echo "At least two independent grounding models are required" >&2
  exit 2
fi

job_ids=()
for spec in "${specs[@]}"; do
  IFS='|' read -r tag model_kind model_path served_name tp extra <<< "${spec}"
  if [[ -z "${tag}" || -z "${model_kind}" || -z "${model_path}" || -z "${served_name}" || -z "${tp}" || -n "${extra:-}" ]]; then
    echo "Invalid grounding spec: ${spec}" >&2
    exit 2
  fi
  output="${GROUNDING_PREFIX}_${tag}.jsonl"
  job_id=$(sbatch --parsable \
    --gres="gpu:${tp}" \
    --array="0-$((NUM_SHARDS - 1))%${NUM_SHARDS}" \
    --export="ALL,GROUNDING_MODEL_TAG=${tag},GROUNDING_MODEL_KIND=${model_kind},GROUNDING_MODEL_PATH=${model_path},GROUNDING_SERVED_NAME=${served_name},TP=${tp},NUM_SHARDS=${NUM_SHARDS},GROUNDING_OUTPUT=${output}" \
    jobs/build/nextgqa/ground_windows.sbatch)
  job_ids+=("${job_id}")
  echo "submitted grounder ${tag}: ${job_id} -> ${output%.jsonl}.*.jsonl"
done

dependency=$(IFS=:; echo "${job_ids[*]}")
EXPECTED_GROUNDERS=${#specs[@]}
if [[ -n "${CLASSIFIER_MODEL_PATH:-}" ]]; then
  judge_job=$(sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --array="0-$((NUM_SHARDS - 1))%${NUM_SHARDS}" \
    --export="ALL,NUM_SHARDS=${NUM_SHARDS},GROUNDINGS=${GROUNDING_PREFIX}_*.jsonl,CLASSIFICATION_OUTPUT=${CLASSIFICATION_OUTPUT},MIN_GROUNDER_SUPPORT=${MIN_GROUNDER_SUPPORT},EXPECTED_GROUNDERS=${EXPECTED_GROUNDERS}" \
    jobs/build/nextgqa/classify_groundings.sbatch)
  echo "submitted ensemble judge: ${judge_job} (after ${dependency})"
else
  echo "CLASSIFIER_MODEL_PATH is unset; grounding jobs were submitted without the judge."
  echo "After completion, set CLASSIFIER_MODEL_PATH and submit classify_groundings.sbatch with GROUNDINGS='${GROUNDING_PREFIX}_*.jsonl'."
fi
