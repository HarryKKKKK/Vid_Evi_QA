#!/bin/bash
#SBATCH --job-name=conflict_vlm_audit
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/conflict_final_audit/vlm_%A_%a.out
#SBATCH --error=logs/conflict_final_audit/vlm_%A_%a.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
OUTPUT_ROOT=${CONFLICT_FINAL_OUTPUT:-reports/conflict_final_audit}
cd "${ROOT}"
mkdir -p "${OUTPUT_ROOT}" logs/conflict_final_audit
export CONFLICT_SOURCE_TASK=all
export EVAL_WORKERS=8
export CONFLICT_QA_PATH="${OUTPUT_ROOT}/prefilter/visual_review_input.jsonl"
export CONFLICT_ALLOW_NULL_ACTION=1

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]]; then
  exec bash etbench_pipeline/conflict/run_conflict_eval_model.sh internvl \
    "${ROOT}/models/InternVL3_5-8B" internvl3.5-8b internvl35_conflict_full_audit \
    "${OUTPUT_ROOT}/internvl35"
else
  # LLaVA-OneVision expands each image into substantially more visual tokens than
  # InternVL. Keep enough evidence coverage while staying below its 32K context.
  export CONFLICT_FRAMES_PER_OCCURRENCE=2
  export CONFLICT_MAX_FRAMES=8
  export CONFLICT_FRAME_WIDTH=448
  exec bash etbench_pipeline/conflict/run_conflict_eval_model.sh llava \
    "${ROOT}/models/LLaVA-OneVision-Qwen2-7B-OV" llava-onevision-qwen2-7b-ov llava_conflict_full_audit \
    "${OUTPUT_ROOT}/llava_onevision"
fi
