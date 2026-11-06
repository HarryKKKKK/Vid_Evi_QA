#!/bin/bash
#SBATCH --job-name=merged_eval
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/merged_eval_%A_%a.out
#SBATCH --error=logs/merged_eval_%A_%a.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"

KINDS=(qwen25 llava internvl)
PATHS=(
  "${ROOT}/models/Qwen2.5-VL-32B-Instruct"
  "${ROOT}/models/LLaVA-OneVision-Qwen2-7B-OV"
  "${ROOT}/models/InternVL3_5-8B"
)
NAMES=(qwen2.5-vl-32b llava-onevision-qwen2-7b-ov internvl3.5-8b)
TAGS=(qwen25vl32b_merged_v1 llava_onevision_merged_v1 internvl35_merged_v1)

i=${SLURM_ARRAY_TASK_ID}
EVAL_WORKERS=8 bash scripts/eval/run_merged_model.sh \
  "${KINDS[$i]}" "${PATHS[$i]}" "${NAMES[$i]}" "${TAGS[$i]}" \
  "results/final_filtered_hierarchical_v1/${TAGS[$i]}"
