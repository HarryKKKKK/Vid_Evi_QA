#!/bin/bash
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/videollama3_full_%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/videollama3_full_%A_%a.err
set -euo pipefail
RELEASE=$1
SHARD=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
export V2_MANIFEST="${RELEASE}/shards/${SHARD}.jsonl"
export V2_RESULT_DIR="${RELEASE}/results/videollama3_7b/shard_${SHARD}"
export V2_MODEL_TAG="videollama3_7b_native_full_v2_2"
exec /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u \
  "${RELEASE}/adapters/run_videollama3_smoke.py" "${RELEASE}"
