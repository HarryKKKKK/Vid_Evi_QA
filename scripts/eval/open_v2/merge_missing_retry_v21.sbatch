#!/bin/bash
#SBATCH --job-name=merge-missing-v21
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/merge_missing_v21_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/merge_missing_v21_%j.err
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
exec /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u \
  scripts/eval/open_v2/merge_missing_retry_v21.py "$@"
