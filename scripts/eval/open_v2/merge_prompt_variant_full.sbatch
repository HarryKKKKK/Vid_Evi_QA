#!/bin/bash
#SBATCH --job-name=merge-prompt-full
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/merge_prompt_full_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/merge_prompt_full_%j.err
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
exec /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u \
  scripts/eval/open_v2/merge_prompt_variant_full.py "$1" "$2" --prompt-version "$3"
