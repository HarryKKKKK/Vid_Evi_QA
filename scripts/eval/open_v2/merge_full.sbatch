#!/bin/bash
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/v22_small_merge_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/v22_small_merge_%j.err
set -euo pipefail
exec /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u "$1/adapters/merge_full.py" "$1" "$2"
