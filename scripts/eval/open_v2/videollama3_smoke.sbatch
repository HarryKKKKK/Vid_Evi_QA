#!/bin/bash
#SBATCH --job-name=videollama3-v22-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/videollama3_v22_smoke_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/videollama3_v22_smoke_%j.err
set -euo pipefail
exec /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u "$1/adapters/run_videollama3_smoke.py" "$1"
