#!/bin/bash
#SBATCH --job-name=frontier-v22-smoke
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/frontier_v22_smoke_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/frontier_v22_smoke_%j.err
set -euo pipefail
exec /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u "$1/adapters/run_frontier_smoke.py" "$1" "$2"
