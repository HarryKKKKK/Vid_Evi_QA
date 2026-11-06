#!/bin/bash
#SBATCH --job-name=v22-smoke-gate
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/v22_smoke_gate_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/v22_smoke_gate_%j.err
set -euo pipefail
exec /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u "$1/adapters/check_smoke_gate.py" "$1" "$2" "$3"
