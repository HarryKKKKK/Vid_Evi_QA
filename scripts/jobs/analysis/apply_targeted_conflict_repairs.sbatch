#!/bin/bash
#SBATCH --job-name=conflict_apply_repairs
#SBATCH -p debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/conflict_apply_repairs_%j.out
#SBATCH --error=results/logs/conflict_apply_repairs_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
cd "${ROOT}"

exec "${PYTHON}" scripts/analysis/apply_targeted_conflict_repairs.py \
  --repair-decisions results/reports/conflict_targeted_review/qwen3vl/options/repair.json \
  --pass-decisions results/reports/conflict_targeted_review/qwen3vl/options/pass.json \
  --first-verification results/reports/conflict_targeted_review/qwen3vl/options/support_verification.jsonl \
  --second-verification results/reports/conflict_targeted_review/repair_recheck/qwen3vl/options/support_verification.jsonl \
  --output-dir results/reports/conflict_targeted_review/final_repaired \
  --min-confidence 0.65
