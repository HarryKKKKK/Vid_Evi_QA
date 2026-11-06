#!/usr/bin/env bash
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
mkdir -p logs results/ablation_results/qwen25_risk_ladder_smoke results/ablation_results/qwen25_risk_ladder_v1

SMOKE_ID=$(LIMIT=2 WORKERS=1 EVIDENCE_CONDITIONS=full RISK_VARIANTS=0 AUDIT_N=1 \
  OUTPUT_DIR=results/ablation_results/qwen25_risk_ladder_smoke \
  sbatch --parsable jobs/experiments/qwen25_risk_ladder.sbatch)

PILOT_ID=$(LIMIT=100 WORKERS=4 EVIDENCE_CONDITIONS=full,no_evidence RISK_VARIANTS=0,1 AUDIT_N=20 \
  OUTPUT_DIR=results/ablation_results/qwen25_risk_ladder_v1 \
  sbatch --parsable --dependency="afterok:${SMOKE_ID}" jobs/experiments/qwen25_risk_ladder.sbatch)

printf 'SMOKE_JOB_ID=%s\n' "${SMOKE_ID}"
printf 'PILOT_JOB_ID=%s\n' "${PILOT_ID}"
printf 'PILOT_DEPENDENCY=afterok:%s\n' "${SMOKE_ID}"
