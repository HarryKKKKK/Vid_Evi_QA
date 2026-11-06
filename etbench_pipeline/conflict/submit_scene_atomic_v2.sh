#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
cd "${ROOT}"
mkdir -p logs reports/final_benchmark_aligned/scene_atomic_v2/verification
VERIFY_JOB=$(sbatch --parsable jobs/build/etbench/verify_scene_atomic_candidates.sbatch)
VERIFY_ID=${VERIFY_JOB%%;*}
FINAL_JOB=$(sbatch --parsable --dependency="afterok:${VERIFY_ID}" \
  jobs/build/etbench/finalize_scene_atomic_verified.sbatch)
echo "SCENE_VERIFY_JOB=${VERIFY_JOB}"
echo "SCENE_FINALIZE_JOB=${FINAL_JOB}"
squeue -j "${VERIFY_ID},${FINAL_JOB%%;*}"
