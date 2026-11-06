#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
SBATCH=${SBATCH:-/cm/shared/apps/slurm/current/bin/sbatch}
SQUEUE=${SQUEUE:-/cm/shared/apps/slurm/current/bin/squeue}
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
cd "${ROOT}"

resolution_job=$(${SBATCH} --parsable --array=0-2%1 \
  scripts/jobs/experiments/qwen25_resolution_prompt_ablation.sbatch)
blind_job=$(${SBATCH} --parsable --array=0-3%1 \
  scripts/jobs/experiments/matched_blind_multimodel.sbatch)
operator_job=$(${SBATCH} --parsable --dependency="afterok:${blind_job}" --array=0-2%1 \
  scripts/jobs/experiments/operator_placebo_multimodel.sbatch)

echo "RESOLUTION_JOB=${resolution_job}"
echo "BLIND_JOB=${blind_job}"
echo "OPERATOR_JOB=${operator_job}"
${SQUEUE} -j "${resolution_job},${blind_job},${operator_job}" \
  -o '%.18i %.28j %.2t %.10M %.24R'
