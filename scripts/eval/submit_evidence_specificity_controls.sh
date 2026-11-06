#!/bin/bash
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
SBATCH=${SBATCH:-/cm/shared/apps/slurm/current/bin/sbatch}
SQUEUE=${SQUEUE:-/cm/shared/apps/slurm/current/bin/squeue}
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
RUN_TAG=${RUN_TAG:-representative_n750_v1}
cd "${ROOT}"
mkdir -p logs

selective_job=$(${SBATCH} --parsable --array=0-14%3 \
  --export=ALL,PROTOCOL=selective,RUN_TAG="${RUN_TAG}" \
  scripts/eval/evidence_specificity_controls.sbatch)
forced_job=$(${SBATCH} --parsable --array=0-14%3 \
  --export=ALL,PROTOCOL=forced,RUN_TAG="${RUN_TAG}" \
  scripts/eval/evidence_specificity_controls.sbatch)

printf 'SELECTIVE_JOB=%s\nFORCED_JOB=%s\n' "${selective_job}" "${forced_job}"
${SQUEUE} -j "${selective_job},${forced_job}" -o '%.18i %.28j %.2t %.10M %.24R'
