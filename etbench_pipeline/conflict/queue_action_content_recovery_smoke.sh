#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
QUEUE_DIR=${QUEUE_DIR:-etbench_pipeline/conflict/action_content_recovery_queue}
POLL_SECONDS=${POLL_SECONDS:-60}
FREE_MEMORY_THRESHOLD_MIB=${FREE_MEMORY_THRESHOLD_MIB:-1024}
MAX_LAUNCH_ATTEMPTS=${MAX_LAUNCH_ATTEMPTS:-2}
MAX_CASES=${MAX_CASES:-2}
RUN_OUTPUT_DIR=${RUN_OUTPUT_DIR:-${QUEUE_DIR}/pilot_2}

cd "${ROOT}"
mkdir -p "${QUEUE_DIR}"
echo $$ >"${QUEUE_DIR}/queue.pid"
echo "$(date -Is) queue_started host=$(hostname) max_cases=${MAX_CASES}" | tee -a "${QUEUE_DIR}/queue_status.log"

attempt=0
while true; do
  mapfile -t free_gpus < <(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | awk -F, -v threshold="${FREE_MEMORY_THRESHOLD_MIB}" '
          {gsub(/ /, "", $1); gsub(/ /, "", $2); if (($2 + 0) < threshold) print $1}'
  )
  if (( ${#free_gpus[@]} >= 2 )); then
    attempt=$((attempt + 1))
    grounding_gpu=${free_gpus[0]}
    qa_gpu=${free_gpus[1]}
    echo "$(date -Is) launch_attempt=${attempt} grounding_gpu=${grounding_gpu} qa_gpu=${qa_gpu}" \
      | tee -a "${QUEUE_DIR}/queue_status.log"
    set +e
    ROOT="${ROOT}" \
    OUTPUT_DIR="${RUN_OUTPUT_DIR}" \
    GROUNDING_GPU="${grounding_gpu}" \
    QA_GPU="${qa_gpu}" \
    MAX_CASES="${MAX_CASES}" \
      bash etbench_pipeline/conflict/run_action_content_recovery_smoke.sh \
      >>"${QUEUE_DIR}/pilot_console.log" 2>&1
    code=$?
    set -e
    if [[ "${code}" -eq 0 ]]; then
      echo "$(date -Is) pilot_complete" | tee -a "${QUEUE_DIR}/queue_status.log"
      exit 0
    fi
    echo "$(date -Is) pilot_failed exit=${code}" | tee -a "${QUEUE_DIR}/queue_status.log"
    if (( attempt >= MAX_LAUNCH_ATTEMPTS )); then
      echo "$(date -Is) queue_stopped launch_attempt_limit_reached" \
        | tee -a "${QUEUE_DIR}/queue_status.log"
      exit "${code}"
    fi
    sleep 300
  else
    state=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ';')
    echo "$(date -Is) waiting free=${#free_gpus[@]} gpu_memory_mib=${state}" \
      >>"${QUEUE_DIR}/queue_status.log"
    sleep "${POLL_SECONDS}"
  fi
done
