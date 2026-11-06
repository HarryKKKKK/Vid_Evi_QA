#!/bin/bash
# Run this on the login node to submit scripts/eval/serve_internvl3_5_8b.sbatch
# from inside a named tmux session, so an SSH drop during submission/monitoring
# doesn't lose your view of the job.
#
# Note: the sbatch job itself already survives SSH disconnects (SLURM runs it
# on a compute node independent of your shell) -- this tmux wrapper exists so
# you can reattach and keep watching/tailing logs live per the cluster's
# "long-running work must run in a named tmux session" convention.
#
# Usage: bash scripts/eval/launch_internvl3_5_8b_tmux.sh
# Reattach later:  tmux attach -t internvl3_5_8b_serve

set -uo pipefail -e

# ---- mandatory per-shell/session convention ----
unset http_proxy && unset https_proxy

SESSION_NAME="internvl3_5_8b_serve"
SBATCH_SCRIPT="scripts/eval/serve_internvl3_5_8b.sbatch"
REPO_DIR="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux session '${SESSION_NAME}' already exists (job may already be submitted/running)."
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    exit 0
fi

mkdir -p "${REPO_DIR}/logs"

tmux new-session -d -s "${SESSION_NAME}" "bash -lc '
    unset http_proxy && unset https_proxy
    cd \"${REPO_DIR}\"
    JOB_ID=\$(sbatch --parsable \"${SBATCH_SCRIPT}\")
    if [ -z \"\${JOB_ID}\" ]; then
        echo \"sbatch submission failed -- see error above (this is likely the TODO_* placeholder check in ${SBATCH_SCRIPT} rejecting an unfilled value).\"
        exec bash
    fi
    echo \"Submitted job \${JOB_ID}. Waiting for its log file to appear...\"
    LOG_FILE=\"logs/internvl3_5_serve_\${JOB_ID}.out\"
    for i in \$(seq 1 30); do
        [ -f \"\${LOG_FILE}\" ] && break
        sleep 2
    done
    echo \"Tailing \${LOG_FILE} (Ctrl-C stops the tail only, NOT the SLURM job -- cancel with: scancel \${JOB_ID})\"
    tail -f \"\${LOG_FILE}\"
    exec bash
'"

echo "Started tmux session '${SESSION_NAME}'."
echo "Reattach any time with: tmux attach -t ${SESSION_NAME}"
