#!/bin/bash
#SBATCH --job-name=internvl3_5_8b_serve
#SBATCH -p debug
#SBATCH --gres=gpu:TODO_GPU_COUNT
#SBATCH --cpus-per-task=TODO_CPUS_PER_TASK
#SBATCH --mem=TODO_MEM
#SBATCH --time=TODO_TIME_LIMIT
#SBATCH --output=logs/internvl3_5_serve_%j.out
#SBATCH --error=logs/internvl3_5_serve_%j.err
#
# The four #SBATCH values above are TODO_* sentinels on purpose (see the
# checklist at the bottom of this file) -- SLURM will refuse to queue the job
# with a clear parse error if you submit before editing them, e.g.:
#   sbatch scripts/eval/serve_internvl3_5_8b.sbatch
# Alternatively override at submit time without editing the file:
#   sbatch --gres=gpu:2 --cpus-per-task=16 --mem=128G --time=10:00:00 \
#       scripts/eval/serve_internvl3_5_8b.sbatch
#
# Deploys OpenGVLab/InternVL3_5-8B via vLLM on the DGX cluster.
#
# Not copied from the Qwen3-VL reference (scripts/eval/eval.sh) because the
# models differ:
#   - Qwen3-VL used VLLM_ATTENTION_BACKEND=TORCH_SDPA; not carried over here
#     since that env var is a Qwen2/3-VL-specific attention-backend workaround,
#     not a general vLLM requirement -- do not assume InternVL3_5 needs it.
#   - --enforce-eager is kept below as a conservative default matching the
#     Qwen3-VL recipe (avoids CUDA-graph capture surprises on a new model),
#     not because it's confirmed necessary for InternVL3_5 -- revisit if you
#     want the throughput that CUDA graphs would give you.
#   - --trust-remote-code is required per InternVL3_5-8B's own model card
#     (custom modeling code), unlike Qwen3-VL.

set -uo pipefail -e

# ---- mandatory per-shell/session convention ----
unset http_proxy && unset https_proxy

echo "CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES:-<unset>}]"
echo "SLURM_JOB_ID=[${SLURM_JOB_ID:-<unset>}]"
echo "SLURM_ARRAY_TASK_ID=[${SLURM_ARRAY_TASK_ID:-<unset>}]"
echo "SLURM_PROCID=[${SLURM_PROCID:-<unset>}]"
echo "SLURM_JOB_GPUS=[${SLURM_JOB_GPUS:-<unset>}]"
echo "SLURM_STEP_GPUS=[${SLURM_STEP_GPUS:-<unset>}]"

# Unique suffix for every file this job writes, so concurrent jobs/tasks never
# race on the same path (required constraint: shard output by SLURM task).
TASK_TAG="${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_${SLURM_PROCID:-0}"

MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/InternVL3_5-8B/"
CONDA_ENV="/aifs4su/hansirui_2nd/harry/envs"

# ==================== values you must confirm before running ====================
# Do not fill these in with a guessed number without saying so -- these are
# left as sentinels deliberately per your instruction not to assume defaults.
SERVED_NAME="internvl3_5-8b"
PORT="TODO_PORT"                       # TODO: pick a port (Qwen3-VL's eval.sh already uses 8000 -- avoid colliding if both run at once)
TP="TODO_TP"                            # TODO: tensor-parallel-size; must equal the GPU count you put in --gres above (checked at runtime below)
MAX_MODEL_LEN="TODO_MAX_MODEL_LEN"      # TODO: fill in after measuring actual token cost of 128 frames x 336px on this model's processor
LIMIT_MM_IMAGE="TODO_LIMIT_MM_IMAGE"    # TODO: same -- max images per prompt, fill in after the measurement above
# ===================================================================================

REQUIRED_PLACEHOLDER_VARS=(PORT TP MAX_MODEL_LEN LIMIT_MM_IMAGE)
missing=()
for var in "${REQUIRED_PLACEHOLDER_VARS[@]}"; do
    val="${!var}"
    if [[ "${val}" == TODO_* ]]; then
        missing+=("${var}")
    fi
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: the following placeholders are still unfilled: ${missing[*]}"
    echo "Edit scripts/eval/serve_internvl3_5_8b.sbatch and replace the TODO_* sentinel values, then resubmit."
    exit 1
fi

RESULTS_DIR="logs/internvl3_5_health"
mkdir -p logs "${RESULTS_DIR}"

echo "Job on node: $(hostname)"
echo "---- nvidia-smi (full) ----"
nvidia-smi

# ---- orphan GPU process check (must run BEFORE starting vLLM) ----
# Rationale: a prior SLURM TIME LIMIT kill on this cluster left a vLLM
# process running and holding VRAM, which then made the next job's server
# fail to allocate memory. Rather than starting vLLM on top of whatever is
# already there, fail loudly and let a human clean it up.
echo "---- checking for orphaned GPU compute processes before starting vLLM ----"
ORPHAN_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv,noheader 2>/dev/null || true)"

if [ -n "${ORPHAN_PROCESSES}" ]; then
    echo "ERROR: found existing GPU compute process(es) on this node BEFORE starting vLLM:"
    echo "${ORPHAN_PROCESSES}"
    echo ""
    echo "This looks like a leftover process from a previously killed job (e.g. a SLURM"
    echo "TIME LIMIT kill that didn't clean up vLLM's children). Refusing to start a new"
    echo "server on top of it -- investigate/kill the orphan manually first, e.g.:"
    echo "  nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
    echo "  kill -9 <pid>"
    exit 1
fi
echo "No pre-existing GPU compute processes found. Proceeding."

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

echo "python: $(which python)"
echo "python version: $(python --version)"

# ---- precondition check: this conda env must actually have GPU vLLM ----
# /aifs4su/hansirui_2nd/harry/envs was documented in this repo's README as
# the CPU/ffmpeg env, not the GPU inference env (envs_qwen3vl was). Using it
# here per your explicit instruction, but verifying it's actually usable for
# GPU inference before spending 20 minutes waiting on a server that can't
# possibly come up.
NUM_VISIBLE_GPUS=$(python -c "
import sys
try:
    import torch
except ImportError as e:
    print(f'FATAL: torch not importable in {sys.prefix}: {e}', file=sys.stderr)
    sys.exit(1)
try:
    import vllm
except ImportError as e:
    print(f'FATAL: vllm not importable in {sys.prefix}: {e}', file=sys.stderr)
    sys.exit(1)
print(f'vllm={vllm.__version__} torch={torch.__version__} cuda_available={torch.cuda.is_available()}', file=sys.stderr)
if not torch.cuda.is_available():
    print('FATAL: torch.cuda.is_available() is False in this conda env -- cannot serve on GPU.', file=sys.stderr)
    sys.exit(1)
print(torch.cuda.device_count())
")
echo "Visible GPU count in this job: ${NUM_VISIBLE_GPUS}"

if [ "${NUM_VISIBLE_GPUS}" != "${TP}" ]; then
    echo "ERROR: TP=${TP} does not match the number of GPUs actually visible to this job (${NUM_VISIBLE_GPUS})."
    echo "Make --gres=gpu:N (SBATCH header) and TP consistent, then resubmit."
    exit 1
fi

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_MODEL="${SERVED_NAME}"

echo "Starting vLLM server (InternVL3_5-8B)..."

vllm serve "${MODEL_PATH}" \
    --trust-remote-code \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "{\"image\": ${LIMIT_MM_IMAGE}}" \
    --enforce-eager \
    --port "${PORT}" \
    --host 0.0.0.0 \
    > "logs/vllm_stdout_${TASK_TAG}.log" 2>&1 &

SERVER_PID=$!
echo "vLLM server PID: ${SERVER_PID} (raw server stdout/stderr: logs/vllm_stdout_${TASK_TAG}.log)"

cleanup() {
    echo "Stopping vLLM server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM to be ready..."
READY=0
for i in $(seq 1 120); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM process died during startup. See logs/vllm_stdout_${TASK_TAG}.log"
        exit 1
    fi
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "${SERVED_NAME}"; then
        echo "vLLM ready after ~$((i * 10))s"
        READY=1
        break
    fi
    echo "Still waiting... $((i * 10))s"
    sleep 10
done

if [ "${READY}" -ne 1 ]; then
    echo "ERROR: vLLM not ready after 20 min, aborting."
    curl -s "http://localhost:${PORT}/v1/models" || true
    exit 1
fi

echo "vLLM server up at ${VLLM_BASE_URL} (served-model-name=${SERVED_NAME})."

# ==================== health check (NOT assumed to succeed) ====================
# Purpose:
#   a) does the server respond at all
#   b) does a top_logprobs field come back in the shape you expect
#   c) does the model/server accept a multi-image content array
# This section only treats a connection-level failure (empty curl response) as
# an error. Whether the JSON shape/content is actually correct is for you to
# read in the log -- it is NOT asserted here, per your instruction not to
# assume success.

HEALTH_LOG="${RESULTS_DIR}/health_check_${TASK_TAG}.json"
echo "Running health check, raw responses will be written to ${HEALTH_LOG}"
{
    echo "===== InternVL3_5-8B health check (task_tag=${TASK_TAG}, $(date -Is)) ====="
} > "${HEALTH_LOG}"

# 1x1 red-pixel JPEG, base64-encoded -- only exercises the image_url content
# block plumbing, not visual understanding quality.
TINY_JPEG_B64="/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAj/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCdABmX/9k="

echo "--- [1] single image + top_logprobs ---" | tee -a "${HEALTH_LOG}"
SINGLE_IMG_RESP=$(curl -s -X POST "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${SERVED_NAME}\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"What color is this image? Answer in one word.\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${TINY_JPEG_B64}\"}}
            ]
        }],
        \"max_tokens\": 16,
        \"logprobs\": true,
        \"top_logprobs\": 5
    }")
echo "${SINGLE_IMG_RESP}" | tee -a "${HEALTH_LOG}"
echo "" >> "${HEALTH_LOG}"

echo "--- [2] multi-image (2 images) content format ---" | tee -a "${HEALTH_LOG}"
MULTI_IMG_RESP=$(curl -s -X POST "http://localhost:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${SERVED_NAME}\",
        \"messages\": [{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"text\", \"text\": \"Are these two images the same? Answer yes or no.\"},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${TINY_JPEG_B64}\"}},
                {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${TINY_JPEG_B64}\"}}
            ]
        }],
        \"max_tokens\": 16
    }")
echo "${MULTI_IMG_RESP}" | tee -a "${HEALTH_LOG}"
echo "" >> "${HEALTH_LOG}"

echo ""
echo "==== HEALTH CHECK RAW OUTPUT ABOVE -- MANUAL VERIFICATION REQUIRED ===="
echo "Full record also written to: ${HEALTH_LOG}"
echo "Please manually confirm:"
echo "  a) both requests returned a real JSON body, not a curl/connection error"
echo "  b) response [1] actually contains a top_logprobs field somewhere under"
echo "     choices[0].logprobs (exact key path may differ by vLLM version --"
echo "     read the raw JSON, don't assume the shape)"
echo "  c) response [2] didn't reject the 2-image content array (confirms"
echo "     InternVL3_5's chat template accepts multi-image input this way)"
echo "This script deliberately does NOT fail the job based on the content of"
echo "these responses -- only a fully empty response is flagged below as a"
echo "connection-level problem."

if [ -z "${SINGLE_IMG_RESP}" ]; then
    echo "WARNING: health check [1] (single image) got an empty response -- likely a connection-level failure, not just a bad answer."
fi
if [ -z "${MULTI_IMG_RESP}" ]; then
    echo "WARNING: health check [2] (multi-image) got an empty response -- likely a connection-level failure, not just a bad answer."
fi

echo "Server staying up for the rest of the job's walltime; run your eval client against ${VLLM_BASE_URL} (VLLM_MODEL=${SERVED_NAME})."
wait "${SERVER_PID}"

# ==================== placeholder checklist (confirm before running) ====================
# #SBATCH --gres=gpu:TODO_GPU_COUNT       -> how many GPUs to allocate
# #SBATCH --cpus-per-task=TODO_CPUS_PER_TASK
# #SBATCH --mem=TODO_MEM
# #SBATCH --time=TODO_TIME_LIMIT
# PORT="TODO_PORT"                        -> avoid colliding with Qwen3-VL's port 8000 if both run concurrently
# TP="TODO_TP"                            -> must equal the GPU count above (checked at runtime, job will abort on mismatch)
# MAX_MODEL_LEN="TODO_MAX_MODEL_LEN"      -> measure actual token cost of 128 frames x 336px on InternVL3_5's processor first
# LIMIT_MM_IMAGE="TODO_LIMIT_MM_IMAGE"    -> same measurement as above
# ==========================================================================================
