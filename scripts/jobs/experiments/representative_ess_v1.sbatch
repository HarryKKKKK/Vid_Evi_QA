#!/bin/bash
#SBATCH --job-name=repr-ess
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/repr_ess_%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/logs/repr_ess_%A_%a.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
QENV=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
NEW_ENV=${ROOT}/.runtime/vllm0191
INDEX=${SLURM_ARRAY_TASK_ID:-0}
KINDS=(qwen3_8b qwen36_35b llava_ov_7b)
if (( INDEX < 0 || INDEX >= ${#KINDS[@]} )); then exit 2; fi
KIND=${KINDS[$INDEX]}
TP=1
MAX_LEN=32768
GPU_UTIL=0.88
EXTRA=()
REQUEST_EXTRA='{}'
MEDIA_MODE=frames
MM_LIMIT='{"image":32,"video":0}'
case "${KIND}" in
  qwen3_8b)
    ENV_DIR=${QENV}; MODEL=Qwen3-VL-8B-Instruct; SERVED=qwen3_8b_ess
    EXTRA+=(--trust-remote-code --mm-processor-kwargs '{"max_pixels":75264}') ;;
  qwen36_35b)
    ENV_DIR=${NEW_ENV}; MODEL=Qwen3.6-35B-A3B-FP8; SERVED=qwen36_35b_ess
    MEDIA_MODE=video; MM_LIMIT='{"image":0,"video":1}'
    REQUEST_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
    EXTRA+=(--trust-remote-code --reasoning-parser qwen3 --media-io-kwargs '{"video":{"num_frames":32}}') ;;
  llava_ov_7b)
    ENV_DIR=${QENV}; MODEL=LLaVA-OneVision-Qwen2-7B-OV; SERVED=llava_ov_7b_ess
    MEDIA_MODE=video; MM_LIMIT='{"image":0,"video":1}'
    EXTRA+=(--media-io-kwargs '{"video":{"num_frames":32}}') ;;
esac

PORT=$((31000 + RANDOM % 1000))
RUN_TAG=${RUN_TAG:-representative_n800_v1}
SAMPLE=${SAMPLE:-results/representative_ess/${RUN_TAG}/sample_n800.json}
OUTPUT=${ROOT}/results/representative_ess/${RUN_TAG}/selective_${KIND}.jsonl
LIMIT=${LIMIT:-800}
WORKERS=${WORKERS:-3}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda VLLM_ATTENTION_BACKEND=TORCH_SDPA VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 PATH="${ENV_DIR}/bin:${QENV}/bin:${PATH}"
cd "${ROOT}"
mkdir -p results/logs "$(dirname "${OUTPUT}")"

setsid "${ENV_DIR}/bin/vllm" serve "${ROOT}/models/${MODEL}" \
  --served-model-name "${SERVED}" --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" --limit-mm-per-prompt "${MM_LIMIT}" \
  --mm-processor-cache-gb 0 --no-enable-prefix-caching \
  --gpu-memory-utilization "${GPU_UTIL}" --enforce-eager --host 127.0.0.1 --port "${PORT}" \
  "${EXTRA[@]}" >"results/logs/repr_ess_${SLURM_JOB_ID}_${INDEX}.vllm.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill -- "-${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 360); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then tail -160 "results/logs/repr_ess_${SLURM_JOB_ID}_${INDEX}.vllm.log" >&2; exit 1; fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED}"; then READY=1; break; fi
  sleep 5
done
[[ "${READY:-0}" -eq 1 ]] || { echo "vLLM readiness timeout" >&2; exit 1; }

"${QENV}/bin/python" -u scripts/analysis/partial_seeded_control_experiment.py \
  --sample "${SAMPLE}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED}" --model-tag "${KIND}" --output "${OUTPUT}" \
  --workers "${WORKERS}" --limit "${LIMIT}" --protocol selective \
  --media-mode "${MEDIA_MODE}" \
  --alphas 0.0 --seeds 17,29,43 --extra-body-json "${REQUEST_EXTRA}" \
  --ffmpeg "${QENV}/bin/ffmpeg"

test -s "${OUTPUT}"
