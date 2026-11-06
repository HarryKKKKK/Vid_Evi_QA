#!/bin/bash
#SBATCH --job-name=scene-atomic-v2
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --array=0-7%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/scene_atomic_v2_%A_%a.out
#SBATCH --error=logs/scene_atomic_v2_%A_%a.err
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
PORT=$((28600 + SLURM_ARRAY_TASK_ID))
NAME=qwen3-vl-scene-atomic-verifier
OUT=reports/final_benchmark_aligned/scene_atomic_v2/verification
cd "${ROOT}"
mkdir -p logs "${OUT}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
LOG="logs/scene_atomic_v2_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.vllm.log"
setsid "${ENV_DIR}/bin/vllm" serve models/Qwen3-VL --served-model-name "${NAME}" \
  --tensor-parallel-size 1 --max-model-len 7936 --limit-mm-per-prompt '{"image":40,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":75264,"use_fast":true}' --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.90 --enforce-eager --port "${PORT}" --host 127.0.0.1 >"${LOG}" 2>&1 &
PID=$!
cleanup(){ kill -- "-${PID}" 2>/dev/null || true; wait "${PID}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 240); do
  kill -0 "${PID}" 2>/dev/null || { tail -120 "${LOG}" >&2; exit 1; }
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${NAME}" && break
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${NAME}"
"${ENV_DIR}/bin/python" -u etbench_pipeline/conflict/verify_scene_atomic_candidates.py \
  --requests reports/final_benchmark_aligned/scene_atomic_v2/scene_rebuild_requests.jsonl \
  --groups etbench_pipeline/conflict/visual_contrast_jobs.jsonl \
  --video-root source_datasets/et_bench/ETBench/videos \
  --output "${OUT}/shard_$(printf '%02d' "${SLURM_ARRAY_TASK_ID}").jsonl" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${NAME}" \
  --num-shards 8 --shard-index "${SLURM_ARRAY_TASK_ID}" --workers 4
