#!/bin/bash
#SBATCH --job-name=verify_conflict_distractors
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/verify_conflict_distractors_%j.out
#SBATCH --error=logs/verify_conflict_distractors_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
PORT=$((28000 + RANDOM % 1000))
NAME=qwen3-vl-distractor-verifier
cd "${ROOT}"
mkdir -p logs reports/final_benchmark_aligned/work
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export PATH="${ENV_DIR}/bin:${PATH}"
LOG=logs/verify_conflict_distractors_${SLURM_JOB_ID}.vllm.log
setsid "${ENV_DIR}/bin/vllm" serve models/Qwen3-VL \
  --served-model-name "${NAME}" --tensor-parallel-size 1 --max-model-len 12288 \
  --limit-mm-per-prompt '{"image":40,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":75264,"use_fast":true}' \
  --mm-processor-cache-gb 0 --gpu-memory-utilization 0.92 --enforce-eager \
  --port "${PORT}" --host 127.0.0.1 >"${LOG}" 2>&1 &
PID=$!
cleanup(){ kill -- "-${PID}" 2>/dev/null || true; wait "${PID}" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 240); do
  if ! kill -0 "${PID}" 2>/dev/null; then tail -160 "${LOG}" >&2; exit 1; fi
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${NAME}" && break
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${NAME}"
exec "${ENV_DIR}/bin/python" -u scripts/analysis/verify_conflict_distractors_vlm.py \
  --requests "${DISTRACTOR_REQUESTS:-reports/final_benchmark_aligned/work/conflict_distractor_requests.jsonl}" \
  --candidates "${DISTRACTOR_CANDIDATES:-reports/final_benchmark_aligned/work/conflict_distractor_candidates.jsonl}" \
  --groups etbench_pipeline/conflict/visual_contrast_jobs.jsonl \
  --video-root source_datasets/et_bench/ETBench/videos \
  --output "${DISTRACTOR_VERIFICATION:-reports/final_benchmark_aligned/work/conflict_distractor_verification.jsonl}" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${NAME}" --workers 4 \
  --frames-per-occurrence 3 --frame-width 448 \
  --key-prefix "${DISTRACTOR_KEY_PREFIX:-base}"
