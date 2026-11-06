#!/bin/bash
#SBATCH --job-name=q25_risk_ladder
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/qwen25_risk_ladder_%j.out
#SBATCH --error=results/logs/qwen25_risk_ladder_%j.err

set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
MODEL_PATH=${ROOT}/models/Qwen2.5-VL-32B-Instruct
SERVED_NAME=qwen25-risk-ladder
PORT=$((19300 + SLURM_JOB_ID % 500))
SAMPLE=${SAMPLE:-results/ablation_results/qwen25_conservatism/sample_n300.json}
OUTPUT_DIR=${OUTPUT_DIR:-results/ablation_results/qwen25_risk_ladder_v1}
LIMIT=${LIMIT:-100}
WORKERS=${WORKERS:-4}
EVIDENCE_CONDITIONS=${EVIDENCE_CONDITIONS:-full,no_evidence}
RISK_VARIANTS=${RISK_VARIANTS:-0,1}
AUDIT_N=${AUDIT_N:-20}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

cd "${ROOT}"
mkdir -p results/logs "${OUTPUT_DIR}"

setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image":32,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":75264,"use_fast":true}' \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager --host 127.0.0.1 --port "${PORT}" \
  >"results/logs/qwen25_risk_ladder_${SLURM_JOB_ID}.vllm.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill -- "-${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -160 "results/logs/qwen25_risk_ladder_${SLURM_JOB_ID}.vllm.log" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}"; then
    READY=1
    break
  fi
  sleep 5
done
if [[ "${READY}" -ne 1 ]]; then
  echo "vLLM readiness timeout" >&2
  exit 1
fi

"${ENV_DIR}/bin/python" -u scripts/analysis/qwen25_risk_ladder.py \
  --sample "${SAMPLE}" \
  --output "${OUTPUT_DIR}/qwen25vl32b.jsonl" \
  --summary "${OUTPUT_DIR}/qwen25vl32b_summary.json" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED_NAME}" \
  --model-tag qwen25vl32b \
  --ffmpeg "${ENV_DIR}/bin/ffmpeg" \
  --limit "${LIMIT}" \
  --workers "${WORKERS}" \
  --evidence-conditions "${EVIDENCE_CONDITIONS}" \
  --risk-variants "${RISK_VARIANTS}" \
  --audit-n "${AUDIT_N}"
