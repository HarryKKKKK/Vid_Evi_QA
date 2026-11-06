#!/bin/bash
#SBATCH --job-name=nextgqa_eval_partial_no_leak
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/nextgqa_eval_partial_no_leak_%j.out
#SBATCH --error=logs/nextgqa_eval_partial_no_leak_%j.err

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
FILTERED_JSON=${FILTERED_JSON:-nextgqa_pipeline/nextgqa_filtered_no_evidence_leak_eval_clean.json}
PARTIAL_DIR=${PARTIAL_DIR:-source_datasets/next_gqa/partial_videos_no_evidence_leak}
RESULTS_DIR=${RESULTS_DIR:-nextgqa_result_qwen3vl_partial_no_evidence_leak}
MODEL_PATH=${MODEL_PATH:-${ROOT}/models/Qwen3-VL}
SERVED_NAME=${SERVED_NAME:-qwen3-vl-partial-no-leak-eval}
ALPHAS=${ALPHAS:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
WORKERS=${WORKERS:-8}
PORT=$((24000 + SLURM_JOB_ID % 1000))

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl
cd "${ROOT}"
mkdir -p logs "${RESULTS_DIR}"

python - "${FILTERED_JSON}" "${PARTIAL_DIR}" "${ALPHAS}" <<'PY'
import json
import sys
from pathlib import Path

metadata_path, video_dir, alpha_spec = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = json.loads(metadata_path.read_text(encoding="utf-8"))
alphas = [float(value) for value in alpha_spec.split(",") if value.strip()]
missing = []
for row in rows:
    for alpha in alphas:
        tag = f"a{round(alpha * 100):03d}"
        path = video_dir / f"{row['video_id']}_q{row['qid']}_c2_noise_{tag}.mp4"
        if not path.is_file():
            missing.append(str(path))
print(
    f"[PREFLIGHT] metadata={len(rows)} alphas={len(alphas)} "
    f"expected={len(rows) * len(alphas)} missing={len(missing)}"
)
if missing:
    raise SystemExit(f"missing partial videos; first paths: {missing[:10]}")
PY

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --limit-mm-per-prompt '{"image": 130, "video": 0}' \
  --mm-processor-kwargs '{"max_pixels": 75264}' \
  --mm-processor-cache-gb 0 \
  --enforce-eager \
  --port "${PORT}" \
  --host 127.0.0.1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Qwen3-VL evaluation server exited during startup" >&2
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "${SERVED_NAME}"; then
    break
  fi
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | grep -q "${SERVED_NAME}"

export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_MODEL="${SERVED_NAME}"

python -u scripts/eval/evaluate.py \
  --mode partial \
  --task classify \
  --filtered-json "${FILTERED_JSON}" \
  --partial-candidates "${FILTERED_JSON}" \
  --partial-video-dir "${PARTIAL_DIR}" \
  --partial-mode noise \
  --alphas "${ALPHAS}" \
  --results-dir "${RESULTS_DIR}" \
  --model-tag qwen3vl_partial_no_evidence_leak \
  --workers "${WORKERS}"

python scripts/eval/partial_metrics.py \
  "${RESULTS_DIR}/partial.classify.json" \
  --condition all \
  --csv "${RESULTS_DIR}/partial_alpha_metrics.csv"
