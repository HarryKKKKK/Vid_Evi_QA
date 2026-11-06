#!/bin/bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 MODEL_KIND MODEL_PATH SERVED_NAME MODEL_TAG RESULTS_DIR" >&2
    exit 2
fi

MODEL_KIND=$1
MODEL_PATH=$2
SERVED_NAME=$3
MODEL_TAG=$4
RESULTS_DIR=$5

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
ENV_DIR=/aifs4su/hansirui_2nd/harry/envs_qwen3vl
FILTERED_JSON=${PARTIAL_FILTERED_JSON:-nextgqa_pipeline/nextgqa_filtered_no_evidence_leak_eval_clean.json}
CANDIDATES=${PARTIAL_CANDIDATES:-nextgqa_pipeline/partial_candidates_n500_seed20260819.json}
PARTIAL_DIR=${PARTIAL_VIDEO_DIR:-source_datasets/next_gqa/partial_videos_no_evidence_leak}
ALPHAS=${PARTIAL_ALPHAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
EXPECTED_ANCHORS=${PARTIAL_EXPECTED_ANCHORS:-500}
EXPECTED_TASKS=${PARTIAL_EXPECTED_TASKS:-4500}
RUN_ID=${SLURM_JOB_ID:-$$}
PORT=$((25000 + RANDOM % 1000))
EVAL_WORKERS=${EVAL_WORKERS:-8}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export PATH="${ENV_DIR}/bin:${PATH}"

case "${MODEL_KIND}" in
    qwen25)
        MAX_MODEL_LEN=32768
        NUM_FRAMES=32
        TP=2
        MEDIA_MODE=frames
        MM_LIMIT='{"image": 32, "video": 0}'
        EXTRA_SERVER_ARGS=(
            --mm-processor-kwargs '{"max_pixels": 75264}'
        )
        ;;
    llava)
        MAX_MODEL_LEN=32768
        NUM_FRAMES=32
        TP=1
        MEDIA_MODE=video
        MM_LIMIT='{"image": 0, "video": 1}'
        EXTRA_SERVER_ARGS=(
            --media-io-kwargs '{"video": {"num_frames": 32}}'
        )
        ;;
    internvl)
        MAX_MODEL_LEN=40960
        NUM_FRAMES=32
        TP=1
        MEDIA_MODE=frames
        MM_LIMIT='{"image": 32, "video": 0}'
        EXTRA_SERVER_ARGS=(
            --trust-remote-code
            --mm-processor-kwargs '{"max_dynamic_patch": 3}'
        )
        ;;
    *)
        echo "Unknown MODEL_KIND=${MODEL_KIND}" >&2
        exit 2
        ;;
esac

cd "${ROOT}"
mkdir -p logs "${RESULTS_DIR}"

"${ENV_DIR}/bin/python" - "${CANDIDATES}" "${PARTIAL_DIR}" "${ALPHAS}" \
    "${EXPECTED_ANCHORS}" "${EXPECTED_TASKS}" <<'PY'
import json
import sys
from pathlib import Path

candidates = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
partial_dir = Path(sys.argv[2])
alphas = [float(value) for value in sys.argv[3].split(",")]
expected_anchors = int(sys.argv[4])
expected_tasks = int(sys.argv[5])
if len(candidates) != expected_anchors or len(candidates) * len(alphas) != expected_tasks:
    raise SystemExit(
        f"unexpected selection: anchors={len(candidates)} alphas={len(alphas)} "
        f"expected_anchors={expected_anchors} expected_tasks={expected_tasks}"
    )
missing = []
for row in candidates:
    for alpha in alphas:
        tag = f"a{round(alpha * 100):03d}"
        path = partial_dir / f"{row['video_id']}_q{row['qid']}_c2_noise_{tag}.mp4"
        if not path.is_file():
            missing.append(str(path))
if missing:
    raise SystemExit(f"missing partial videos: count={len(missing)} first={missing[:10]}")
print(f"PARTIAL_PREFLIGHT_OK anchors={len(candidates)} tasks={len(candidates) * len(alphas)}")
PY

LOG_FILE="logs/partial_${MODEL_TAG}_${RUN_ID}.vllm.log"
setsid "${ENV_DIR}/bin/vllm" serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "${MM_LIMIT}" \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --port "${PORT}" \
    --host 127.0.0.1 \
    "${EXTRA_SERVER_ARGS[@]}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
    kill -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 240); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM exited during startup; tail of ${LOG_FILE}:" >&2
        tail -160 "${LOG_FILE}" >&2
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null \
        | grep -q "${SERVED_NAME}"; then
        READY=1
        break
    fi
    sleep 5
done
if [[ "${READY}" -ne 1 ]]; then
    echo "vLLM readiness timeout: ${SERVED_NAME}" >&2
    exit 1
fi

export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export VLLM_MODEL="${SERVED_NAME}"

"${ENV_DIR}/bin/python" -u scripts/eval/evaluate.py \
    --mode partial \
    --task classify \
    --filtered-json "${FILTERED_JSON}" \
    --partial-candidates "${CANDIDATES}" \
    --partial-video-dir "${PARTIAL_DIR}" \
    --partial-mode noise \
    --alphas "${ALPHAS}" \
    --results-dir "${RESULTS_DIR}" \
    --model-tag "${MODEL_TAG}" \
    --num-frames "${NUM_FRAMES}" \
    --media-mode "${MEDIA_MODE}" \
    --frame-width 560 \
    --sampling uniform \
    --workers "${EVAL_WORKERS}"

"${ENV_DIR}/bin/python" - "${RESULTS_DIR}" "${EXPECTED_ANCHORS}" "${EXPECTED_TASKS}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

result_path = Path(sys.argv[1]) / "partial.classify.json"
expected_anchors = int(sys.argv[2])
expected_tasks = int(sys.argv[3])
rows = json.loads(result_path.read_text(encoding="utf-8"))
keys = {
    (str(row.get("video_id")), str(row.get("qid")), float(row.get("alpha")))
    for row in rows
}
per_alpha = Counter(round(float(row.get("alpha")), 1) for row in rows)
if len(rows) != expected_tasks or len(keys) != expected_tasks:
    raise SystemExit(
        f"incomplete/duplicate partial result: expected={expected_tasks} "
        f"rows={len(rows)} unique_keys={len(keys)}"
    )
expected_alpha_count = expected_tasks // expected_anchors
if set(per_alpha.values()) != {expected_anchors} or len(per_alpha) != expected_alpha_count:
    raise SystemExit(f"unexpected alpha counts: {dict(sorted(per_alpha.items()))}")
if any(len(row.get("sampled_seconds", [])) != 32 for row in rows):
    raise SystemExit("at least one result does not record exactly 32 sampled frames")
print(f"FULL_PARTIAL_OK rows={len(rows)} per_alpha={dict(sorted(per_alpha.items()))}")
PY

"${ENV_DIR}/bin/python" scripts/eval/update_model_results.py \
    || echo "WARNING: failed to refresh MODEL_RESULTS.md" >&2
