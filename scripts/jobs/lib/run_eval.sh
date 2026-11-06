#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo "Usage: $0 CONDITION MODEL_KIND MODEL_PATH SERVED_NAME MODEL_TAG RESULTS_DIR" >&2
    exit 2
fi
MODE=$1; MODEL_KIND=$2; MODEL_PATH=$3; SERVED_NAME=$4; MODEL_TAG=$5; RESULTS_DIR=$6
case "$MODE" in sufficient|insufficient|partial|merged) ;; *) echo "Unknown condition: $MODE" >&2; exit 2 ;; esac
if [[ "$MODEL_KIND" == qwen3 && "$MODE" != merged ]]; then
    echo "qwen3 is supported by the merged runner only" >&2; exit 2
fi
LIB_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${VID_EVI_ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
ENV_DIR=${VID_EVI_ENV:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl}
EVAL_CODE_DIR=${EVAL_CODE_DIR:-${ROOT}/scripts/eval}
FILTERED_JSON=${FILTERED_JSON:-results/build/nextgqa/nextgqa_filtered_no_evidence_leak_eval_clean.json}
if [[ "$MODE" == partial ]]; then FILTERED_JSON=${PARTIAL_FILTERED_JSON:-$FILTERED_JSON}; fi
VIDEO_DIR=${VIDEO_DIR:-source_datasets/next_gqa/videos}
FREEZE_DIR=${FREEZE_DIR:-source_datasets/next_gqa/freeze_videos_no_evidence_leak}
MANIFEST=${MERGED_MANIFEST:-results/reports/final_benchmark_aligned/final/all_condition_records.jsonl}
CANDIDATES=${PARTIAL_CANDIDATES:-results/build/nextgqa/partial_candidates_n500_seed20260819.json}
PARTIAL_DIR=${PARTIAL_VIDEO_DIR:-source_datasets/next_gqa/partial_videos_no_evidence_leak}
ALPHAS=${PARTIAL_ALPHAS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
EXPECTED_ANCHORS=${PARTIAL_EXPECTED_ANCHORS:-500}
EXPECTED_TASKS=${PARTIAL_EXPECTED_TASKS:-4500}
RUN_ID=${SLURM_JOB_ID:-$$}
PORT=${VLLM_PORT:-$((25000 + RANDOM % 1000))}
EVAL_WORKERS=${EVAL_WORKERS:-8}
EVAL_TASK=${EVAL_TASK:-classify}
QWEN_USE_FAST=${QWEN_USE_FAST:-true}
if [[ "$EVAL_TASK" != classify && ( "$EVAL_TASK" != qa || "$MODE" != sufficient ) ]]; then
    echo "Only sufficient supports EVAL_TASK=qa; other conditions use classify" >&2; exit 2
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 PYTHONNOUSERSITE=1
export CUDA_HOME=/usr/local/cuda VLLM_ATTENTION_BACKEND=TORCH_SDPA VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ALLREDUCE_USE_SYMM_MEM=0 PATH="${ENV_DIR}/bin:${PATH}"
MODEL_CONFIG="${LIB_DIR}/vllm_models.sh"
[[ -f "$MODEL_CONFIG" ]] || MODEL_CONFIG="${ROOT}/scripts/configs/vllm_models.sh"
source "$MODEL_CONFIG"
cd "$ROOT"
export TORCHINDUCTOR_CACHE_DIR="${ROOT}/results/.cache/torchinductor"
export TRITON_CACHE_DIR="${ROOT}/results/.cache/triton"
[[ -x "$ENV_DIR/bin/python" && -x "$ENV_DIR/bin/vllm" ]] || { echo "Missing Python/vLLM runtime: $ENV_DIR" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 2; }
if [[ "$MODE" == merged ]]; then
    "$ENV_DIR/bin/python" "$EVAL_CODE_DIR/check_v2_run.py" "$MANIFEST"
else
    [[ -s "$FILTERED_JSON" ]] || { echo "Missing filtered manifest: $FILTERED_JSON" >&2; exit 2; }
fi
if [[ "$MODE" == partial ]]; then
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

fi
mkdir -p results/logs "${RESULTS_DIR}"
LOG_FILE="results/logs/${MODE}_${MODEL_TAG}_${RUN_ID}.vllm.log"
source "${LIB_DIR}/vllm_service.sh"
case "$MODE" in
sufficient)
"${ENV_DIR}/bin/python" -u "${EVAL_CODE_DIR}/evaluate.py" \
    --mode sufficient \
    --task "${EVAL_TASK}" \
    --filtered-json "${FILTERED_JSON}" \
    --video-dir "${VIDEO_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --model-tag "${MODEL_TAG}" \
    --num-frames "${NUM_FRAMES}" \
    --media-mode "${MEDIA_MODE}" \
    --frame-width 560 \
    --sampling uniform \
    --workers "${EVAL_WORKERS}"

"${ENV_DIR}/bin/python" - "${FILTERED_JSON}" "${RESULTS_DIR}" "${EVAL_TASK}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sum(
    str(row.get("evidence_condition") or "").strip().lower() == "sufficient"
    for row in manifest
)
task = sys.argv[3]
result_path = Path(sys.argv[2]) / f"sufficient.{task}.json"
rows = json.loads(result_path.read_text(encoding="utf-8"))
if len(rows) != expected:
    raise SystemExit(
        f"incomplete sufficient result: expected={expected} actual={len(rows)}"
    )
if any(len(row.get("sampled_seconds", [])) != 32 for row in rows):
    raise SystemExit("at least one result does not record exactly 32 sampled frames")
print(f"FULL_SUFFICIENT_OK count={len(rows)} result={result_path}")
PY

"${ENV_DIR}/bin/python" "${EVAL_CODE_DIR}/update_model_results.py" \
    || echo "WARNING: failed to refresh results/reports/results/MODEL_RESULTS.md" >&2
;;
insufficient)
"${ENV_DIR}/bin/python" -u "${EVAL_CODE_DIR}/evaluate.py" \
    --mode insufficient \
    --task classify \
    --filtered-json "${FILTERED_JSON}" \
    --insufficient-video-dir "${FREEZE_DIR}" \
    --results-dir "${RESULTS_DIR}" \
    --model-tag "${MODEL_TAG}" \
    --num-frames "${NUM_FRAMES}" \
    --media-mode "${MEDIA_MODE}" \
    --frame-width 560 \
    --sampling uniform \
    --workers "${EVAL_WORKERS}"

"${ENV_DIR}/bin/python" - "${FILTERED_JSON}" "${RESULTS_DIR}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = sum(
    str(row.get("evidence_condition", "")).lower() == "sufficient"
    for row in manifest
)
result_path = Path(sys.argv[2]) / "insufficient.classify.json"
rows = json.loads(result_path.read_text(encoding="utf-8"))
keys = {(str(row.get("video_id")), str(row.get("qid"))) for row in rows}
if len(rows) != expected or len(keys) != expected:
    raise SystemExit(
        f"incomplete/duplicate insufficient result: expected={expected} "
        f"rows={len(rows)} unique_keys={len(keys)}"
    )
if any(len(row.get("sampled_seconds", [])) != 32 for row in rows):
    raise SystemExit("at least one result does not record exactly 32 sampled frames")
print(f"FULL_INSUFFICIENT_OK count={len(rows)} result={result_path}")
PY

"${ENV_DIR}/bin/python" "${EVAL_CODE_DIR}/update_model_results.py" \
    || echo "WARNING: failed to refresh results/MODEL_RESULTS.md" >&2
;;
partial)
"${ENV_DIR}/bin/python" -u "${EVAL_CODE_DIR}/evaluate.py" \
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

"${ENV_DIR}/bin/python" "${EVAL_CODE_DIR}/update_model_results.py" \
    || echo "WARNING: failed to refresh results/MODEL_RESULTS.md" >&2
;;
merged)
"${ENV_DIR}/bin/python" -u "${EVAL_CODE_DIR}/evaluate.py" \
    --manifest-jsonl "${MANIFEST}" --task classify \
    --video-dir "${VIDEO_DIR}" \
    --results-dir "${RESULTS_DIR}" --model-tag "${MODEL_TAG}" \
    --num-frames "${NUM_FRAMES}" --media-mode "${MEDIA_MODE}" \
    --frame-width 560 --sampling uniform --workers "${EVAL_WORKERS}"

"${ENV_DIR}/bin/python" "${EVAL_CODE_DIR}/check_v2_run.py" "${MANIFEST}" \
    --result "${RESULTS_DIR}/merged.classify.json"
;;
esac
