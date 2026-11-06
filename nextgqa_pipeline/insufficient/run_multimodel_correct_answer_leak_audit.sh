#!/usr/bin/env bash
# Reuse the claimed-evidence grounding and semantic leak classifier for all
# insufficient-condition downstream model runs. Safe to restart: each stage
# uses latest-record semantics and skips completed keys.

set -euo pipefail

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
PYTHON=${PYTHON:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python}
FFMPEG=${FFMPEG:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/ffmpeg}
FFPROBE=${FFPROBE:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/ffprobe}
NUM_SHARDS=${NUM_SHARDS:-8}
GROUND_ENDPOINT=${GROUND_ENDPOINT:-qwen35_grounder,http://127.0.0.1:8001/v1,/aifs4su/models/Qwen3.5-35B-A3B-FP8}
JUDGE_ENDPOINT=${JUDGE_ENDPOINT:-qwen35_judge,http://127.0.0.1:8002/v1,/aifs4su/models/Qwen3.5-35B-A3B-FP8}
PREFIX=${PREFIX:-nextgqa_pipeline/insufficient/multimodel_correct_answer}
STAGE=${STAGE:-all}
LEGACY_QWEN3_CLASSIFICATIONS=${LEGACY_QWEN3_CLASSIFICATIONS:-nextgqa_pipeline/insufficient/correct_answer_evidence_classifications8.*.jsonl}

cd "${ROOT}"

RESULTS=(
  nextgqa_result_qwen3vl_no_evidence_leak/insufficient.classify.json
  nextgqa_result_qwen25_32f_insufficient/insufficient.classify.json
  nextgqa_result_llava_onevision_32f_video_insufficient/insufficient.classify.json
  nextgqa_result_internvl35_32f_patch3_insufficient/insufficient.classify.json
)

result_args=()
for result in "${RESULTS[@]}"; do
  result_args+=(--results "${result}")
done

# Qwen3-VL already has a complete 2,320-record semantic audit in the legacy
# files above. Ground only the three outstanding models by default.
pending_result_args=(
  --results "${RESULTS[1]}"
  --results "${RESULTS[2]}"
  --results "${RESULTS[3]}"
)

run_parallel_stage() {
  local command=$1
  shift
  local pids=()
  local failed=0
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    "${PYTHON}" -u "${command}" "$@" --num-shards "${NUM_SHARDS}" --shard "${shard}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if ((failed)); then
    echo "[FATAL] one or more shards failed in ${command}" >&2
    return 1
  fi
}

if [[ "${STAGE}" == preflight || "${STAGE}" == all ]]; then
  "${PYTHON}" nextgqa_pipeline/insufficient/ground_correct_answer_evidence.py \
    --preflight "${result_args[@]}"
fi

if [[ "${STAGE}" == ground || "${STAGE}" == all ]]; then
  run_parallel_stage nextgqa_pipeline/insufficient/ground_correct_answer_evidence.py \
    "${pending_result_args[@]}" \
    --output "${PREFIX}_groundings.jsonl" \
    --endpoint "${GROUND_ENDPOINT}" \
    --ffmpeg "${FFMPEG}" --ffprobe "${FFPROBE}"
fi

if [[ "${STAGE}" == classify || "${STAGE}" == all ]]; then
  run_parallel_stage nextgqa_pipeline/insufficient/classify_groundings.py \
    --groundings "${PREFIX}_groundings.*.jsonl" \
    --output "${PREFIX}_classifications.jsonl" \
    --endpoint "${JUDGE_ENDPOINT}" --disable-thinking --max-tokens 500
fi

if [[ "${STAGE}" == summarize || "${STAGE}" == all ]]; then
  "${PYTHON}" nextgqa_pipeline/insufficient/summarize_correct_answer_evidence.py \
    --classifications "${PREFIX}_classifications.*.jsonl" \
    --output "${PREFIX}_semantic_summary.json"
  "${PYTHON}" scripts/analysis/export_insufficient_correct_claims.py \
    --result insufficient:Qwen3-VL-32B="${RESULTS[0]}" \
    --result insufficient:Qwen2.5-VL-32B="${RESULTS[1]}" \
    --result insufficient:LLaVA-OneVision-7B="${RESULTS[2]}" \
    --result insufficient:InternVL3.5-8B="${RESULTS[3]}" \
    --classifications "${LEGACY_QWEN3_CLASSIFICATIONS}" \
    --classifications "${PREFIX}_classifications.*.jsonl" \
    --output-jsonl "${PREFIX}_claims.jsonl" \
    --output-report "${PREFIX}_report.json"
fi
