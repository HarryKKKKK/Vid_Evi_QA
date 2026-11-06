#!/usr/bin/env bash
# Wait for sharded claimed-evidence grounding jobs, then run the semantic judge
# and write a combined summary. The script is safe to restart: both Python
# stages resume from their JSONL outputs.

set -u

ROOT=${ROOT:-/aifs4su/hansirui_2nd/harry/Vid_Evi_QA}
PYTHON=${PYTHON:-/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python}
NUM_SHARDS=${NUM_SHARDS:-4}
JUDGE_ENDPOINT=${JUDGE_ENDPOINT:-qwen35_9b_judge,http://127.0.0.1:18007/v1,Qwen3.5-9B}
GROUNDING_GLOB=${GROUNDING_GLOB:-results/build/nextgqa/insufficient/correct_answer_evidence_groundings*.jsonl}
CLASSIFICATION_OUTPUT=${CLASSIFICATION_OUTPUT:-results/build/nextgqa/insufficient/correct_answer_evidence_classifications.jsonl}
SUMMARY_OUTPUT=${SUMMARY_OUTPUT:-results/build/nextgqa/insufficient/correct_answer_evidence_summary.json}

cd "${ROOT}"
mkdir -p results/logs

for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    pid_file="results/logs/correct_answer_grounding.${shard}.pid"
    if [[ ! -f "${pid_file}" ]]; then
        echo "[FATAL] missing grounding pid file: ${pid_file}" >&2
        exit 1
    fi
    pid=$(<"${pid_file}")
    while kill -0 "${pid}" 2>/dev/null; do
        sleep 30
    done
done

echo "[GROUNDING COMPLETE] launching ${NUM_SHARDS} judge shards"
pids=()
for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    "${PYTHON}" -u pipelines/nextgqa/insufficient/classify_groundings.py \
        --groundings "${GROUNDING_GLOB}" \
        --output "${CLASSIFICATION_OUTPUT}" \
        --endpoint "${JUDGE_ENDPOINT}" \
        --disable-thinking \
        --max-tokens 500 \
        --num-shards "${NUM_SHARDS}" \
        --shard "${shard}" \
        >"results/logs/correct_answer_judge.${shard}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
done

"${PYTHON}" pipelines/nextgqa/insufficient/summarize_correct_answer_evidence.py \
    --classifications "results/build/nextgqa/insufficient/correct_answer_evidence_classifications*.jsonl" \
    --output "${SUMMARY_OUTPUT}" \
    >results/logs/correct_answer_evidence_summary.log 2>&1

echo "[DONE] judge_failed=${failed} summary=${SUMMARY_OUTPUT}"
exit "${failed}"
