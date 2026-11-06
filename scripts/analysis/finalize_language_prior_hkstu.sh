#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 PID [PID ...]" >&2
    exit 2
fi

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python

while true; do
    alive=0
    for pid in "$@"; do
        if kill -0 "${pid}" 2>/dev/null; then
            alive=1
            break
        fi
    done
    if [[ "${alive}" -eq 0 ]]; then
        break
    fi
    sleep 60
done

cd "${ROOT}"

# The first pass records unparsable blind refusals as retryable errors. Rerun
# after all workers exit with the current normalization logic; successful rows
# are skipped and only failed rows are regenerated/classified.
"${PYTHON}" scripts/analysis/screen_language_priors_offline.py \
    --backend qwen \
    --model-path /aifs4su/hansirui_2nd/harry/models/Qwen3-8B \
    --model-name qwen3-8b-text-prior \
    --permutations 5 \
    --batch-size 8 \
    --max-new-tokens 64 \
    --threads 64 \
    --output-dir reports/language_prior_screen/full_5279_qwen_cpu

"${PYTHON}" scripts/analysis/screen_language_priors_offline.py \
    --backend llama \
    --model-path /aifs4su/hansirui_2nd/harry/models/Meta-Llama-3.1-8B-Instruct-Nous \
    --model-name llama31-8b-text-prior \
    --permutations 5 \
    --batch-size 8 \
    --max-new-tokens 64 \
    --threads 64 \
    --output-dir reports/language_prior_screen/full_5279_llama_cpu

"${PYTHON}" scripts/analysis/merge_language_prior_runs.py \
    --input reports/language_prior_screen/full_5279/raw_predictions.jsonl \
    --input reports/language_prior_screen/full_5279_qwen_cpu/raw_predictions.jsonl \
    --input reports/language_prior_screen/full_5279_llama_cpu/raw_predictions.jsonl \
    --output-dir reports/language_prior_screen/combined_5279

exec "${PYTHON}" scripts/analysis/summarize_retained_benchmark.py \
    --exclusions reports/language_prior_screen/combined_5279/removal_candidates.json \
    --output-dir reports/language_prior_screen/combined_5279/retained_benchmark_statistics
