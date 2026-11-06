#!/bin/bash
set -euo pipefail

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
EXCLUSIONS="${ROOT}/reports/language_prior_screen/combined_5279/removal_candidates.json"
OUTPUT="${ROOT}/reports/language_prior_screen/combined_5279/retained_benchmark_statistics"

while [[ ! -s "${EXCLUSIONS}" ]]; do
    sleep 60
done

cd "${ROOT}"
exec "${PYTHON}" scripts/analysis/summarize_retained_benchmark.py \
    --exclusions "${EXCLUSIONS}" \
    --output-dir "${OUTPUT}"
