#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {gpt|qwen|llama|aggregate}" >&2
    exit 2
fi

ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python
OUTPUT_DIR=${LANGUAGE_PRIOR_OUTPUT_DIR:-${ROOT}/reports/language_prior_screen/full_5279}
PERMUTATIONS=${LANGUAGE_PRIOR_PERMUTATIONS:-5}
WORKERS=${LANGUAGE_PRIOR_WORKERS:-8}
MODE=$1

cd "${ROOT}"

case "${MODE}" in
    gpt)
        KEY_FILE=${XHUB_KEY_FILE:-${HOME}/.config/vid_evi_qa/xhub_api_key}
        if [[ ! -s "${KEY_FILE}" ]]; then
            echo "Missing protected API key file: ${KEY_FILE}" >&2
            exit 1
        fi
        OPENAI_API_KEY=
        # A protected key file need not end in a newline; read still assigns the
        # value but returns non-zero at EOF, so tolerate that status explicitly.
        IFS= read -r OPENAI_API_KEY < "${KEY_FILE}" || true
        if [[ -z "${OPENAI_API_KEY}" ]]; then
            echo "Protected API key file is empty: ${KEY_FILE}" >&2
            exit 1
        fi
        export OPENAI_API_KEY
        export OPENAI_BASE_URL=${OPENAI_BASE_URL:-https://api3.xhub.chat/v1}
        export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
        export GPT_API_MODE=chat
        exec "${PYTHON}" scripts/analysis/screen_language_priors.py \
            --backends gpt \
            --permutations "${PERMUTATIONS}" \
            --workers "${WORKERS}" \
            --output-dir "${OUTPUT_DIR}"
        ;;
    qwen)
        export QWEN_BASE_URL=${QWEN_BASE_URL:-http://127.0.0.1:18101/v1}
        export QWEN_MODEL=${QWEN_MODEL:-qwen3-8b-text-prior}
        exec "${PYTHON}" scripts/analysis/screen_language_priors.py \
            --backends qwen \
            --permutations "${PERMUTATIONS}" \
            --workers "${WORKERS}" \
            --output-dir "${OUTPUT_DIR}"
        ;;
    llama)
        export LLAMA_BASE_URL=${LLAMA_BASE_URL:-http://127.0.0.1:18102/v1}
        export LLAMA_MODEL=${LLAMA_MODEL:-llama31-8b-text-prior}
        exec "${PYTHON}" scripts/analysis/screen_language_priors.py \
            --backends llama \
            --permutations "${PERMUTATIONS}" \
            --workers "${WORKERS}" \
            --output-dir "${OUTPUT_DIR}"
        ;;
    aggregate)
        export QWEN_MODEL=${QWEN_MODEL:-qwen3-8b-text-prior}
        export LLAMA_MODEL=${LLAMA_MODEL:-llama31-8b-text-prior}
        export OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o}
        exec "${PYTHON}" scripts/analysis/screen_language_priors.py \
            --aggregate-only \
            --backends qwen,llama,gpt \
            --permutations "${PERMUTATIONS}" \
            --output-dir "${OUTPUT_DIR}"
        ;;
    *)
        echo "Unknown mode: ${MODE}" >&2
        exit 2
        ;;
esac
