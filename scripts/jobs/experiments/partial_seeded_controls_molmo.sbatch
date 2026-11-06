#!/bin/bash
#SBATCH --job-name=partial-molmo
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=results/logs/partial_molmo_%j.out
#SBATCH --error=results/logs/partial_molmo_%j.err

set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
PYTHON=${ROOT}/.runtime/molmo2_transformers4571/bin/python
RUN_MODE=${RUN_MODE:-selective}
LIMIT=${LIMIT:-100}
RUN_TAG=${RUN_TAG:-pilot_v1}
SAMPLE=${SAMPLE:-results/ablation_results/qwen25_conservatism/sample_n300.json}
OUTPUT=${ROOT}/results/partial_seeded_controls/${RUN_TAG}/${RUN_MODE}_molmo2_o_7b.jsonl

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_HOME=/usr/local/cuda
cd "${ROOT}"
mkdir -p results/logs "$(dirname "${OUTPUT}")"
"${PYTHON}" -u scripts/analysis/partial_seeded_control_molmo.py \
  --sample "${SAMPLE}" --model-path "${ROOT}/models/Molmo2-O-7B" \
  --model-tag molmo2_o_7b --output "${OUTPUT}" --protocol "${RUN_MODE}" \
  --limit "${LIMIT}" --ffmpeg /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/ffmpeg
