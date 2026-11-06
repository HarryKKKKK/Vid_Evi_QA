#!/bin/bash
#SBATCH --job-name=eval-v2
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/eval_v2_%j.out
#SBATCH --error=logs/eval_v2_%j.err
set -euo pipefail
ROOT=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA
cd "${ROOT}"
RELEASE=$1
KIND=$2
case "${KIND}" in
  qwen25) MODEL=Qwen2.5-VL-32B-Instruct; SERVED=qwen2.5-vl-32b; TAG=qwen25vl32b_merged_v2 ;;
  qwen3) MODEL=Qwen3-VL; SERVED=qwen3-vl; TAG=qwen3vl_merged_v2 ;;
  internvl) MODEL=InternVL3_5-8B; SERVED=internvl3.5-8b; TAG=internvl35_merged_v2 ;;
  llava) MODEL=LLaVA-OneVision-Qwen2-7B-OV; SERVED=llava-onevision-qwen2-7b-ov; TAG=llava_onevision_merged_v2 ;;
  *) echo "Unknown model kind: ${KIND}" >&2; exit 2 ;;
esac
/aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python "${RELEASE}/code/manage_v2_release.py" verify "${RELEASE}"
export EVAL_CODE_DIR="${RELEASE}/code"
export MERGED_MANIFEST="${RELEASE}/all_condition_records.jsonl"
export EVAL_WORKERS=8
bash "${EVAL_CODE_DIR}/run_merged_model.sh" "${KIND}" "${ROOT}/models/${MODEL}" "${SERVED}" "${TAG}" \
  "${RELEASE}/results/${TAG}"
