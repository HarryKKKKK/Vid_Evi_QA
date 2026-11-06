#!/bin/bash
#SBATCH --partition=llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/v22_small_%A_%a.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/v22_small_%A_%a.err
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 RELEASE MODEL_KIND" >&2
    exit 2
fi
RELEASE=$1
KIND=$2
SHARD=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
export V2_MANIFEST="${RELEASE}/shards/${SHARD}.jsonl"
export V2_RESULT_DIR="${RELEASE}/results/${KIND}/shard_${SHARD}"
export V2_MODEL_TAG="${KIND}_full_v2_2"
if [[ "${KIND}" == "molmo2_o_7b" ]]; then
    export MOLMO_JSON_CONSTRAINT=1
    export V2_WORKERS=1
else
    export V2_WORKERS=4
fi
exec /aifs4su/hansirui_2nd/harry/envs_qwen3vl/bin/python -u \
    "${RELEASE}/adapters/run_smoke.py" "${RELEASE}" "${KIND}"
