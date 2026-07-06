#!/bin/bash
#SBATCH --job-name=freeze_verify
#SBATCH -p debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/verify_%j.out
#SBATCH --error=logs/verify_%j.err

set -uo pipefail
mkdir -p logs

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

TARGET_DIR="source_datasets/cg_bench/freeze_videos"
OK_LIST="freeze_verify_ok.txt"
BROKEN_LIST="freeze_verify_broken.txt"

: > "${OK_LIST}"
: > "${BROKEN_LIST}"

TOTAL=$(find "${TARGET_DIR}" -maxdepth 1 -type f -name "*_freeze.mp4" | wc -l)
echo "[START] target_dir=${TARGET_DIR} total_files=${TOTAL}"

COUNT=0

find "${TARGET_DIR}" -maxdepth 1 -type f -name "*_freeze.mp4" | while read -r f; do
    COUNT=$((COUNT + 1))

    ERR_MSG=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${f}" 2>&1 >/dev/null)
    STATUS=$?

    if [ "${STATUS}" -eq 0 ] && [ -z "${ERR_MSG}" ]; then
        echo "${f}" >> "${OK_LIST}"
        echo "[${COUNT}/${TOTAL}] OK    ${f}"
    else
        echo "${f}|${ERR_MSG}" >> "${BROKEN_LIST}"
        echo "[${COUNT}/${TOTAL}] BROKEN ${f} :: ${ERR_MSG}"
    fi
done

OK_N=$(wc -l < "${OK_LIST}")
BROKEN_N=$(wc -l < "${BROKEN_LIST}")

echo "[DONE] ok=${OK_N} broken=${BROKEN_N}"
echo "[DONE] ok list     -> ${OK_LIST}"
echo "[DONE] broken list -> ${BROKEN_LIST}"