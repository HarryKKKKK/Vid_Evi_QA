#!/bin/bash
#SBATCH --job-name=insuff_freeze
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/freeze_%A_%a.out
#SBATCH --error=logs/freeze_%A_%a.err

#NUM_SHARDS=8 sbatch --array=0-7%8 scripts/vqa/build_freeze_cgbench.sh
set -uo pipefail
mkdir -p logs
unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

python scripts/eval/check_video_frame_extraction.py \
  --mode insufficient \
  --insufficient-video-dir source_datasets/next_gqa/freeze_videos \
  --full-scan