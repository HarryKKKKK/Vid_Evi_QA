#!/bin/bash
#SBATCH --job-name=pilot_frames
#SBATCH -p debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/frames_%j.out
#SBATCH --error=logs/frames_%j.err

set -euo pipefail
mkdir -p logs

unset http_proxy
unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

# —— 先把"有没有 ffmpeg"变成日志事实,没有就直接失败 ——
echo "ffmpeg path: $(which ffmpeg || echo NOT_FOUND)"
which ffmpeg >/dev/null || { echo "FATAL: ffmpeg not in this conda env"; exit 1; }
ffmpeg -version | head -1

V="source_datasets/cg_bench/videos/BV1yG411C7Xi.mp4"
test -f "$V" || { echo "FATAL: video file not found: $V"; exit 1; }

OUT="frames_BV1yG411C7Xi"
mkdir -p "$OUT"

# 证据段1: 370–390s,2fps;证据段2: 395–410s,2fps
ffmpeg -nostdin -y -ss 370 -i "$V" -t 20 -vf fps=2 -q:v 3 "$OUT/e1_%03d.jpg"
ffmpeg -nostdin -y -ss 395 -i "$V" -t 15 -vf fps=2 -q:v 3 "$OUT/e2_%03d.jpg"

echo "frame count: $(ls "$OUT" | wc -l)"
ls -la "$OUT" | head