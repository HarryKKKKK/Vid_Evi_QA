#!/bin/bash
#SBATCH --job-name=insuff_black
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/black_%A_%a.out
#SBATCH --error=logs/black_%A_%a.err

#NUM_SHARDS=8 sbatch --array=0-7%8 scripts/build_insufficient_cgbench.sh
set -uo pipefail
mkdir -p logs
unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

# NUM_SHARDS 必须等于提交时 --array 的规模。
NUM_SHARDS=${NUM_SHARDS:-8}

# 脚本内部线程池并发：jobs 个 ffmpeg，每个 threads 线程，jobs*threads<=16。
python cgbench_pipeline/build_insufficient.py \
    --num-shards "${NUM_SHARDS}" \
    --jobs 4 \
    --threads 4 \
    --x264-preset veryfast \
    --crf 18

# --------------------------------------------------------------------------
# 提交（NUM_SHARDS 要和 --array 的 0-(N-1) 对齐）：
#   NUM_SHARDS=8 sbatch --array=0-7%8 run_blackout.sbatch
#
# 每个 array task = 1 个 shard，内部再开 4 个并发 ffmpeg。
# 8 shard × 4 jobs = 32 路并发。断点续跑：不加 --overwrite，已存在自动跳过。
# report 按 shard 写成 insufficient_build_report.<shard>.jsonl
# --------------------------------------------------------------------------
