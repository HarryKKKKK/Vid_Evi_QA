#!/bin/bash
#SBATCH --job-name=partial_noise
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/partial_%A_%a.out
#SBATCH --error=logs/partial_%A_%a.err

#NUM_SHARDS=8 sbatch --array=0-7%8 scripts/vqa/build_partial_cgbench.sh
set -uo pipefail
mkdir -p logs
unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

# NUM_SHARDS 必须等于提交时 --array 的规模。
NUM_SHARDS=${NUM_SHARDS:-8}

# CPU 编码(libx264)，不用 GPU；jobs 4 × threads 4 = 16 = cpus-per-task。
# python -u scripts/vqa/build_partial.py \
#     --num-shards "${NUM_SHARDS}" \
#     --jobs 4 \
#     --threads 4 \
#     --mode noise \
#     --levels 0.0,0.05,0.1,0.15,0.2,0.3,0.5 \
#     --x264-preset veryfast \
#     --crf 18

python -u scripts/vqa/build_partial.py \
    --num-shards "${NUM_SHARDS}" \
    --jobs 4 \
    --threads 4 \
    --mode noise \
    --levels 0.6,0.7,0.8,0.9,1.0 \
    --x264-preset veryfast \
    --crf 18

# --------------------------------------------------------------------------
# 提交（NUM_SHARDS 要和 --array 的 0-(N-1) 对齐）：
#   NUM_SHARDS=8 sbatch --array=0-7%8 scripts/vqa/build_partial_cgbench.sh
#
# 8 shard × 4 jobs = 32 路 CPU 并发。断点续跑：不加 --overwrite，已存在自动跳过。
# report 按 shard 写成 partial_build_report.<shard>.jsonl
# manifest 按 shard 写成 partial.manifest.<shard>.jsonl
#
# 注意：纯 CPU，不申请 GPU（共享 DGX 别占卡）。若确实想用 nvenc 提速，
#   给命令加 --gpu，并把 --array 的 %N 调小（如 %2）避免一次占满整台卡。
#
# 单条 / 小批量测试（不走 sbatch，直接本机跑）：
#   python scripts/vqa/build_partial.py --index 0 --mode noise --levels 0.0,0.5
#   python scripts/vqa/build_partial.py --all --limit 5 --jobs 2 --mode noise
# --------------------------------------------------------------------------
