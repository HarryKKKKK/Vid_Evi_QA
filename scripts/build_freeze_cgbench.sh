#!/bin/bash
#SBATCH --job-name=insuff_freeze
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/freeze_%A_%a.out
#SBATCH --error=logs/freeze_%A_%a.err

#NUM_SHARDS=8 sbatch --array=0-7%8 scripts/build_freeze_cgbench.sh
set -uo pipefail
mkdir -p logs
unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

# NUM_SHARDS 必须等于提交时 --array 的规模。
NUM_SHARDS=${NUM_SHARDS:-8}

# CPU 编码(libx264)，不用 GPU；jobs 4 × threads 4 = 16 = cpus-per-task。
# --freeze-source pre   : evidence 前一帧（默认）
# --freeze-source first : evidence 区间第一帧
python -u cgbench_pipeline/build_freeze.py \
    --num-shards "${NUM_SHARDS}" \
    --jobs 4 \
    --threads 4 \
    --freeze-source pre \
    --x264-preset veryfast \
    --crf 18

# --------------------------------------------------------------------------
# 提交（NUM_SHARDS 要和 --array 的 0-(N-1) 对齐）：
#   NUM_SHARDS=8 sbatch --array=0-7%8 run_freeze.sbatch
#
# 8 shard × 4 jobs = 32 路 CPU 并发。断点续跑：不加 --overwrite，已存在自动跳过。
# report 按 shard 写成 freeze_build_report.<shard>.jsonl
#
# 注意：纯 CPU，不申请 GPU（共享 DGX 别占卡）。若确实想用 nvenc 提速，
#   给命令加 --gpu，并把 --array 的 %N 调小（如 %2）避免一次占满整台卡。
#
# 单条 / 小批量测试：
#   python cgbench_pipeline/build_freeze.py --index 0 --freeze-source pre
#   python cgbench_pipeline/build_freeze.py --all --limit 5 --jobs 2
# --------------------------------------------------------------------------
