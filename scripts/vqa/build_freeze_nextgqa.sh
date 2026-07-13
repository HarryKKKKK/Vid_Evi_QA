#!/bin/bash
#SBATCH --job-name=nextgqa_freeze
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/nextgqa_freeze_%A_%a.out
#SBATCH --error=logs/nextgqa_freeze_%A_%a.err

#NUM_SHARDS=8 sbatch --array=0-7%8 scripts/vqa/build_freeze_nextgqa.sh
set -uo pipefail
mkdir -p logs
unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

# NUM_SHARDS 必须等于提交时 --array 的规模。
NUM_SHARDS=${NUM_SHARDS:-8}

# CPU 编码(libx264)，不用 GPU；jobs 4 × threads 4 = 16 = cpus-per-task。
# --json/--video-dir/--output-dir/--report 都指向 NExT-GQA 的路径，和
# build_freeze_cgbench.sh（跑 cgbench_filtered.json）完全分开，不会互相覆盖。
# --video-id-mapping 不用传，build_freeze.py 默认就指向 NExT-GQA 官方的
# map_vid_vidorID.json（只有 source_dataset=="nextgqa" 的条目会用到它）。
# --freeze-source pre   : evidence 前一帧（默认）
# --freeze-source first : evidence 区间第一帧
python -u scripts/vqa/build_freeze.py \
    --json nextgqa_pipeline/nextgqa_filtered.json \
    --video-dir source_datasets/next_gqa/videos \
    --output-dir source_datasets/next_gqa/freeze_videos \
    --report nextgqa_pipeline/nextgqa_freeze_build_report.jsonl \
    --num-shards "${NUM_SHARDS}" \
    --jobs 4 \
    --threads 4 \
    --freeze-source pre \
    --x264-preset veryfast \
    --crf 18

# --------------------------------------------------------------------------
# 提交前先做只读的 preflight（不生成任何视频，只检查 989 个视频能不能被
# 正确解析成路径；不用 sbatch，直接在登录节点跑，几秒钟出结果）：
#   python scripts/vqa/build_freeze.py \
#     --json nextgqa_pipeline/nextgqa_filtered.json \
#     --video-dir source_datasets/next_gqa/videos \
#     --output-dir source_datasets/next_gqa/freeze_videos \
#     --preflight
#
# 提交（NUM_SHARDS 要和 --array 的 0-(N-1) 对齐）：
#   NUM_SHARDS=8 sbatch --array=0-7%8 scripts/vqa/build_freeze_nextgqa.sh
#
# 8 shard × 4 jobs = 32 路 CPU 并发。断点续跑：不加 --overwrite，已存在自动跳过。
# report 按 shard 写成 nextgqa_pipeline/nextgqa_freeze_build_report.<shard>.jsonl
#
# 注意：纯 CPU，不申请 GPU（共享 DGX 别占卡）。若确实想用 nvenc 提速，
#   给命令加 --gpu，并把 --array 的 %N 调小（如 %2）避免一次占满整台卡。
#
# 单条 / 小批量测试（不走 sbatch，直接本机跑，先验证 ffmpeg 命令本身没问题）：
#   python scripts/vqa/build_freeze.py \
#     --json nextgqa_pipeline/nextgqa_filtered.json \
#     --video-dir source_datasets/next_gqa/videos \
#     --output-dir source_datasets/next_gqa/freeze_videos \
#     --all --limit 5 --jobs 2 --freeze-source pre
# --------------------------------------------------------------------------
