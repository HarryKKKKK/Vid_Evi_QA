#!/bin/bash
#SBATCH --job-name=partial_c2
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/partial_%A_%a.out
#SBATCH --error=logs/partial_%A_%a.err

#NUM_SHARDS=8 sbatch --array=0-7%8 cgbench_pipeline/build_partial_cgbench.sh
set -uo pipefail
mkdir -p logs
unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

# NUM_SHARDS 必须等于提交时 --array 的规模。分片对象是 (candidate x alpha) 任务。
NUM_SHARDS=${NUM_SHARDS:-8}
MODE=${MODE:-noise}

# 纯 CPU (libx264)；jobs 4 x threads 4 = 16 = cpus-per-task。
python -u cgbench_pipeline/build_partial.py \
    --mode "${MODE}" \
    --num-shards "${NUM_SHARDS}" \
    --jobs 4 \
    --threads 4 \
    --x264-preset veryfast \
    --crf 18

# --------------------------------------------------------------------------
# 提交（NUM_SHARDS 要和 --array 的 0-(N-1) 对齐）：
#   NUM_SHARDS=8 sbatch --array=0-7%8 cgbench_pipeline/build_partial_cgbench.sh
#   MODE=blur NUM_SHARDS=8 sbatch --array=0-7%8 cgbench_pipeline/build_partial_cgbench.sh
#
# 8 shard x 4 jobs = 32 路并发，任务总数 = 245 candidates x 5 levels = 1225。
# 断点续跑：不加 --overwrite，已存在自动跳过。
# report 按 shard 写成 partial_build_report.<shard>.jsonl
# manifest 按 shard 写成 partial.manifest.<shard>.jsonl
#   (video_id, qid, question, answer, evidence_intervals, mode, alpha, condition, output)
#
# 单条 / 小批量本地测试（先 --dry-run 看 ffmpeg 命令，再去掉跑几条看效果）：
#   python cgbench_pipeline/build_partial.py --index 0 --dry-run
#   python cgbench_pipeline/build_partial.py --all --limit 3 --jobs 1
#   python cgbench_pipeline/build_partial.py --all --limit 3 --mode blur --jobs 1
#
# alpha -> U 映射 (eq. 4): 1.0=C1(不改), 0.75/0.5/0.25=C2, 0.0=C3。
# 强度公式 (eq. 8-9): noise alls = noise_max*(1-alpha); blur sigma = blur_max*(1-alpha)。
# 默认 noise_max=60 / blur_max=18，可用 --noise-max / --blur-max 调整档位强弱。
# --------------------------------------------------------------------------
