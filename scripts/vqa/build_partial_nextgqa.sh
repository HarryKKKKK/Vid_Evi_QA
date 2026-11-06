#!/bin/bash
#SBATCH --job-name=nextgqa_partial
#SBATCH -p debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/nextgqa_partial_%A_%a.out
#SBATCH --error=logs/nextgqa_partial_%A_%a.err

#NUM_SHARDS=8 sbatch --array=0-7%8 scripts/vqa/build_partial_nextgqa.sh
set -uo pipefail
mkdir -p logs
unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

# --------------------------------------------------------------------------
# PREREQUISITE — do not submit this until both of these exist:
#   nextgqa_result/sufficient.qa.json
#   nextgqa_result/insufficient.qa.json
# then generate the NExT-GQA candidate list (sufficient answered correctly,
# insufficient answered wrong) with:
#   python scripts/eval/compare_suff_insuff.py \
#     --sufficient nextgqa_result/sufficient.qa.json \
#     --insufficient nextgqa_result/insufficient.qa.json \
#     --out nextgqa_result/suff_correct_insuff_wrong.json
# Without that file, --candidates below joins to zero entries against
# --meta (different video_id namespace than the CG-Bench default) and this
# job will just do nothing for its full 10-hour walltime, not error out.
# --------------------------------------------------------------------------

# NUM_SHARDS 必须等于提交时 --array 的规模。
NUM_SHARDS=${NUM_SHARDS:-8}

# CPU 编码(libx264)，不用 GPU；jobs 4 × threads 4 = 16 = cpus-per-task。
# --candidates/--meta/--video-dir/--output-dir/--manifest/--report 都指向
# NExT-GQA 的路径，和 build_partial_cgbench.sh（跑 cgbench_filtered.json）
# 完全分开，不会互相覆盖。--video-id-mapping 不用传，build_partial.py 默认
# 就指向 NExT-GQA 官方的 map_vid_vidorID.json（只有 source_dataset=="nextgqa"
# 的条目会用到它）。
python -u scripts/vqa/build_partial.py \
    --candidates nextgqa_result/suff_correct_insuff_wrong.json \
    --meta nextgqa_pipeline/nextgqa_filtered.json \
    --video-dir source_datasets/next_gqa/videos \
    --output-dir source_datasets/next_gqa/partial_videos \
    --manifest nextgqa_pipeline/nextgqa_partial_manifest.jsonl \
    --report nextgqa_pipeline/nextgqa_partial_build_report.jsonl \
    --num-shards "${NUM_SHARDS}" \
    --jobs 4 \
    --threads 4 \
    --mode noise \
    --levels 0.0,0.05,0.1,0.15,0.2,0.3,0.5 \
    --x264-preset veryfast \
    --crf 18

# --------------------------------------------------------------------------
# 提交前先做只读检查：
#
# 1) 有多少候选（sufficient 对、insufficient 错的 (video_id, qid) 对）：
#   python scripts/vqa/build_partial.py \
#     --candidates nextgqa_result/suff_correct_insuff_wrong.json \
#     --meta nextgqa_pipeline/nextgqa_filtered.json \
#     --print-count
#
# 2) 这些候选对应的视频能不能被正确解析成路径（不生成任何视频）：
#   python scripts/vqa/build_partial.py \
#     --candidates nextgqa_result/suff_correct_insuff_wrong.json \
#     --meta nextgqa_pipeline/nextgqa_filtered.json \
#     --video-dir source_datasets/next_gqa/videos \
#     --output-dir source_datasets/next_gqa/partial_videos \
#     --preflight
#
# 提交（NUM_SHARDS 要和 --array 的 0-(N-1) 对齐）：
#   NUM_SHARDS=8 sbatch --array=0-7%8 scripts/vqa/build_partial_nextgqa.sh
#
# 8 shard × 4 jobs = 32 路 CPU 并发。断点续跑：不加 --overwrite，已存在自动跳过。
# report 按 shard 写成 nextgqa_pipeline/nextgqa_partial_build_report.<shard>.jsonl
# manifest 按 shard 写成 nextgqa_pipeline/nextgqa_partial_manifest.<shard>.jsonl
#
# 注意：纯 CPU，不申请 GPU（共享 DGX 别占卡）。若确实想用 nvenc 提速，
#   给命令加 --gpu，并把 --array 的 %N 调小（如 %2）避免一次占满整台卡。
#
# 单条 / 小批量测试（不走 sbatch，直接本机跑）：
#   python scripts/vqa/build_partial.py \
#     --candidates nextgqa_result/suff_correct_insuff_wrong.json \
#     --meta nextgqa_pipeline/nextgqa_filtered.json \
#     --video-dir source_datasets/next_gqa/videos \
#     --output-dir source_datasets/next_gqa/partial_videos \
#     --index 0 --mode noise --levels 0.0,0.5
#   python scripts/vqa/build_partial.py \
#     --candidates nextgqa_result/suff_correct_insuff_wrong.json \
#     --meta nextgqa_pipeline/nextgqa_filtered.json \
#     --video-dir source_datasets/next_gqa/videos \
#     --output-dir source_datasets/next_gqa/partial_videos \
#     --all --limit 5 --jobs 2 --mode noise
# --------------------------------------------------------------------------
