#!/bin/bash
#SBATCH --job-name=insuff_bench
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

# 跑帧冻结版的校验：sbatch run_test_freeze.sbatch
# 可用环境变量覆盖：
#   N=5 FREEZE_SOURCE=pre PICK_MULTI=1 sbatch run_test_freeze.sbatch
#   QIDS="5218 66 79"                 sbatch run_test_freeze.sbatch   # 测指定 qid
set -uo pipefail
mkdir -p logs
unset http_proxy; unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs
cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

# 测试用的本地输出目录（节点本地 /tmp，跑完即用即弃）。
OUT_DIR=${OUT_DIR:-/tmp/freeze_test_${SLURM_JOB_ID:-local}}

# 组装参数：默认前 5 条、pre 模式。
ARGS=( --out-dir "${OUT_DIR}" --freeze-source "${FREEZE_SOURCE:-pre}" )
if [[ -n "${QIDS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=( --qids ${QIDS} )
else
    ARGS+=( --n "${N:-5}" )
    [[ "${PICK_MULTI:-0}" == "1" ]] && ARGS+=( --pick-multi )
fi

echo "[RUN] test_freeze.py ${ARGS[*]}"
python scripts/tmp/test_freeze.py "${ARGS[@]}"

# 注：本测试调用 build_freeze.py 时用的是 CPU 编码(libx264)，不依赖 GPU；
# 头部 --gres=gpu:1 仅为与你给的模板一致，去掉也能跑。
# 完整四项校验需要 pillow/numpy；缺失时自动退化为 freezedetect 验冻结。
