#!/bin/bash
#SBATCH --job-name=insuff_mask
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mask_%A_%a.out
#SBATCH --error=logs/mask_%A_%a.err

set -euo pipefail
mkdir -p logs

unset http_proxy
unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

python cgbench_pipeline/build_insufficient.py --gpu