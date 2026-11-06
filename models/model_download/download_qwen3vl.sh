#!/bin/bash
#SBATCH --job-name=download_qwen3vl
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/download_qwen3vl_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/download_qwen3vl_%j.err

unset http_proxy
unset https_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before downloading}"

python /aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/model_download/download_qwen3vl.py
