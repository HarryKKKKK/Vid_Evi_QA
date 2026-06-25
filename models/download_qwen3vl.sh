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

source /aifs4su/hansirui_2nd/harry/envs/bin/activate

export HF_TOKEN=

python /aifs4su/hansirui_2nd/harry/Vid_Evi_QA/download_qwen3vl.py
