#!/bin/bash
#SBATCH --job-name=download_internvl3_5_8b
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/download_internvl3_5_8b_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/download_internvl3_5_8b_%j.err

unset http_proxy
unset https_proxy
unset ftp_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset FTP_PROXY
unset ALL_PROXY
unset no_proxy
unset NO_PROXY

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs

export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before downloading}"

python /aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/model_download/download_internvl3_5.py