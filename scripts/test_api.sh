#!/bin/bash
#SBATCH --job-name=test_api
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/mask_%A_%a.out
#SBATCH --error=logs/mask_%A_%a.err

set -euo pipefail
mkdir -p logs

export XHUB_API_KEY="sk-XfZIzSKfM44gFDcMk6IZNgLSixUgSHTFtXyNROsq7bKYRNMu"

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
unset all_proxy

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

echo "HOSTNAME=$(hostname)"
echo "Testing network..."
curl -v https://api3.xhub.chat/v1/models || true
curl -v https://www.google.com || tr

python scripts/test_api.py