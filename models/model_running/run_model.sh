#!/bin/bash
#SBATCH --job-name=qwen3vl_serve
#SBATCH -p debug              
#SBATCH --gres=gpu:2                
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00            
#SBATCH --output=logs/serve_%j.out
#SBATCH --error=logs/serve_%j.err

set -uo pipefail

MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/Qwen3-VL"
PORT=8000
TP=2

export HF_HUB_OFFLINE=1

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
conda activate /aifs4su/hansirui_2nd/harry/envs

cd /aifs4su/hansirui_2nd/harry/Vid_Evi_QA

echo "=================================================="
echo "Service node hostname : $(hostname)"
echo "Service URL (in-job)  : http://localhost:${PORT}/v1"
echo "Service URL (external): http://$(hostname):${PORT}/v1"
echo "=================================================="
nvidia-smi

vllm serve "${MODEL_PATH}" \
    --served-model-name qwen3-vl \
    --tensor-parallel-size ${TP} \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 40, "video": 0}' \
    --mm-processor-kwargs '{"max_pixels": 200704}' \
    --port ${PORT} \
    --host 0.0.0.0