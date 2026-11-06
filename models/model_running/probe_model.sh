#!/bin/bash
#SBATCH --job-name=probe_internvl3_5
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/probe_internvl3_5_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/probe_internvl3_5_%j.err
#
# 用法:
#   sbatch probe_internvl3_5.sbatch.sh
#   sbatch probe_internvl3_5.sbatch.sh 8192 1 130          # 传参改 MAX_MODEL_LEN/TP/LIMIT_IMAGES
#   sbatch probe_internvl3_5.sbatch.sh 16384 1 130 /path/到/真实抽帧.jpg
#
# 位置参数(全部可选,不传则用下面的默认起点值):
#   $1 = MAX_MODEL_LEN   (默认: 16384)
#   $2 = TP               (默认: 1)
#   $3 = LIMIT_IMAGES     (默认: 130)
#   $4 = IMAGE_PATH       (默认: 空,health check 会用占位图,只能验证 server 存活,
#                           不能反映真实抽帧场景 -- 强烈建议传一张真实图片)
#
# 这几个默认值只是探测起点,不是我验证过的推荐值 -- 如果启动报错(OOM 等),
# 说明这组值不合适,改参数重新 sbatch 提交,不代表脚本本身有问题。
#
# 结果需要你自己去 logs/probe_internvl3_5_<jobid>.out 里读:
#   - vLLM 是否正常启动(还是 OOM / 报错退出)
#   - health check 打出来的 logprobs / top_logprobs 字段是否符合预期
#   - 如果做了下面的批量图片压测,vLLM 报错信息里给出的 token 数上限提示

set -uo pipefail

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

MAX_MODEL_LEN="${1:-16384}"
TP="${2:-1}"
LIMIT_IMAGES="${3:-130}"
IMAGE_PATH="${4:-}"

PORT=8000
MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/InternVL3_5-8B"
SERVED_NAME="internvl3_5-8b"
WORKDIR="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA"
HEALTH_CHECK_SCRIPT="${WORKDIR}/models/model_running/probe_inter.py"   # TODO(需确认): 实际存放路径

echo "===== 本次探测参数 ====="
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}  TP=${TP}  LIMIT_IMAGES=${LIMIT_IMAGES}"
echo "IMAGE_PATH=${IMAGE_PATH:-<未提供,将用占位图>}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "========================"

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
# 切到 envs_qwen3vl: 已通过实际报错确认这个环境的 vLLM 0.11.0 认识
# InternVLChatModel 架构("Resolved architecture: InternVLChatModel" 出现在
# 上一次的日志里),而 envs 里的 vLLM 0.7.0 连 CLI 参数格式都对不上,
# 大概率也不支持这个架构。
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl

# 强制用 TORCH_SDPA 而不是默认的 FlashAttention 后端。
# 依据: 上一次报错("the provided PTX was compiled with an unsupported
# toolchain")定位在视觉编码器(InternViT)调用 flash_attn_varlen_func 时,
# 日志里也确认了 "Using Flash Attention backend on V1 engine"。你原有的
# Qwen3-VL 脚本一直都设了这两个变量并且跑通过,这里补回来是修正我之前
# 遗漏的地方,不是新的未验证尝试。
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_MM_ATTENTION_BACKEND=TORCH_SDPA

cd "${WORKDIR}"

echo "Job on node: $(hostname)"
echo "===== 启动前 GPU 状态 ====="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

ORPHAN_PROC=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)
if [[ -n "$ORPHAN_PROC" ]]; then
    echo "[FATAL] 检测到 GPU 上已有进程占用,可能是孤儿进程,脚本中止:" >&2
    echo "$ORPHAN_PROC" >&2
    echo "        请联系集群管理员或确认后手动处理,不要在这种状态下继续探测。" >&2
    exit 1
fi

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "[FATAL] 端口 ${PORT} 已被占用,脚本中止。" >&2
    exit 1
fi

# ---- 启动 vLLM server(后台,日志走 sbatch 的 stdout,即本文件顶部的 --output) ----
echo ""
echo "===== 启动 vLLM server (InternVL3_5-8B) ====="

vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt "{\"image\": ${LIMIT_IMAGES}, \"video\": 0}" \
    --enforce-eager \
    --port "${PORT}" \
    --host 0.0.0.0 &

SERVER_PID=$!

cleanup() {
    echo ""
    echo "===== 停止 vLLM server (pid=${SERVER_PID}) ====="
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM to be ready (最多等 10 分钟)..."
READY=0

for i in $(seq 1 60); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[FATAL] vLLM 进程在启动过程中退出,很可能是这组参数不合适(比如 OOM)。" >&2
        echo "        请查看本文件顶部同名的 .err 日志获取完整报错。" >&2
        exit 1
    fi

    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "${SERVED_NAME}"; then
        echo "vLLM ready after ~$((i * 10))s"
        READY=1
        break
    fi

    sleep 10
done

if [[ "${READY}" -ne 1 ]]; then
    echo "[FATAL] 10 分钟内未就绪,中止。" >&2
    exit 1
fi

# ---- 自动跑一次基础 health check(单图 + 双图) ----
echo ""
echo "===== 运行 health check(单图 + 双图)====="
if [[ ! -f "${HEALTH_CHECK_SCRIPT}" ]]; then
    echo "[WARNING] 找不到 ${HEALTH_CHECK_SCRIPT},请确认实际路径并修改本脚本里 HEALTH_CHECK_SCRIPT 变量。" >&2
else
    if [[ -n "${IMAGE_PATH}" ]]; then
        python "${HEALTH_CHECK_SCRIPT}" --port "${PORT}" --model-name "${SERVED_NAME}" --image "${IMAGE_PATH}"
    else
        echo "[NOTE] 未提供图片路径,用占位图测试,只能验证 server 是否响应。"
        python "${HEALTH_CHECK_SCRIPT}" --port "${PORT}" --model-name "${SERVED_NAME}"
    fi
fi

# ---- 批量图片压测:模拟真实 128 帧场景下的 token 开销,帮助判断 MAX_MODEL_LEN 是否够 ----
# 注意: 这里用同一张图片重复 LIMIT_IMAGES 次来估算 token 数量级,不能完全代表
# 真实视频抽帧的内容差异,但通常图像 token 开销主要取决于分辨率和数量,
# 用于探测"这个 MAX_MODEL_LEN 装不装得下 N 张图" 是合理的替代方式。
# 如果这一步报错,vLLM 通常会在报错信息里给出实际需要的 token 数,可据此调大
# MAX_MODEL_LEN 重新提交。
echo ""
echo "===== 批量图片压测(LIMIT_IMAGES=${LIMIT_IMAGES} 张)====="

python3 - "$PORT" "$SERVED_NAME" "$LIMIT_IMAGES" "$IMAGE_PATH" <<'PYEOF'
import base64
import io
import json
import sys

import requests

try:
    from PIL import Image
except ImportError:
    Image = None

port, served_name, limit_images, image_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

if image_path:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
else:
    if Image is None:
        print("[ERROR] 未安装 Pillow 且未提供 --image,无法生成占位图。", file=sys.stderr)
        sys.exit(1)
    img = Image.new("RGB", (336, 336), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

content = [{"type": "text", "text": "These are frames from a video."}]
for i in range(limit_images):
    content.append({"type": "text", "text": f"Frame at {i}.0s:"})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
content.append({"type": "text", "text": "How many frames were shown? Answer with a number only."})

payload = {
    "model": served_name,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 16,
}

resp = requests.post(f"http://localhost:{port}/v1/chat/completions", json=payload, timeout=300)
print(f"HTTP status: {resp.status_code}")
try:
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
except json.JSONDecodeError:
    print("[ERROR] 响应不是合法 JSON,原始内容:")
    print(resp.text[:3000])
PYEOF

echo ""
echo "===== 探测结束 ====="
echo "请查看本次 job 的 .out / .err 日志(logs/probe_internvl3_5_${SLURM_JOB_ID:-<jobid>}.out/.err),"
echo "确认: 1) server 是否正常启动  2) health check 的 logprobs 字段是否符合预期"
echo "      3) 批量图片压测是否报错,如报错,错误信息里的 token 数提示可用来调整 MAX_MODEL_LEN。"
echo "确认好数值后,把 MAX_MODEL_LEN / TP / LIMIT_IMAGES 抄到 evaluate_internvl3_5.sh 对应位置,"
echo "并删掉那个文件里挡住流程的 exit 1。"