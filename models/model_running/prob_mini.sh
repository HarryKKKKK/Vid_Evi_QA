#!/bin/bash
#SBATCH --job-name=probe_minicpmv45
#SBATCH -p llm
#SBATCH --qos=llm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/probe_minicpmv45_%j.out
#SBATCH --error=/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/logs/probe_minicpmv45_%j.err
#
# 用法:
#   sbatch probe_minicpmv45.sbatch.sh
#   sbatch probe_minicpmv45.sbatch.sh 16384 1 130
#   sbatch probe_minicpmv45.sbatch.sh 16384 1 130 /path/到/真实抽帧.jpg
#
# 位置参数(全部可选,不传则用下面的默认起点值):
#   $1 = MAX_MODEL_LEN   (默认: 16384,纯粹是探测起点,不是推荐值)
#   $2 = TP               (默认: 1)
#   $3 = LIMIT_IMAGES     (默认: 130,对齐 evaluate.py 里的图片上限)
#   $4 = IMAGE_PATH       (默认: 空,用占位图,建议传真实抽帧图片)
#
# 已确认的事实(不是猜测):
#   - vLLM 0.11.0(envs_qwen3vl 环境)原生支持 MiniCPMV4_5 这个架构
#     (vLLM 官方文档 minicpmv.py 的 _SUPPORT_VERSION 字典里显式列出
#     (4, 5): MiniCPMV4_5),不像 InternVL3.5 那样需要绕版本坑。
#   - vLLM 官方模型支持表里 MiniCPMV 的模态标注是 "T + IE+ + VE+",
#     同时支持图片和视频输入,所以这里沿用你现有 evaluate.py 的
#     "抽帧当图片发" 调用方式,不需要另外写视频输入的代码路径。
#
# 未确认、需要这次探测验证的:
#   - --limit-mm-per-prompt 的 JSON 格式在 MiniCPM-V-4_5 上是否和
#     InternVL3.5 一样能正常解析(两者跑在同一个 vLLM 版本上,理论上
#     解析规则是 vLLM CLI 层面的、不分模型,但没有专门验证过)。
#   - 是否需要 --trust-remote-code(vLLM 有原生实现,可能不需要,但
#     HF checkpoint 的 tokenizer/processor 部分可能仍需要,保留这个
#     flag 更安全,如果报 "unexpected argument" 之类的错再去掉)。
#   - max_position_embeddings 实际是多少,32 帧场景下 token 开销多少。

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
MODEL_PATH="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/MiniCPM-V-4_5"
SERVED_NAME="minicpm-v-4_5"
WORKDIR="/aifs4su/hansirui_2nd/harry/Vid_Evi_QA"
HEALTH_CHECK_SCRIPT="${WORKDIR}/models/model_running/probe_inter.py"  

echo "===== 本次探测参数 ====="
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}  TP=${TP}  LIMIT_IMAGES=${LIMIT_IMAGES}"
echo "IMAGE_PATH=${IMAGE_PATH:-<未提供,将用占位图>}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<unset>}"
echo "========================"

source /home/hansirui_2nd/anaconda3/etc/profile.d/conda.sh
# envs_qwen3vl: 已确认这个环境的 vLLM 0.11.0 原生认识 MiniCPMV4_5 架构
# (见上方注释依据),且 TORCH_SDPA 后端在这个环境上对 InternVL3.5 跑通过,
# 这里沿用同一套后端设置作为起点,如果遇到类似 PTX 工具链报错,已经
# 知道怎么处理。
conda activate /aifs4su/hansirui_2nd/harry/envs_qwen3vl

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
    exit 1
fi

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
    echo "[FATAL] 端口 ${PORT} 已被占用,脚本中止。" >&2
    exit 1
fi

echo ""
echo "===== 启动 vLLM server (MiniCPM-V-4_5) ====="

# --limit-mm-per-prompt 用 JSON 格式(沿用 InternVL3.5 在同一 vLLM 0.11.0
# 上验证过的格式),显式把 video 配额设为 0(本次走图片输入路径,不用
# native video)。--trust-remote-code 保留但未验证是否必要,见上方注释。
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
        echo "[FATAL] vLLM 进程在启动过程中退出,请查看本文件顶部同名的 .err 日志获取完整报错。" >&2
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

echo ""
echo "===== 运行 health check(单图 + 双图)====="
if [[ ! -f "${HEALTH_CHECK_SCRIPT}" ]]; then
    echo "[WARNING] 找不到 ${HEALTH_CHECK_SCRIPT},请确认实际路径。" >&2
else
    if [[ -n "${IMAGE_PATH}" ]]; then
        python "${HEALTH_CHECK_SCRIPT}" --port "${PORT}" --model-name "${SERVED_NAME}" --image "${IMAGE_PATH}"
    else
        echo "[NOTE] 未提供图片路径,用占位图测试,只能验证 server 是否响应。"
        python "${HEALTH_CHECK_SCRIPT}" --port "${PORT}" --model-name "${SERVED_NAME}"
    fi
fi

echo ""
echo "===== 批量图片压测(LIMIT_IMAGES=${LIMIT_IMAGES} 张)====="
# 与 InternVL3.5 探测时同样的逻辑: 用同一张图重复 LIMIT_IMAGES 次,
# 估算 32 帧真实场景下大致的 token 数量级。注意 evaluate.py 默认只用
# 32 帧(NUM_FRAMES=32),这里测 130 张只是为了对齐 --limit-mm-per-prompt
# 的上限设置,不代表正式跑批会用满 130 张 -- 正式跑批的真实 token 开销
# 需要另外用 32 张真实抽帧图片验证(参考 InternVL3.5 那次,占位图数值
# 和真实抽帧数值可能差好几倍,不能直接信这里的占位图结果)。

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
echo "确认: 1) server 是否正常启动(尤其确认是否也需要 VLLM_ATTENTION_BACKEND=TORCH_SDPA"
echo "         才能绕开类似 InternVL3.5 遇到的 PTX 报错,这里已经预先加上了,若仍报错需要"
echo "         进一步排查)"
echo "      2) --trust-remote-code 是否报 unexpected argument 之类的错(如果报错,说明"
echo "         vLLM 原生实现不需要这个 flag,去掉即可)"
echo "      3) --limit-mm-per-prompt 的 JSON 格式是否被正常解析"
echo "      4) 批量图片压测的 prompt token 数,作为 MAX_MODEL_LEN 的参考起点"
echo "确认好后,把结果抄到 evaluate_minicpmv45.sh 对应位置。"