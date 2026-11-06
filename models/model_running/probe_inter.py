#!/usr/bin/env python3
"""
InternVL3_5-8B vLLM server 健康检查脚本。

这个脚本只做验证和打印,不对"是否符合预期"做任何自动判定 —
top_logprobs 是否正常返回、多图 content 格式是否兼容,都需要你
人工核对下面打印出来的原始响应。

用法:
    python health_check_internvl3_5_8b.py --port 8001 --image /path/to/test.jpg

如果没有现成的测试图片,脚本会用一张 1x1 的占位图,但这只能验证
server 是否响应,不能代表真实多帧场景下的行为 —— 请务必再用一张
你实际 pipeline 会用到的真实抽帧图片测一遍。
"""

import argparse
import base64
import io
import json
import sys

import requests

try:
    from PIL import Image
except ImportError:
    Image = None


def make_placeholder_image_b64() -> str:
    if Image is None:
        print("[ERROR] 未安装 Pillow,无法生成占位图,请用 --image 指定一张真实图片。", file=sys.stderr)
        sys.exit(1)
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_single_image_check(base_url: str, model_name: str, image_b64: str, top_logprobs_n: int):
    print("\n===== 测试 1: 单图请求 + top_logprobs =====")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 64,
        "logprobs": True,
        "top_logprobs": top_logprobs_n,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    print(f"HTTP status: {resp.status_code}")
    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[ERROR] 响应不是合法 JSON,原始内容如下:")
        print(resp.text)
        return None
    print(json.dumps(data, indent=2, ensure_ascii=False)[:4000])

    # 只做存在性检查,不代表"正确",请人工核对上面打印的原始结构
    choice = (data.get("choices") or [{}])[0]
    has_logprobs = "logprobs" in choice and choice["logprobs"] is not None
    print(f"\n[CHECK] 响应中是否包含 logprobs 字段: {has_logprobs}")
    if not has_logprobs:
        print("[ATTENTION] 未返回 logprobs,你的 p(gold)/ECE 计算依赖这个字段,"
              "需要确认是这个 checkpoint 本身不支持,还是请求参数写法有误。")
    return data


def run_multi_image_check(base_url: str, model_name: str, image_b64: str):
    print("\n===== 测试 2: 多图请求(验证 content 格式兼容性) =====")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image-1 and Image-2 are frames from a video. Describe what changes."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "max_tokens": 64,
    }
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    print(f"HTTP status: {resp.status_code}")
    try:
        data = resp.json()
    except json.JSONDecodeError:
        print("[ERROR] 响应不是合法 JSON,原始内容如下:")
        print(resp.text)
        return None
    print(json.dumps(data, indent=2, ensure_ascii=False)[:4000])
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", default="InternVL3_5-8B",
                         help="必须和部署脚本里 --served-model-name 一致,否则请求会 404/400")
    parser.add_argument("--image", default=None, help="真实测试图片路径,强烈建议提供")
    parser.add_argument("--top-logprobs", type=int, default=20,
                         help="与你现有 pipeline 的 top_logprobs=20 保持一致,按需修改")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    print(f"===== 检查 server 是否存活: {base_url}/v1/models =====")
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        print(f"HTTP status: {resp.status_code}")
        print(resp.text)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 无法连接到 server: {e}", file=sys.stderr)
        sys.exit(1)

    if args.image:
        image_b64 = load_image_b64(args.image)
    else:
        print("\n[WARNING] 未提供 --image,使用占位图,仅能验证 server 是否响应,"
              "不能代表真实抽帧场景下的行为。")
        image_b64 = make_placeholder_image_b64()

    run_single_image_check(base_url, args.model_name, image_b64, args.top_logprobs)
    run_multi_image_check(base_url, args.model_name, image_b64)

    print("\n===== 完成 =====")
    print("以上结果需要你人工核对: logprobs 字段是否存在、top_logprobs 数量是否等于请求值、"
          "多图 content 是否被正确解析(而非被截断或报错)。脚本本身不对这些做最终判定。")


if __name__ == "__main__":
    main()