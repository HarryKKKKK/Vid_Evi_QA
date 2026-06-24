# test_multi_image.py
# 目的:仅测"一条消息里发 N 张 base64 帧"的可行性与上限。
# 已确证字段来源:单帧 base64 (data:image/jpeg;base64) 返回 200;T1 解析路径;T2/T3 错误体读法。
import os, json, base64, time, urllib.request, urllib.error, glob

URL = "https://api3.xhub.chat/v1/chat/completions"   # T1 已确证
KEY = os.environ["XHUB_API_KEY"]
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

MODEL = "gpt-4o"                       # T1/T2/单帧 已确证此串可用且多模态
FRAME_DIR = "frames_BV1yG411C7Xi"      # ← 按你服务器实际路径改
LADDER = [1, 4, 8, 16, 32, 64]         # 逐级加压的帧数
SLEEP_BETWEEN = 5                      # 级间停顿秒数(速率限制未知,保守)

# 取已存在的帧文件(e1_*, e2_* 一起,按文件名排序),不抽新帧
frames = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
if not frames:
    raise SystemExit(f"no .jpg found under {FRAME_DIR} (按你实际路径改 FRAME_DIR)")
print(f"found {len(frames)} frames under {FRAME_DIR}")

def b64_image_block(path):
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}

def run_level(n):
    use = frames[:n]
    if len(use) < n:
        print(f"[skip] level {n}: only {len(use)} frames available")
        return
    content = [{"type": "text",
                "text": f"You are shown {n} video frames in order. "
                        f"Briefly describe how the scene changes across them."}]
    content += [b64_image_block(p) for p in use]
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}]}
    body = json.dumps(payload).encode()

    approx_mb = len(body) / 1024 / 1024
    print(f"\n===== LEVEL n={n} | request payload ≈ {approx_mb:.2f} MB =====")
    req = urllib.request.Request(URL, data=body, headers=HEADERS, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            dt = time.time() - t0
            data = json.loads(r.read().decode())
            txt = data.get("choices", [{}])[0].get("message", {}).get("content")
            usage = data.get("usage", {})
            print(f"HTTP 200 | {dt:.1f}s | prompt_tokens={usage.get('prompt_tokens')} "
                  f"completion_tokens={usage.get('completion_tokens')}")
            print("CONTENT:", (txt or "")[:500])
    except urllib.error.HTTPError as e:
        dt = time.time() - t0
        print(f"HTTP ERROR {e.code} | {dt:.1f}s")
        print("ERROR BODY:", e.read().decode()[:1500])   # 原样打印,不预判原因
    except Exception as e:
        print(f"ERROR | {time.time()-t0:.1f}s :", repr(e))

for n in LADDER:
    run_level(n)
    time.sleep(SLEEP_BETWEEN)

print("\n[done] 逐级结果见上。上限 = 第一个出现 ERROR 的 n 之前的那一级。")