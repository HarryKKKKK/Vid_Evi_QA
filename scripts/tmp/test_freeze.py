#!/usr/bin/env python3
"""
测试帧冻结版 build_freeze.py：验证输出视频的
  (1) 时长是否和源视频一致（无漂移）
  (2) evidence 区间是否真的被"冻结"——区间内两帧彼此几乎逐像素相同
  (3) 冻结用的是不是对的那一帧——区间内一帧 ≈ 源视频在 freeze 时刻的帧
        （--freeze-source pre  -> start-1/fps；first -> start，与构建脚本一致）
  (4) 非 evidence 区间是否保留原画面——输出帧 ≈ 源视频同时刻帧
不修改你的脚本，只调用它（按 entry index），再用 ffprobe/ffmpeg(+PIL) 校验。

用法：
  python test_freeze.py --n 5                         # 测前 5 条
  python test_freeze.py --qids 5218 66 79             # 测指定 qid
  python test_freeze.py --n 5 --pick-multi            # 优先挑 evidence 段数多的
  python test_freeze.py --n 5 --freeze-source first   # 测 first 模式

判定（都可用命令行覆盖阈值）：
  时长   |diff| <= --tol            (默认 0.5s)
  冻结   frz_MAD  <= --frz-tol      (默认 1.5，区间内两帧差异，越小越冻)
  对帧   match_MAD<= --match-tol    (默认 8.0，区间帧 vs 源 freeze 帧)
  非ev   nonev_MAD<= --nonev-tol    (默认 8.0，输出 vs 源 同时刻，重编码噪声)
MAD = 两帧灰度平均绝对差(0-255)。没装 PIL 时退化用 ffmpeg freezedetect 验冻结，
对帧/非ev 检查跳过（装一下：pip install pillow numpy 可获得完整校验）。
"""
import argparse, json, os, subprocess, tempfile, sys

SCRIPT = "scripts/vqa/build_freeze.py"
JSON   = "cgbench_pipeline/cgbench_filtered.json"
VDIR   = "source_datasets/cg_bench/videos"

try:
    from PIL import Image
    import numpy as np
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


def run(c):
    return subprocess.run(c, capture_output=True, text=True)


def probe_dur(path):
    o = run(["ffprobe", "-v", "error", "-of", "json",
             "-show_format", "-show_streams", path])
    try:
        info = json.loads(o.stdout or "{}")
    except Exception:
        return None, None, None
    fd = None
    try:
        fd = float(info["format"]["duration"])
    except Exception:
        pass
    vd = nf = None
    for st in info.get("streams", []):
        if st.get("codec_type") == "video":
            try: vd = float(st.get("duration"))
            except Exception: pass
            try: nf = int(st.get("nb_frames"))
            except Exception: pass
            break
    return fd, vd, nf


def probe_fps(path):
    o = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path])
    s = o.stdout.strip()
    try:
        if "/" in s:
            n, d = s.split("/"); n, d = float(n), float(d)
            return n / d if d else None
        return float(s) if s else None
    except Exception:
        return None


def freeze_ts(start, fps, mode):
    if mode == "first":
        return max(0.0, start)
    dt = (1.0 / fps) if (fps and fps > 0) else (1.0 / 30.0)
    return max(0.0, start - dt)


def resolve_input(video_id):
    for f in os.listdir(VDIR):
        if os.path.splitext(f)[0] == video_id:
            return os.path.join(VDIR, f)
    return None


def extract(video, t, png):
    """抽 video 在 t 秒的一帧到 png。-ss 放 -i 前，与构建脚本抽冻结帧方式一致。"""
    run(["ffmpeg", "-y", "-ss", f"{t:g}", "-i", video,
         "-frames:v", "1", "-an", "-update", "1", png])
    return os.path.isfile(png) and os.path.getsize(png) > 0


def load_gray(png):
    return np.asarray(Image.open(png).convert("L"), dtype=np.float64)


def mad_at(video_a, ta, video_b, tb, tmp):
    """抽两帧算灰度平均绝对差；尺寸不一致时取公共区域。返回 None 表示抽帧失败。"""
    pa = os.path.join(tmp, "a.png"); pb = os.path.join(tmp, "b.png")
    if not extract(video_a, ta, pa) or not extract(video_b, tb, pb):
        return None
    A, B = load_gray(pa), load_gray(pb)
    h = min(A.shape[0], B.shape[0]); w = min(A.shape[1], B.shape[1])
    return float(np.abs(A[:h, :w] - B[:h, :w]).mean())


def freezedetect_segments(video, noise="-60dB", d=0.2):
    """无 PIL 时的退化方案：ffmpeg freezedetect，返回 [(start,end), ...]。"""
    o = run(["ffmpeg", "-i", video, "-vf",
             f"freezedetect=n={noise}:d={d},metadata=print",
             "-an", "-f", "null", "-"])
    segs = []
    cur_start = None
    for line in o.stderr.splitlines():
        if "freeze_start" in line:
            try: cur_start = float(line.split(":")[-1])
            except Exception: cur_start = None
        elif "freeze_end" in line and cur_start is not None:
            try:
                segs.append((cur_start, float(line.split(":")[-1])))
            except Exception:
                pass
            cur_start = None
    return segs


def pick_nonev_t(ivs, dur, fps):
    """挑一个不在任何 evidence 区间、且离边界有余量的时刻。"""
    margin = (2.0 / fps) if (fps and fps > 0) else 0.1
    for cand in [5.0, (dur or 60) / 2, (dur or 60) * 0.8, (dur or 60) * 0.3]:
        if cand <= 0 or (dur and cand >= dur):
            continue
        inside = any(float(iv["start"]) - margin <= cand <= float(iv["end"]) + margin
                     for iv in ivs)
        if not inside:
            return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--qids", type=int, nargs="+", default=None)
    ap.add_argument("--pick-multi", action="store_true",
                    help="优先挑 evidence 段数多的条目")
    ap.add_argument("--freeze-source", choices=["pre", "first"], default="pre",
                    help="传给构建脚本，并据此核对'对帧'检查")
    ap.add_argument("--tol", type=float, default=0.5, help="时长容差(秒)")
    ap.add_argument("--frz-tol", type=float, default=1.5,
                    help="区间内两帧 MAD 上限（越小越冻）")
    ap.add_argument("--match-tol", type=float, default=8.0,
                    help="区间帧 vs 源 freeze 帧 MAD 上限")
    ap.add_argument("--nonev-tol", type=float, default=8.0,
                    help="非ev 输出 vs 源 同时刻 MAD 上限")
    ap.add_argument("--out-dir", default="/tmp/freeze_test")
    a = ap.parse_args()

    full = json.load(open(JSON))

    if a.qids:
        qset = set(a.qids)
        sample = [(i, e) for i, e in enumerate(full)
                  if e.get("qid") in qset and e.get("evidence_intervals")]
    else:
        idxd = [(i, e) for i, e in enumerate(full) if e.get("evidence_intervals")]
        if a.pick_multi:
            idxd.sort(key=lambda x: len(x[1]["evidence_intervals"]), reverse=True)
        sample = idxd[:a.n]

    os.makedirs(a.out_dir, exist_ok=True)
    if not HAVE_PIL:
        print("[WARN] 未检测到 PIL/numpy：仅做 时长 + freezedetect 冻结检查；"
              "对帧/非ev 检查跳过。装 pillow numpy 可获完整校验。\n")

    print(f"mode={a.freeze_source}")
    print(f"{'qid':>7} {'segs':>4} {'src_dur':>9} {'out_dur':>9} {'diff':>8} "
          f"{'frames':>7} {'frz':>6} {'match':>6} {'nonev':>6} {'verdict':>8}")
    print("-" * 96)

    all_ok = True
    for idx, e in sample:
        vid = e["video_id"]; qid = e["qid"]; ivs = e["evidence_intervals"]
        inp = resolve_input(vid)
        if not inp:
            print(f"{qid:>7}  MISSING video {vid}"); all_ok = False; continue
        src_fd, _src_vd, _ = probe_dur(inp)
        fps = probe_fps(inp)

        out = os.path.join(a.out_dir, f"{vid}_q{qid}_freeze.mp4")
        if os.path.exists(out):
            os.remove(out)
        r = run(["python", SCRIPT, "--index", str(idx),
                 "--output-dir", a.out_dir,
                 "--freeze-source", a.freeze_source,
                 "--x264-preset", "veryfast", "--crf", "18",
                 "--report", os.path.join(a.out_dir, "r.jsonl"),
                 "--overwrite"])
        if not os.path.exists(out):
            tail = (r.stderr or r.stdout or "")[-200:]
            print(f"{qid:>7}  BUILD FAILED: {tail}")
            all_ok = False; continue

        out_fd, _out_vd, out_nf = probe_dur(out)
        diff = (out_fd or 0) - (src_fd or 0)
        dur_ok = abs(diff) <= a.tol

        # 第一个 evidence 区间用于冻结/对帧检查
        es, ee = float(ivs[0]["start"]), float(ivs[0]["end"])
        L = ee - es

        frz_s = match_s = nonev_s = "n/a"
        frz_ok = match_ok = nonev_ok = True

        with tempfile.TemporaryDirectory() as tmp:
            if HAVE_PIL:
                # (2) 冻结：区间内 30% 与 70% 两帧应几乎相同
                if L >= 0.25:
                    t1, t2 = es + 0.3 * L, es + 0.7 * L
                    m = mad_at(out, t1, out, t2, tmp)
                    if m is not None:
                        frz_s = f"{m:.1f}"; frz_ok = m <= a.frz_tol
                else:
                    frz_s = "short"  # 区间太短，跳过两帧比对，靠 match 判

                # (3) 对帧：区间中点输出帧 ≈ 源视频 freeze 时刻帧
                ft = freeze_ts(es, fps, a.freeze_source)
                m = mad_at(out, (es + ee) / 2, inp, ft, tmp)
                if m is not None:
                    match_s = f"{m:.1f}"; match_ok = m <= a.match_tol

                # (4) 非ev：输出帧 ≈ 源同时刻帧（取 ±1 帧最小值，抗 seek 抖动）
                nt = pick_nonev_t(ivs, src_fd, fps)
                if nt is not None:
                    dt = (1.0 / fps) if (fps and fps > 0) else 0.033
                    cands = [mad_at(out, nt, inp, nt + k * dt, tmp)
                             for k in (-1, 0, 1)]
                    cands = [c for c in cands if c is not None]
                    if cands:
                        m = min(cands)
                        nonev_s = f"{m:.1f}"; nonev_ok = m <= a.nonev_tol
            else:
                # 退化：freezedetect 验证每个区间是否被检出为冻结段
                segs = freezedetect_segments(out)
                covered = all(
                    any(s <= (float(iv["start"]) + float(iv["end"])) / 2 <= en
                        for s, en in segs)
                    for iv in ivs)
                frz_s = "det✓" if covered else "det✗"
                frz_ok = covered

        verdict = "OK" if (dur_ok and frz_ok and match_ok and nonev_ok) else "FAIL"
        if verdict == "FAIL":
            all_ok = False
        print(f"{qid:>7} {len(ivs):>4} {src_fd:>9.3f} {out_fd:>9.3f} {diff:>+8.3f} "
              f"{str(out_nf):>7} {frz_s:>6} {match_s:>6} {nonev_s:>6} {verdict:>8}")

    print("-" * 96)
    print("全部通过 ✅" if all_ok else "有问题，看上面的 FAIL 行 ❌")
    print(f"\n生成的测试视频在 {a.out_dir}，可手动播放抽查：evidence 段是否定格、"
          f"画面是否是 {a.freeze_source} 那一帧、段外是否正常。")
    print("frz=区间内两帧差(越小越冻)，match=区间帧vs源freeze帧，nonev=输出vs源同时刻。")


if __name__ == "__main__":
    main()