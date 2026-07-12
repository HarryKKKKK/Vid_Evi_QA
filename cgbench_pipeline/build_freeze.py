#!/usr/bin/env python3
"""
Build "insufficient evidence" videos by FREEZING a single frame over each
question's evidence_intervals (instead of blacking them out), while keeping
everything else (duration, non-evidence frames/audio) unchanged.

帧冻结策略（与遮黑版几乎同构）：
  * 对每个（合并后的）evidence 区间，先用一次轻量 ffmpeg 抽出一张"冻结帧"：
      - --freeze-source pre   : evidence 开始前的一帧（默认）
      - --freeze-source first : evidence 区间的第一帧
    存成临时 PNG。
  * 主编码时把这张 PNG 用 overlay=...:enable='between(t,s,e)' 叠在该区间上，
    等价于把区间内画面替换为这一张静止帧。
  * 时长精确保留：单输入流、单遍编码、时间戳不动；区间外画面逐帧不变。
  * 音频默认在区间内静音（volume=0），与遮黑版一致；--keep-audio 可保留原音。

One output video is produced PER QUESTION (per qid), because the same video_id
can map to multiple questions with different evidence intervals.

Schema: video_id, qid, evidence_intervals[ {start, end, description} ]

Parallelism / sharding / resume：与遮黑版完全一致
  * --jobs N      run N ffmpeg encodes concurrently in THIS process
  * --threads T   libx264 threads per ffmpeg; keep jobs*threads <= cpus-per-task
  * --num-shards / --shard   split entries across slurm array tasks (round-robin)
Resumable: existing outputs are skipped unless --overwrite.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_JSON = "cgbench_pipeline/cgbench_filtered.json"
# DEFAULT_JSON = "cgbench_pipeline/cgbench_filtered.broken_subset.json"
DEFAULT_VIDEO_DIR = "source_datasets/cg_bench/videos"
DEFAULT_OUTPUT_DIR = "source_datasets/cg_bench/freeze_videos"
OUTPUT_EXT = ".mp4"

_print_lock = threading.Lock()

# 缓存每个视频的 fps / 是否有音频，避免同一 video_id 多个 qid 重复 ffprobe。
_probe_lock = threading.Lock()
_fps_cache = {}
_audio_cache = {}


def log(msg):
    with _print_lock:
        print(msg, flush=True)


def fmt_num(x):
    return f"{float(x):g}"


def merge_intervals(intervals):
    spans = []
    for iv in intervals:
        s = float(iv["start"])
        e = float(iv["end"])
        if e < s:
            s, e = e, s
        spans.append((s, e))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def build_enable_expr(merged):
    parts = [f"between(t,{fmt_num(s)},{fmt_num(e)})" for s, e in merged]
    return "+".join(parts)


def build_stem_index(video_dir):
    index = {}
    if not os.path.isdir(video_dir):
        raise SystemExit(f"[FATAL] video dir not found: {video_dir}")
    for name in os.listdir(video_dir):
        p = os.path.join(video_dir, name)
        if os.path.isfile(p):
            stem = os.path.splitext(name)[0]
            index.setdefault(stem, []).append(p)
    return index


def resolve_input(stem_index, video_id):
    paths = stem_index.get(video_id, [])
    if len(paths) == 1:
        return "ok", paths[0]
    if len(paths) == 0:
        return "missing", None
    return "ambiguous", paths


def has_audio_stream(path, ffprobe="ffprobe"):
    with _probe_lock:
        if path in _audio_cache:
            return _audio_cache[path]
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    val = out.returncode == 0 and out.stdout.strip() != ""
    with _probe_lock:
        _audio_cache[path] = val
    return val


def probe_fps(path, ffprobe="ffprobe"):
    """Return avg fps as float, or None if it can't be determined."""
    with _probe_lock:
        if path in _fps_cache:
            return _fps_cache[path]
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    fps = None
    s = out.stdout.strip()
    try:
        if "/" in s:
            num, den = s.split("/")
            num, den = float(num), float(den)
            if den != 0:
                fps = num / den
        elif s:
            fps = float(s)
    except ValueError:
        fps = None
    with _probe_lock:
        _fps_cache[path] = fps
    return fps


def check_binaries(args):
    for name, val in (("ffmpeg", args.ffmpeg), ("ffprobe", args.ffprobe)):
        if shutil.which(val) is None and not os.path.isfile(val):
            raise SystemExit(
                f"[FATAL] '{val}' not found on PATH. Install it, e.g.\n"
                f"        conda install -p $CONDA_PREFIX -c conda-forge ffmpeg\n"
                f"        or pass --{name} /abs/path/to/{name}")
    if args.gpu:
        enc = subprocess.run([args.ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True)
        if "h264_nvenc" not in (enc.stdout + enc.stderr):
            raise SystemExit(
                "[FATAL] --gpu requested but this ffmpeg has no 'h264_nvenc'.\n"
                "        Drop --gpu to use CPU (libx264).")


def freeze_timestamp(start, fps, args):
    """要抽取的冻结帧时间戳。"""
    if args.freeze_source == "first":
        return max(0.0, start)
    # pre: evidence 开始前的一帧
    dt = (1.0 / fps) if (fps and fps > 0) else (1.0 / 30.0)
    return max(0.0, start - dt)


def extract_freeze_frame(inp, ts, out_png, args):
    """抽一张静止帧到 PNG。-ss 放在 -i 前：从最近关键帧解到 ts，快且足够准。"""
    cmd = [args.ffmpeg, "-y", "-ss", fmt_num(ts), "-i", inp,
           "-frames:v", "1", "-an", "-update", "1", out_png]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0 and os.path.isfile(out_png) \
        and os.path.getsize(out_png) > 0
    return ok, proc.stderr[-400:]


def build_ffmpeg_cmd(inp, out, merged, freeze_pngs, audio, args):
    """
    inputs : 0 = 原视频；1..N = 每个区间对应的冻结帧 PNG（-loop 1 无限单帧）。
    filter : 链式 overlay，每个区间用自己的 enable 窗口把 PNG 叠上去；
             音频用 volume=0 在所有区间静音（除非 --keep-audio）。
    """
    cmd = [args.ffmpeg, "-y", "-i", inp]
    for png in freeze_pngs:
        cmd += ["-loop", "1", "-i", png]

    chains = []
    cur = "0:v"
    for k, (s, e) in enumerate(merged):
        in_idx = k + 1                      # PNG 输入序号
        label = f"v{k+1}"
        # shortest=1 是关键：-loop 1 的 PNG 是无限流，不加它 overlay 不会在
        # 主视频结束时停止，会跟着无限图像流一直生成、永不收尾（实测 5s 源
        # 被滚成 596s+ 仍未结束），最终被 SLURM 超时杀掉。
        chains.append(
            f"[{cur}][{in_idx}:v]overlay=x=0:y=0:shortest=1:"
            f"enable='between(t,{fmt_num(s)},{fmt_num(e)})'[{label}]"
        )
        cur = label
    fc = ";".join(chains)

    do_audio = audio and not args.keep_audio
    if do_audio:
        enable_expr = build_enable_expr(merged)
        fc += f";[0:a]volume=volume=0:enable='{enable_expr}'[aout]"

    cmd += ["-filter_complex", fc, "-map", f"[{cur}]"]
    if audio:
        cmd += ["-map", "[aout]"] if do_audio else ["-map", "0:a"]

    if args.gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", args.nvenc_preset,
                "-tune", "hq", "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-crf", str(args.crf),
                "-preset", args.x264_preset, "-threads", str(args.threads)]
    cmd += ["-pix_fmt", "yuv420p"]

    if audio:
        if do_audio:
            cmd += ["-c:a", "aac", "-b:a", args.audio_bitrate]
        else:
            cmd += ["-c:a", "copy"]   # --keep-audio：原音直拷
    else:
        cmd += ["-an"]

    cmd += ["-movflags", "+faststart", out]
    return cmd


def load_entries(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"[FATAL] {json_path} is not a JSON array.")
    return data


def get_shard(args):
    if args.shard is not None:
        return args.shard
    t = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(t) if t is not None else None


def select_indices(n, args):
    if args.index is not None:
        idxs = [args.index]
    elif args.all:
        idxs = list(range(n))
    else:
        shard = get_shard(args)
        if args.num_shards and shard is not None:
            idxs = [i for i in range(n) if i % args.num_shards == shard]
        else:
            idxs = list(range(n))
    if args.limit is not None:
        idxs = idxs[:args.limit]
    return idxs


def out_name(video_id, qid):
    return f"{video_id}_q{qid}_freeze{OUTPUT_EXT}"


def process_entry(i, entries, stem_index, args):
    """Pure worker -> report record. Safe to run in a thread pool."""
    e = entries[i]
    vid = e["video_id"]
    qid = e["qid"]
    intervals = e.get("evidence_intervals", [])
    rec = {"index": i, "video_id": vid, "qid": qid,
           "freeze_source": args.freeze_source, "status": None}

    if not intervals:
        rec["status"] = "skip_no_intervals"
        return rec

    status, val = resolve_input(stem_index, vid)
    if status == "missing":
        rec["status"] = "missing_video"
        return rec
    if status == "ambiguous":
        rec["status"] = "ambiguous_video"; rec["matches"] = val
        return rec
    inp = val

    out = os.path.join(args.output_dir, out_name(vid, qid))
    if os.path.isfile(out) and not args.overwrite:
        rec["status"] = "skip_exists"; rec["output"] = out
        return rec

    merged = merge_intervals(intervals)
    audio = has_audio_stream(inp, args.ffprobe)
    fps = probe_fps(inp, args.ffprobe)
    rec["has_audio"] = audio

    # ---- 抽冻结帧（每个区间一张），放到一个临时目录里 ----
    tmpdir = tempfile.mkdtemp(prefix=f"freeze_{qid}_", dir=args.tmp_dir)
    try:
        freeze_pngs = []
        for k, (s, _e) in enumerate(merged):
            ts = freeze_timestamp(s, fps, args)
            png = os.path.join(tmpdir, f"frz_{k}.png")
            ok, err = extract_freeze_frame(inp, ts, png, args)
            if not ok:
                rec["status"] = "freeze_extract_error"
                rec["stderr_tail"] = err
                return rec
            freeze_pngs.append(png)

        cmd = build_ffmpeg_cmd(inp, out, merged, freeze_pngs, audio, args)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.isfile(out):
            rec["status"] = "ok"; rec["output"] = out
        else:
            rec["status"] = "ffmpeg_error"
            rec["stderr_tail"] = proc.stderr[-800:]
            # 失败时清掉半成品，便于断点续跑
            if os.path.isfile(out):
                try:
                    os.remove(out)
                except OSError:
                    pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return rec


def run_preflight(entries, args):
    stem_index = build_stem_index(args.video_dir)
    missing, ambiguous, found, seen = [], [], 0, set()
    for e in entries:
        vid = e["video_id"]
        if vid in seen:
            continue
        seen.add(vid)
        status, val = resolve_input(stem_index, vid)
        if status == "ok":
            found += 1
        elif status == "missing":
            missing.append(vid)
        else:
            ambiguous.append((vid, val))
    print(f"[PREFLIGHT] video dir: {args.video_dir}")
    print(f"[PREFLIGHT] unique video_ids: {len(seen)} | resolved: {found} | "
          f"missing: {len(missing)} | ambiguous: {len(ambiguous)}")
    for vid in missing[:20]:
        print(f"  MISSING: {vid}")
    for vid, paths in ambiguous[:20]:
        print(f"  AMBIGUOUS: {vid} -> {paths}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    # ---- 帧冻结特有 ----
    ap.add_argument("--freeze-source", choices=["pre", "first"], default="pre",
                    help="pre=evidence 前一帧（默认）；first=区间第一帧。")
    ap.add_argument("--keep-audio", action="store_true",
                    help="区间内保留原音（默认与遮黑版一致：静音）。")
    ap.add_argument("--tmp-dir", default=None,
                    help="冻结帧 PNG 的临时目录（默认系统 temp）。")
    # ---- 编码 ----
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--cq", type=int, default=19)
    ap.add_argument("--nvenc-preset", default="p5")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--x264-preset", default="veryfast")
    ap.add_argument("--audio-bitrate", default="192k")
    ap.add_argument("--threads", type=int, default=4,
                    help="libx264 threads per ffmpeg. Keep jobs*threads <= cpus.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Concurrent ffmpeg encodes within this process.")
    ap.add_argument("--num-shards", type=int, default=None,
                    help="Total shards (= slurm array size). Shard id from "
                         "--shard or SLURM_ARRAY_TASK_ID; round-robin split.")
    ap.add_argument("--shard", type=int, default=None,
                    help="This shard id (overrides SLURM_ARRAY_TASK_ID).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="Process every entry, ignoring sharding.")
    ap.add_argument("--index", type=int, default=None,
                    help="Process only this single entry index.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap entries processed (after sharding); for tests.")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--print-count", action="store_true")
    ap.add_argument("--report", default="freeze_build_report.jsonl")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    args = ap.parse_args()

    entries = load_entries(args.json)

    if args.print_count:
        print(len(entries))
        return
    if args.preflight:
        run_preflight(entries, args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    check_binaries(args)
    stem_index = build_stem_index(args.video_dir)
    indices = select_indices(len(entries), args)
    shard = get_shard(args)

    log(f"[START] entries={len(entries)} this_task={len(indices)} "
        f"jobs={args.jobs} threads={args.threads} shard={shard} "
        f"num_shards={args.num_shards} freeze_source={args.freeze_source}")

    report = []
    total = len(indices)
    done = 0

    def emit(rec):
        nonlocal done
        done += 1
        brief = {k: rec.get(k) for k in ("index", "video_id", "qid", "status")}
        log(f"[{done}/{total}] {brief}")

    if args.jobs <= 1:
        for i in indices:
            rec = process_entry(i, entries, stem_index, args)
            report.append(rec); emit(rec)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(process_entry, i, entries, stem_index, args): i
                    for i in indices}
            for fut in as_completed(futs):
                rec = fut.result()
                report.append(rec); emit(rec)

    # Per-shard report file to avoid concurrent-append corruption across tasks.
    report_path = args.report
    if shard is not None and not args.all and args.index is None:
        base, ext = os.path.splitext(args.report)
        report_path = f"{base}.{shard}{ext}"

    with open(report_path, "a", encoding="utf-8") as f:
        for rec in report:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    counts = Counter(r["status"] for r in report)
    log(f"[DONE] processed {len(report)} | {dict(counts)} | "
        f"report -> {report_path}")


if __name__ == "__main__":
    main()