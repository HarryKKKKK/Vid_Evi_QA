#!/usr/bin/env python3
"""
Build "insufficient evidence" videos by blacking out the video frames AND
silencing the audio inside each question's evidence_intervals, while keeping
everything else (duration, non-evidence frames/audio) unchanged.

One output video is produced PER QUESTION (per qid), because the same video_id
can map to multiple questions with different evidence intervals.

Schema: video_id, qid, evidence_intervals[ {start, end, description} ]

Encoding strategy: single-pass full re-encode + drawbox (black) + volume=0 on
the evidence intervals. Duration is preserved exactly (single stream, single
pass). Non-evidence regions are visually identical but re-compressed.

Parallelism:
  * --jobs N      run N ffmpeg encodes concurrently in THIS process (thread pool;
                  ffmpeg runs as a subprocess so the GIL is released while waiting)
  * --threads T   libx264 threads per ffmpeg; keep jobs*threads <= cpus-per-task
  * --num-shards / --shard   split entries across slurm array tasks
                  (shard from --shard or SLURM_ARRAY_TASK_ID; round-robin)
Resumable: existing outputs are skipped unless --overwrite.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_JSON = "cgbench_pipeline/cgbench_filtered.json"
DEFAULT_VIDEO_DIR = "source_datasets/cg_bench/videos"
DEFAULT_OUTPUT_DIR = "source_datasets/cg_bench/insufficient_videos"
OUTPUT_EXT = ".mp4"

_print_lock = threading.Lock()


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
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return out.returncode == 0 and out.stdout.strip() != ""


def check_binaries(args):
    import shutil
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


def build_ffmpeg_cmd(inp, out, enable_expr, audio, args):
    vf = f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='{enable_expr}'"
    af = f"volume=volume=0:enable='{enable_expr}'"

    cmd = [args.ffmpeg, "-y", "-i", inp, "-vf", vf]
    if audio:
        cmd += ["-af", af]

    if args.gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", args.nvenc_preset,
                "-tune", "hq", "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-crf", str(args.crf),
                "-preset", args.x264_preset, "-threads", str(args.threads)]
    cmd += ["-pix_fmt", "yuv420p"]

    if audio:
        cmd += ["-c:a", "aac", "-b:a", args.audio_bitrate]
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
    return f"{video_id}_q{qid}_insufficient{OUTPUT_EXT}"


def process_entry(i, entries, stem_index, args):
    """Pure worker -> report record. Safe to run in a thread pool."""
    e = entries[i]
    vid = e["video_id"]
    qid = e["qid"]
    intervals = e.get("evidence_intervals", [])
    rec = {"index": i, "video_id": vid, "qid": qid, "status": None}

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
    expr = build_enable_expr(merged)
    audio = has_audio_stream(inp, args.ffprobe)
    rec["has_audio"] = audio

    cmd = build_ffmpeg_cmd(inp, out, expr, audio, args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and os.path.isfile(out):
        rec["status"] = "ok"; rec["output"] = out
    else:
        rec["status"] = "ffmpeg_error"
        rec["stderr_tail"] = proc.stderr[-800:]
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
    ap.add_argument("--report", default="insufficient_build_report.jsonl")
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
        f"num_shards={args.num_shards}")

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