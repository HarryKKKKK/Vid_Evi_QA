#!/usr/bin/env python3
"""
Build "insufficient evidence" videos by blacking out the video frames AND
silencing the audio inside each question's evidence_intervals, while keeping
everything else (duration, non-evidence frames/audio) unchanged.

One output video is produced PER QUESTION (per qid), because the same video_id
can map to multiple questions with different evidence intervals.

This script makes NO assumptions about field names beyond the confirmed schema:
    video_id, qid, evidence_intervals[ {start, end, description} ]
The on-disk filename is assumed to be   {video_id}.mp4   (see --preflight).

Encoding strategy: full re-encode + drawbox (black) + volume=0 on the evidence
intervals. Non-evidence regions are visually identical but re-compressed.
"""

import argparse
import json
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Defaults use the EXACT paths you provided (verbatim, not normalized).
# ---------------------------------------------------------------------------
DEFAULT_JSON = "cgbench_pipeline/cgbench_filtered.json"
# Confirmed via your check output: files live here, named EXACTLY by video_id
# with NO extension (e.g. a file literally named "aTV3damvdZs"). Stem-matching
# handles this: splitext("aTV3damvdZs") -> stem "aTV3damvdZs".
DEFAULT_VIDEO_DIR = "source_datasets/cg_bench/videos"
DEFAULT_OUTPUT_DIR = "source_datasets/cg_bench/insufficient_videos"
OUTPUT_EXT = ".mp4"   # output container; set to "" if your pipeline wants extensionless outputs


def fmt_num(x):
    """Format a number for an ffmpeg expression: 6.0 -> '6', 7.5 -> '7.5'."""
    return f"{float(x):g}"


def merge_intervals(intervals):
    """
    intervals: list of dicts each with 'start' and 'end'.
    Returns a sorted list of merged (start, end) tuples; touching intervals
    such as [6,7] and [7,8] are merged into [6,8].
    """
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
    """
    Map filename-stem -> list of file paths, exactly like your check script
    (f.stem for f in dir if f.is_file()). Extension is NOT assumed, so a video
    stored as .mp4 / .mkv / .webm all resolve by stem == video_id.
    """
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
    """Return ('ok', path) / ('missing', None) / ('ambiguous', [paths])."""
    paths = stem_index.get(video_id, [])
    if len(paths) == 1:
        return "ok", paths[0]
    if len(paths) == 0:
        return "missing", None
    return "ambiguous", paths


def has_audio_stream(path, ffprobe="ffprobe"):
    """Return True if the file has at least one audio stream."""
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return out.returncode == 0 and out.stdout.strip() != ""


def check_binaries(args):
    """Fail fast with a clear message if ffmpeg/ffprobe (and NVENC) are missing."""
    import shutil
    for name, val in (("ffmpeg", args.ffmpeg), ("ffprobe", args.ffprobe)):
        if shutil.which(val) is None and not os.path.isfile(val):
            raise SystemExit(
                f"[FATAL] '{val}' not found on PATH. Install it into your env, e.g.\n"
                f"        conda install -p $CONDA_PREFIX -c conda-forge ffmpeg\n"
                f"        or pass --{name} /abs/path/to/{name}")
    if args.gpu:
        enc = subprocess.run([args.ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True)
        if "h264_nvenc" not in (enc.stdout + enc.stderr):
            raise SystemExit(
                "[FATAL] --gpu requested but this ffmpeg has no 'h264_nvenc'.\n"
                "        Use an ffmpeg built with NVENC, or drop --gpu to use CPU (libx264).")


def build_ffmpeg_cmd(inp, out, enable_expr, audio, args):
    vf = (f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:"f"enable='{enable_expr}'")
    af = f"volume=volume=0:enable='{enable_expr}'"

    cmd = [args.ffmpeg, "-y", "-i", inp, "-vf", vf]
    if audio:
        cmd += ["-af", af]

    if args.gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", args.nvenc_preset,
                "-tune", "hq", "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-crf", str(args.crf), "-preset", args.x264_preset]
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


def select_indices(n, args):
    """Decide which entry indices this process handles (slurm array aware)."""
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if args.index is not None:
        return [args.index]
    if task is not None and not args.all:
        return [int(task)]
    return list(range(n))


def out_name(video_id, qid):
    return f"{video_id}_q{qid}_insufficient{OUTPUT_EXT}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--gpu", action="store_true",
                    help="Use NVENC (run under slurm with a GPU).")
    ap.add_argument("--cq", type=int, default=19, help="NVENC quality (lower=better).")
    ap.add_argument("--nvenc-preset", default="p5")
    ap.add_argument("--crf", type=int, default=18, help="libx264 quality (lower=better).")
    ap.add_argument("--x264-preset", default="medium")
    ap.add_argument("--audio-bitrate", default="192k")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-encode even if the output already exists.")
    ap.add_argument("--all", action="store_true",
                    help="Process every entry even if SLURM_ARRAY_TASK_ID is set.")
    ap.add_argument("--index", type=int, default=None,
                    help="Process only this single entry index.")
    ap.add_argument("--preflight", action="store_true",
                    help="Only check video_id -> file resolution; no encoding.")
    ap.add_argument("--print-count", action="store_true",
                    help="Print the number of entries and exit (for slurm --array).")
    ap.add_argument("--report", default="insufficient_build_report.jsonl")
    ap.add_argument("--ffmpeg", default="ffmpeg",
                    help="ffmpeg binary (name on PATH or absolute path).")
    ap.add_argument("--ffprobe", default="ffprobe",
                    help="ffprobe binary (name on PATH or absolute path).")
    args = ap.parse_args()

    entries = load_entries(args.json)

    if args.print_count:
        print(len(entries))
        return

    # -------------------- preflight: resolution check only -----------------
    if args.preflight:
        stem_index = build_stem_index(args.video_dir)
        missing, ambiguous, found = [], [], 0
        seen = set()
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
        if missing:
            print("[PREFLIGHT] first unresolved video_ids "
                  "(if most are unresolved, the dir or naming rule differs):")
            for vid in missing[:20]:
                print(f"  MISSING: {vid}")
        for vid, paths in ambiguous[:20]:
            print(f"  AMBIGUOUS: {vid} -> {paths}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    check_binaries(args)
    stem_index = build_stem_index(args.video_dir)
    indices = select_indices(len(entries), args)

    report = []
    for i in indices:
        e = entries[i]
        vid = e["video_id"]
        qid = e["qid"]
        intervals = e.get("evidence_intervals", [])

        rec = {"index": i, "video_id": vid, "qid": qid, "status": None}

        if not intervals:
            rec["status"] = "skip_no_intervals"
            report.append(rec); print(rec); continue

        status, val = resolve_input(stem_index, vid)
        if status == "missing":
            rec["status"] = "missing_video"
            report.append(rec); print(rec); continue
        if status == "ambiguous":
            rec["status"] = "ambiguous_video"; rec["matches"] = val
            report.append(rec); print(rec); continue
        inp = val

        out = os.path.join(args.output_dir, out_name(vid, qid))
        if os.path.isfile(out) and not args.overwrite:
            rec["status"] = "skip_exists"; rec["output"] = out
            report.append(rec); print(rec); continue

        merged = merge_intervals(intervals)
        expr = build_enable_expr(merged)
        audio = has_audio_stream(inp, args.ffprobe)
        rec["merged_intervals"] = merged
        rec["has_audio"] = audio

        cmd = build_ffmpeg_cmd(inp, out, expr, audio, args)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.isfile(out):
            rec["status"] = "ok"; rec["output"] = out
        else:
            rec["status"] = "ffmpeg_error"
            rec["stderr_tail"] = proc.stderr[-800:]
        report.append(rec); print({k: rec[k] for k in ("index", "video_id", "qid", "status")})

    # Per-array-task report to avoid concurrent-append corruption when many
    # slurm array tasks write at once. Merge later with: cat report*.jsonl
    report_path = args.report
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None and not args.all and args.index is None:
        base, ext = os.path.splitext(args.report)
        report_path = f"{base}.{task}{ext}"

    with open(report_path, "a", encoding="utf-8") as f:
        for rec in report:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in report if r["status"] == "ok")
    print(f"[DONE] processed {len(report)} | ok {n_ok} | "
          f"report written to {report_path}")


if __name__ == "__main__":
    main()