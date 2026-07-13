#!/usr/bin/env python3
"""
Build "Partially Sufficient" (C2) dose-response videos by applying Gaussian noise
or Gaussian blur INSIDE each question's evidence_intervals only, at a discrete
series of retained-evidence fractions alpha, while keeping everything else
(duration, audio, non-evidence frames) unchanged.

CHANGED (this version): the "noise" mode no longer uses ffmpeg's additive
`noise=alls=` filter (which lets underlying structure bleed through even at
max strength). Instead it BLENDS the original frame with a per-pixel fully
random noise frame, with blend opacity = (1 - alpha):

    alpha=1.0 (C1) -> opacity=0   -> pixel-identical to original
    alpha=0.0 (C3) -> opacity=1   -> evidence region is 100% random noise,
                                     original content fully unrecoverable
    0 < alpha < 1  -> linear blend, still a smooth dose-response

This guarantees the "alpha=0 must be completely unrecognizable" requirement
regardless of how bright/dark/high-contrast the original footage is (additive
noise strength that "looks enough" on one clip can fail on another; this
blend approach is content-independent by construction).

"blur" mode is unchanged (gblur inside the interval); if you also need blur
to guarantee full unrecognizability at alpha=0, the same blend-based pattern
can be applied there (blend original with a very-heavily-blurred copy) --
ask and it can be added the same way.

U(alpha) = C1 if alpha==1, C3 if alpha==0, else C2 (eq. 4).

Input candidates are {video_id, qid} pairs that already passed the evidence-
dependence screen (Algorithm 1, lines 1-11) -- by default the
sufficient-correct/insufficient-wrong set. These are joined against the full
anchor metadata to recover evidence_intervals / question / answer.

One output video is produced per (question, alpha level); each screened anchor
yields a 5-point dose-response curve (Option 2 of Section 5.2 / Algorithm 1).

Parallelism / sharding / resume: same conventions as build_insufficient.py /
build_freeze.py.
  * --jobs N      run N ffmpeg encodes concurrently in THIS process
  * --threads T   libx264 threads per ffmpeg; keep jobs*threads <= cpus-per-task
  * --num-shards / --shard   split (candidate x level) tasks across slurm array
                  tasks (shard from --shard or SLURM_ARRAY_TASK_ID; round-robin)
Resumable: existing outputs are skipped unless --overwrite.
"""

import argparse
import json
import os
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_CANDIDATES = "cgbench_result/suff_correct_insuff_wrong.json"
DEFAULT_META = "cgbench_pipeline/cgbench_filtered.json"
DEFAULT_VIDEO_DIR = "source_datasets/cg_bench/videos"
DEFAULT_OUTPUT_DIR = "source_datasets/cg_bench/partial_videos"
DEFAULT_MANIFEST = "cgbench_result/partial.manifest.jsonl"
OUTPUT_EXT = ".mp4"
DEFAULT_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

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


def get_video_dims(path, ffprobe="ffprobe"):
    """Return (width, height, fps_str) for the first video stream.
    fps_str is kept as-is (e.g. '24000/1001') since ffmpeg lavfi 'rate='
    accepts fractional rate strings directly.
    """
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    parts = out.stdout.strip().split(",")
    if len(parts) < 3:
        raise RuntimeError(f"could not probe dimensions for {path}: {out.stdout!r} {out.stderr!r}")
    width, height, fps = int(parts[0]), int(parts[1]), parts[2]
    return width, height, fps


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


def load_json_array(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"[FATAL] {path} is not a JSON array.")
    return data


def join_entries(candidates_path, meta_path):
    keys = [(e["video_id"], e["qid"]) for e in load_json_array(candidates_path)]
    meta = {(e["video_id"], e["qid"]): e for e in load_json_array(meta_path)}
    entries, missing = [], []
    for vid, qid in keys:
        e = meta.get((vid, qid))
        if e is None:
            missing.append((vid, qid))
        else:
            entries.append(e)
    if missing:
        log(f"[WARN] {len(missing)} candidate pairs not found in {meta_path} "
            f"(showing up to 10): {missing[:10]}")
    return entries


def alpha_tag(alpha):
    return f"a{round(alpha * 100):03d}"


def condition_for_alpha(alpha):
    if alpha >= 1.0:
        return "C1"
    if alpha <= 0.0:
        return "C3"
    return "C2"


def build_blur_vf(alpha, args, enable_expr):
    """Unchanged: additive gblur inside the interval."""
    sigma = args.blur_max * (1.0 - alpha)
    if sigma <= 0:
        return None
    return f"gblur=sigma={sigma:g}:enable='{enable_expr}'"


def build_noise_filter_complex(alpha, enable_expr, width, height, fps):
    """
    Blend the main video with a fully random per-pixel noise frame, only
    inside the evidence interval(s), with opacity = 1 - alpha.

    alpha=0  -> opacity=1 -> evidence region is pure random noise (unrecoverable)
    alpha=1  -> opacity=0 -> pixel-identical to original (filter is a no-op there)

    Returns a filter_complex string producing an output pad named [vout].
    """
    # NOTE: ffmpeg's blend filter `all_opacity` is the weight of the FIRST
    # input (the original video), not the second (the noise). Verified
    # empirically: opacity=1 -> 100% original, opacity=0 -> 100% noise.
    # So opacity must equal alpha directly (not 1 - alpha).
    opacity = round(alpha, 4)
    # NOTE: previously used `geq=random(...)`. geq evaluates a per-pixel
    # expression every frame and is extremely slow for long videos (can be
    # much slower than realtime). Using the native `noise` filter on a flat
    # gray source is C-implemented and fast, and at max strength on a flat
    # base it still saturates into full television-static-like noise.
    return (
        f"color=size={width}x{height}:rate={fps}:color=gray,"
        f"format=yuv420p,"
        f"noise=alls=100:allf=t+u[noise];"
        f"[0:v][noise]blend=all_mode='normal':all_opacity={opacity}:"
        f"enable='{enable_expr}':shortest=1[vout]"
    )


def build_ffmpeg_cmd(inp, out, audio, args, mode, filter_complex=None, vf=None):
    """
    Two paths:
      - mode == "noise": use -filter_complex (the noise source is generated
        in-graph via lavfi 'color'+'geq', no extra -i needed) and map [vout].
      - mode == "blur": use the simple -vf path, unchanged.
    """
    if mode == "noise":
        cmd = [args.ffmpeg, "-y", "-i", inp,
               "-filter_complex", filter_complex, "-map", "[vout]"]
        if audio:
            cmd += ["-map", "0:a"]
    else:
        cmd = [args.ffmpeg, "-y", "-i", inp]
        if vf:
            cmd += ["-vf", vf]

    if args.gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", args.nvenc_preset,
                "-tune", "hq", "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0"]
    else:
        cmd += ["-c:v", "libx264", "-crf", str(args.crf),
                "-preset", args.x264_preset, "-threads", str(args.threads)]
    cmd += ["-pix_fmt", "yuv420p"]

    cmd += ["-c:a", "copy"] if audio else ["-an"]
    cmd += ["-movflags", "+faststart", out]
    return cmd


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


def out_name(video_id, qid, mode, alpha):
    return f"{video_id}_q{qid}_c2_{mode}_{alpha_tag(alpha)}{OUTPUT_EXT}"


def process_level(entry, alpha, stem_index, args):
    vid, qid = entry["video_id"], entry["qid"]
    intervals = entry.get("evidence_intervals", [])
    rec = {"video_id": vid, "qid": qid, "mode": args.mode, "alpha": alpha,
           "condition": condition_for_alpha(alpha), "status": None}

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

    out = os.path.join(args.output_dir, out_name(vid, qid, args.mode, alpha))
    rec["output"] = out
    if os.path.isfile(out) and not args.overwrite:
        rec["status"] = "skip_exists"
        return rec

    merged = merge_intervals(intervals)
    expr = build_enable_expr(merged)
    audio = has_audio_stream(inp, args.ffprobe)
    rec["has_audio"] = audio

    if args.mode == "noise":
        try:
            width, height, fps = get_video_dims(inp, args.ffprobe)
        except RuntimeError as e:
            rec["status"] = "probe_error"
            rec["stderr_tail"] = str(e)
            return rec
        fc = build_noise_filter_complex(alpha, expr, width, height, fps)
        cmd = build_ffmpeg_cmd(inp, out, audio, args, "noise", filter_complex=fc)
    else:
        vf = build_blur_vf(alpha, args, expr)
        cmd = build_ffmpeg_cmd(inp, out, audio, args, "blur", vf=vf)

    if args.dry_run:
        rec["status"] = "dry_run"; rec["cmd"] = " ".join(cmd)
        return rec

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and os.path.isfile(out):
        rec["status"] = "ok"
    else:
        rec["status"] = "ffmpeg_error"
        rec["stderr_tail"] = proc.stderr[-800:]
        if os.path.isfile(out):
            try:
                os.remove(out)
            except OSError:
                pass
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
    ap.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    ap.add_argument("--meta", default=DEFAULT_META)
    ap.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--mode", choices=["noise", "blur"], default="noise")
    ap.add_argument("--levels", default=",".join(str(a) for a in DEFAULT_LEVELS),
                    help="Comma-separated retained-evidence fractions alpha in [0,1].")
    ap.add_argument("--blur-max", type=float, default=18.0,
                    help="ffmpeg `gblur` sigma at alpha=0 (blur mode only).")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--cq", type=int, default=19)
    ap.add_argument("--nvenc-preset", default="p5")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--x264-preset", default="veryfast")
    ap.add_argument("--threads", type=int, default=4,
                    help="libx264 threads per ffmpeg. Keep jobs*threads <= cpus.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Concurrent ffmpeg encodes within this process.")
    ap.add_argument("--num-shards", type=int, default=None,
                    help="Total shards (= slurm array size). Shard id from "
                         "--shard or SLURM_ARRAY_TASK_ID; round-robin split "
                         "over (candidate x level) tasks.")
    ap.add_argument("--shard", type=int, default=None,
                    help="This shard id (overrides SLURM_ARRAY_TASK_ID).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="Process every candidate, ignoring sharding.")
    ap.add_argument("--index", type=int, default=None,
                    help="Process only this single candidate (all its levels).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N candidates (after sharding, "
                         "if any). Use this for a quick timing/sanity test, "
                         "e.g. --limit 3. Mutually exclusive with --full.")
    ap.add_argument("--full", action="store_true",
                    help="Explicitly process the full candidate list (no "
                         "--limit). This is the default behavior when "
                         "--limit is omitted; the flag exists so call sites "
                         "(e.g. sbatch scripts) can state their intent "
                         "clearly instead of relying on an implicit default.")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--print-count", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print ffmpeg commands without running them.")
    ap.add_argument("--report", default="partial_build_report.jsonl")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    args = ap.parse_args()

    if args.limit is not None and args.full:
        raise SystemExit("[FATAL] --limit and --full are mutually exclusive; "
                          "pass --limit N for a quick test on the first N "
                          "candidates, or --full to process everything.")

    levels = [float(x) for x in args.levels.split(",") if x != ""]
    entries = join_entries(args.candidates, args.meta)

    if args.print_count:
        print(len(entries))
        return
    if args.preflight:
        run_preflight(entries, args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.dry_run:
        check_binaries(args)
    stem_index = build_stem_index(args.video_dir)
    indices = select_indices(len(entries), args)
    shard = get_shard(args)

    mode_desc = f"LIMIT={args.limit}" if args.limit is not None else "FULL"
    log(f"[START] candidates={len(entries)} this_task={len(indices)} run_mode={mode_desc} "
        f"levels={levels} mode={args.mode} jobs={args.jobs} threads={args.threads} "
        f"shard={shard} num_shards={args.num_shards}")

    tasks = [(entries[i], alpha) for i in indices for alpha in levels]
    report = []
    total = len(tasks)
    done = 0

    def emit(rec):
        nonlocal done
        done += 1
        brief = {k: rec.get(k) for k in ("video_id", "qid", "alpha", "condition", "status")}
        log(f"[{done}/{total}] {brief}")

    if args.jobs <= 1:
        for entry, alpha in tasks:
            rec = process_level(entry, alpha, stem_index, args)
            report.append(rec); emit(rec)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(process_level, entry, alpha, stem_index, args): (entry, alpha)
                    for entry, alpha in tasks}
            for fut in as_completed(futs):
                rec = fut.result()
                report.append(rec); emit(rec)

    report_path = args.report
    if shard is not None and not args.all and args.index is None:
        base, ext = os.path.splitext(args.report)
        report_path = f"{base}.{shard}{ext}"
    with open(report_path, "a", encoding="utf-8") as f:
        for rec in report:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_key = {(e["video_id"], e["qid"]): e for e in entries}
    manifest_rows = []
    for rec in report:
        if rec["status"] not in ("ok", "skip_exists", "dry_run"):
            continue
        e = by_key[(rec["video_id"], rec["qid"])]
        manifest_rows.append({
            "video_id": e["video_id"],
            "qid": e["qid"],
            "question": e.get("question"),
            "choices": e.get("choices"),
            "answer": e.get("answer"),
            "evidence_intervals": e.get("evidence_intervals"),
            "mode": rec["mode"],
            "alpha": rec["alpha"],
            "condition": rec["condition"],
            "output": rec.get("output"),
        })
    manifest_path = args.manifest
    if shard is not None and not args.all and args.index is None:
        base, ext = os.path.splitext(args.manifest)
        manifest_path = f"{base}.{shard}{ext}"
    with open(manifest_path, "a", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(r["status"] for r in report)
    log(f"[DONE] processed {len(report)} | {dict(counts)} | "
        f"report -> {report_path} | manifest -> {manifest_path}")


if __name__ == "__main__":
    main()