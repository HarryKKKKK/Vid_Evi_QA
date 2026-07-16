#!/usr/bin/env python3
"""
CG-Bench evaluation pipeline (concurrent).

Task modes:
  --task qa
      Pure multiple-choice QA. The model must choose one option.

  --task classify
      Answerability classification:
      ANSWERABLE / PARTIALLY_ANSWERABLE / UNANSWERABLE.

Evidence-condition modes (--mode):
  sufficient      C1 anchors, full video.
  hallucination   CG-Bench Hallucination split, full video.
  insufficient    C3, masked/frozen-evidence video (*_freeze.mp4).
  partial         C2, temporal-ablation dose-response videos
                   (*_c2_<mode>_a<NNN>.mp4), one task per (anchor, alpha).
  all             sufficient + hallucination + insufficient + partial.
                   NOTE: partial multiplies its candidate set by
                   len(--alphas) (default 7) LLM calls per anchor, so
                   `--mode all` is now substantially more calls than
                   before partial was added. Use --limit while testing.

Partial (C2) construction:
  Candidate anchors come from --partial-candidates (default:
  cgbench_result/suff_correct_insuff_wrong.json), a list of
  {"video_id": ..., "qid": ...} pairs. These are joined against
  FILTERED_JSON by (video_id, qid) -- the same join pattern used in
  build_partial.py's join_entries() -- to recover question/choices/answer/
  evidence_intervals. Each joined anchor is then expanded into one task per
  alpha in --alphas (default: 0.0,0.05,0.1,0.15,0.2,0.3,0.5), and the video
  path for each task is built as:
      partial_videos/{video_id}_q{qid}_c2_{partial_mode}_{alpha_tag}.mp4
  where alpha_tag / condition_for_alpha are copied verbatim from
  build_partial.py (f"a{round(alpha*100):03d}"; C1 if alpha>=1, C3 if
  alpha<=0, else C2) so the naming/condition logic stays identical between
  the two scripts.

Resume / dedup key:
  sufficient/hallucination/insufficient results are still deduplicated by
  (video_id, qid). Partial results are deduplicated by
  (video_id, qid, alpha) instead -- a plain (video_id, qid) key would
  collapse all alpha levels of the same anchor onto a single jsonl line
  and silently drop the rest of the dose-response curve.

Incremental saving:
  Each finished result is immediately appended to a .jsonl file.
  At the end, an ordered .json file is also written for compatibility.

Multi-model tagging (--model-tag, required):
  Every output record carries a "model_tag" field (e.g. "qwen3vl",
  "internvl3_5") identifying which model produced it. This is required
  (no default) so that a missing/forgotten --model-tag fails loudly with a
  CLI error instead of silently writing an ambiguous or wrong tag into the
  results. This does NOT by itself let two different models safely share
  the same --results-dir: the resume/dedup key in _record_key() is still
  (video_id, qid[, alpha]) only, so if you point two different models at
  the same --results-dir, the second model's run will treat the first
  model's records as "already done" and skip them, not overwrite them with
  its own results. --model-tag is for *after-the-fact* identification of
  which model a record came from (e.g. if you inspect a file and need to
  confirm), not a substitute for using separate --results-dir per model.

Other datasets (e.g. NExT-GQA):
  --filtered-json / --video-dir / --results-dir override the CG-Bench
  defaults below, so this same script can run against
  nextgqa_pipeline/nextgqa_filtered.json + source_datasets/next_gqa/videos/
  without touching CG-Bench's own eval.sh invocation (which never passes
  these flags, so it keeps using the CG-Bench paths unchanged).

  Video path resolution for --mode sufficient/hallucination is dataset-aware
  via each entry's `source_dataset` field: "cgbench" entries keep the flat
  {video_id}.mp4 naming used today; "nextgqa" entries are resolved through
  --video-id-mapping (defaults to NExT-GQA's own map_vid_vidorID.json),
  which maps video_id to the {folder}/{vidorID} relative path NExT-GQA
  videos are actually stored under (see nextgqa_pipeline/README.md).

  --mode insufficient uses --insufficient-video-dir instead (default:
  CG-Bench's freeze_videos/); its {video_id}_q{qid}_freeze.mp4 naming (from
  build_freeze.py) is flat for both datasets, so no --video-id-mapping
  lookup is needed there -- just point --insufficient-video-dir at
  source_datasets/next_gqa/freeze_videos for a NExT-GQA run.

  --mode partial is still CG-Bench-only (no NExT-GQA partial/C2 videos
  exist yet).

  Example (sufficient):
    python scripts/eval/evaluate.py --mode sufficient --task qa \
      --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
      --video-dir source_datasets/next_gqa/videos \
      --results-dir nextgqa_result \
      --model-tag qwen3vl \
      --limit 5

  Example (insufficient):
    python scripts/eval/evaluate.py --mode insufficient --task qa \
      --filtered-json nextgqa_pipeline/nextgqa_filtered.json \
      --insufficient-video-dir source_datasets/next_gqa/freeze_videos \
      --results-dir nextgqa_result \
      --model-tag qwen3vl \
      --limit 5
"""

import argparse
import base64
import json
import os
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


# -- Hyperparameters ----------------------------------------------------------

NUM_FRAMES = 32
FRAME_WIDTH = 560
EVIDENCE_FRACTION = 0.25
WORKERS = 8
DEFAULT_ALPHAS = "0.0,0.05,0.1,0.15,0.2,0.3,0.5"
DEFAULT_PARTIAL_MODE = "noise"


# -- Paths --------------------------------------------------------------------

BASE_DIR = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
FILTERED_JSON = BASE_DIR / "cgbench_pipeline" / "cgbench_filtered.json"
VIDEOS_DIR = BASE_DIR / "source_datasets" / "cg_bench" / "videos"
INSUFFICIENT_DIR = BASE_DIR / "source_datasets" / "cg_bench" / "freeze_videos"
PARTIAL_VIDEOS_DIR = BASE_DIR / "source_datasets" / "cg_bench" / "partial_videos"
PARTIAL_CANDIDATES_JSON = BASE_DIR / "cgbench_result" / "suff_correct_insuff_wrong.json"
RESULTS_DIR = BASE_DIR / "cgbench_result"

# Only consulted for entries whose source_dataset == "nextgqa" (see
# process_entry()); CG-Bench entries never touch this. Defaults to
# NExT-GQA's own official mapping file so a NExT-GQA run works out of the
# box with just --filtered-json/--video-dir/--results-dir overridden.
NEXTGQA_MAPPING_JSON = (
    BASE_DIR / "source_datasets" / "next_gqa" / "NExT-GQA" / "datasets" / "nextgqa" / "map_vid_vidorID.json"
)


# -- LLM client ---------------------------------------------------------------

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("VLLM_MODEL", "qwen3-vl")

_client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
    max_retries=0,
    timeout=120.0,
)


# -- Partial (C2) helpers -- copied verbatim from build_partial.py's
#    alpha_tag()/condition_for_alpha() so naming/condition logic can never
#    drift between the two scripts. -------------------------------------------

def alpha_tag(alpha: float) -> str:
    return f"a{round(alpha * 100):03d}"


def condition_for_alpha(alpha: float) -> str:
    if alpha >= 1.0:
        return "C1"
    if alpha <= 0.0:
        return "C3"
    return "C2"


def parse_alphas(spec: str) -> list[float]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(float(tok))
    if not out:
        raise SystemExit(f"[FATAL] --alphas produced an empty list from {spec!r}")
    return out


def load_partial_entries(filtered_json_path: Path, candidates_path: Path) -> list[dict]:
    """
    Join --partial-candidates ({"video_id","qid"} pairs) against
    FILTERED_JSON by (video_id, qid). Same join pattern as
    build_partial.py's join_entries(): unresolved pairs are reported and
    dropped, not silently substituted or guessed.
    """
    if not filtered_json_path.is_file():
        raise SystemExit(f"[FATAL] filtered meta json not found: {filtered_json_path}")
    if not candidates_path.is_file():
        raise SystemExit(f"[FATAL] partial candidates json not found: {candidates_path}")

    with open(filtered_json_path, encoding="utf-8") as f:
        meta_all = json.load(f)
    meta_by_key = {(e["video_id"], e["qid"]): e for e in meta_all}

    with open(candidates_path, encoding="utf-8") as f:
        candidates = json.load(f)

    entries, missing = [], []
    for c in candidates:
        key = (c["video_id"], c["qid"])
        e = meta_by_key.get(key)
        if e is None:
            missing.append(key)
        else:
            entries.append(e)

    if missing:
        print(
            f"[WARN] {len(missing)} pairs from {candidates_path.name} not found "
            f"in {filtered_json_path.name} (showing up to 10): {missing[:10]}",
            flush=True,
        )
    return entries


def load_video_id_mapping(path: Path) -> dict:
    """Load a {video_id: relative_path_without_extension} mapping.

    Used only to resolve video paths for entries whose source_dataset is
    "nextgqa" (see process_entry()). Returns {} if the file doesn't exist so
    that pure CG-Bench runs never fail just because this optional file isn't
    present in a given environment.
    """
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_with_alphas(entries: list[dict], alphas: list[float], partial_mode: str) -> list[dict]:
    """One task dict per (entry, alpha). Each task is a shallow copy of the
    meta entry plus "alpha" and "partial_mode" -- process_entry() branches
    on entry.get("alpha") to pick the partial_videos/ naming scheme, and
    these two keys ride along into the final saved result via {**entry, ...}.
    """
    expanded = []
    for e in entries:
        for a in alphas:
            e2 = dict(e)
            e2["alpha"] = a
            e2["partial_mode"] = partial_mode
            expanded.append(e2)
    return expanded


# -- Timestamp planning -------------------------------------------------------

def _video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def plan_timestamps(
    duration: float,
    num_frames: int,
    *,
    sampling: str = "uniform",
    evidence_intervals: list[dict] | None = None,
    evidence_fraction: float = EVIDENCE_FRACTION,
) -> list[float]:
    if sampling == "evidence" and evidence_intervals:
        n_evidence = max(1, int(round(num_frames * evidence_fraction)))
        n_global = max(0, num_frames - n_evidence)

        timestamps: list[float] = []
        total_ev = sum(iv["end"] - iv["start"] for iv in evidence_intervals)

        if total_ev <= 0:
            for iv in evidence_intervals:
                timestamps.append(iv["start"])
        else:
            for iv in evidence_intervals:
                seg = iv["end"] - iv["start"]
                k = max(1, round(n_evidence * seg / total_ev))
                for j in range(k):
                    timestamps.append(iv["start"] + (j + 0.5) * seg / k)

        for i in range(n_global):
            timestamps.append((i + 0.5) * duration / max(1, n_global))

        timestamps = sorted({round(t, 2) for t in timestamps})
        timestamps = [
            min(max(t, 0.0), max(duration - 0.01, 0.0))
            for t in timestamps
        ]
        return timestamps

    return [(i + 0.5) * duration / num_frames for i in range(num_frames)]


def extract_frames(
    video_path: Path,
    timestamps: list[float],
    frame_width: int,
) -> list[str]:
    frames_b64 = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, ts in enumerate(timestamps):
            frame_path = os.path.join(tmpdir, f"frame_{i:04d}.jpg")

            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-y",
                    "-ss", str(ts),
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-vf", f"scale={frame_width}:-2",
                    "-q:v", "3",
                    frame_path,
                ],
                capture_output=True,
                check=True,
            )

            with open(frame_path, "rb") as f:
                frames_b64.append(base64.b64encode(f.read()).decode())

    return frames_b64


def sample_frames(
    video_path: Path,
    num_frames: int,
    frame_width: int = FRAME_WIDTH,
    *,
    sampling: str = "uniform",
    evidence_intervals: list[dict] | None = None,
    evidence_fraction: float = EVIDENCE_FRACTION,
) -> tuple[list[str], list[float]]:
    duration = _video_duration(video_path)

    timestamps = plan_timestamps(
        duration,
        num_frames,
        sampling=sampling,
        evidence_intervals=evidence_intervals,
        evidence_fraction=evidence_fraction,
    )

    frames_b64 = extract_frames(video_path, timestamps, frame_width)
    return frames_b64, timestamps


# -- Prompt construction ------------------------------------------------------

def build_instruction(
    question: str,
    choices: list[str],
    *,
    task: str = "qa",
) -> str:
    options_block = ""

    if choices:
        labels = "\n".join(
            f"{chr(65 + i)}. {choice}"
            for i, choice in enumerate(choices)
        )
        options_block = f"\n\nOptions:\n{labels}"

    if task == "qa":
        return f"""You are answering a multiple-choice question about a video.

You are given frames sampled from the video. Each image is preceded by a line "Frame at <t>s:" giving its approximate timestamp in seconds from the start of the video.

Choose the best answer based on the visual evidence in the frames.

Important rules:
- You MUST choose exactly one option.
- Use the timestamps when reporting evidence timing.

Return ONLY a valid JSON object and nothing else, using straight double quotes:
{{
  "answer": "the option letter, e.g. A",
  "evidence": "brief visual evidence supporting the answer",
  "evidence_seconds": [start_second, end_second] or null,
  "reason": "one short sentence"
}}

Question: {question}{options_block}"""

    if task == "classify":
        return f"""You are determining whether the sampled video frames contain sufficient visual evidence to answer a multiple-choice question.

You are given frames sampled from the video. Each image is preceded by a line "Frame at <t>s:" giving its approximate timestamp in seconds from the start of the video.

Your task is to answer the question if it is answerable, otherwise output UNANSWERABLE.

Important rules:
- If the frames contain sufficient evidence, choose the best-supported option.
- If the frames do not contain sufficient visual evidence to determine a unique answer, output UNANSWERABLE.
- Use the timestamps when reporting evidence timing.

Return ONLY a valid JSON object and nothing else, using straight double quotes:
{{
  "label": "ANSWERABLE | UNANSWERABLE",
  "answer": "the option letter if ANSWERABLE, otherwise null",
  "evidence": "brief visual evidence supporting your decision",
  "evidence_seconds": [start_second, end_second] or null,
  "reason": "one short sentence"
}}

Question: {question}{options_block}"""

    raise ValueError(f"Unknown task: {task}")


def print_prompt_preview(task: str) -> None:
    demo_question = "What is the main action shown in the video?"
    demo_choices = [
        "A person is cooking.",
        "A person is driving.",
        "A person is playing music.",
        "A person is reading.",
    ]

    prompt = build_instruction(
        demo_question,
        demo_choices,
        task=task,
    )

    print("\n" + "=" * 80)
    print(f"PROMPT PREVIEW task={task}")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n", flush=True)


# -- LLM call -----------------------------------------------------------------

def call_llm(
    frames_b64: list[str],
    timestamps: list[float],
    question: str,
    choices: list[str],
    *,
    task: str = "qa",
) -> str:
    content = []

    for b64, ts in zip(frames_b64, timestamps):
        content.append({
            "type": "text",
            "text": f"Frame at {ts:.1f}s:",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
            },
        })

    content.append({
        "type": "text",
        "text": build_instruction(question, choices, task=task),
    })

    resp = _client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )

    return resp.choices[0].message.content


# -- Output parsing -----------------------------------------------------------

def parse_model_json(raw: str) -> dict:
    out = {"raw": raw}

    if raw is None:
        out["parse_error"] = "empty response"
        return out

    cleaned = raw.strip()
    cleaned = re.sub(
        r"^```(?:json)?|```$",
        "",
        cleaned,
        flags=re.MULTILINE,
    ).strip()

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    candidate_json = match.group(0) if match else cleaned

    try:
        parsed = json.loads(candidate_json)

        if isinstance(parsed, dict):
            out.update(parsed)
        else:
            out["parse_error"] = "JSON was not an object"

    except json.JSONDecodeError as e:
        out["parse_error"] = f"invalid JSON: {e}"

    return out


# -- Per-entry worker ---------------------------------------------------------

def process_entry(
    idx: int,
    entry: dict,
    video_dir: Path,
    *,
    insufficient: bool,
    num_frames: int,
    frame_width: int,
    sampling: str,
    evidence_fraction: float,
    task: str,
    model_tag: str,
    video_id_mapping: dict | None = None,
) -> dict:
    video_id = entry["video_id"]
    qid = entry["qid"]
    alpha = entry.get("alpha")  # only present on partial(C2) expanded entries

    if alpha is not None:
        partial_mode = entry.get("partial_mode", DEFAULT_PARTIAL_MODE)
        video_file = video_dir / f"{video_id}_q{qid}_c2_{partial_mode}_{alpha_tag(alpha)}.mp4"
    elif insufficient:
        video_file = video_dir / f"{video_id}_q{qid}_freeze.mp4"
    elif entry.get("source_dataset") == "nextgqa":
        # NExT-GQA videos are stored under a {folder}/{vidorID}.mp4 layout
        # (see nextgqa_pipeline/README.md), not flat {video_id}.mp4 like
        # CG-Bench, so they need the official video_id -> path mapping.
        mapped = (video_id_mapping or {}).get(str(video_id))
        if mapped is None:
            return {
                "idx": idx,
                "status": "skip",
                "video_id": video_id,
                "qid": qid,
                "alpha": alpha,
                "reason": "missing_video_id_mapping",
            }
        video_file = video_dir / f"{mapped}.mp4"
    else:
        video_file = video_dir / f"{video_id}.mp4"

    if not video_file.exists():
        return {
            "idx": idx,
            "status": "skip",
            "video_id": video_id,
            "qid": qid,
            "alpha": alpha,
            "reason": "missing_video",
        }

    ev_intervals = entry.get("evidence_intervals") if not insufficient else None

    try:
        frames, timestamps = sample_frames(
            video_file,
            num_frames,
            frame_width,
            sampling=sampling,
            evidence_intervals=ev_intervals,
            evidence_fraction=evidence_fraction,
        )

        raw_response = call_llm(
            frames,
            timestamps,
            entry["question"],
            entry.get("choices", []),
            task=task,
        )

    except Exception as e:
        return {
            "idx": idx,
            "status": "error",
            "video_id": video_id,
            "qid": qid,
            "alpha": alpha,
            "reason": str(e),
        }

    parsed = parse_model_json(raw_response)

    if alpha is not None:
        # constant "partial" label per user's instruction, mirroring the
        # constant "insufficient" label below. The finer C1/C2/C3 tag for
        # a given alpha is still recoverable via condition_for_alpha(alpha)
        # if ever needed downstream -- it is deliberately NOT duplicated
        # into evidence_condition since alpha is already stored separately.
        evidence_condition = "partial"
    elif insufficient:
        evidence_condition = "insufficient"
    else:
        evidence_condition = entry.get("evidence_condition")

    result = {
        **entry,
        "task": task,
        "sampling": sampling,
        "model_tag": model_tag,
        "evidence_condition": evidence_condition,
        "sampled_seconds": [round(t, 2) for t in timestamps],
        "model_response": raw_response,
        "model_parsed": parsed,
    }

    if task == "qa":
        short_status = parsed.get("answer", "?")
    else:
        short_status = parsed.get("label", "?")

    return {
        "idx": idx,
        "status": "ok",
        "video_id": video_id,
        "qid": qid,
        "alpha": alpha,
        "result": result,
        "short_status": short_status,
    }


# -- Evaluation loop ----------------------------------------------------------

def append_jsonl(path: Path, obj: dict) -> None:
    """Append one JSON object to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _record_key(rec: dict):
    """(video_id, qid) for sufficient/hallucination/insufficient records;
    (video_id, qid, alpha) for partial(C2) records. A plain (video_id, qid)
    key would collapse every alpha level of the same anchor onto one jsonl
    line and silently drop the rest of the dose-response curve.

    NOTE: this key does NOT include model_tag. If you point two different
    models at the same --results-dir, the second model's run will see the
    first model's records as "already done" (same video_id/qid[/alpha]) and
    skip them rather than adding its own -- --model-tag alone does not make
    it safe to share --results-dir across models. Use separate
    --results-dir per model.
    """
    if rec.get("alpha") is not None:
        return (rec["video_id"], rec["qid"], rec["alpha"])
    return (rec["video_id"], rec["qid"])


def load_existing_keys(jsonl_path: Path) -> dict:
    """
    读取已有的 jsonl 文件，返回 {key: 该行原始 dict} 的映射，key 见
    _record_key()（非 partial 为 (video_id, qid)，partial 为
    (video_id, qid, alpha)）。用于判断某个 entry 是否已经跑过。
    如果文件不存在，返回空字典。
    如果某一行 JSON 解析失败，跳过该行但不删除、不修改原文件。
    """
    existing = {}

    if not jsonl_path.exists():
        return existing

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARN] {jsonl_path.name} line {line_no} is not valid JSON, "
                    f"skipped (file left unmodified).",
                    flush=True,
                )
                continue

            if "video_id" not in rec or "qid" not in rec:
                print(
                    f"[WARN] {jsonl_path.name} line {line_no} missing "
                    f"'video_id' or 'qid' key, skipped.",
                    flush=True,
                )
                continue

            existing[_record_key(rec)] = rec

    return existing


def run_evaluation(
    entries: list[dict],
    video_dir: Path,
    output_path: Path,
    *,
    model_tag: str,
    insufficient: bool = False,
    num_frames: int = NUM_FRAMES,
    frame_width: int = FRAME_WIDTH,
    sampling: str = "uniform",
    evidence_fraction: float = EVIDENCE_FRACTION,
    workers: int = WORKERS,
    task: str = "qa",
    video_id_mapping: dict | None = None,
) -> None:
    total = len(entries)
    done = 0
    print_lock = threading.Lock()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_path.with_suffix(".jsonl")
    skipped_jsonl_path = output_path.with_suffix(".skipped.jsonl")

    # -- 读取已有结果，按 key 建立索引，用于跳过已完成的条目 ------------------
    existing_by_key = load_existing_keys(jsonl_path)

    # skipped_jsonl_path 每次运行仍然清空重建，只记录本次运行新产生的 skip/error。
    if skipped_jsonl_path.exists():
        skipped_jsonl_path.unlink()

    results_by_idx: dict[int, dict] = {}
    skipped = []

    to_submit: dict[int, dict] = {}
    already_done = 0

    for idx, entry in enumerate(entries):
        key = _record_key(entry)
        if key in existing_by_key:
            results_by_idx[idx] = existing_by_key[key]
            already_done += 1
        else:
            to_submit[idx] = entry

    with print_lock:
        print(
            f"[RESUME] total={total} already_in_{jsonl_path.name}={already_done} "
            f"to_run={len(to_submit)}",
            flush=True,
        )

    run_total = len(to_submit)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_entry,
                idx,
                entry,
                video_dir,
                insufficient=insufficient,
                num_frames=num_frames,
                frame_width=frame_width,
                sampling=sampling,
                evidence_fraction=evidence_fraction,
                task=task,
                model_tag=model_tag,
                video_id_mapping=video_id_mapping,
            ): idx
            for idx, entry in to_submit.items()
        }

        for fut in as_completed(futures):
            out = fut.result()
            done += 1

            vid = out["video_id"]
            qid = out["qid"]
            alpha = out.get("alpha")
            alpha_suffix = f" alpha={alpha}" if alpha is not None else ""

            if out["status"] == "ok":
                result = out["result"]
                results_by_idx[out["idx"]] = result

                append_jsonl(jsonl_path, result)

                msg = (
                    f"[{done}/{run_total}] OK    {vid} qid={qid}{alpha_suffix}  "
                    f"{task}={out['short_status']}  "
                    f"saved_to={jsonl_path.name}"
                )

            elif out["status"] == "skip":
                skip_obj = {
                    "idx": out["idx"],
                    "video_id": vid,
                    "qid": qid,
                    "alpha": alpha,
                    "reason": out["reason"],
                }

                skipped.append(skip_obj)
                append_jsonl(skipped_jsonl_path, skip_obj)

                msg = f"[{done}/{run_total}] SKIP  missing video: {vid} qid={qid}{alpha_suffix}"

            else:
                skip_obj = {
                    "idx": out["idx"],
                    "video_id": vid,
                    "qid": qid,
                    "alpha": alpha,
                    "reason": out["reason"],
                }

                skipped.append(skip_obj)
                append_jsonl(skipped_jsonl_path, skip_obj)

                msg = f"[{done}/{run_total}] ERROR {vid} qid={qid}{alpha_suffix}: {out['reason']}"

            with print_lock:
                print(msg, flush=True)

    results = [results_by_idx[i] for i in sorted(results_by_idx)]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nIncremental results saved during run -> {jsonl_path}")
    print(f"Final ordered JSON saved -> {output_path}")

    if skipped:
        skip_path = output_path.with_suffix(".skipped.json")

        with open(skip_path, "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)

        print(f"Incremental skipped/errors -> {skipped_jsonl_path}")
        print(f"Final skipped/errors -> {skip_path}")


# -- Entry point --------------------------------------------------------------

MODES = ("sufficient", "hallucination", "insufficient", "partial", "all")
TASKS = ("qa", "classify")


def main():
    parser = argparse.ArgumentParser(
        description="CG-Bench evaluation pipeline"
    )

    parser.add_argument("--mode", choices=MODES, default="all")

    parser.add_argument(
        "--filtered-json", type=Path, default=FILTERED_JSON,
        help="Filtered metadata JSON to evaluate (default: CG-Bench's "
             f"{FILTERED_JSON}). Pass nextgqa_pipeline/nextgqa_filtered.json "
             "for a NExT-GQA run.",
    )
    parser.add_argument(
        "--video-dir", type=Path, default=VIDEOS_DIR,
        help="Directory containing sufficient-condition videos (default: "
             f"CG-Bench's {VIDEOS_DIR}).",
    )
    parser.add_argument(
        "--insufficient-video-dir", type=Path, default=INSUFFICIENT_DIR,
        help="Directory containing insufficient/freeze-condition videos, "
             "i.e. {video_id}_q{qid}_freeze.mp4 files built by build_freeze.py "
             f"(default: CG-Bench's {INSUFFICIENT_DIR}). Pass "
             "source_datasets/next_gqa/freeze_videos for a NExT-GQA run. "
             "This naming is flat regardless of source_dataset, so no "
             "--video-id-mapping lookup is needed here.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Where result .json/.jsonl files are written (default: "
             f"CG-Bench's {RESULTS_DIR}). Use a separate directory per model "
             "(e.g. cgbench_result_internvl3_5) -- see the module docstring's "
             "'Multi-model tagging' note: --model-tag alone does not make it "
             "safe to share --results-dir across models.",
    )
    parser.add_argument(
        "--video-id-mapping", type=Path, default=NEXTGQA_MAPPING_JSON,
        help="video_id -> relative-path mapping, only consulted for entries "
             "with source_dataset == 'nextgqa' (default: NExT-GQA's own "
             "map_vid_vidorID.json). Irrelevant for CG-Bench runs.",
    )

    parser.add_argument(
        "--task",
        choices=TASKS,
        default="qa",
        help=(
            "qa = pure multiple-choice answering; "
            "classify = answerability classification"
        ),
    )

    parser.add_argument(
        "--model-tag",
        required=True,
        help=(
            "Identifier written into every output record's 'model_tag' "
            "field (e.g. 'qwen3vl', 'internvl3_5'). Required -- no default "
            "-- so a forgotten --model-tag fails as a CLI error instead of "
            "silently tagging results ambiguously. Does NOT make it safe to "
            "point two different models at the same --results-dir (see "
            "module docstring); use a separate --results-dir per model."
        ),
    )

    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--frame-width", type=int, default=FRAME_WIDTH)

    parser.add_argument(
        "--sampling",
        choices=("uniform", "evidence"),
        default="uniform",
    )

    parser.add_argument(
        "--evidence-fraction",
        type=float,
        default=EVIDENCE_FRACTION,
    )

    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--alphas",
        default=DEFAULT_ALPHAS,
        help=(
            "Comma-separated alpha levels for --mode partial. "
            f"Default: {DEFAULT_ALPHAS}"
        ),
    )
    parser.add_argument(
        "--partial-mode",
        choices=("noise", "blur"),
        default=DEFAULT_PARTIAL_MODE,
        help="Degradation operator used when the partial videos were built "
             "(must match how build_partial.py named the files).",
    )
    parser.add_argument(
        "--partial-candidates",
        type=Path,
        default=PARTIAL_CANDIDATES_JSON,
        help="JSON array of {video_id, qid} pairs selecting which anchors "
             "to evaluate under --mode partial.",
    )

    args = parser.parse_args()

    try:
        _client.models.list()
        print(f"Connected to vLLM at {VLLM_BASE_URL} model={MODEL_NAME} model_tag={args.model_tag}\n")

    except Exception as e:
        raise SystemExit(
            f"Cannot reach vLLM server at {VLLM_BASE_URL}: {e}\n"
            f"Check VLLM_BASE_URL / VLLM_MODEL / server job."
        )

    data = json.load(open(args.filtered_json, encoding="utf-8"))
    video_id_mapping = load_video_id_mapping(args.video_id_mapping)

    sufficient = [
        d for d in data
        if d["evidence_condition"] == "sufficient"
    ]

    hallucination = [
        d for d in data
        if d["evidence_condition"] == "Hallucination"
    ]

    if args.limit is not None:
        sufficient = sufficient[:args.limit]
        hallucination = hallucination[:args.limit]
        print(f"*** DEBUG MODE: first {args.limit} entries per condition ***")

    print(
        f"Loaded {len(data)} entries from {args.filtered_json} "
        f"(sufficient={len(sufficient)}, hallucination={len(hallucination)})"
    )

    print(
        f"task={args.task}  model_tag={args.model_tag}  "
        f"num_frames={args.num_frames}  "
        f"frame_width={args.frame_width}  "
        f"sampling={args.sampling}  "
        f"workers={args.workers}\n"
        f"video_dir={args.video_dir}  results_dir={args.results_dir}\n"
    )

    print_prompt_preview(args.task)

    suffix = f".{args.task}.json"

    if args.mode in ("sufficient", "all"):
        print("=== sufficient ===")
        run_evaluation(
            sufficient,
            args.video_dir,
            args.results_dir / f"sufficient{suffix}",
            model_tag=args.model_tag,
            num_frames=args.num_frames,
            frame_width=args.frame_width,
            sampling=args.sampling,
            evidence_fraction=args.evidence_fraction,
            workers=args.workers,
            task=args.task,
            video_id_mapping=video_id_mapping,
        )

    if args.mode in ("hallucination", "all"):
        print("\n=== hallucination ===")
        run_evaluation(
            hallucination,
            args.video_dir,
            args.results_dir / f"hallucination{suffix}",
            model_tag=args.model_tag,
            num_frames=args.num_frames,
            frame_width=args.frame_width,
            sampling=args.sampling,
            evidence_fraction=args.evidence_fraction,
            workers=args.workers,
            task=args.task,
            video_id_mapping=video_id_mapping,
        )

    if args.mode in ("insufficient", "all"):
        print("\n=== insufficient ===")
        run_evaluation(
            sufficient,
            args.insufficient_video_dir,
            args.results_dir / f"insufficient{suffix}",
            model_tag=args.model_tag,
            insufficient=True,
            num_frames=args.num_frames,
            frame_width=args.frame_width,
            sampling=args.sampling,
            evidence_fraction=args.evidence_fraction,
            workers=args.workers,
            task=args.task,
            video_id_mapping=video_id_mapping,
        )

    if args.mode in ("partial", "all"):
        print("\n=== partial (C2) ===")
        alphas = parse_alphas(args.alphas)
        partial_entries = load_partial_entries(args.filtered_json, args.partial_candidates)

        if args.limit is not None:
            partial_entries = partial_entries[:args.limit]
            print(f"*** DEBUG MODE: first {args.limit} partial anchors ***")

        partial_tasks = expand_with_alphas(partial_entries, alphas, args.partial_mode)

        print(
            f"partial anchors={len(partial_entries)} alphas={alphas} "
            f"partial_mode={args.partial_mode} -> total_tasks={len(partial_tasks)}"
        )

        run_evaluation(
            partial_tasks,
            PARTIAL_VIDEOS_DIR,
            args.results_dir / f"partial{suffix}",
            model_tag=args.model_tag,
            num_frames=args.num_frames,
            frame_width=args.frame_width,
            sampling=args.sampling,
            evidence_fraction=args.evidence_fraction,
            workers=args.workers,
            task=args.task,
            video_id_mapping=video_id_mapping,
        )


if __name__ == "__main__":
    main()