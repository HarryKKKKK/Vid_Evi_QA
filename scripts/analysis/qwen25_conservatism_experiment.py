#!/usr/bin/env python3
"""Build and run a matched Qwen2.5-VL abstention-mechanism experiment."""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import subprocess
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("original", "diagnostic", "calibrated", "forced_choice", "answer_first", "evidence_first", "aeu_logprobs")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--qwen-sufficient", type=Path, default=ROOT / "nextgqa_result_qwen25_32f_sufficient/sufficient.classify.json")
    p.add_argument("--qwen-insufficient", type=Path, default=ROOT / "nextgqa_result_qwen25_32f_insufficient/insufficient.classify.json")
    p.add_argument("--qwen3-sufficient", type=Path, default=ROOT / "nextgqa_result_qwen3vl_partial_no_evidence_leak/sufficient_alpha1.classify.json")
    p.add_argument("--llava-sufficient", type=Path, default=ROOT / "nextgqa_result_llava_onevision_32f_video_sufficient/sufficient.classify.json")
    p.add_argument("--intern-sufficient", type=Path, default=ROOT / "nextgqa_result_internvl35_32f_patch3_sufficient/sufficient.classify.json")
    p.add_argument("--primary-n", type=int, default=200)
    p.add_argument("--answered-control-n", type=int, default=50)
    p.add_argument("--insufficient-control-n", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--base-url", default="http://127.0.0.1:18025/v1")
    p.add_argument("--model", default="qwen2.5-vl-32b-conservatism")
    p.add_argument("--output", type=Path, default=ROOT / "ablation_results/qwen25_conservatism/results.jsonl")
    p.add_argument("--source-video-dir", type=Path, default=ROOT / "source_datasets/next_gqa/videos")
    p.add_argument("--insufficient-video-dir", type=Path, default=ROOT / "source_datasets/next_gqa/freeze_videos_no_evidence_leak")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--frame-width", type=int, default=560)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="Run only the first N prepared diagnostic rows.")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="Comma-separated subset of experiment conditions.",
    )
    p.add_argument("--experiment-tag", default="baseline")
    p.add_argument("--max-pixels", type=int)
    return p.parse_args()


def selected_conditions(spec: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in spec.split(",") if part.strip())
    unknown = sorted(set(values) - set(CONDITIONS))
    if not values or unknown:
        raise SystemExit(f"invalid --conditions={spec!r}; unknown={unknown}, allowed={CONDITIONS}")
    return values


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: dict) -> tuple[str, str]:
    return str(row["video_id"]), str(row["qid"])


def prediction(row: dict) -> tuple[str, str | None]:
    parsed = row.get("model_parsed") or {}
    label = str(parsed.get("label") or "").strip().upper()
    value = str(parsed.get("answer") or "").strip().upper()
    match = re.search(r"\b([A-E])\b", value)
    return label, match.group(1) if match else None


def correct(row: dict) -> bool:
    label, answer = prediction(row)
    return label == "ANSWERABLE" and answer == str(row.get("answer", "")).upper()


def refused(row: dict) -> bool:
    return prediction(row)[0] == "UNANSWERABLE"


def stratified_take(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("question_type") or "UNKNOWN")].append(row)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < n and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(selected) < n:
                selected.append(groups[name].pop())
    return selected


def prepare(a: argparse.Namespace) -> None:
    rng = random.Random(a.seed)
    qs = {key(r): r for r in load(a.qwen_sufficient)}
    qi = {key(r): r for r in load(a.qwen_insufficient)}
    others = [{key(r): r for r in load(path)} for path in (a.qwen3_sufficient, a.llava_sufficient, a.intern_sufficient)]
    common = set(qs).intersection(*(set(x) for x in others))
    primary = [qs[k] for k in common if refused(qs[k]) and sum(correct(x[k]) for x in others) >= 2]
    answered = [r for r in qs.values() if correct(r)]
    insuff = [r for r in qi.values() if refused(r)]
    chosen = []
    for group, rows, n, video_kind in (
        ("sufficient_refused_others_correct", primary, a.primary_n, "source"),
        ("sufficient_answered_correct_control", answered, a.answered_control_n, "source"),
        ("insufficient_refused_control", insuff, a.insufficient_control_n, "insufficient"),
    ):
        for row in stratified_take(rows, n, rng):
            chosen.append({
                "experiment_group": group, "video_kind": video_kind,
                "video_id": str(row["video_id"]), "qid": str(row["qid"]),
                "question_type": row.get("question_type"), "question": row["question"],
                "choices": row["choices"], "answer": row["answer"],
                "sampled_seconds": row["sampled_seconds"],
                "baseline_model_parsed": row.get("model_parsed"),
            })
    a.sample.parent.mkdir(parents=True, exist_ok=True)
    a.sample.write_text(json.dumps(chosen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sample": str(a.sample), "n": len(chosen), "groups": dict(Counter(x["experiment_group"] for x in chosen))}, indent=2))


def options(row: dict) -> str:
    return "\n".join(f"{chr(65+i)}. {v}" for i, v in enumerate(row["choices"]))


def selective_prompt(row: dict, *, calibrated: bool = False) -> str:
    """Return the canonical selective-answer schema used by the headline evaluation."""
    if not calibrated:
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

Question: {row['question']}

Options:
{options(row)}"""


def diagnostic_prompt(row: dict) -> str:
    """Single-pass, answer-first diagnostic with a bounded visible-evidence trace."""
    return f"""You are determining whether the sampled video frames contain sufficient visual evidence to answer a multiple-choice question.

You are given frames sampled from the video. Each image is preceded by a line "Frame at <t>s:" giving its approximate timestamp in seconds from the start of the video.

Make the final decision directly from the sampled video frames. Use only evidence that is directly visible. Do not use general knowledge, assumptions, or events that are merely likely.

Decision rules:
- If the frames provide sufficient visual evidence to determine exactly one option, set "label" to "ANSWERABLE" and "answer" to that option letter.
- If the relevant event is not visible, is ambiguous, or does not distinguish the options, set "label" to "UNANSWERABLE" and "answer" to null.
- State the decision fields before the evidence fields.
- Give at most 3 concise evidence observations. Include approximate timestamps when possible.
- Do not invent hidden events or unsupported details.

Return ONLY a valid JSON object and nothing else, using straight double quotes:
{{
  "label": "ANSWERABLE or UNANSWERABLE",
  "answer": "A-E if ANSWERABLE, otherwise null",
  "evidence": [
    {{"timestamp_seconds": 0.0, "observation": "one concise directly visible fact"}}
  ],
  "reason": "one concise evidence-based explanation of the decision"
}}

Question: {row['question']}

Options:
{options(row)}"""

    decision_rule = (
        "If meaningful visual evidence distinguishes one option from the others, choose the "
        "best-supported option. Output UNANSWERABLE only when the options cannot reasonably "
        "be distinguished from the sampled frames."
    )
    return f"""You are determining whether the sampled video frames contain sufficient visual evidence to answer a multiple-choice question.

You are given frames sampled from the video. Each image is preceded by a line "Frame at <t>s:" giving its approximate timestamp in seconds from the start of the video.

Your task is to answer the question if it is answerable, otherwise output UNANSWERABLE.

Important rules:
- {decision_rule}
- The "label" field MUST be exactly "ANSWERABLE" or "UNANSWERABLE".
- If label is "ANSWERABLE", "answer" MUST be exactly one option letter from A to E.
- If label is "UNANSWERABLE", "answer" MUST be null.
- Use the timestamps when reporting evidence timing.

Return ONLY a valid JSON object and nothing else, using straight double quotes:
{{
  "label": "ANSWERABLE or UNANSWERABLE",
  "answer": "A-E if ANSWERABLE, otherwise null",
  "evidence": "brief visual evidence supporting the decision",
  "evidence_seconds": [start_second, end_second] or null,
  "reason": "a concise evidence-based justification"
}}

Question: {row['question']}

Options:
{options(row)}"""


def forced_choice_prompt(row: dict) -> str:
    return f"""Choose the best-supported option from the sampled video frames.

Important rules:
- You MUST answer with exactly one option letter from A to E.
- Use only visible evidence in the sampled frames.
- Use timestamps when reporting evidence timing.

Return ONLY a valid JSON object and nothing else:
{{
  "answer": "A-E",
  "evidence": "brief visual evidence supporting the answer",
  "evidence_seconds": [start_second, end_second] or null,
  "reason": "a concise evidence-based justification"
}}

Question: {row['question']}

Options:
{options(row)}"""


def evidence_observation_prompt(row: dict) -> str:
    return f"""Inspect the sampled video frames and record only visible facts relevant to the question below.

Do not guess an answer. Do not decide whether the question is answerable. You are not given the answer choices. Do not infer events that are not visibly supported.

Return ONLY a valid JSON object:
{{
  "visual_observations": [
    {{"timestamp_seconds": 0.0, "observation": "one concise visible fact"}}
  ],
  "coverage_gaps": ["important event or detail not visible in the sampled frames"]
}}

Question: {row['question']}"""


def evidence_decision_prompt(row: dict, observation_text: str) -> str:
    return f"""Decide whether the multiple-choice question is answerable using ONLY the visual-observation record below.

Important rules:
- The "label" field MUST be exactly "ANSWERABLE" or "UNANSWERABLE".
- If label is "ANSWERABLE", "answer" MUST be exactly one option letter from A to E.
- If label is "UNANSWERABLE", "answer" MUST be null.
- Assess every option briefly, without inventing evidence.
- Provide a concise reasoning summary, not hidden chain-of-thought or unsupported speculation.

Return ONLY a valid JSON object:
{{
  "option_assessment": {{
    "A": "supported, contradicted, or not shown - brief reason",
    "B": "supported, contradicted, or not shown - brief reason",
    "C": "supported, contradicted, or not shown - brief reason",
    "D": "supported, contradicted, or not shown - brief reason",
    "E": "supported, contradicted, or not shown - brief reason"
  }},
  "label": "ANSWERABLE or UNANSWERABLE",
  "answer": "A-E if ANSWERABLE, otherwise null",
  "reasoning_summary": "concise evidence-based explanation of the decision"
}}

Visual-observation record:
{observation_text}

Question: {row['question']}

Options:
{options(row)}"""


def image_content(frames: list[str], times: list[float]) -> list[dict]:
    out = []
    for data, ts in zip(frames, times):
        out.extend(({"type": "text", "text": f"Frame at {ts:.1f}s:"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}))
    return out


def extract(row: dict, a: argparse.Namespace) -> list[str]:
    if row["video_kind"] == "insufficient":
        video = a.insufficient_video_dir / f"{row['video_id']}_q{row['qid']}_freeze.mp4"
    else:
        matches = list(a.source_video_dir.rglob(f"{row['video_id']}.*"))
        if not matches:
            raise FileNotFoundError(row["video_id"])
        video = matches[0]
    if not video.is_file():
        raise FileNotFoundError(video)
    frames = []
    with tempfile.TemporaryDirectory(prefix="qwen25_policy_") as tmp:
        for i, ts in enumerate(row["sampled_seconds"]):
            path = Path(tmp) / f"{i:03d}.jpg"
            subprocess.run([a.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-ss", str(ts), "-i", str(video), "-frames:v", "1", "-vf", f"scale={a.frame_width}:-2", "-q:v", "3", str(path)], check=True)
            frames.append(base64.b64encode(path.read_bytes()).decode())
    return frames


def call(client: OpenAI, model: str, content: list[dict] | str, *, max_tokens: int = 500, logprobs: bool = False):
    kw = dict(model=model, messages=[{"role": "user", "content": content}], temperature=0, max_tokens=max_tokens)
    if logprobs:
        kw.update(logprobs=True, top_logprobs=20)
    response = client.chat.completions.create(**kw)
    choice = response.choices[0]
    return {"text": choice.message.content, "logprobs": choice.logprobs.model_dump() if choice.logprobs else None, "usage": response.usage.model_dump() if response.usage else None}


def multimodal(frames: list[str], row: dict, prompt: str) -> list[dict]:
    return image_content(frames, row["sampled_seconds"]) + [{"type": "text", "text": prompt}]


def run_item(
    row: dict, a: argparse.Namespace, completed: set[str], conditions: tuple[str, ...]
) -> list[dict]:
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=a.timeout)
    item = f"{row['video_id']}_q{row['qid']}"
    pending = [c for c in conditions if f"{item}::{c}" not in completed]
    if not pending:
        return []
    frames = extract(row, a)
    opts = options(row)
    common = f"Question: {row['question']}\n\nOptions:\n{opts}"
    records = []
    forced = None
    if "forced_choice" in pending or "answer_first" in pending:
        forced = call(client, a.model, multimodal(frames, row, forced_choice_prompt(row)))
        if "forced_choice" in pending:
            records.append({"key": f"{item}::forced_choice", "condition": "forced_choice", "response": forced})
    prompts = {
        "original": selective_prompt(row),
        "diagnostic": diagnostic_prompt(row),
        "calibrated": selective_prompt(row, calibrated=True),
    }
    for condition in ("original", "diagnostic", "calibrated"):
        if condition in pending:
            records.append({"key": f"{item}::{condition}", "condition": condition, "response": call(client, a.model, multimodal(frames, row, prompts[condition]))})
    if "answer_first" in pending:
        provisional = (forced or {}).get("text")
        prompt = f"A first pass committed to this provisional answer: {provisional}\nNow independently judge whether the frames provide sufficient evidence. Return JSON with provisional_answer, label (ANSWERABLE or UNANSWERABLE), final_answer, evidence, evidence_seconds, reason.\n\n{common}"
        records.append({"key": f"{item}::answer_first", "condition": "answer_first", "provisional": forced, "response": call(client, a.model, multimodal(frames, row, prompt))})
    if "evidence_first" in pending:
        evidence = call(client, a.model, multimodal(frames, row, evidence_observation_prompt(row)))
        decision_prompt = evidence_decision_prompt(row, evidence["text"])
        records.append({"key": f"{item}::evidence_first", "condition": "evidence_first", "evidence_stage": evidence, "response": call(client, a.model, decision_prompt)})
    if "aeu_logprobs" in pending:
        prompt = "Output exactly one character: A-E for the best-supported answer, or U only if the options cannot reasonably be distinguished.\n\n" + common
        records.append({"key": f"{item}::aeu_logprobs", "condition": "aeu_logprobs", "response": call(client, a.model, multimodal(frames, row, prompt), max_tokens=1, logprobs=True)})
    return [{
        **{k: row[k] for k in ("experiment_group", "video_kind", "video_id", "qid", "question_type", "question", "choices", "answer")},
        "experiment_tag": a.experiment_tag,
        "max_pixels": a.max_pixels,
        **record,
        "status": "ok",
    } for record in records]


def main() -> None:
    a = args()
    conditions = selected_conditions(a.conditions)
    if a.prepare_only or not a.sample.exists():
        prepare(a)
        if a.prepare_only:
            return
    rows = json.loads(a.sample.read_text(encoding="utf-8"))
    if a.limit is not None:
        if a.limit <= 0:
            raise SystemExit("--limit must be a positive integer")
        rows = rows[:a.limit]
    completed = set()
    if a.output.exists():
        for line in a.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line); completed.add(str(value.get("key")))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_item, row, a, completed, conditions): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                records = future.result()
            except Exception as exc:
                records = [{"key": f"{row['video_id']}_q{row['qid']}::ITEM_ERROR", "video_id": row["video_id"], "qid": row["qid"], "status": "error", "error": f"{type(exc).__name__}: {exc}"}]
            with lock, a.output.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(row["video_id"], row["qid"], [r.get("condition", r["status"]) for r in records], flush=True)


if __name__ == "__main__":
    main()
