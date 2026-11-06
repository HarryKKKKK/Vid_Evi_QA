#!/usr/bin/env python3
"""Run a matched risk-threshold diagnostic for Qwen2.5-VL.

The primary pass keeps the output to one constrained character.  Forced
choice (A--E) measures answer ability/confidence; risk cells (A--E or R)
measure the answer/refuse policy.  Explanations are collected only in a
separate post-hoc audit so that they cannot change the primary decision.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
LETTERS = tuple("ABCDE")
RISK_LEVELS = (
    {"risk_id": "tau000", "s_cor": 1.0, "s_inc": 0.0, "s_ref": 0.0, "tau": 0.0},
    {"risk_id": "tau020", "s_cor": 1.0, "s_inc": -0.25, "s_ref": 0.0, "tau": 0.2},
    {"risk_id": "tau050", "s_cor": 1.0, "s_inc": -1.0, "s_ref": 0.0, "tau": 0.5},
    {"risk_id": "tau080", "s_cor": 1.0, "s_inc": -4.0, "s_ref": 0.0, "tau": 0.8},
    {"risk_id": "tau095", "s_cor": 1.0, "s_inc": -19.0, "s_ref": 0.0, "tau": 0.95},
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--model-tag", default="qwen25vl32b")
    p.add_argument("--source-video-dir", type=Path, default=ROOT / "source_datasets/next_gqa/videos")
    p.add_argument("--no-evidence-video-dir", type=Path, default=ROOT / "source_datasets/next_gqa/freeze_videos_no_evidence_leak")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--frame-width", type=int, default=560)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--evidence-conditions", default="full,no_evidence")
    p.add_argument("--risk-variants", default="0,1")
    p.add_argument("--audit-n", type=int, default=20)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--main-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def item_key(row: dict) -> str:
    return f"{row['video_id']}_q{row['qid']}"


def options(row: dict) -> str:
    if len(row["choices"]) != 5:
        raise ValueError(f"{item_key(row)} has {len(row['choices'])} choices, expected 5")
    return "\n".join(f"{letter}. {value}" for letter, value in zip(LETTERS, row["choices"]))


def forced_prompt(row: dict) -> str:
    return f"""Choose the single best answer using the supplied video frames.
You must answer even if uncertain. Output exactly one character: A, B, C, D, or E.

Question: {row['question']}

Options:
{options(row)}"""


def risk_prompt(row: dict, risk: dict, variant: int) -> str:
    score = (
        f"correct answer = {risk['s_cor']:g} points; incorrect answer = "
        f"{risk['s_inc']:g} points; refusal = {risk['s_ref']:g} points"
    )
    if variant == 0:
        lead = (
            "Maximize your expected score under this scoring rule: " + score + ". "
            "Answer with A, B, C, D, or E, or refuse with R."
        )
    elif variant == 1:
        lead = (
            "Select the action with the highest expected number of points. The payoffs are: "
            + score + ". Choose one option letter A-E, or choose R to decline answering."
        )
    else:
        raise ValueError(f"unsupported prompt variant: {variant}")
    return f"""{lead}
Return exactly one character and no explanation.

Question: {row['question']}

Options:
{options(row)}"""


def source_video(row: dict, a: argparse.Namespace, evidence: str) -> Path:
    if evidence == "no_evidence":
        path = a.no_evidence_video_dir / f"{row['video_id']}_q{row['qid']}_freeze.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    if evidence != "full":
        raise ValueError(f"unsupported evidence condition: {evidence}")
    matches = list(a.source_video_dir.rglob(f"{row['video_id']}.*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one source video for {row['video_id']}, found {len(matches)}")
    return matches[0]


def extract_frames(row: dict, evidence: str, a: argparse.Namespace) -> list[str]:
    timestamps = row.get("sampled_seconds")
    if not isinstance(timestamps, list) or len(timestamps) != 32:
        raise ValueError(f"{item_key(row)} does not contain exactly 32 sampled_seconds")
    video = source_video(row, a, evidence)
    encoded: list[str] = []
    with tempfile.TemporaryDirectory(prefix="qwen25_risk_") as temp_dir:
        temp = Path(temp_dir)
        for index, second in enumerate(timestamps):
            image = temp / f"{index:03d}.jpg"
            subprocess.run(
                [a.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-ss", str(second),
                 "-i", str(video), "-frames:v", "1", "-vf", f"scale={a.frame_width}:-2",
                 "-q:v", "3", str(image)],
                check=True,
            )
            encoded.append(base64.b64encode(image.read_bytes()).decode("ascii"))
    return encoded


def multimodal(frames: list[str], timestamps: list[float], prompt: str) -> list[dict]:
    content: list[dict] = []
    for frame, second in zip(frames, timestamps):
        content.append({"type": "text", "text": f"Frame at {float(second):.2f}s:"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + frame}})
    content.append({"type": "text", "text": prompt})
    return content


def constrained_call(client: OpenAI, a: argparse.Namespace, content: list[dict], allowed: tuple[str, ...]) -> dict:
    response = client.chat.completions.create(
        model=a.model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=max(10, len(allowed)),
        extra_body={
            "structured_outputs": {"choice": list(allowed), "disable_fallback": True},
            "return_token_ids": True,
        },
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if text not in allowed:
        raise ValueError(f"constrained output {text!r} is not in {allowed}")
    dumped = choice.logprobs.model_dump() if choice.logprobs else None
    if not dumped or len(dumped.get("content") or []) != 1:
        raise ValueError(f"expected exactly one generated token with logprobs, got {dumped!r}")
    top = dumped["content"][0].get("top_logprobs") or []
    probability = {}
    for value in top:
        token = str(value.get("token", "")).strip()
        if token in allowed:
            probability[token] = math.exp(float(value["logprob"]))
    missing = sorted(set(allowed) - set(probability))
    if missing:
        raise ValueError(f"top_logprobs omitted constrained choices: {missing}; payload={top!r}")
    total = sum(probability.values())
    normalized = {token: probability[token] / total for token in allowed}
    return {
        "decision": text,
        "choice_probabilities": normalized,
        "logprobs": dumped,
        "usage": response.usage.model_dump() if response.usage else None,
    }


def explanation_call(client: OpenAI, a: argparse.Namespace, content: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=a.model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return {"text": text, "parsed": parsed, "usage": response.usage.model_dump() if response.usage else None}


def load_completed(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            completed.add(str(row["key"]))
    return completed


def main_records(row: dict, evidence: str, a: argparse.Namespace, completed: set[str], variants: tuple[int, ...]) -> list[dict]:
    prefix = f"{a.model_tag}::{item_key(row)}::{evidence}"
    specifications: list[tuple[str, str, tuple[str, ...], dict]] = [
        (f"{prefix}::forced", forced_prompt(row), LETTERS, {"condition": "forced"})
    ]
    for risk in RISK_LEVELS:
        for variant in variants:
            specifications.append((
                f"{prefix}::risk::{risk['risk_id']}::v{variant}",
                risk_prompt(row, risk, variant),
                LETTERS + ("R",),
                {"condition": "risk", "prompt_variant": variant, **risk},
            ))
    pending = [spec for spec in specifications if spec[0] not in completed]
    if not pending:
        return []
    frames = extract_frames(row, evidence, a)
    client = OpenAI(base_url=a.base_url, api_key="EMPTY", max_retries=0, timeout=a.timeout)
    records = []
    common = {
        "model_tag": a.model_tag,
        "video_id": str(row["video_id"]),
        "qid": str(row["qid"]),
        "question_type": row.get("question_type"),
        "experiment_group": row.get("experiment_group"),
        "gold_answer": str(row["answer"]).upper(),
        "evidence_condition": evidence,
        "sampled_seconds": row["sampled_seconds"],
    }
    for key, prompt, allowed, metadata in pending:
        try:
            result = constrained_call(client, a, multimodal(frames, row["sampled_seconds"], prompt), allowed)
            records.append({"key": key, **common, **metadata, "status": "ok", "response": result})
        except Exception as exc:
            records.append({"key": key, **common, **metadata, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return records


def append_records(path: Path, rows: list[dict], lock: threading.Lock) -> None:
    if not rows:
        return
    with lock, path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def run_main(rows: list[dict], a: argparse.Namespace, evidence_conditions: tuple[str, ...], variants: tuple[int, ...]) -> None:
    completed = load_completed(a.output)
    tasks = [(row, evidence) for row in rows for evidence in evidence_conditions]
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(main_records, row, evidence, a, completed, variants): (row, evidence) for row, evidence in tasks}
        for future in as_completed(futures):
            row, evidence = futures[future]
            try:
                records = future.result()
            except Exception as exc:
                key = f"{a.model_tag}::{item_key(row)}::{evidence}::TASK_ERROR"
                records = [{"key": key, "model_tag": a.model_tag, "video_id": str(row["video_id"]),
                            "qid": str(row["qid"]), "evidence_condition": evidence,
                            "status": "error", "error": f"{type(exc).__name__}: {exc}"}]
            append_records(a.output, records, lock)
            print(item_key(row), evidence, f"records={len(records)}", flush=True)


def latest_ok_rows(path: Path) -> dict[str, dict]:
    latest = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("status") == "ok":
                    latest[str(row["key"])] = row
    return latest


def select_audit(rows: list[dict], a: argparse.Namespace, n: int) -> list[tuple[dict, dict, dict]]:
    results = latest_ok_rows(a.output)
    selected = []
    for row in rows:
        prefix = f"{a.model_tag}::{item_key(row)}::full"
        forced = results.get(f"{prefix}::forced")
        if not forced or forced["response"]["decision"] != str(row["answer"]).upper():
            continue
        refusal = None
        for risk_id in ("tau000", "tau020", "tau050", "tau080", "tau095"):
            candidate = results.get(f"{prefix}::risk::{risk_id}::v0")
            if candidate and candidate["response"]["decision"] == "R":
                refusal = candidate
                break
        if refusal:
            selected.append((row, forced, refusal))
        if len(selected) >= n:
            break
    return selected


def run_audit(rows: list[dict], a: argparse.Namespace, n: int) -> None:
    if n <= 0:
        return
    selected = select_audit(rows, a, n)
    completed = load_completed(a.output)
    lock = threading.Lock()
    for row, forced, refusal in selected:
        for evidence in ("full", "no_evidence"):
            key = f"{a.model_tag}::{item_key(row)}::{evidence}::reasoning_audit::{refusal['risk_id']}"
            if key in completed:
                continue
            frames = extract_frames(row, evidence, a)
            client = OpenAI(base_url=a.base_url, api_key="EMPTY", max_retries=0, timeout=a.timeout)
            prompt = f"""This is a post-hoc decision audit. Give a concise report of the evidence and decision criteria; do not claim to reveal hidden chain-of-thought.

The forced-answer pass selected {forced['response']['decision']}. Under scoring correct={refusal['s_cor']:g}, incorrect={refusal['s_inc']:g}, refusal={refusal['s_ref']:g}, the risk pass refused.

Return one JSON object with exactly these fields:
{{
  "visible_evidence": "facts visible in the supplied frames",
  "missing_or_ambiguous_evidence": "what prevents certainty, or none",
  "best_answer": "A-E",
  "confidence_percent": 0,
  "would_answer_under_this_scoring": true,
  "decision_basis": "brief explanation",
  "used_elimination": true
}}

Question: {row['question']}

Options:
{options(row)}"""
            try:
                response = explanation_call(client, a, multimodal(frames, row["sampled_seconds"], prompt))
                record = {"key": key, "model_tag": a.model_tag, "video_id": str(row["video_id"]),
                          "qid": str(row["qid"]), "question_type": row.get("question_type"),
                          "experiment_group": row.get("experiment_group"), "gold_answer": str(row["answer"]).upper(),
                          "evidence_condition": evidence, "condition": "reasoning_audit",
                          "audit_trigger_risk_id": refusal["risk_id"], "forced_decision": forced["response"]["decision"],
                          "status": "ok", "response": response}
            except Exception as exc:
                record = {"key": key, "model_tag": a.model_tag, "video_id": str(row["video_id"]),
                          "qid": str(row["qid"]), "evidence_condition": evidence,
                          "condition": "reasoning_audit", "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            append_records(a.output, [record], lock)
            print(item_key(row), evidence, "reasoning_audit", record["status"], flush=True)


def write_summary(path: Path, output: Path) -> None:
    rows = list(latest_ok_rows(output).values())
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        if row.get("condition") not in {"forced", "risk"}:
            continue
        group = (row.get("evidence_condition"), row.get("condition"), row.get("risk_id"), row.get("prompt_variant"))
        groups.setdefault(group, []).append(row)
    table = []
    for group, values in sorted(groups.items(), key=lambda x: str(x[0])):
        answered = [r for r in values if r["response"]["decision"] != "R"]
        correct = [r for r in answered if r["response"]["decision"] == r["gold_answer"]]
        entry = {
            "evidence_condition": group[0], "condition": group[1], "risk_id": group[2],
            "prompt_variant": group[3], "n": len(values), "answered": len(answered),
            "correct": len(correct), "coverage": len(answered) / len(values),
            "overall_accuracy": len(correct) / len(values),
            "selective_accuracy": len(correct) / len(answered) if answered else None,
        }
        if group[1] == "risk":
            first = values[0]
            score = sum(first["s_ref"] if r["response"]["decision"] == "R" else
                        first["s_cor"] if r["response"]["decision"] == r["gold_answer"] else first["s_inc"]
                        for r in values)
            entry.update({k: first[k] for k in ("tau", "s_cor", "s_inc", "s_ref")})
            entry["mean_utility"] = score / len(values)
        table.append(entry)
    payload = {
        "source": str(output),
        "record_count": len(rows),
        "error_count": sum(1 for line in output.read_text(encoding="utf-8").splitlines()
                           if line.strip() and json.loads(line).get("status") == "error") if output.is_file() else 0,
        "risk_levels": RISK_LEVELS,
        "metrics": table,
        "reasoning_audit_count": sum(r.get("condition") == "reasoning_audit" for r in rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    a = parse_args()
    if a.limit <= 0 or a.workers <= 0:
        raise SystemExit("--limit and --workers must be positive")
    evidence = tuple(value.strip() for value in a.evidence_conditions.split(",") if value.strip())
    if not evidence or set(evidence) - {"full", "no_evidence"}:
        raise SystemExit(f"invalid --evidence-conditions={a.evidence_conditions!r}")
    variants = tuple(int(value) for value in a.risk_variants.split(",") if value.strip())
    if not variants or set(variants) - {0, 1}:
        raise SystemExit(f"invalid --risk-variants={a.risk_variants!r}")
    rows = json.loads(a.sample.read_text(encoding="utf-8"))[:a.limit]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if not a.audit_only:
        run_main(rows, a, evidence, variants)
    if not a.main_only:
        run_audit(rows, a, a.audit_n)
    write_summary(a.summary, a.output)
    print(json.dumps({"output": str(a.output), "summary": str(a.summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
