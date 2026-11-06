#!/usr/bin/env python3
"""GPT adjudication of visually reviewed conflict QA."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


SYSTEM = """You are the final quality controller for an evidence-conflict video-QA benchmark.
The candidate has already been checked by two independent vision-language models. You must decide whether it is publication-ready.

KEEP only if ALL conditions hold:
1. The question is grammatical, unambiguous except for its intentionally unspecified occurrence, and every option answers exactly the same semantic slot.
2. At least two options give meaningfully different, occurrence-dependent answers. They need not be maximally contradictory; modest differences in scene detail are acceptable when they distinguish occurrences and remain useful answers. Exact paraphrases or effectively identical answers are not acceptable.
3. Every option has concrete support from at least one supplied visual-audit report. Disagreement from a weaker visual model is a warning, not an automatic failure.
4. For surrounding-event questions, each answer really describes the requested immediately-before or immediately-after event, not the core action or a distant event.
5. For temporal-localization questions, intervals represent distinguishable occurrences. Small boundary overlap is a warning but not an automatic failure.
6. The item is useful as a multiple-valid-answer conflict QA without requiring unstated assumptions.

Choose DROP for clear QA defects, unsupported options, essentially duplicate answers, wrong temporal direction, or unusable ambiguity. Do not drop a usable QA merely because its conflict is not maximally strong.
Return exactly one JSON object with keys:
decision: KEEP or DROP
confidence: number from 0 to 1
failure_types: array chosen from [question_option_mismatch, compatible_answers, paraphrase_or_duplicate, subset_superset, evidence_mismatch, occurrence_mismatch, temporal_direction_error, overlapping_intervals, ambiguous_or_weak, unverifiable, other]
reason: concise English explanation."""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--pre-gpt-rejected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api3.xhub.chat/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--keep-confidence", type=float, default=0.65)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args()


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        raise ValueError("response has no JSON object")
    value = json.loads(match.group(0))
    decision = str(value.get("decision", "")).upper()
    if decision not in {"KEEP", "DROP"}:
        raise ValueError("invalid decision")
    value["decision"] = decision
    value["confidence"] = float(value.get("confidence", 0))
    return value


def compact_candidate(item: dict, permutation: int) -> dict:
    qa = item["qa"]
    options = list(qa.get("options", []))
    if permutation:
        random.Random(f"{qa['id']}:{permutation}:20260829").shuffle(options)
    visual = {}
    for model, report in item.get("visual_audits", {}).items():
        if not report:
            visual[model] = {"status": "missing"}
            continue
        visual[model] = {
            "predicted_label": report.get("predicted_label"),
            "predicted_option_ids": report.get("predicted_option_ids"),
            "option_set_exact": report.get("option_set_exact"),
            "parsed_response": report.get("parsed_response"),
        }
    text = item.get("text_audit") or {}
    return {
        "id": qa["id"],
        "official_action": qa.get("official_action"),
        "conflict_type": qa.get("conflict_type"),
        "grounding_field": qa.get("grounding_field"),
        "question": qa.get("question"),
        "options": options,
        "text_audit": text.get("judgment"),
        "independent_visual_audits": visual,
        "review_flags": item.get("review_flags", []),
        "visually_supported_option_ids": item.get("visually_supported_option_ids", []),
    }


def main() -> None:
    args = arguments()
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    pre_rejected = json.loads(args.pre_gpt_rejected.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "gpt_judgments.jsonl"
    latest = {}
    if raw_path.is_file():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                latest[str(row["decision_key"])] = row
    completed = {key for key, row in latest.items() if row.get("status") == "ok"}
    tasks = [
        (item, permutation)
        for item in candidates
        for permutation in range(args.permutations)
        if f"{item['id']}::p{permutation}" not in completed
    ]
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=180, max_retries=0)
    lock = threading.Lock()

    def judge(task):
        item, permutation = task
        decision_key = f"{item['id']}::p{permutation}"
        payload = compact_candidate(item, permutation)
        error = None
        for attempt in range(args.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0,
                    max_tokens=384,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                return {
                    "decision_key": decision_key, "id": item["id"], "permutation": permutation,
                    "model": args.model, "status": "ok", "judgment": parse_json(raw), "raw_response": raw,
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(min(2 ** attempt, 15))
        return {"decision_key": decision_key, "id": item["id"], "permutation": permutation,
                "model": args.model, "status": "error", "error": error}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(judge, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            latest[row["decision_key"]] = row
            with lock, raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(tasks):
                print(f"[{index}/{len(tasks)}]", flush=True)

    candidate_by_id = {str(item["id"]): item for item in candidates}
    decisions, kept, gpt_dropped = [], [], []
    for item_id, item in candidate_by_id.items():
        rows = [latest.get(f"{item_id}::p{index}") for index in range(args.permutations)]
        keep = all(
            row and row.get("status") == "ok"
            and row["judgment"]["decision"] == "KEEP"
            and row["judgment"]["confidence"] >= args.keep_confidence
            for row in rows
        )
        decision = {"id": item_id, "final_keep": keep, "gpt_passes": rows}
        decisions.append(decision)
        if keep:
            kept.append(item["qa"])
        else:
            gpt_dropped.append({"qa": item["qa"], "final_decision": decision})

    with (args.output_dir / "final_conflict_qa.jsonl").open("w", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "final_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "gpt_dropped.json").write_text(json.dumps(gpt_dropped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "all_rejected.json").write_text(json.dumps({
        "pre_gpt": pre_rejected, "gpt": gpt_dropped
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failure_counts = Counter(
        failure
        for row in gpt_dropped
        for result in row["final_decision"]["gpt_passes"]
        if result and result.get("status") == "ok"
        for failure in result["judgment"].get("failure_types", [])
    )
    summary = {
        "original": len(candidates) + len(pre_rejected),
        "pre_gpt_candidates": len(candidates),
        "pre_gpt_rejected": len(pre_rejected),
        "final_kept": len(kept),
        "gpt_rejected": len(gpt_dropped),
        "final_keep_rate": len(kept) / (len(candidates) + len(pre_rejected)) if candidates or pre_rejected else 0,
        "gpt_failure_type_counts": dict(failure_counts.most_common()),
        "policy": f"keep only when {args.permutations}/{args.permutations} GPT passes say KEEP with confidence >= {args.keep_confidence}",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
