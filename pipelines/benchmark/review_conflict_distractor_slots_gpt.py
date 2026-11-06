#!/usr/bin/env python3
"""Text-review visually absent conflict distractors and emit slot-fit overrides."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/review_conflict_distractor_slots_gpt.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/review_conflict_distractor_slots_gpt.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


SYSTEM = """You review candidate distractors for a video-QA benchmark.
The candidates have already been checked against every relevant video interval and found NOT_SUPPORTED.
Your only task is to decide which candidates can grammatically and semantically answer exactly the
same question slot as the known valid answers.

Absence from the video is NOT a reason to reject a candidate; absence is required for a distractor.
Judge the answer slot from the question and valid-answer form, not from whether the candidate is related
to the official action. For example, any plausible physical object can fill "What is being dropped?",
and any grammatical event phrase can fill "What happens before/after ...?"; the event need not cause or
resemble the official action. Reject a candidate only if it has the wrong answer type or grammatical
form, is nonsensical, duplicates/paraphrases a valid answer, creates genuine ambiguity, or is a
meta-answer. Shared color, material, or broad object class alone is not duplication.
Select the requested number of strongest candidates, copying their text exactly.
The confidence is confidence in your slot-fit selection. If you select the requested number, use
confidence >= 0.8; otherwise select fewer candidates and use confidence < 0.8.
Return only JSON:
{"selected":["exact candidate text"],"confidence":0.0,"rejected_reasons":{"candidate":"reason"}}"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        raise ValueError("no JSON object")
    return json.loads(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unresolved", type=Path, required=True)
    parser.add_argument("--verification", type=Path, action="append", required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--output-verification", type=Path, required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api3.xhub.chat/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-visual-confidence", type=float, default=0.8)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    unresolved = json.loads(args.unresolved.read_text(encoding="utf-8"))
    by_id: dict[str, list[dict]] = {}
    for path in args.verification:
        for row in read_jsonl(path):
            if (str(row.get("verdict", "")).upper() == "NOT_SUPPORTED"
                    and float(row.get("confidence", 0)) >= args.min_visual_confidence):
                by_id.setdefault(row["id"], []).append(row)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=180, max_retries=0)
    lock = threading.Lock()
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        row["id"]: row for row in read_jsonl(args.output_review)
    } if args.output_review.exists() else {}

    def review(item: dict) -> dict:
        qa = item["qa"]
        valid = {str(option["answer"]).strip().casefold() for option in qa["options"]}
        seen = set()
        candidates = []
        source_rows = {}
        for row in sorted(by_id.get(item["id"], []),
                          key=lambda value: (-float(value["confidence"]), value["verification_key"])):
            text = str(row.get("candidate", "")).strip()
            key = text.casefold()
            if not text or key in valid or key in seen:
                continue
            seen.add(key)
            candidates.append(text)
            source_rows[key] = row
        payload = {
            "question": qa["question"],
            "official_action": qa.get("official_action"),
            "conflict_type": qa["conflict_type"],
            "semantic_field": qa.get("grounding_field"),
            "known_valid_answers": [option["answer"] for option in qa["options"]],
            "needed": int(item["needed"]),
            "visually_not_supported_candidates": candidates,
        }
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
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                parsed = parse_json(raw)
                selected = []
                for value in parsed.get("selected", []):
                    exact = str(value).strip()
                    source = source_rows.get(exact.casefold())
                    if source and exact.casefold() not in {x.casefold() for x in selected}:
                        selected.append(source["candidate"])
                needed = int(item["needed"])
                confidence = float(parsed.get("confidence", 0))
                return {
                    "id": item["id"],
                    "needed": needed,
                    "status": "ok" if len(selected) >= needed and confidence >= 0.8 else "insufficient",
                    "selected": selected[:needed],
                    "confidence": confidence,
                    "candidate_count": len(candidates),
                    "source_rows": {
                        text: source_rows[text.casefold()] for text in selected[:needed]
                    },
                    "raw_response": raw,
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(min(2 ** attempt, 15))
        return {
            "id": item["id"], "needed": int(item["needed"]), "status": "error",
            "selected": [], "candidate_count": len(candidates), "error": error,
        }

    pending = [
        item for item in unresolved
        if (completed.get(item["id"], {}).get("status") != "ok"
            or float(completed[item["id"]].get("confidence", 0)) < 0.8)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(review, item): item["id"] for item in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed[result["id"]] = result
            with lock, args.output_review.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 25 == 0 or index == len(pending):
                print(f"reviewed {index}/{len(pending)}", flush=True)

    overrides = []
    for item_id, review_row in sorted(completed.items()):
        if review_row.get("status") != "ok":
            continue
        for candidate in review_row["selected"]:
            source = review_row["source_rows"][candidate]
            digest = hashlib.sha256((item_id + "\0" + candidate).encode()).hexdigest()[:12]
            confidence = min(float(source["confidence"]), float(review_row.get("confidence", 0)))
            overrides.append({
                "verification_key": f"{item_id}::text_slot_override::{digest}",
                "id": item_id,
                "candidate_index": int(source.get("candidate_index", 0)),
                "candidate": candidate,
                "slot_fit": True,
                "verdict": "NOT_SUPPORTED",
                "confidence": confidence,
                "judgment": {
                    "slot_fit": True,
                    "verdict": "NOT_SUPPORTED",
                    "confidence": confidence,
                    "text_slot_review_model": args.model,
                    "visual_verification_key": source["verification_key"],
                    "review_method": "independent_text_slot_fit_plus_visual_absence",
                },
            })
    with args.output_verification.open("w", encoding="utf-8") as stream:
        for row in overrides:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "unresolved_input": len(unresolved),
        "review_ok": sum(row.get("status") == "ok" for row in completed.values()),
        "review_not_ok": sum(row.get("status") != "ok" for row in completed.values()),
        "override_distractors": len(overrides),
    }
    args.output_review.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
