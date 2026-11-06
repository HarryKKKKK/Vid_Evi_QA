#!/usr/bin/env python3
"""Generate plausible same-slot distractor candidates for conflict QA."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


SYSTEM = """You create candidate distractors for a five-option video-QA benchmark.
Given a question and its known valid answers, generate the requested number of concise candidate
answers that fit exactly the same grammatical and semantic answer slot but are meaningfully different
from every valid answer. Candidates should be plausible for this kind of video, not absurd, not a
paraphrase, not a subset/superset of a valid answer, and must not contain meta choices such as
"none of the above". They will later be checked against the video, so do not claim they are definitely
absent. Return only JSON: {"candidates":["..."],"notes":["..."]}."""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        raise ValueError("no JSON object")
    value = json.loads(match.group(0))
    candidates = []
    for item in value.get("candidates", []):
        text = str(item).strip()
        if text and text.casefold() not in {x.casefold() for x in candidates}:
            candidates.append(text)
    value["candidates"] = candidates
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api3.xhub.chat/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is required")
    requests = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {row["id"]: row for row in read_jsonl(args.output)} if args.output.exists() else {}
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=180, max_retries=0)
    lock = threading.Lock()

    def generate(item: dict) -> dict:
        payload = {
            "question": item["question"], "official_action": item.get("official_action"),
            "conflict_type": item["conflict_type"], "semantic_field": item.get("grounding_field"),
            "known_valid_answers": item["valid_answers"],
            "candidate_count": item["candidate_count"],
            "do_not_repeat": item.get("avoid_candidates", []),
        }
        error = None
        for attempt in range(args.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    temperature=0.4, max_tokens=512, response_format={"type": "json_object"})
                raw = response.choices[0].message.content or ""
                parsed = parse(raw)
                status = "ok" if len(parsed["candidates"]) >= item["needed"] else "insufficient"
                return {"id": item["id"], "status": status, "needed": item["needed"],
                        "candidates": parsed["candidates"], "raw_response": raw}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.max_retries:
                    time.sleep(min(2 ** attempt, 15))
        return {"id": item["id"], "status": "error", "needed": item["needed"], "error": error,
                "candidates": []}

    pending = [item for item in requests if item["id"] not in completed or completed[item["id"]].get("status") != "ok"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate, item): item["id"] for item in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed[result["id"]] = result
            with lock, args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 50 == 0 or index == len(pending):
                print(f"generated {index}/{len(pending)}", flush=True)
    good = sum(row.get("status") == "ok" for row in completed.values())
    print(json.dumps({"requests": len(requests), "ok": good, "not_ok": len(requests) - good}, indent=2))


if __name__ == "__main__":
    main()
