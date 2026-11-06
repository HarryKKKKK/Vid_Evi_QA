#!/usr/bin/env python3
"""Verify generated conflict distractors against every official occurrence."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/verify_conflict_distractors_vlm.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/verify_conflict_distractors_vlm.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(_LAYOUT_LEGACY_FILE).resolve().parents[2]
CONFLICT_DIR = ROOT / "results/build/etbench" / "conflict"
for path in (ROOT, CONFLICT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from pipelines.etbench.conflict.run_conflict_pipeline_v2 import encode_frames


PROMPT = """You are validating one proposed distractor for a video-QA benchmark. The frames cover
ALL official occurrences relevant to the question, grouped and timestamped by occurrence.

Question: {question}
Official action category: {official_action}
Semantic field: {grounding_field}
Known valid answers (for identifying the answer slot only): {valid_answers}
Candidate distractor: {candidate}

Decide two things independently:
1. slot_fit: can the candidate grammatically and semantically answer exactly the same question slot?
2. video verdict: is the candidate actually supported as an answer in ANY shown official occurrence?

SUPPORTED means it is visibly a valid answer somewhere and therefore is a bad distractor.
NOT_SUPPORTED means all shown occurrences adequately contradict it or show different content.
UNCERTAIN means visibility is insufficient to establish absence. Treat the official action wording as
a fixed benchmark category; do not reject or accept based only on preferring a more specific action name.

Return only JSON:
{{"slot_fit":true,"verdict":"SUPPORTED|NOT_SUPPORTED|UNCERTAIN","confidence":0.0,
"supporting_instance_ids":[],"visible_evidence":"...","reason":"..."}}
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        raise ValueError("no JSON object")
    return json.loads(match.group(0))


def points(start: float, end: float, count: int, label: str) -> list[tuple[float, str]]:
    start, end = float(start), float(end)
    return [(start + (i + 0.5) * max(0.001, end - start) / count, label) for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames-per-occurrence", type=int, default=3)
    parser.add_argument("--frame-width", type=int, default=448)
    parser.add_argument("--key-prefix", default="base")
    args = parser.parse_args()

    requests = {row["id"]: row for row in read_jsonl(args.requests)}
    candidates = {row["id"]: row for row in read_jsonl(args.candidates) if row.get("status") == "ok"}
    groups = {row["job_id"]: row for row in read_jsonl(args.groups)}
    tasks = []
    for item_id, request in requests.items():
        generated = candidates.get(item_id, {}).get("candidates", [])
        for index, candidate in enumerate(generated):
            tasks.append((f"{item_id}::{args.key_prefix}::{index}", request, index, candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {row["verification_key"]: row for row in read_jsonl(args.output)} if args.output.exists() else {}
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300, max_retries=2)
    lock = threading.Lock()

    def verify(task) -> dict:
        key, request, index, candidate = task
        group = groups[request["parent_job_id"]]
        frame_points = []
        for instance in group["instances"]:
            if request["conflict_type"] == "surrounding_event":
                field = request.get("grounding_field")
                region = instance["pre"] if field == "event_before" else instance["post"]
                frame_points.extend(points(region[0], region[1], args.frames_per_occurrence,
                                           f"{instance['instance_id']} requested phase"))
                frame_points.extend(points(instance["core"][0], instance["core"][1], 1,
                                           f"{instance['instance_id']} core boundary"))
            else:
                frame_points.extend(points(instance["core"][0], instance["core"][1],
                                           args.frames_per_occurrence, instance["instance_id"]))
        frames = encode_frames(args.video_root / group["video"], frame_points, args.frame_width)
        content = []
        for second, phase, encoded in frames:
            content.append({"type": "text", "text": f"{phase} at {second:.3f}s"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        content.append({"type": "text", "text": PROMPT.format(
            question=request["question"], official_action=request.get("official_action"),
            grounding_field=request.get("grounding_field"), valid_answers=request["valid_answers"],
            candidate=candidate)})
        try:
            response = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": content}], temperature=0,
                max_tokens=512, response_format={"type": "json_object"})
            raw = response.choices[0].message.content or ""
            judgment = parse(raw)
            verdict = str(judgment.get("verdict", "UNCERTAIN")).upper()
            if verdict not in {"SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"}:
                verdict = "UNCERTAIN"
            return {"verification_key": key, "id": request["id"], "candidate_index": index,
                    "candidate": candidate, "slot_fit": bool(judgment.get("slot_fit")),
                    "verdict": verdict, "confidence": float(judgment.get("confidence", 0)),
                    "judgment": judgment, "raw_response": raw}
        except Exception as exc:
            return {"verification_key": key, "id": request["id"], "candidate_index": index,
                    "candidate": candidate, "slot_fit": False, "verdict": "UNCERTAIN",
                    "confidence": 0.0, "error": f"{type(exc).__name__}: {exc}"}

    pending = [task for task in tasks if task[0] not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(verify, task): task[0] for task in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed[result["verification_key"]] = result
            with lock, args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 50 == 0 or index == len(pending):
                print(f"verified {index}/{len(pending)}", flush=True)
    summary = {
        "tasks": len(tasks), "completed": len(completed),
        "good_distractors": sum(r["slot_fit"] and r["verdict"] == "NOT_SUPPORTED" and r["confidence"] >= .8
                                for r in completed.values()),
        "supported_bad": sum(r["verdict"] == "SUPPORTED" for r in completed.values()),
        "uncertain_or_bad_slot": sum(not r["slot_fit"] or r["verdict"] == "UNCERTAIN" for r in completed.values()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
