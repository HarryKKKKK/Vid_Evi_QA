#!/usr/bin/env python3
"""Visually verify atomic scene-option candidates in resumable shards."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/verify_scene_atomic_candidates.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'etbench_pipeline/conflict/verify_scene_atomic_candidates.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'etbench_pipeline/conflict', _layout_file.parent):
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
sys.path.insert(0, str(ROOT / "etbench_pipeline" / "conflict"))
from run_conflict_pipeline_v2 import encode_frames


PROMPT = """Verify one atomic scene-context answer against frames from ALL official occurrences.
Question: {question}
Scene field: {field}
Candidate answer: {answer}

Return JSON only:
{{"slot_fit":true,"atomic":true,"verdict":"SUPPORTED|NOT_SUPPORTED|UNCERTAIN",
"confidence":0.0,"supporting_instance_ids":[],"semantic_key":"short canonical entity/attribute",
"visible_evidence":"...","reason":"..."}}

SUPPORTED: visibly answers the question in at least one occurrence.
NOT_SUPPORTED: every occurrence is sufficiently visible and shows that this candidate is not an answer.
UNCERTAIN: absence cannot be established. atomic must be false if the candidate contains multiple independently
checkable objects/attributes. semantic_key must collapse paraphrases and synonyms to one concise concept.
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def points(start: float, end: float, count: int, label: str) -> list[tuple[float, str]]:
    return [(float(start) + (i + .5) * max(.001, float(end) - float(start)) / count, label)
            for i in range(count)]


def parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object")
    return json.loads(match.group(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames-per-occurrence", type=int, default=3)
    parser.add_argument("--frame-width", type=int, default=448)
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index")

    groups = {x["job_id"]: x for x in read_jsonl(args.groups)}
    requests = read_jsonl(args.requests)
    all_tasks = [(request, candidate) for request in requests for candidate in request["atomic_candidates"]]
    tasks = [task for index, task in enumerate(all_tasks) if index % args.num_shards == args.shard_index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {x["candidate_id"]: x for x in read_jsonl(args.output)} if args.output.exists() else {}
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300, max_retries=2)
    lock = threading.Lock()

    def verify(task: tuple[dict, dict]) -> dict:
        request, candidate = task
        group = groups[request["parent_job_id"]]
        frame_points = []
        for instance in group["instances"]:
            frame_points.extend(points(instance["core"][0], instance["core"][1],
                                       args.frames_per_occurrence, instance["instance_id"]))
        frames = encode_frames(args.video_root / group["video"], frame_points, args.frame_width)
        content = []
        for second, instance_id, encoded in frames:
            content.extend([
                {"type": "text", "text": f"{instance_id} at {second:.3f}s"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ])
        content.append({"type": "text", "text": PROMPT.format(
            question=request["question"], field=request.get("grounding_field"), answer=candidate["answer"]
        )})
        base = {"qa_id": request["id"], "candidate_id": candidate["candidate_id"],
                "answer": candidate["answer"], "source_option_id": candidate["source_option_id"],
                "source_is_valid": candidate["source_is_valid"]}
        try:
            response = client.chat.completions.create(
                model=args.model, messages=[{"role": "user", "content": content}], temperature=0,
                max_tokens=512, response_format={"type": "json_object"})
            raw = response.choices[0].message.content or ""
            value = parse(raw)
            verdict = str(value.get("verdict", "UNCERTAIN")).upper()
            if verdict not in {"SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"}:
                verdict = "UNCERTAIN"
            return {**base, "slot_fit": bool(value.get("slot_fit")), "atomic": bool(value.get("atomic")),
                    "verdict": verdict, "confidence": float(value.get("confidence", 0)),
                    "supporting_instance_ids": value.get("supporting_instance_ids", []),
                    "semantic_key": str(value.get("semantic_key", "")).strip(),
                    "judgment": value, "raw_response": raw}
        except Exception as exc:
            return {**base, "slot_fit": False, "atomic": False, "verdict": "UNCERTAIN",
                    "confidence": 0.0, "semantic_key": "", "error": f"{type(exc).__name__}: {exc}"}

    pending = [task for task in tasks if task[1]["candidate_id"] not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(verify, task) for task in pending]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            completed[row["candidate_id"]] = row
            with lock, args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 50 == 0 or index == len(pending):
                print(f"shard {args.shard_index}: {index}/{len(pending)}", flush=True)
    summary = {"shard": args.shard_index, "tasks": len(tasks), "completed": len(completed),
               "supported": sum(x["verdict"] == "SUPPORTED" for x in completed.values()),
               "not_supported": sum(x["verdict"] == "NOT_SUPPORTED" for x in completed.values()),
               "uncertain": sum(x["verdict"] == "UNCERTAIN" for x in completed.values())}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
