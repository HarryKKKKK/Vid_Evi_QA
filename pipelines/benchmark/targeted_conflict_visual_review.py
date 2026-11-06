#!/usr/bin/env python3
"""Repair missing TEM actions and verify conflict evidence one option/interval at a time."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/targeted_conflict_visual_review.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/targeted_conflict_visual_review.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFLICT_DIR = ROOT / "etbench_pipeline" / "conflict"
if str(CONFLICT_DIR) not in sys.path:
    sys.path.insert(0, str(CONFLICT_DIR))

from etbench_pipeline.conflict.run_conflict_pipeline_v2 import encode_frames


ACTION_PROMPT = """You are identifying the common core action in multiple official video intervals.
Every labeled interval is supposed to contain another occurrence of the same action. Inspect ALL
intervals, not only the reference/anchor interval. Return the most salient shared physical action,
ignoring differences in objects, actors, background, and manner.

Use a short English gerund phrase such as "turning something off" or "putting something on top of
something". Do not describe the scene and do not choose an action visible in only one interval.
If the intervals do not visibly share one action, return INCONSISTENT. If motion is not clear enough,
return UNCERTAIN.

Return only JSON:
{"status":"CONSISTENT|INCONSISTENT|UNCERTAIN","action":"short gerund phrase or null",
 "confidence":0.0,"interval_checks":[{"instance_id":"...","matches":true,"description":"..."}],
 "reason":"..."}
"""


OPTION_PROMPT = """You are verifying one proposed answer against one specifically annotated video
evidence region. This is verification, not multiple-choice selection. First describe only what is
visibly relevant, then decide whether the proposed answer is supported.

Question: {question}
Proposed answer ({option_id}): {answer}
Conflict type: {conflict_type}
Requested relation/field: {grounding_field}
Annotated occurrence: {instance_id}
Annotated core interval: {core_interval}
Region shown for verification: {shown_region}

Rules:
- SUPPORTED means the shown visual content directly supports this answer to this question.
- NOT_SUPPORTED means it clearly shows something incompatible, the event is in the wrong temporal
  direction, or the answer is absent despite adequate visibility.
- UNCERTAIN means the visual evidence is occluded, too sparse, or otherwise insufficient.
- For before/after questions, distinguish the requested surrounding event from the core action.
- Treat the official action wording as a fixed benchmark category. Do not reject an answer merely
  because the visible action admits a more specific name (for example, capping a bottle can satisfy
  "covering something with something"). Judge whether the proposed answer fills the question's
  requested semantic slot under that fixed category.
- For temporal-localization questions, the proposed answer is the interval itself. Mark it supported
  when the benchmark action broadly occurs in that interval; do not demand an unstated object or
  relation stricter than the official category.
- If the content plausibly matches but only the naming/granularity is debatable, use UNCERTAIN rather
  than NOT_SUPPORTED.
- Do not compare this option with other options and do not assume it is true because it was proposed.

Return only JSON:
{{"verdict":"SUPPORTED|NOT_SUPPORTED|UNCERTAIN","confidence":0.0,
 "best_interval":[0.0,0.0],"visible_evidence":"...",
 "temporal_relation_correct":true,"failure_type":"none|absent|contradicted|wrong_phase|unclear",
 "reason":"..."}}
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", flags=re.S)
    if not match:
        raise ValueError("response contains no JSON object")
    return json.loads(match.group(0))


def points(start: float, end: float, count: int, label: str) -> list[tuple[float, str]]:
    start, end = float(start), float(end)
    if end < start:
        start, end = end, start
    if end - start < 1e-4:
        return [(start, label)]
    return [
        (start + (index + 0.5) * (end - start) / count, label)
        for index in range(max(1, count))
    ]


def interval(value) -> tuple[float, float]:
    if isinstance(value, str):
        values = re.findall(r"-?\d+(?:\.\d+)?", value)
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError(f"invalid interval: {value!r}")
    if len(values) < 2:
        raise ValueError(f"invalid interval: {value!r}")
    return float(values[0]), float(values[1])


def media(frames: list[tuple[float, str, str]], prompt: str) -> list[dict]:
    content: list[dict] = []
    for second, phase, encoded in frames:
        content.append({"type": "text", "text": f"{phase} frame at {second:.3f}s"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    content.append({"type": "text", "text": prompt})
    return content


def request(client: OpenAI, model: str, content: list[dict], max_tokens: int = 640) -> tuple[dict, str]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    return parse_json(raw), raw


def rewrite_question(question: str, action: str) -> str:
    return re.sub(r"\bthe official action\b|\bofficial action\b", action, question, flags=re.I)


def common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frame-width", type=int, default=448)


def action_mode(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.qa)
    groups = {row["job_id"]: row for row in read_jsonl(args.groups)}
    needed = {
        row["parent_job_id"]
        for row in rows
        if row.get("official_action") is None and "official action" in row.get("question", "").lower()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "action_inference.jsonl"
    completed = {row["parent_job_id"]: row for row in read_jsonl(raw_path)} if raw_path.exists() else {}
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300, max_retries=2)
    lock = threading.Lock()

    def infer(parent_id: str) -> dict:
        group = groups[parent_id]
        instances = list(group.get("instances", []))
        if len(instances) < 2:
            return {"parent_job_id": parent_id, "status": "SKIPPED_LT2", "action": None,
                    "confidence": 0.0, "instance_count": len(instances)}
        all_points: list[tuple[float, str]] = []
        shown = []
        for item in instances:
            start, end = interval(item["core"])
            shown.append({"instance_id": item["instance_id"], "interval": [start, end]})
            all_points.extend(points(start, end, args.frames_per_interval,
                                     f"{item['instance_id']} official interval"))
        frames = encode_frames(args.video_root / group["video"], all_points, args.frame_width)
        prompt = ACTION_PROMPT + "\nOfficial intervals:\n" + json.dumps(shown, ensure_ascii=False)
        try:
            parsed, raw = request(client, args.model, media(frames, prompt))
            status = str(parsed.get("status", "UNCERTAIN")).upper()
            action = str(parsed.get("action") or "").strip() or None
            if status != "CONSISTENT":
                action = None
            return {"parent_job_id": parent_id, "status": status, "action": action,
                    "confidence": float(parsed.get("confidence", 0)), "instance_count": len(instances),
                    "intervals": shown, "judgment": parsed, "raw_response": raw}
        except Exception as exc:
            return {"parent_job_id": parent_id, "status": "ERROR", "action": None,
                    "confidence": 0.0, "instance_count": len(instances),
                    "error": f"{type(exc).__name__}: {exc}"}

    pending = sorted(needed - completed.keys())
    if args.limit is not None:
        pending = pending[:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(infer, parent_id): parent_id for parent_id in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed[result["parent_job_id"]] = result
            with lock, raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 20 == 0 or index == len(pending):
                print(f"action {index}/{len(pending)}", flush=True)

    repaired, unresolved = [], []
    for row in rows:
        row = json.loads(json.dumps(row))
        if row["parent_job_id"] in needed:
            result = completed.get(row["parent_job_id"], {})
            if (result.get("status") == "CONSISTENT" and result.get("action")
                    and float(result.get("confidence", 0)) >= args.min_action_confidence):
                row["official_action"] = result["action"]
                row["question"] = rewrite_question(row["question"], result["action"])
                row.setdefault("repair_provenance", {})["official_action"] = {
                    "method": "all_official_intervals_vlm", "confidence": result["confidence"]}
            else:
                unresolved.append({"qa_id": row["id"], "parent_job_id": row["parent_job_id"],
                                   "result": result})
        repaired.append(row)
    with (args.output_dir / "qa_with_repaired_actions.jsonl").open("w", encoding="utf-8") as stream:
        for row in repaired:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "unresolved_actions.json").write_text(
        json.dumps(unresolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "parents_needing_action": len(needed),
        "parents_consistent": sum(r.get("status") == "CONSISTENT" and r.get("action") is not None
                                  and float(r.get("confidence", 0)) >= args.min_action_confidence
                                  for key, r in completed.items() if key in needed),
        "parents_skipped_lt2": sum(r.get("status") == "SKIPPED_LT2" for r in completed.values()),
        "qa_unresolved": len(unresolved),
    }
    (args.output_dir / "action_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def option_mode(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.qa)
    groups = {row["job_id"]: row for row in read_jsonl(args.groups)}
    tasks = []
    for qa in rows:
        if qa.get("official_action") is None and "official action" in qa.get("question", "").lower():
            continue
        group = groups[qa["parent_job_id"]]
        instances = {item["instance_id"]: item for item in group.get("instances", [])}
        for option in qa.get("options", []):
            for support_index, support in enumerate(option.get("supported_by", [])):
                key = f"{qa['id']}::{option['option_id']}::{support_index}"
                tasks.append((key, qa, option, support_index, support, group, instances))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "support_verification.jsonl"
    completed = {row["verification_key"]: row for row in read_jsonl(raw_path)} if raw_path.exists() else {}
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=300, max_retries=2)
    lock = threading.Lock()

    def verify(task) -> dict:
        key, qa, option, support_index, support, group, instances = task
        instance_id = support.get("instance_id")
        instance = instances.get(instance_id)
        core = interval(support.get("interval") or (instance or {}).get("core"))
        phase = "core"
        region = core
        frame_points: list[tuple[float, str]]
        if qa.get("conflict_type") == "surrounding_event" and instance:
            field = qa.get("grounding_field")
            if field == "event_before":
                phase, region = "before", interval(instance.get("pre", core))
            elif field == "event_after":
                phase, region = "after", interval(instance.get("post", core))
            frame_points = points(region[0], region[1], args.frames_per_interval, phase)
            frame_points += points(core[0], core[1], max(2, args.frames_per_interval // 3), "core-boundary")
        else:
            start = max(0.0, core[0] - args.padding)
            end = min(float(group.get("video_duration", core[1] + args.padding)), core[1] + args.padding)
            region = (start, end)
            frame_points = points(start, end, args.frames_per_interval, phase)
        frames = encode_frames(args.video_root / group["video"], frame_points, args.frame_width)
        prompt = OPTION_PROMPT.format(
            question=qa["question"], option_id=option["option_id"], answer=option["answer"],
            conflict_type=qa.get("conflict_type"), grounding_field=qa.get("grounding_field"),
            instance_id=instance_id, core_interval=list(core), shown_region=list(region))
        try:
            parsed, raw = request(client, args.model, media(frames, prompt))
            verdict = str(parsed.get("verdict", "UNCERTAIN")).upper()
            if verdict not in {"SUPPORTED", "NOT_SUPPORTED", "UNCERTAIN"}:
                verdict = "UNCERTAIN"
            return {"verification_key": key, "qa_id": qa["id"], "option_id": option["option_id"],
                    "support_index": support_index, "instance_id": instance_id,
                    "annotated_interval": list(core), "shown_region": list(region),
                    "verdict": verdict, "confidence": float(parsed.get("confidence", 0)),
                    "judgment": parsed, "raw_response": raw}
        except Exception as exc:
            return {"verification_key": key, "qa_id": qa["id"], "option_id": option["option_id"],
                    "support_index": support_index, "instance_id": instance_id,
                    "annotated_interval": list(core), "shown_region": list(region),
                    "verdict": "UNCERTAIN", "confidence": 0.0,
                    "error": f"{type(exc).__name__}: {exc}"}

    pending = [task for task in tasks if task[0] not in completed]
    if args.limit is not None:
        pending = pending[:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(verify, task): task[0] for task in pending}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            completed[result["verification_key"]] = result
            with lock, raw_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 50 == 0 or index == len(pending):
                print(f"support {index}/{len(pending)}", flush=True)

    by_qa: dict[str, list[dict]] = {}
    for result in completed.values():
        by_qa.setdefault(result["qa_id"], []).append(result)
    decisions, pass_rows, repair_rows, review_rows = [], [], [], []
    for qa in rows:
        reports = by_qa.get(qa["id"], [])
        option_results = {}
        for option in qa.get("options", []):
            current = [r for r in reports if r["option_id"] == option["option_id"]]
            supported = [r for r in current if r["verdict"] == "SUPPORTED"
                         and r["confidence"] >= args.min_support_confidence]
            uncertain = [r for r in current if r["verdict"] == "UNCERTAIN"
                         or r["confidence"] < args.min_support_confidence]
            option_results[option["option_id"]] = {
                "status": "PASS" if supported else ("REVIEW" if uncertain else "FAIL"),
                "supported_instances": [r["instance_id"] for r in supported],
                "reports": current,
            }
        passed = [key for key, value in option_results.items() if value["status"] == "PASS"]
        uncertain = [key for key, value in option_results.items() if value["status"] == "REVIEW"]
        all_ids = [str(x) for x in qa.get("valid_option_ids", [])]
        unresolved_action = qa.get("official_action") is None and "official action" in qa.get("question", "").lower()
        if unresolved_action or uncertain:
            decision = "REVIEW"
        elif len(passed) >= 2 and set(passed) == set(all_ids):
            decision = "PASS"
        elif len(passed) >= 2:
            decision = "REPAIR"
        else:
            # One VLM is not sufficient authority for destructive deletion.
            decision = "REVIEW"
        item = {"id": qa["id"], "decision": decision, "passed_option_ids": passed,
                "review_option_ids": uncertain, "option_results": option_results, "qa": qa}
        decisions.append(item)
        {"PASS": pass_rows, "REPAIR": repair_rows, "REVIEW": review_rows}[decision].append(item)
    (args.output_dir / "qa_decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, values in (("pass", pass_rows), ("repair", repair_rows), ("review", review_rows)):
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"qa_total": len(rows), "support_tasks": len(tasks), "support_completed": len(completed),
               "pass": len(pass_rows), "repair": len(repair_rows), "review": len(review_rows), "drop": 0}
    (args.output_dir / "option_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    action = sub.add_parser("actions")
    common_args(action)
    action.add_argument("--frames-per-interval", type=int, default=4)
    action.add_argument("--min-action-confidence", type=float, default=0.70)
    action.add_argument("--limit", type=int)
    option = sub.add_parser("options")
    common_args(option)
    option.add_argument("--frames-per-interval", type=int, default=8)
    option.add_argument("--padding", type=float, default=0.75)
    option.add_argument("--min-support-confidence", type=float, default=0.65)
    option.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.mode == "actions":
        action_mode(args)
    else:
        option_mode(args)


if __name__ == "__main__":
    main()
