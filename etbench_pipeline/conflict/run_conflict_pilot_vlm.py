#!/usr/bin/env python3
"""Run the E.T. Bench conflict-QA pilot against an OpenAI-compatible VLM."""

from __future__ import annotations

import argparse
import base64
import json
import re
from itertools import combinations
from pathlib import Path

import cv2
from openai import OpenAI


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def sample_times(instance: dict, duration: float) -> list[tuple[float, str]]:
    core_start, core_end = map(float, instance["core"])
    pre_start = max(float(instance["pre"][0]), core_start - 1.0)
    post_end = min(float(instance["post"][1]), core_end + 1.0, duration)

    def centers(start: float, end: float, count: int, phase: str) -> list[tuple[float, str]]:
        if end <= start or count <= 0:
            return []
        width = (end - start) / count
        return [(start + (i + 0.5) * width, phase) for i in range(count)]

    points = centers(pre_start, core_start, 2, "pre")
    points += centers(core_start, core_end, 8, "core")
    points += centers(core_end, post_end, 2, "post")
    return [(max(0.0, min(t, duration - 0.001)), phase) for t, phase in points]


def encode_frames(video_path: Path, points: list[tuple[float, str]], width: int) -> list[tuple[float, str, str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    output = []
    try:
        for second, phase in points:
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cannot decode {video_path} at {second:.3f}s")
            height, original_width = frame.shape[:2]
            if original_width > width:
                frame = cv2.resize(frame, (width, round(height * width / original_width)), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                raise RuntimeError(f"cannot encode frame: {video_path} at {second:.3f}s")
            output.append((second, phase, base64.b64encode(encoded).decode("ascii")))
    finally:
        cap.release()
    return output


class Model:
    def __init__(self, base_url: str, model: str, raw_path: Path):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=240, max_retries=2)
        self.model = model
        self.raw_path = raw_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)

    def call(self, request_id: str, content, *, max_tokens: int = 1400) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        raw = {
            "request_id": request_id,
            "model": self.model,
            "response_id": response.id,
            "usage": response.usage.model_dump() if response.usage else None,
            "content": text,
        }
        with self.raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
        return parse_json(text)


def run_instances(args, model: Model, jobs: list[dict]) -> list[dict]:
    result_path = args.output_dir / "visual_instance_results.jsonl"
    results = read_jsonl(result_path) if result_path.exists() else []
    completed = {row["instance_job_id"] for row in results}
    for number, job in enumerate(jobs, 1):
        if job["instance_job_id"] in completed:
            print(f"INSTANCE {number}/{len(jobs)} {job['instance_job_id']} [resume]", flush=True)
            continue
        video_path = args.video_root / job["video"]
        points = sample_times(job["instance"], float(job["video_duration"]))
        frames = encode_frames(video_path, points, args.frame_width)
        content = [{"type": "text", "text": (
            "You are independently describing ONE occurrence of a repeated video event. "
            "Use only the supplied chronological frames. Frames marked CORE belong to the target occurrence; "
            "PRE/POST are at most one second of local context. Do not compare with or speculate about other occurrences. "
            f"Action hint: {job['anchor_action_hint']}. Target interval: {job['instance']['core']} seconds. "
            "Return strict JSON only. Every slot must have value, confidence from 0 to 1, and a short visual_basis. "
            "Use not_visible when unsupported. Required schema: " + json.dumps(job["output_schema"], ensure_ascii=False)
        )}]
        for second, phase, image_b64 in frames:
            content.append({"type": "text", "text": f"{phase.upper()} frame at {second:.3f}s"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
        parsed = model.call(job["instance_job_id"], content)
        parsed["instance_job_id"] = job["instance_job_id"]
        parsed["parent_job_id"] = job["parent_job_id"]
        parsed["sampled_frames"] = [{"second": round(t, 3), "phase": p} for t, p, _ in frames]
        results.append(parsed)
        print(f"INSTANCE {number}/{len(jobs)} {job['instance_job_id']}", flush=True)
        write_jsonl(result_path, results)
    return results


def slot_value(result: dict, slot: str) -> tuple[str, float, str]:
    raw = result.get("slots", {}).get(slot, {})
    if not isinstance(raw, dict):
        return str(raw), 0.0, ""
    try:
        confidence = float(raw.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    return str(raw.get("value", "")), confidence, str(raw.get("visual_basis", ""))


def candidate_pairs(groups: list[dict], results: list[dict], threshold: float) -> list[dict]:
    by_parent = {}
    for row in results:
        by_parent.setdefault(row["parent_job_id"], []).append(row)
    jobs = []
    missing = {"", "not_visible", "unknown", "unclear", "none", "null", "n/a"}
    for group in groups:
        instances = {x["instance_id"]: x for x in group["instances"]}
        for a, b in combinations(by_parent.get(group["job_id"], []), 2):
            differences = []
            for slot in group["requested_slots"]:
                if slot == "anchor_action":
                    continue
                va, ca, ba = slot_value(a, slot)
                vb, cb, bb = slot_value(b, slot)
                if min(ca, cb) < threshold or va.strip().lower() in missing or vb.strip().lower() in missing:
                    continue
                if re.sub(r"\W+", " ", va.lower()).strip() == re.sub(r"\W+", " ", vb.lower()).strip():
                    continue
                differences.append({"slot": slot, "value_a": va, "value_b": vb,
                                    "confidence_a": ca, "confidence_b": cb,
                                    "visual_basis_a": ba, "visual_basis_b": bb})
            if not differences:
                continue
            ida = a["instance_job_id"].rsplit("__", 1)[-1]
            idb = b["instance_job_id"].rsplit("__", 1)[-1]
            jobs.append({"pair_job_id": f"{group['job_id']}__{ida}__{idb}", "parent_job_id": group["job_id"],
                         "anchor_action_hint": group["anchor_action_hint"], "instance_a": instances[ida],
                         "instance_b": instances[idb], "candidate_differences": differences})
    return jobs


def run_pair_judging(args, model: Model, groups: list[dict], results: list[dict]) -> list[dict]:
    jobs = candidate_pairs(groups, results, args.min_confidence)
    write_jsonl(args.output_dir / "pair_judging_jobs.jsonl", jobs)
    judged = []
    for number, job in enumerate(jobs, 1):
        prompt = (
            "Judge candidate differences between two independently described occurrences of the same event. "
            "Accept only visually grounded differences that can be mutually exclusive answers to one natural question. "
            "Reject synonyms, granularity differences, uncertain properties, and mismatched referents. "
            "Return strict JSON: {pair_job_id:string, common_anchor_action:string, accepted_differences:[{slot:string," 
            "value_a:string,value_b:string,same_event_semantics:boolean,mutually_exclusive:boolean," 
            "question_template:string,confidence:number,reason:string}]}. Candidate: " + json.dumps(job, ensure_ascii=False)
        )
        parsed = model.call(job["pair_job_id"], prompt)
        parsed["pair_job_id"] = job["pair_job_id"]
        parsed["parent_job_id"] = job["parent_job_id"]
        parsed["instance_a"] = job["instance_a"]
        parsed["instance_b"] = job["instance_b"]
        judged.append(parsed)
        print(f"PAIR {number}/{len(jobs)} {job['pair_job_id']}", flush=True)
        write_jsonl(args.output_dir / "pair_judgment_results.jsonl", judged)
    return judged


def run_qa(model: Model, groups: list[dict], judged: list[dict], output_dir: Path) -> list[dict]:
    output = []
    for group in groups:
        accepted = []
        for pair in judged:
            if pair.get("parent_job_id") != group["job_id"]:
                continue
            for diff in pair.get("accepted_differences", []):
                if diff.get("same_event_semantics") and diff.get("mutually_exclusive") and float(diff.get("confidence", 0)) >= .75:
                    accepted.append({"pair_job_id": pair["pair_job_id"], "instance_a": pair["instance_a"],
                                     "instance_b": pair["instance_b"], **diff})
        prompt = (
            "Construct one high-quality conflict-evidence QA item from repeated occurrences of one event. "
            "The ambiguous question must omit occurrence/time identifiers and have at least two mutually exclusive, "
            "individually supported answers. Also create two resolved controls, each explicitly identifying one interval. "
            "If evidence is inadequate, return status=reject and explain why. Strict JSON schema: "
            "{parent_job_id:string,status:'accept'|'reject',anchor_action:string,ambiguous_question:string," 
            "conflicting_answers:[{answer:string,instance_id:string,interval:[number,number]}]," 
            "resolved_controls:[{question:string,answer:string,instance_id:string,interval:[number,number]}]," 
            "conflict_slot:string,rationale:string}. Group: " + json.dumps({
                "parent_job_id": group["job_id"], "anchor_action": group["anchor_action_hint"],
                "instances": group["instances"], "accepted_differences": accepted}, ensure_ascii=False)
        )
        parsed = model.call(group["job_id"] + "__qa", prompt, max_tokens=1800)
        parsed["parent_job_id"] = group["job_id"]
        output.append(parsed)
        print(f"QA {len(output)}/{len(groups)} {group['job_id']}", flush=True)
        (output_dir / "conflict_qa_pilot_3.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-width", type=int, default=560)
    parser.add_argument("--min-confidence", type=float, default=.70)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_model_responses.jsonl"
    jobs, groups = read_jsonl(args.instances), read_jsonl(args.groups)
    model = Model(args.base_url, args.model, raw_path)
    results = run_instances(args, model, jobs)
    judged = run_pair_judging(args, model, groups, results)
    qa = run_qa(model, groups, judged, args.output_dir)
    summary = {"model": args.model, "instances": len(results), "pair_jobs": len(judged),
               "qa_items": len(qa), "accepted_qa": sum(x.get("status") == "accept" for x in qa)}
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("DONE " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
