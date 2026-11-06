#!/usr/bin/env python3
"""Validate model-generated conflict QA and emit evidence-locked QA records."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/validate_conflict_qa.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'etbench_pipeline/conflict/validate_conflict_qa.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'etbench_pipeline/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
from pathlib import Path


GENERIC_OBJECTS = {"object", "item", "something", "container", "thing", "items", "objects"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def instance_id(pair_job_id: str, side: str) -> str:
    parts = pair_job_id.split("__")
    return parts[-2] if side == "a" else parts[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--pair-results", type=Path, required=True)
    parser.add_argument("--model-qa", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = {row["job_id"]: row for row in read_jsonl(args.groups)}
    pairs = read_jsonl(args.pair_results)
    model_qa = {row["parent_job_id"]: row for row in json.loads(args.model_qa.read_text(encoding="utf-8"))}
    output = []

    for parent_id, group in groups.items():
        candidates = []
        for pair in pairs:
            if pair.get("parent_job_id") != parent_id:
                continue
            for diff in pair.get("accepted_differences", []):
                va, vb = str(diff.get("value_a", "")).strip(), str(diff.get("value_b", "")).strip()
                # Parent groups are constructed from one E.T. Bench action label, so
                # common event semantics is guaranteed upstream. InternVL sometimes
                # interpreted `same_event_semantics` as "same object" and set it false
                # precisely when the object values differed; do not reuse that broken
                # flag as an acceptance gate.
                if diff.get("slot") != "object" or not diff.get("mutually_exclusive"):
                    continue
                if not va or not vb or va.lower() in GENERIC_OBJECTS or vb.lower() in GENERIC_OBJECTS:
                    continue
                confidence = float(diff.get("confidence", 0))
                if confidence < .85:
                    continue
                candidates.append((confidence, pair, diff))

        original = model_qa.get(parent_id, {})
        model_valid = False
        for _, _, diff in candidates:
            answers = {str(x.get("answer", "")).strip().lower() for x in original.get("conflicting_answers", [])}
            expected = {str(diff["value_a"]).strip().lower(), str(diff["value_b"]).strip().lower()}
            if original.get("conflict_slot") == "object" and answers == expected:
                model_valid = True
                break

        if not candidates:
            output.append({
                "parent_job_id": parent_id,
                "status": "reject",
                "reason": "No high-confidence, non-generic object contrast survived evidence locking.",
                "model_generated_qa_valid": model_valid,
            })
            continue

        _, pair, diff = sorted(candidates, key=lambda x: (-x[0], x[1]["pair_job_id"]))[0]
        ida, idb = instance_id(pair["pair_job_id"], "a"), instance_id(pair["pair_job_id"], "b")
        instances = {x["instance_id"]: x for x in group["instances"]}
        action = group["anchor_action_hint"]
        question = f"What object was involved when the person was {action}?"
        answers = []
        controls = []
        for iid, value in ((ida, diff["value_a"]), (idb, diff["value_b"])):
            interval = instances[iid]["core"]
            answers.append({"answer": value, "instance_id": iid, "interval": interval})
            controls.append({
                "question": f"During {interval[0]}-{interval[1]} seconds, what object was involved when the person was {action}?",
                "answer": value,
                "instance_id": iid,
                "interval": interval,
            })
        output.append({
            "parent_job_id": parent_id,
            "status": "accept",
            "anchor_action": action,
            "ambiguous_question": question,
            "conflicting_answers": answers,
            "resolved_controls": controls,
            "conflict_slot": "object",
            "evidence_pair_id": pair["pair_job_id"],
            "pair_confidence": diff["confidence"],
            "pair_same_event_semantics_reported": diff.get("same_event_semantics"),
            "same_anchor_by_construction": True,
            "model_generated_qa_valid": model_valid,
            "validation": "Answers are copied exactly from one accepted object-difference pair; interval controls are copied from the source manifests.",
        })

    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total": len(output), "accept": sum(x["status"] == "accept" for x in output),
                      "reject": sum(x["status"] == "reject" for x in output),
                      "original_model_qa_valid": sum(x.get("model_generated_qa_valid", False) for x in output)}))


if __name__ == "__main__":
    main()
