#!/usr/bin/env python3
"""Prepare and finalize an atomic-option rebuild of scene-context QA.

Preparation is deliberately conservative.  Supported compound answers may be
split into supported atomic candidates.  Unsupported compound distractors are
only suggestions: falsity of a conjunction does not imply falsity of every
member, so they must be checked against the video before publication.
"""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/scene_atomic_rebuild.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/etbench/conflict/scene_atomic_rebuild.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/etbench/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


LETTERS = "ABCDE"
COLOR_AND_COLOR = re.compile(
    r"\b(black|white|red|blue|green|yellow|purple|brown|gray|grey|orange|pink)\s+and\s+"
    r"(black|white|red|blue|green|yellow|purple|brown|gray|grey|orange|pink)\b",
    re.I,
)
SYNONYMS = {
    "grey": "gray", "sofa": "couch", "television": "tv", "telly": "tv",
    "bottle": "bottle", "waterbottle": "bottle", "cellphone": "phone",
    "mobile": "phone", "sneaker": "shoe", "slipper": "shoe",
}
STOP = {"a", "an", "the", "there", "is", "are", "visible", "shown", "seen"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def canonical(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))
    normalized = [SYNONYMS.get(word, word) for word in words if word not in STOP]
    return " ".join(normalized)


def near_duplicate(left: str, right: str) -> bool:
    a, b = set(canonical(left).split()), set(canonical(right).split())
    if not a or not b:
        return True
    if a <= b or b <= a:
        return True
    return len(a & b) / len(a | b) >= 0.72


def atomic_parts(text: str) -> list[str]:
    """Return conservative split suggestions, not truth labels."""
    protected: dict[str, str] = {}

    def hide(match: re.Match) -> str:
        key = f"__COLORPAIR{len(protected)}__"
        protected[key] = match.group(0)
        return key

    value = COLOR_AND_COLOR.sub(hide, text.strip().strip(". ;,"))
    chunks = re.split(r"\s*[;,]\s*|\s+(?:and|plus)\s+", value, flags=re.I)
    parts = []
    for chunk in chunks:
        chunk = chunk.strip(" .;,")
        for key, original in protected.items():
            chunk = chunk.replace(key, original)
        if chunk:
            parts.append(chunk)
    return parts or [text.strip()]


def fingerprint(record: dict) -> str:
    payload = {
        "id": record.get("id"),
        "question": record.get("question"),
        "options": [{"option_id": x.get("option_id"), "answer": x.get("answer")} for x in record.get("options", [])],
        "valid_option_ids": record.get("valid_option_ids", []),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def prepare(source: Path, output_dir: Path) -> dict:
    rows = read_jsonl(source)
    non_scene = [copy.deepcopy(row) for row in rows if row.get("conflict_type") != "scene_context"]
    scene = [row for row in rows if row.get("conflict_type") == "scene_context"]
    requests, candidates = [], []
    compound_records = 0
    for record in scene:
        record_candidates = []
        has_compound = False
        for option in record["options"]:
            parts = atomic_parts(option["answer"])
            has_compound |= len(parts) > 1
            for index, part in enumerate(parts):
                item = {
                    "candidate_id": f"{record['id']}::{option['option_id']}::{index}",
                    "answer": part,
                    "canonical_answer": canonical(part),
                    "source_option_id": option["option_id"],
                    "source_answer": option["answer"],
                    "source_is_valid": bool(option.get("is_valid")),
                    "supported_by": option.get("supported_by", []) if option.get("is_valid") else [],
                    "truth_status": "supported" if option.get("is_valid") else "needs_visual_verification",
                }
                record_candidates.append(item)
                candidates.append({"qa_id": record["id"], **item})
        compound_records += int(has_compound)
        duplicate_pairs = []
        for i, left in enumerate(record_candidates):
            for right in record_candidates[i + 1:]:
                if near_duplicate(left["answer"], right["answer"]):
                    duplicate_pairs.append([left["candidate_id"], right["candidate_id"]])
        requests.append({
            "id": record["id"],
            "parent_job_id": record["parent_job_id"],
            "video": record.get("video"),
            "question": record["question"],
            "grounding_field": record.get("grounding_field"),
            "source_fingerprint": fingerprint(record),
            "atomic_candidates": record_candidates,
            "possible_duplicate_pairs": duplicate_pairs,
            "required_output": {
                "option_count": 5,
                "atomic_claims_only": True,
                "unique_semantic_group_per_option": True,
                "visual_verification_required": True,
                "minimum_supported_options": 2,
            },
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "conflict_qa_five_options.current.jsonl", non_scene)
    write_jsonl(output_dir / "scene_v1_removed.jsonl", scene)
    write_jsonl(output_dir / "scene_atomic_candidates.jsonl", candidates)
    write_jsonl(output_dir / "scene_rebuild_requests.jsonl", requests)
    summary = {
        "status": "scene_withheld_pending_visual_verification",
        "source": str(source),
        "source_records": len(rows),
        "current_non_scene_records": len(non_scene),
        "removed_scene_records": len(scene),
        "scene_records_with_detected_compounds": compound_records,
        "atomic_candidate_suggestions": len(candidates),
        "publication_rule": "No scene record is returned to current until it has exactly five visually verified atomic, semantically unique options.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def validate_verified_scene(record: dict) -> None:
    options = record.get("options", [])
    if len(options) != 5 or [x.get("option_id") for x in options] != list(LETTERS):
        raise ValueError(f"{record.get('id')}: options must be exactly A-E")
    groups = []
    for option in options:
        if option.get("verification_status") != "verified":
            raise ValueError(f"{record.get('id')}: every option must be visually verified")
        if len(atomic_parts(option.get("answer", ""))) != 1:
            raise ValueError(f"{record.get('id')}: compound option remains: {option.get('answer')}")
        group = str(option.get("semantic_group_id", "")).strip()
        if not group:
            raise ValueError(f"{record.get('id')}: missing semantic_group_id")
        groups.append(group)
    if len(set(groups)) != 5:
        raise ValueError(f"{record.get('id')}: repeated semantic groups")
    for i, left in enumerate(options):
        for right in options[i + 1:]:
            if near_duplicate(left["answer"], right["answer"]):
                raise ValueError(f"{record.get('id')}: possible paraphrase options: {left['answer']} / {right['answer']}")
    valid_ids = [x["option_id"] for x in options if x.get("is_valid")]
    if len(valid_ids) < 2 or sorted(valid_ids) != sorted(record.get("valid_option_ids", [])):
        raise ValueError(f"{record.get('id')}: inconsistent supported option IDs")


def finalize(current: Path, verified_scene: Path, output: Path) -> dict:
    non_scene = read_jsonl(current)
    scene = read_jsonl(verified_scene)
    for record in scene:
        if record.get("conflict_type") != "scene_context":
            raise ValueError(f"{record.get('id')}: verified file contains a non-scene record")
        validate_verified_scene(record)
        record["qa_fingerprint"] = fingerprint(record)
        record["alignment_version"] = "five_option_atomic_scene_v2"
    ids = [x["id"] for x in non_scene + scene]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate QA IDs in final dataset")
    write_jsonl(output, non_scene + scene)
    return {"status": "complete", "records": len(ids), "scene_records": len(scene), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--current", type=Path, required=True)
    finish.add_argument("--verified-scene", type=Path, required=True)
    finish.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.output_dir) if args.command == "prepare" else finalize(args.current, args.verified_scene, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
