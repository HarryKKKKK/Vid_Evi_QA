"""Pure contrast selection and evidence locking for E.T. Bench conflict QA v2."""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/conflict_v2_core.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'etbench_pipeline/conflict/conflict_v2_core.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'etbench_pipeline/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import json
import re
from collections import defaultdict
from typing import Any


MISSING = {"", "none", "null", "unknown", "unclear", "not visible", "not_visible", "n/a"}
SCENE_FIELDS = ("location", "salient_background", "actor_appearance", "co_participants")
TEMPORAL_FIELDS = ("event_before", "event_after")
CATEGORY_ORDER = ("action_content", "scene_context", "surrounding_event", "temporal_localization")


def normalize_value(value: Any) -> str:
    if isinstance(value, list):
        value = ", ".join(sorted(str(x).strip() for x in value if str(x).strip()))
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text.rstrip(". ")


def parse_grounded(raw: Any) -> tuple[str, float, str]:
    if not isinstance(raw, dict):
        return str(raw or "").strip(), 0.0, ""
    value = raw.get("value", "")
    if isinstance(value, list):
        value = ", ".join(str(x).strip() for x in value if str(x).strip())
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return str(value).strip(), max(0.0, min(1.0, confidence)), str(raw.get("visual_basis", "")).strip()


def _intervals(group: dict) -> dict[str, list[float]]:
    return {row["instance_id"]: row["core"] for row in group["instances"]}


def build_contrast(
    group: dict,
    groundings: list[dict],
    *,
    category: str,
    container: str,
    field: str,
    min_confidence: float,
) -> dict | None:
    """Build a multi-answer contrast only when at least two grounded values differ."""
    interval_by_id = _intervals(group)
    grouped: dict[str, list[dict]] = defaultdict(list)
    display_value: dict[str, str] = {}
    confidences: list[float] = []
    for row in groundings:
        instance_id = row.get("instance_id")
        if instance_id not in interval_by_id:
            continue
        value, confidence, basis = parse_grounded(row.get(container, {}).get(field))
        normalized = normalize_value(value)
        if normalized in MISSING or confidence < min_confidence:
            continue
        display_value.setdefault(normalized, value)
        grouped[normalized].append({
            "instance_id": instance_id,
            "interval": interval_by_id[instance_id],
            "confidence": confidence,
            "visual_basis": basis,
        })
        confidences.append(confidence)
    if len(grouped) < 2:
        return None
    alternatives = [
        {"answer": display_value[key], "support": supports}
        for key, supports in sorted(grouped.items(), key=lambda item: item[0])
    ]
    return {
        "parent_job_id": group["job_id"],
        "category": category,
        "field": field,
        "official_action": group["anchor_action_hint"],
        "alternatives": alternatives,
        "score": round(sum(confidences) / len(confidences), 4),
    }


def choose_best(candidates: list[dict], preferred_fields: list[str]) -> dict | None:
    if not candidates:
        return None
    rank = {field: i for i, field in enumerate(preferred_fields)}
    return sorted(candidates, key=lambda row: (-row["score"], rank.get(row["field"], 999), row["field"]))[0]


def action_candidate(group: dict, groundings: list[dict], role_schema: dict, min_confidence: float) -> dict | None:
    roles = [str(row["name"]) for row in role_schema.get("roles", []) if row.get("name")]
    candidates = [
        build_contrast(group, groundings, category="action_content", container="action_content",
                       field=role, min_confidence=min_confidence)
        for role in roles
    ]
    return choose_best([x for x in candidates if x], roles)


def candidate_lists(group: dict, core_rows: list[dict], temporal_rows: list[dict], role_schema: dict,
                    min_confidence: float) -> dict[str, list[dict]]:
    """Enumerate every candidate in semantic priority order for fallback traversal."""
    roles = [str(row["name"]) for row in role_schema.get("roles", []) if row.get("name")]
    action = [
        build_contrast(group, core_rows, category="action_content", container="action_content",
                       field=role, min_confidence=min_confidence)
        for role in roles
    ]
    scene = [
        build_contrast(group, core_rows, category="scene_context", container="scene_content",
                       field=field, min_confidence=min_confidence)
        for field in SCENE_FIELDS
    ]
    temporal = [
        build_contrast(group, temporal_rows, category="surrounding_event", container="temporal_content",
                       field=field, min_confidence=min_confidence)
        for field in TEMPORAL_FIELDS
    ]
    time = time_candidate(group)
    return {
        "action_content": [x for x in action if x],
        "scene_context": [x for x in scene if x],
        "surrounding_event": [x for x in temporal if x],
        "temporal_localization": [time] if time else [],
    }


def scene_candidate(group: dict, groundings: list[dict], min_confidence: float) -> dict | None:
    candidates = [
        build_contrast(group, groundings, category="scene_context", container="scene_content",
                       field=field, min_confidence=min_confidence)
        for field in SCENE_FIELDS
    ]
    return choose_best([x for x in candidates if x], list(SCENE_FIELDS))


def temporal_candidate(group: dict, groundings: list[dict], min_confidence: float) -> dict | None:
    candidates = [
        build_contrast(group, groundings, category="surrounding_event", container="temporal_content",
                       field=field, min_confidence=min_confidence)
        for field in TEMPORAL_FIELDS
    ]
    return choose_best([x for x in candidates if x], list(TEMPORAL_FIELDS))


def time_candidate(group: dict) -> dict | None:
    if len(group.get("instances", [])) < 2:
        return None
    alternatives = []
    for row in group["instances"]:
        start, end = row["core"]
        alternatives.append({
            "answer": f"{start:g}-{end:g} seconds",
            "support": [{"instance_id": row["instance_id"], "interval": row["core"],
                         "confidence": 1.0, "visual_basis": "Official E.T. Bench target interval."}],
        })
    return {
        "parent_job_id": group["job_id"],
        "category": "temporal_localization",
        "field": "official_interval",
        "official_action": group["anchor_action_hint"],
        "alternatives": alternatives,
        "score": 1.0,
    }


def select_candidates(group: dict, core_rows: list[dict], temporal_rows: list[dict], role_schema: dict,
                      min_confidence: float) -> tuple[list[dict], list[dict]]:
    """Return at most one candidate per category plus explicit skip records."""
    attempts = {
        "action_content": action_candidate(group, core_rows, role_schema, min_confidence),
        "scene_context": scene_candidate(group, core_rows, min_confidence),
        "surrounding_event": temporal_candidate(group, temporal_rows, min_confidence),
        "temporal_localization": time_candidate(group),
    }
    selected, skipped = [], []
    for category in CATEGORY_ORDER:
        if attempts[category] is None:
            skipped.append({"parent_job_id": group["job_id"], "category": category,
                            "reason": "fewer_than_two_distinct_high_confidence_values"})
        else:
            selected.append(attempts[category])
    return selected[:4], skipped


def apply_semantic_groups(candidate: dict, decision: dict) -> tuple[dict | None, str | None]:
    """Merge synonymous alternatives by index without allowing new answer text."""
    if not decision.get("genuinely_distinct"):
        return None, "text_judge_found_no_mutually_exclusive_values"
    raw_groups = decision.get("groups")
    if not isinstance(raw_groups, list):
        return None, "invalid_semantic_groups"
    count = len(candidate["alternatives"])
    seen: list[int] = []
    groups: list[list[int]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, list) or not raw_group:
            return None, "invalid_semantic_groups"
        try:
            group = [int(x) for x in raw_group]
        except (TypeError, ValueError):
            return None, "invalid_semantic_groups"
        if any(x < 0 or x >= count for x in group):
            return None, "invalid_semantic_groups"
        groups.append(group)
        seen.extend(group)
    if sorted(seen) != list(range(count)) or len(set(seen)) != count or len(groups) < 2:
        return None, "invalid_or_single_semantic_partition"
    merged = dict(candidate)
    merged["alternatives"] = []
    for group in groups:
        first = candidate["alternatives"][group[0]]
        support = []
        for index in group:
            support.extend(candidate["alternatives"][index]["support"])
        merged["alternatives"].append({"answer": first["answer"], "support": support})
    merged["semantic_judge_reason"] = str(decision.get("reason", ""))
    return merged, None


def attach_locked_options(candidate: dict, question: str, model_metadata: dict) -> dict:
    options = []
    for index, alternative in enumerate(candidate["alternatives"]):
        options.append({
            "option_id": chr(ord("A") + index),
            "answer": alternative["answer"],
            "supported_by": alternative["support"],
        })
    return {
        "id": f"{candidate['parent_job_id']}__{candidate['category']}",
        "parent_job_id": candidate["parent_job_id"],
        "official_action": candidate["official_action"],
        "conflict_type": candidate["category"],
        "grounding_field": candidate["field"],
        "question": question.strip(),
        "options": options,
        "valid_option_ids": [row["option_id"] for row in options],
        "answer_mode": "multiple_valid_answers",
        "status": "accepted",
        "provenance": model_metadata,
    }


def validate_record(record: dict, candidate: dict, group: dict) -> list[str]:
    errors = []
    if not str(record.get("question", "")).strip():
        errors.append("empty_question")
    if len(record.get("options", [])) < 2:
        errors.append("fewer_than_two_options")
    expected_answers = {normalize_value(x["answer"]) for x in candidate["alternatives"]}
    actual_answers = {normalize_value(x.get("answer")) for x in record.get("options", [])}
    if actual_answers != expected_answers:
        errors.append("answers_not_evidence_locked")
    expected_intervals = {tuple(x["core"]) for x in group["instances"]}
    for option in record.get("options", []):
        for support in option.get("supported_by", []):
            if tuple(support.get("interval", [])) not in expected_intervals:
                errors.append("non_official_interval")
    if len(record.get("valid_option_ids", [])) != len(record.get("options", [])):
        errors.append("valid_option_set_mismatch")
    return sorted(set(errors))


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
