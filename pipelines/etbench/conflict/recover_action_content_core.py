"""Pure selection and validation helpers for missing-action recovery."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/etbench/conflict/recover_action_content_core.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/etbench/conflict/recover_action_content_core.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/etbench/conflict', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


from collections import defaultdict
from typing import Any


MISSING_ACTIONS = {"", "none", "null", "unknown", "unclear", "not_visible", "n/a"}


def _missing(value: Any) -> bool:
    return str(value or "").strip().lower() in MISSING_ACTIONS


def _has_action_content(row: dict) -> bool:
    content = row.get("action_content")
    return isinstance(content, dict) and bool(content)


def build_recovery_audit(
    groups: list[dict],
    schemas: list[dict],
    core_rows: list[dict],
    skipped_rows: list[dict],
    existing_qa: list[dict],
) -> list[dict]:
    """Classify every group; eligible means action-content was never truly attempted."""
    schema_by_parent = {row.get("parent_job_id"): row for row in schemas}
    core_by_parent: dict[str, list[dict]] = defaultdict(list)
    skip_by_parent: dict[str, list[dict]] = defaultdict(list)
    qa_by_parent: dict[str, list[dict]] = defaultdict(list)
    for row in core_rows:
        core_by_parent[str(row.get("parent_job_id"))].append(row)
    for row in skipped_rows:
        skip_by_parent[str(row.get("parent_job_id"))].append(row)
    for row in existing_qa:
        qa_by_parent[str(row.get("parent_job_id"))].append(row)

    audit = []
    for group in groups:
        parent_id = str(group["job_id"])
        schema = schema_by_parent.get(parent_id, {})
        parent_core = core_by_parent.get(parent_id, [])
        reasons = []
        if not _missing(group.get("anchor_action_hint")):
            reasons.append("action_hint_already_present")
        if schema.get("roles"):
            reasons.append("action_roles_already_present")
        if any(_has_action_content(row) for row in parent_core):
            reasons.append("action_content_grounding_already_present")
        if not parent_core:
            reasons.append("no_previous_core_grounding")
        if len(group.get("instances", [])) < 2:
            reasons.append("fewer_than_two_occurrences")
        if any(row.get("conflict_type") == "action_content" for row in qa_by_parent.get(parent_id, [])):
            reasons.append("accepted_action_content_already_exists")
        has_expected_skip = any(
            row.get("category") == "action_content"
            and row.get("reason") == "no_candidate_survived_traversal"
            for row in skip_by_parent.get(parent_id, [])
        )
        if not has_expected_skip:
            reasons.append("not_marked_as_failed_action_traversal")
        audit.append({
            "parent_job_id": parent_id,
            "source_task": group.get("source_task"),
            "source_idx": group.get("source_idx"),
            "video": group.get("video"),
            "occurrence_count": len(group.get("instances", [])),
            "eligible": not reasons,
            "exclusion_reasons": reasons,
        })
    return audit


def validate_recovered_action(
    detection: dict,
    expected_instance_ids: list[str],
    min_confidence: float,
) -> list[str]:
    """Require one visible action that every official occurrence actually matches."""
    errors = []
    action = detection.get("shared_action")
    if _missing(action):
        errors.append("missing_shared_action")
    if detection.get("consistent_across_occurrences") is not True:
        errors.append("not_consistent_across_occurrences")
    try:
        confidence = float(detection.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < min_confidence:
        errors.append("action_confidence_below_threshold")
    per_instance = detection.get("per_instance")
    if not isinstance(per_instance, list):
        errors.append("invalid_per_instance_evidence")
        return sorted(set(errors))
    observed_ids = [str(row.get("instance_id")) for row in per_instance if isinstance(row, dict)]
    if sorted(observed_ids) != sorted(expected_instance_ids) or len(set(observed_ids)) != len(observed_ids):
        errors.append("instance_coverage_mismatch")
    if any(row.get("matches_shared_action") is not True for row in per_instance if isinstance(row, dict)):
        errors.append("one_or_more_occurrences_do_not_match")
    if any(not str(row.get("visual_basis", "")).strip() for row in per_instance if isinstance(row, dict)):
        errors.append("missing_instance_visual_basis")
    return sorted(set(errors))
