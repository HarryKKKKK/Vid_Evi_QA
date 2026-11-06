from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "etbench_pipeline" / "conflict"))

from recover_action_content_core import build_recovery_audit, validate_recovered_action


def test_only_never_attempted_missing_action_is_eligible() -> None:
    groups = [
        {"job_id": "missing", "source_task": "tem", "anchor_action_hint": None,
         "instances": [{"instance_id": "a"}, {"instance_id": "b"}]},
        {"job_id": "semantic_reject", "source_task": "tal", "anchor_action_hint": "lifting a glass",
         "instances": [{"instance_id": "a"}, {"instance_id": "b"}]},
        {"job_id": "accepted", "source_task": "tal", "anchor_action_hint": "folding clothing",
         "instances": [{"instance_id": "a"}, {"instance_id": "b"}]},
    ]
    schemas = [
        {"parent_job_id": "missing", "roles": []},
        {"parent_job_id": "semantic_reject", "roles": [{"name": "manipulated_object"}]},
        {"parent_job_id": "accepted", "roles": [{"name": "folded_item"}]},
    ]
    core = [
        {"parent_job_id": "missing", "instance_id": "a", "action_content": {}},
        {"parent_job_id": "missing", "instance_id": "b", "action_content": {}},
        {"parent_job_id": "semantic_reject", "instance_id": "a",
         "action_content": {"manipulated_object": {"value": "glass"}}},
        {"parent_job_id": "accepted", "instance_id": "a",
         "action_content": {"folded_item": {"value": "shirt"}}},
    ]
    skips = [
        {"parent_job_id": "missing", "category": "action_content",
         "reason": "no_candidate_survived_traversal"},
        {"parent_job_id": "semantic_reject", "category": "action_content",
         "reason": "no_candidate_survived_traversal"},
    ]
    existing = [{"parent_job_id": "accepted", "conflict_type": "action_content"}]
    audit = build_recovery_audit(groups, schemas, core, skips, existing)
    by_id = {row["parent_job_id"]: row for row in audit}
    assert by_id["missing"]["eligible"] is True
    assert by_id["semantic_reject"]["eligible"] is False
    assert "action_hint_already_present" in by_id["semantic_reject"]["exclusion_reasons"]
    assert "action_content_grounding_already_present" in by_id["semantic_reject"]["exclusion_reasons"]
    assert by_id["accepted"]["eligible"] is False
    assert "accepted_action_content_already_exists" in by_id["accepted"]["exclusion_reasons"]


def test_recovered_action_requires_all_occurrences_and_visual_support() -> None:
    detection = {
        "shared_action": "placing a small object on a table",
        "consistent_across_occurrences": True,
        "confidence": .93,
        "per_instance": [
            {"instance_id": "a", "matches_shared_action": True, "visual_basis": "object contacts table"},
            {"instance_id": "b", "matches_shared_action": True, "visual_basis": "object placed on table"},
        ],
    }
    assert validate_recovered_action(detection, ["a", "b"], .85) == []


def test_recovered_action_rejects_partial_or_low_confidence_match() -> None:
    detection = {
        "shared_action": "holding something",
        "consistent_across_occurrences": False,
        "confidence": .6,
        "per_instance": [
            {"instance_id": "a", "matches_shared_action": True, "visual_basis": "visible"},
            {"instance_id": "b", "matches_shared_action": False, "visual_basis": "different action"},
        ],
    }
    errors = validate_recovered_action(detection, ["a", "b"], .85)
    assert "not_consistent_across_occurrences" in errors
    assert "action_confidence_below_threshold" in errors
    assert "one_or_more_occurrences_do_not_match" in errors
