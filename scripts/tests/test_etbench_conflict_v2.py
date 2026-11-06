from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines/etbench" / "conflict"))

from conflict_v2_core import apply_semantic_groups, attach_locked_options, candidate_lists, select_candidates, validate_record


def field(value: str, confidence: float = .95) -> dict:
    return {"value": value, "confidence": confidence, "visual_basis": "visible in core frames"}


GROUP = {
    "job_id": "sample",
    "anchor_action_hint": "lifting something and placing it back down",
    "instances": [
        {"instance_id": "occurrence_0", "core": [1.0, 2.0]},
        {"instance_id": "occurrence_1", "core": [4.0, 5.0]},
        {"instance_id": "occurrence_2", "core": [7.0, 8.0]},
    ],
}
ROLE_SCHEMA = {"roles": [{"name": "lifted_object"}]}


def test_four_categories_are_selected_when_each_differs() -> None:
    core = [
        {"instance_id": "occurrence_0", "action_content": {"lifted_object": field("yellow mug")},
         "scene_content": {"location": field("kitchen")}},
        {"instance_id": "occurrence_1", "action_content": {"lifted_object": field("black cup")},
         "scene_content": {"location": field("dining room")}},
        {"instance_id": "occurrence_2", "action_content": {"lifted_object": field("yellow mug")},
         "scene_content": {"location": field("kitchen")}},
    ]
    temporal = [
        {"instance_id": "occurrence_0", "temporal_content": {"event_before": field("reaches for mug")}},
        {"instance_id": "occurrence_1", "temporal_content": {"event_before": field("opens cupboard")}},
        {"instance_id": "occurrence_2", "temporal_content": {"event_before": field("reaches for mug")}},
    ]
    selected, skipped = select_candidates(GROUP, core, temporal, ROLE_SCHEMA, .75)
    assert [x["category"] for x in selected] == [
        "action_content", "scene_context", "surrounding_event", "temporal_localization"
    ]
    assert skipped == []
    assert len(selected) == 4


def test_equal_categories_are_skipped_but_time_remains() -> None:
    core = [
        {"instance_id": row["instance_id"], "action_content": {"lifted_object": field("same mug")},
         "scene_content": {"location": field("same kitchen")}}
        for row in GROUP["instances"]
    ]
    temporal = [
        {"instance_id": row["instance_id"], "temporal_content": {"event_before": field("same reach")}}
        for row in GROUP["instances"]
    ]
    selected, skipped = select_candidates(GROUP, core, temporal, ROLE_SCHEMA, .75)
    assert [x["category"] for x in selected] == ["temporal_localization"]
    assert {x["category"] for x in skipped} == {"action_content", "scene_context", "surrounding_event"}


def test_low_confidence_difference_is_skipped() -> None:
    core = [
        {"instance_id": "occurrence_0", "action_content": {"lifted_object": field("yellow mug", .6)},
         "scene_content": {}},
        {"instance_id": "occurrence_1", "action_content": {"lifted_object": field("black cup", .95)},
         "scene_content": {}},
    ]
    selected, _ = select_candidates(GROUP, core, [], ROLE_SCHEMA, .75)
    assert all(x["category"] != "action_content" for x in selected)


def test_options_and_intervals_are_evidence_locked() -> None:
    candidate = {
        "parent_job_id": "sample",
        "category": "action_content",
        "field": "lifted_object",
        "official_action": GROUP["anchor_action_hint"],
        "alternatives": [
            {"answer": "yellow mug", "support": [{"instance_id": "occurrence_0", "interval": [1.0, 2.0]}]},
            {"answer": "black cup", "support": [{"instance_id": "occurrence_1", "interval": [4.0, 5.0]}]},
        ],
    }
    record = attach_locked_options(candidate, "What object was lifted and placed back down?", {"qa_model": "test"})
    assert record["valid_option_ids"] == ["A", "B"]
    assert validate_record(record, candidate, GROUP) == []
    record["options"][0]["answer"] = "invented answer"
    assert "answers_not_evidence_locked" in validate_record(record, candidate, GROUP)


def test_semantic_groups_merge_synonyms_without_inventing_answers() -> None:
    candidate = {
        "alternatives": [
            {"answer": "yellow mug", "support": [{"instance_id": "occurrence_0"}]},
            {"answer": "a yellow cup", "support": [{"instance_id": "occurrence_1"}]},
            {"answer": "black cup", "support": [{"instance_id": "occurrence_2"}]},
        ]
    }
    merged, error = apply_semantic_groups(candidate, {
        "genuinely_distinct": True,
        "groups": [[0, 1], [2]],
        "reason": "yellow mug/cup are synonyms here",
    })
    assert error is None
    assert [x["answer"] for x in merged["alternatives"]] == ["yellow mug", "black cup"]
    assert len(merged["alternatives"][0]["support"]) == 2


def test_semantic_gate_can_skip_false_difference() -> None:
    candidate = {"alternatives": [{"answer": "mug", "support": []}, {"answer": "a cup", "support": []}]}
    merged, error = apply_semantic_groups(candidate, {
        "genuinely_distinct": False,
        "groups": [[0, 1]],
        "reason": "synonyms",
    })
    assert merged is None
    assert error == "text_judge_found_no_mutually_exclusive_values"


def test_action_candidates_follow_role_priority_not_confidence() -> None:
    role_schema = {"roles": [{"name": "removed_item"}, {"name": "source_container"}]}
    core = [
        {"instance_id": "occurrence_0", "action_content": {
            "removed_item": field("orange bottle", .8), "source_container": field("white bag with handles", 1.0)},
         "scene_content": {}},
        {"instance_id": "occurrence_1", "action_content": {
            "removed_item": field("yellow box", .8), "source_container": field("white paper bag", 1.0)},
         "scene_content": {}},
    ]
    lists = candidate_lists(GROUP, core, [], role_schema, .75)
    assert [x["field"] for x in lists["action_content"]] == ["removed_item", "source_container"]


def test_scene_candidates_are_available_for_runner_fallback() -> None:
    core = [
        {"instance_id": "occurrence_0", "action_content": {"lifted_object": field("same mug")},
         "scene_content": {"location": field("kitchen")}},
        {"instance_id": "occurrence_1", "action_content": {"lifted_object": field("same mug")},
         "scene_content": {"location": field("garden")}},
    ]
    lists = candidate_lists(GROUP, core, [], ROLE_SCHEMA, .75)
    assert lists["action_content"] == []
    assert [x["field"] for x in lists["scene_context"]] == ["location"]
