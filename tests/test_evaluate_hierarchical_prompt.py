from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def load_evaluator():
    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")
        openai_stub.OpenAI = lambda *args, **kwargs: object()
        sys.modules["openai"] = openai_stub
    path = ROOT / "scripts" / "eval" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("vid_evi_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_conflict_evaluator():
    if "openai" not in sys.modules:
        openai_stub = types.ModuleType("openai")
        openai_stub.OpenAI = lambda *args, **kwargs: object()
        sys.modules["openai"] = openai_stub
    if "run_conflict_pipeline_v2" not in sys.modules:
        pipeline_stub = types.ModuleType("run_conflict_pipeline_v2")
        pipeline_stub.encode_frames = lambda *args, **kwargs: []
        sys.modules["run_conflict_pipeline_v2"] = pipeline_stub
    conflict_dir = ROOT / "etbench_pipeline" / "conflict"
    sys.path.insert(0, str(conflict_dir))
    path = conflict_dir / "evaluate_conflict_v2.py"
    spec = importlib.util.spec_from_file_location("vid_evi_conflict_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classify_prompt_uses_user_v2_five_options() -> None:
    evaluator = load_evaluator()
    prompt = evaluator.build_instruction(
        "What happened?",
        [f"choice {letter}" for letter in "ABCDE"],
        task="classify",
    )

    assert '"A", "B", "C", "D", or "E"' in prompt
    assert '|' not in prompt
    assert "option_probabilities_if_answered" not in prompt
    assert "option_id" not in prompt
    assert "three hierarchical" not in prompt.lower()
    assert "CONFLICTING_EVIDENCE" in prompt


def test_grouped_evidence_is_flattened_for_legacy_metrics() -> None:
    evaluator = load_evaluator()
    raw = json.dumps({
        "label": "ANSWERABLE",
        "answer": "B",
        "evidence": [
            {
                "group_id": "G1",
                "evidence_seconds": [[1.0, 2.0], [5.0, 6.0]],
                "quality": 1.0,
                "description": "two relevant moments",
            }
        ],
    })

    parsed = evaluator.parse_model_json(raw)
    assert parsed["evidence_seconds"] == [
        {"start": 1.0, "end": 2.0},
        {"start": 5.0, "end": 6.0},
    ]


def test_benchmark_metadata_normalizes_analysis_fields() -> None:
    evaluator = load_evaluator()
    metadata = evaluator.build_benchmark_metadata(
        {
            "source_dataset": "NExTGQA",
            "question_type": "TN",
            "source_task": "Temporal",
            "alpha": 0.4,
            "partial_mode": "noise",
        },
        "C2",
    )

    assert metadata == {
        "evidence_condition": "partial",
        "source_dataset": "nextgqa",
        "question_type": "TN",
        "source_task": "Temporal",
        "question_subtype": None,
        "expected_behavior": None,
        "alpha": 0.4,
        "partial_mode": "noise",
    }


def test_conflict_evaluator_uses_same_unprompted_evidence_schema() -> None:
    evaluator = load_conflict_evaluator()
    options = [
        {"option_id": "A", "answer": "red cup"},
        {"option_id": "B", "answer": "blue cup"},
        {"option_id": "C", "answer": "green cup"},
        {"option_id": "D", "answer": "white cup"},
        {"option_id": "E", "answer": "black cup"},
    ]
    prompt = evaluator.build_prompt("Which cup?", options)
    assert "option_id" not in prompt
    assert '"A", "B", "C", "D", or "E"' in prompt
    assert '|' not in prompt

    scored = evaluator.score({
        "label": "UNANSWERABLE",
        "answer": "R",
        "reject_reason": "CONFLICTING_EVIDENCE",
        "decision_probabilities": {"ANSWERABLE": 0.1, "UNANSWERABLE": 0.9},
        "evidence": [
            {"group_id": "G1", "evidence_seconds": [[1, 1]], "description": "Option A: visible red cup."},
            {"group_id": "G2", "evidence_seconds": [[1, 1]], "description": "Option B: visible blue cup."},
        ],
        "reason": "Two distinct options are supported.",
        "option_probabilities_if_answered": {"A": 0.45, "B": 0.44, "C": 0.11},
    }, ["A", "B"])
    assert scored["label_correct"] is True
    assert scored["option_set_exact"] is True
    assert scored["expected_option_probability_mass"] == 0.89


def test_multiple_correct_options_in_one_frame_trigger_rejection() -> None:
    prompts = [
        load_evaluator().build_instruction('What is visible?', ['A carrot','A chair','A cup','A hat','A fork'], task='classify'),
        load_conflict_evaluator().build_prompt('What is visible?', [
            {'option_id': letter, 'answer': choice} for letter, choice in zip('ABCDE', ['A carrot','A chair','A cup','A hat','A fork'])]),
    ]
    for prompt in prompts:
        assert 'Their time intervals may overlap.' in prompt
        assert 'at least two distinct options satisfy the question' in prompt
        assert 'one group for each distinct supported answer' in prompt
        assert 'simultaneously in the same frame' in prompt
        assert 'different intervals support at least two incompatible' not in prompt
