"""Frozen v2 prompt and conservative resume/schema helpers."""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/src/vid_evi_qa/prompts/prompt_v2.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/prompt_v2.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import hashlib
import json
import math

PROMPT_VERSION = "v2.2"
TEMPLATE = '''Answer the multiple-choice question using only the supplied video frames. Each frame is labeled with its timestamp in seconds.

Apply this decision rule before writing JSON:

1. Determine the set S of answer options supported by the visible frames.
2. If S contains no options, answer R with INSUFFICIENT_EVIDENCE.
3. If S contains exactly one option, answer that option.
4. If S contains two or more distinct options, answer R with CONFLICTING_EVIDENCE.

CONFLICTING_EVIDENCE means multiple options satisfy the question; the options do not need to contradict each other. They may be supported at different times or simultaneously in the same frame.

- INSUFFICIENT_EVIDENCE: not enough information to determine the answer.
- CONFLICTING_EVIDENCE: at least two distinct options satisfy the question.

Evidence groups represent supported answer options, not individual frames. Create at most one group per supported option. Merge adjacent timestamps into intervals; never create one group per frame. Use at most five groups and at most three intervals per group. Every interval must be a two-number array inside the evidence_seconds array, as in [[2.0,3.0]], even when there is only one interval.

Return exactly one JSON object with these six fields: label, answer, decision_probabilities, reject_reason, evidence, reason.

Field rules:
- label must contain exactly one string: "ANSWERABLE" or "UNANSWERABLE". Never concatenate the values.
- For ANSWERABLE, answer must be exactly one letter: "A", "B", "C", "D", or "E". Set reject_reason to JSON null.
- For UNANSWERABLE, answer must be "R". Set reject_reason to exactly one string: "INSUFFICIENT_EVIDENCE" or "CONFLICTING_EVIDENCE".
- decision_probabilities must contain two numeric values between 0 and 1, named ANSWERABLE and UNANSWERABLE. They must sum to 1; do not set both to zero.
- Each evidence group must have a unique group_id from "G1" through "G5", one to three numeric evidence_seconds intervals, and a brief description beginning with the supported option ID, such as "Option B: ...".
- For a conflict, provide one group for each distinct supported answer, with at least two groups total. Their time intervals may overlap.
- reason must be one short sentence. Use an empty evidence list when nothing relevant is shown.

The following are fictional format examples, not answers to the current question. Do not copy their labels, answers, probabilities, timestamps, or descriptions. Determine your own values from the supplied frames, and return only one JSON object without Markdown.

Example 1: one uniquely supported answer.
{"label":"ANSWERABLE","answer":"C","decision_probabilities":{"ANSWERABLE":0.9,"UNANSWERABLE":0.1},"reject_reason":null,"evidence":[{"group_id":"G1","evidence_seconds":[[2.0,3.0]],"description":"Option C: the visible action supports this option alone."}],"reason":"Only option C is supported by the frames."}

Example 2: no relevant evidence.
{"label":"UNANSWERABLE","answer":"R","decision_probabilities":{"ANSWERABLE":0.1,"UNANSWERABLE":0.9},"reject_reason":"INSUFFICIENT_EVIDENCE","evidence":[],"reason":"The requested information is not visible."}

Example 3: two distinct options supported at the same time.
{"label":"UNANSWERABLE","answer":"R","decision_probabilities":{"ANSWERABLE":0.05,"UNANSWERABLE":0.95},"reject_reason":"CONFLICTING_EVIDENCE","evidence":[{"group_id":"G1","evidence_seconds":[[4.0,5.0]],"description":"Option A: the described object is visible."},{"group_id":"G2","evidence_seconds":[[4.0,5.0]],"description":"Option D: the different described object is also visible."}],"reason":"Both A and D satisfy the question, so there is no unique answer."}

Example 4: two distinct options supported at different times.
{"label":"UNANSWERABLE","answer":"R","decision_probabilities":{"ANSWERABLE":0.05,"UNANSWERABLE":0.95},"reject_reason":"CONFLICTING_EVIDENCE","evidence":[{"group_id":"G1","evidence_seconds":[[1.0,2.0]],"description":"Option B: the described action occurs earlier."},{"group_id":"G2","evidence_seconds":[[8.0,9.0]],"description":"Option E: the other described action occurs later."}],"reason":"Both B and E satisfy the question at different times, so there is no unique answer."}

Question: {question}

Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
E. {option_E}'''
PROMPT_SHA256 = hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()


# The transport adapters use this schema for constrained decoding when the
# backend supports it. response_errors() remains the authoritative validator,
# including cross-field consistency and probability-sum checks that a grammar
# may not enforce.
RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["ANSWERABLE", "UNANSWERABLE"]},
        "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E", "R"]},
        "decision_probabilities": {
            "type": "object",
            "properties": {
                "ANSWERABLE": {"type": "number", "minimum": 0, "maximum": 1},
                "UNANSWERABLE": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["ANSWERABLE", "UNANSWERABLE"],
            "additionalProperties": False,
        },
        "reject_reason": {
            "anyOf": [
                {"type": "null"},
                {"type": "string", "enum": ["INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"]},
            ]
        },
        "evidence": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "group_id": {"type": "string", "pattern": "^G[1-5]$"},
                    "evidence_seconds": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number", "minimum": 0},
                        },
                    },
                    "description": {"type": "string", "pattern": "^Option [A-E]: .+", "maxLength": 240},
                },
                "required": ["group_id", "evidence_seconds", "description"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 240},
    },
    "required": ["label", "answer", "decision_probabilities", "reject_reason", "evidence", "reason"],
    "additionalProperties": False,
}


def build_prompt(question, choices):
    if len(choices) != 5:
        raise ValueError("v2 requires exactly five choices")
    # Replace the final placeholders only; braces in JSON/question are literal.
    prefix, _ = TEMPLATE.rsplit("Question: {question}", 1)
    return prefix + "Question: " + question + "\n\nOptions:\n" + "\n".join(
        f"{letter}. {choice}" for letter, choice in zip("ABCDE", choices)
    )


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def record_identity(row):
    if row.get("record_id"):
        return str(row["record_id"])
    condition = row.get("evidence_condition")
    if condition in ("conflict", "conflicting", "conflicting_evidence") and row.get("id"):
        return f"{row.get('source_dataset', 'unknown')}::{row['id']}::conflict"
    return None


def response_errors(parsed):
    """Validate format, not factual correctness or the model's confidence choice."""
    if not isinstance(parsed, dict):
        return ["response is not an object"]
    errors = []
    required = {"label", "answer", "decision_probabilities", "reject_reason", "evidence", "reason"}
    if not required.issubset(parsed):
        errors.append("missing required fields")
    label, answer, reject = parsed.get("label"), parsed.get("answer"), parsed.get("reject_reason")
    if label == "ANSWERABLE":
        if answer not in list("ABCDE") or reject is not None:
            errors.append("inconsistent answerable decision")
    elif label == "UNANSWERABLE":
        if answer != "R" or reject not in ("INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE"):
            errors.append("inconsistent rejection")
    else:
        errors.append("invalid label")
    def number(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    probs = parsed.get("decision_probabilities")
    if (not isinstance(probs, dict) or set(probs) != {"ANSWERABLE", "UNANSWERABLE"}
            or not all(number(x) and 0 <= x <= 1 for x in probs.values())
            or not math.isclose(sum(probs.values()), 1, abs_tol=1e-6)):
        errors.append("invalid decision probabilities")
    evidence = parsed.get("evidence")
    if not isinstance(evidence, list):
        errors.append("evidence must be a list")
    else:
        if len(evidence) > 5:
            errors.append("too many evidence groups")
        seen = set()
        for group in evidence:
            if not isinstance(group, dict):
                errors.append("evidence group must be an object")
                continue
            gid = group.get("group_id")
            if gid not in {"G1", "G2", "G3", "G4", "G5"} or gid in seen:
                errors.append("invalid or duplicate group_id")
            else:
                seen.add(gid)
            description = group.get("description")
            if (not isinstance(description, str) or not description.strip()
                    or not description.startswith(tuple(f"Option {x}: " for x in "ABCDE"))):
                errors.append("missing evidence description")
            intervals = group.get("evidence_seconds")
            if not isinstance(intervals, list) or not intervals:
                errors.append("missing evidence intervals")
            elif len(intervals) > 3:
                errors.append("too many evidence intervals")
            elif any(not isinstance(pair, list) or len(pair) != 2
                     or not all(number(x) for x in pair) or not 0 <= pair[0] <= pair[1]
                     for pair in intervals):
                errors.append("invalid evidence interval")
        if reject == "CONFLICTING_EVIDENCE" and len(evidence) < 2:
            errors.append("conflict requires separate supported-answer groups")
    if not isinstance(parsed.get("reason"), str) or not parsed["reason"].strip():
        errors.append("missing reason")
    if parsed.get("parse_error"):
        errors.append("JSON parse error")
    return errors


def reusable(result, expected_fingerprint):
    return (result.get("prompt_version") == PROMPT_VERSION
            and result.get("prompt_sha256") == PROMPT_SHA256
            and result.get("input_fingerprint") == expected_fingerprint
            and not response_errors(result.get("model_parsed")))
