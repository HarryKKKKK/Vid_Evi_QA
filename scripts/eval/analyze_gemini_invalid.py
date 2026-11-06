#!/usr/bin/env python3
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
sys.path.insert(0, str(ROOT / "results/open_v2_deployment_20260902_r4/eval"))
from prompt_v2 import response_errors

out = ROOT / "results/gemini25pro_v2_sample800_20260919"
rows = json.loads((out / "merged.classify.json").read_text(encoding="utf-8"))
invalid = [(r, response_errors(r.get("model_parsed"))) for r in rows]
invalid = [(r, errors) for r, errors in invalid if errors]
missing = [json.loads(line) for line in (out / "merged.classify.skipped.jsonl").read_text(
    encoding="utf-8").splitlines() if line.strip()]

counts = Counter()
error_sets = Counter()
examples = {}
for row, errors in invalid:
    parsed = row.get("model_parsed") or {}
    raw = str(parsed.get("raw") or row.get("model_response") or "")
    label = parsed.get("label") if parsed.get("label") in ("ANSWERABLE", "UNANSWERABLE") else None
    answer = parsed.get("answer") if parsed.get("answer") in list("ABCDER") else None
    if not label:
        match = re.search(r'"label"\s*:\s*"(ANSWERABLE|UNANSWERABLE)"', raw)
        label = match.group(1) if match else None
    if not answer:
        match = re.search(r'"answer"\s*:\s*"([A-ER])"', raw)
        answer = match.group(1) if match else None
    reject = re.search(
        r'"reject_reason"\s*:\s*(null|"(?:INSUFFICIENT_EVIDENCE|CONFLICTING_EVIDENCE)")', raw)
    if label:
        counts["label_extractable"] += 1
    if answer:
        counts["answer_extractable"] += 1
    if label and answer:
        counts["label_answer_extractable"] += 1
    if label and answer and ((label == "ANSWERABLE" and answer in "ABCDE") or
                             (label == "UNANSWERABLE" and answer == "R")):
        counts["decision_pair_consistent"] += 1
    if reject:
        counts["reject_field_extractable"] += 1
    if label and answer and reject:
        counts["core_three_extractable"] += 1
    counts["json_parse_error" if parsed.get("parse_error") else "json_parsed_but_schema_invalid"] += 1
    error_sets[tuple(errors)] += 1
    examples.setdefault(tuple(errors), {"record_id": row.get("record_id"), "raw": raw[:500]})

api_failures = Counter()
for row in missing:
    reason = str(row.get("reason", ""))
    if "timeout" in reason.lower():
        api_failures["timeout"] += 1
    elif "PROHIBITED_CONTENT" in reason:
        api_failures["prohibited_content"] += 1
    else:
        api_failures["other"] += 1

print(json.dumps({
    "invalid_response_records": len(invalid),
    "no_response_api_failures": len(missing),
    "recovery": dict(counts),
    "validation_error_sets": [
        {"count": count, "errors": list(key), "example": examples[key]}
        for key, count in error_sets.most_common()
    ],
    "api_failures": dict(api_failures),
}, ensure_ascii=False, indent=2))
