#!/usr/bin/env python3
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")


def answer(row):
    parsed = row.get("model_parsed") or {}
    value = str(parsed.get("answer") or "").strip().upper()
    if value in "ABCDER" and len(value) == 1:
        return value
    raw = str(row.get("model_response") or parsed.get("raw") or "")
    values = set(re.findall(r'''["']answer["']\s*:\s*["']([A-ER])["']''', raw, re.I))
    if len(values) == 1:
        return next(iter(values)).upper()
    labels = set(re.findall(r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE)["']''', raw, re.I))
    if labels == {"UNANSWERABLE"}:
        return "R"
    return None


def base_id(row):
    return str(row.get("id") or "")


def audit(name, rows, expected_prompt):
    unique = {}
    for row in rows:
        unique[str(row.get("record_id") or (base_id(row) + "::" + str(row.get("evidence_condition"))))] = row
    rows = list(unique.values())
    by = defaultdict(dict)
    for row in rows:
        cond = str(row.get("evidence_condition") or "")
        by[base_id(row)][cond] = row
    pairs = [(v["sufficient"], v["insufficient"]) for v in by.values()
             if "sufficient" in v and "insufficient" in v]
    parseable_pairs = sum(answer(s) is not None and answer(i) is not None for s, i in pairs)
    print(json.dumps({
        "name": name,
        "returned_unique": len(rows),
        "conditions": dict(Counter(r.get("evidence_condition") for r in rows)),
        "prompt_versions": dict(Counter(r.get("prompt_version") for r in rows)),
        "matches_main_prompt": all(r.get("prompt_version") == expected_prompt for r in rows),
        "complete_sufficient_insufficient_pairs": len(pairs),
        "parseable_complete_pairs": parseable_pairs,
        "conflict_records": sum(r.get("evidence_condition") == "conflict" for r in rows),
        "loose_parseable": sum(answer(r) is not None for r in rows),
    }, ensure_ascii=False))


gpt = []
for path in (ROOT / "results/gpt4o_v2_full_20260914/results/gpt4o").glob("shard_*/merged.classify.jsonl"):
    gpt.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
gemini = [json.loads(line) for line in (ROOT / "results/gemini25pro_v2_sample800_20260919/merged.classify.jsonl").read_text(
    encoding="utf-8").splitlines() if line.strip()]
audit("gpt4o", gpt, "v2.2")
audit("gemini25pro", gemini, "v2.2")
