#!/usr/bin/env python3
import glob
import json
import re
from collections import Counter


def answer(row):
    parsed = row.get("model_parsed") or {}
    value = parsed.get("answer")
    if isinstance(value, str) and value.strip().upper() in set("ABCDER"):
        return value.strip().upper()
    raw = str(row.get("model_response") or parsed.get("raw") or "")
    values = set(re.findall(r'''["']answer["']\s*:\s*["']([A-ER])["']''', raw, re.I))
    if len(values) == 1:
        return next(iter(values)).upper()
    labels = set(re.findall(r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE)["']''', raw, re.I))
    if {x.upper() for x in labels} == {"UNANSWERABLE"}:
        return "R"
    return None


rows = []
for path in glob.glob("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/results/gemini25pro_main_table_add479_20260921/results/shard_*/merged.classify.jsonl"):
    rows += [json.loads(x) for x in open(path) if x.strip()]
unparsed = [row for row in rows if answer(row) is None]
print(json.dumps({
    "returned": len(rows),
    "parseable": len(rows) - len(unparsed),
    "unparsed": len(unparsed),
    "unparsed_finish_reasons": dict(Counter((row.get("api_logprobs") or {}).get("finish_reason") for row in unparsed)),
    "unparsed_record_ids": [row.get("record_id") for row in unparsed],
}, indent=2))
