#!/usr/bin/env python3
"""Merge the recovered Gemini outputs and compute sampled endpoint metrics."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
BASE = ROOT / "results/closed_paper_sample_20260920/gemini25pro_loose_parseable.jsonl"
RETRY = ROOT / "results/gemini25pro_v2_retry49_20260920/merged.classify.jsonl"
OUT = ROOT / "results/closed_paper_sample_20260920"
VALID = set("ABCDER")
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_REPLICATES = 10_000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def loose_answer(row: dict) -> str | None:
    parsed = row.get("model_parsed") or {}
    value = parsed.get("answer")
    if isinstance(value, str) and value.strip().upper() in VALID:
        return value.strip().upper()
    raw = str(row.get("model_response") or parsed.get("raw") or "").strip()
    values = {x.upper() for x in re.findall(
        r'''["']answer["']\s*:\s*["']([A-ER])["']''', raw, re.I)}
    if len(values) == 1:
        return next(iter(values))
    labels = {x.upper() for x in re.findall(
        r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE)["']''', raw, re.I)}
    if labels == {"UNANSWERABLE"}:
        return "R"
    return None


def correct_reason(row: dict) -> bool:
    value = (row.get("model_parsed") or {}).get("reject_reason")
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z]+", "_", value.strip().lower()).strip("_")
    return normalized in {"insufficient", "insufficient_evidence", "evidence_insufficient"}


def cluster_id(row: dict) -> str:
    return f"{row.get('source_dataset')}::{row.get('video_id')}"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(rows: list[dict], metric_functions: dict[str, object]) -> dict:
    clusters = defaultdict(list)
    for row in rows:
        clusters[cluster_id(row)].append(row)
    cluster_names = sorted(clusters)
    estimates = {
        name: sum(function(row) for row in rows) / len(rows)
        for name, function in metric_functions.items()
    }
    rng = random.Random(BOOTSTRAP_SEED)
    draws = {name: [] for name in metric_functions}
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = []
        for _ in cluster_names:
            sampled.extend(clusters[rng.choice(cluster_names)])
        for name, function in metric_functions.items():
            draws[name].append(sum(function(row) for row in sampled) / len(sampled))
    return {
        "n": len(rows),
        "source_video_clusters": len(cluster_names),
        "estimates_percent": {name: round(100 * value, 4) for name, value in estimates.items()},
        "ci_percent": {
            name: [round(100 * percentile(values, 0.025), 4),
                   round(100 * percentile(values, 0.975), 4)]
            for name, values in draws.items()
        },
    }


base = read_jsonl(BASE)
retry = read_jsonl(RETRY)
assert len(base) == 751
assert len(retry) == 49
rows = base + retry
assert len({row["record_id"] for row in rows}) == 800
assert all(loose_answer(row) is not None for row in rows)

for row in rows:
    row["paper_loose_answer"] = loose_answer(row)

conditions = defaultdict(list)
for row in rows:
    condition = row["evidence_condition"]
    if condition == "partially_sufficient":
        condition = "partial"
    conditions[condition].append(row)
assert {name: len(values) for name, values in conditions.items()} == {
    "sufficient": 267, "partial": 266, "insufficient": 267}

sufficient = summarize(conditions["sufficient"], {
    "qa_accuracy": lambda row: loose_answer(row) in set(row.get("valid_option_ids") or []),
    "coverage": lambda row: loose_answer(row) in set("ABCDE"),
})
insufficient = summarize(conditions["insufficient"], {
    "reject": lambda row: loose_answer(row) == "R",
    "reason": lambda row: loose_answer(row) == "R" and correct_reason(row),
})
partial = summarize(conditions["partial"], {
    "answer_rate": lambda row: loose_answer(row) in set("ABCDE"),
    "qa_accuracy": lambda row: loose_answer(row) in set(row.get("valid_option_ids") or []),
})
partial_by_alpha = {
    str(alpha): summarize([row for row in conditions["partial"] if float(row["alpha"]) == alpha], {
        "answer_rate": lambda row: loose_answer(row) in set("ABCDE"),
        "qa_accuracy": lambda row: loose_answer(row) in set(row.get("valid_option_ids") or []),
    })
    for alpha in (0.2, 0.4, 0.5, 0.6)
}

destination = OUT / "gemini25pro_sample800_final.jsonl"
payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
destination.write_text(payload, encoding="utf-8")
summary = {
    "records": len(rows),
    "condition_counts": dict(Counter(
        "partial" if row["evidence_condition"] == "partially_sufficient" else row["evidence_condition"]
        for row in rows)),
    "all_loose_parseable": True,
    "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                  "unit": "source_video", "interval": "percentile_95"},
    "sufficient": sufficient,
    "insufficient": insufficient,
    "partial": partial,
    "partial_by_alpha": partial_by_alpha,
}
(OUT / "gemini25pro_sample800_metrics.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
