#!/usr/bin/env python3
"""Create Table-6-compatible loose metrics for complete frozen-manifest runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from compute_paired_behavior import loose_answer, state


def norm_reason(value):
    return re.sub(r"[^a-z]+", "_", str(value or "").strip().lower()).strip("_")


def condition(row):
    return str((row.get("benchmark_metadata") or {}).get("evidence_condition")
               or row.get("evidence_condition") or "unknown").lower()


def pct(n, d):
    return round(100 * n / d, 2) if d else None


def load(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summarize(path):
    counts = defaultdict(Counter)
    total = Counter()
    for row in load(path):
        c = condition(row)
        answer = loose_answer(row)
        gold = frozenset(row.get("valid_option_ids") or [])
        s = state(answer, gold)
        reason = norm_reason((row.get("model_parsed") or {}).get("reject_reason"))
        counts[c]["n"] += 1
        counts[c][s] += 1
        if answer == "R" and reason in {"insufficient", "insufficient_evidence", "evidence_insufficient"}:
            counts[c]["insufficient_reason"] += 1
        if answer == "R" and reason in {"conflict", "conflicting", "conflicting_evidence", "multiple_valid_answers"}:
            counts[c]["conflict_reason"] += 1
        total["n"] += 1
        total["parsed"] += answer is not None
    s, i, c = counts["sufficient"], counts["insufficient"], counts["conflict"]
    suff_cov = pct(s["C"] + s["W"], s["n"])
    suff_acc = pct(s["C"], s["n"])
    suff_reject = pct(s["R"], s["n"])
    ins_reject = pct(i["R"], i["n"])
    return {
        "records": total["n"], "parse_pct": pct(total["parsed"], total["n"]),
        "sufficient": {"n": s["n"], "qa_accuracy": suff_acc, "coverage": suff_cov,
                       "conditional_accuracy": pct(s["C"], s["C"] + s["W"]),
                       "reject": suff_reject, "unparsed": pct(s["U"], s["n"])},
        "insufficient": {"n": i["n"], "reject": ins_reject,
                         "reason": pct(i["insufficient_reason"], i["n"]),
                         "unparsed": pct(i["U"], i["n"]),
                         "delta_reject": round(ins_reject - suff_reject, 2)},
        "conflict": {"n": c["n"], "reject": pct(c["R"], c["n"]),
                     "detect": pct(c["conflict_reason"], c["n"]),
                     "single": pct(c["C"] + c["W"], c["n"]),
                     "unparsed": pct(c["U"], c["n"])},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=[], help="NAME=JSONL")
    p.add_argument("--paired")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    paired = json.loads(Path(a.paired).read_text(encoding="utf-8")) if a.paired else {}
    result = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        value = summarize(path)
        if name in paired:
            metrics = paired[name]["sufficient_to_insufficient"]["metrics"]
            value["paired"] = {
                "joint": metrics.get("joint_sufficient_correct_insufficient_reject", {}).get("pct"),
                "joint_reason": metrics.get("joint_reason_aware_success", {}).get("pct"),
            }
        result[name] = value
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
