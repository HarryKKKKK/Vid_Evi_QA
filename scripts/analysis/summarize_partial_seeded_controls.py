#!/usr/bin/env python3
"""Summarize the 100-QA seeded evidence/non-evidence control experiment."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def parse_response(row):
    text = str((row.get("response") or {}).get("text") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return None


def state(row):
    parsed = parse_response(row)
    if not parsed:
        return "U"
    answer = str(parsed.get("answer") or "").strip().upper()
    if row["protocol"] == "selective":
        label = str(parsed.get("label") or "").strip().upper()
        if label == "UNANSWERABLE":
            return "R"
        if label != "ANSWERABLE" or answer not in "ABCDE":
            return "U"
    elif answer not in "ABCDE":
        return "U"
    return "C" if answer == str(row["answer"]).strip().upper() else "W"


def mean(values):
    return sum(values) / len(values) if values else None


def pct(value):
    return None if value is None else round(100 * value, 2)


def load_files(root):
    groups = {
        "selective": list((root / "pilot100_v1").glob("*.jsonl"))
                     + list((root / "molmo_pilot100_v1").glob("*.jsonl")),
        "forced": list((root / "forced_pilot100_v1").glob("*.jsonl"))
                  + list((root / "molmo_forced_pilot100_v1").glob("*.jsonl")),
    }
    rows = []
    for protocol, paths in groups.items():
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    row["protocol"] = protocol
                    row["_state"] = state(row)
                    rows.append(row)
    return rows


def summarize(rows):
    models = sorted({r["model_tag"] for r in rows})
    result = {"models": {}, "record_count": len(rows)}
    for model in models:
        mrows = [r for r in rows if r["model_tag"] == model and r.get("status") == "ok"]
        out = {"records": len(mrows), "selective": {}, "forced": {}, "paired": {}}
        for protocol in ("selective", "forced"):
            prows = [r for r in mrows if r["protocol"] == protocol]
            buckets = defaultdict(list)
            for r in prows:
                key = "original" if r["condition"] == "original" else f"{r['condition']}_a{r['alpha']:.1f}"
                buckets[key].append(r)
            for key, values in sorted(buckets.items()):
                states = [r["_state"] for r in values]
                out[protocol][key] = {
                    "n": len(values), "correct": pct(mean([x == "C" for x in states])),
                    "answer_rate": pct(mean([x in ("C", "W") for x in states])),
                    "reject_rate": pct(mean([x == "R" for x in states])),
                    "invalid_rate": pct(mean([x == "U" for x in states])),
                }

        selective = [r for r in mrows if r["protocol"] == "selective"]
        index = {(r["video_id"], r["qid"], r["condition"], r.get("alpha"), r.get("noise_seed")): r
                 for r in selective}
        specificity = {}
        for alpha in (0.2, 0.4, 0.5, 0.6):
            pairs = []
            for key, e in index.items():
                vid, qid, condition, a, seed = key
                if condition != "evidence" or a != alpha:
                    continue
                n = index.get((vid, qid, "non_evidence", a, seed))
                if n:
                    pairs.append((e["_state"], n["_state"]))
            specificity[f"a{alpha:.1f}"] = {
                "n_pairs": len(pairs),
                "accuracy_non_evidence_minus_evidence_pp": round(100 * mean([(n == "C") - (e == "C") for e, n in pairs]), 2),
                "reject_evidence_minus_non_evidence_pp": round(100 * mean([(e == "R") - (n == "R") for e, n in pairs]), 2),
            }
        out["paired"]["evidence_specificity"] = specificity

        seed_instability = {}
        for condition in ("evidence", "non_evidence"):
            for alpha in (0.2, 0.4, 0.5, 0.6):
                by_item = defaultdict(dict)
                for r in selective:
                    if r["condition"] == condition and r["alpha"] == alpha:
                        by_item[(r["video_id"], r["qid"])][r["noise_seed"]] = r["_state"]
                disagreements = []
                any_switch = []
                for values in by_item.values():
                    if len(values) == 3:
                        states = list(values.values())
                        disagreements.extend(a != b for a, b in itertools.combinations(states, 2))
                        any_switch.append(len(set(states)) > 1)
                seed_instability[f"{condition}_a{alpha:.1f}"] = {
                    "items": len(any_switch),
                    "pairwise_state_disagreement_pct": pct(mean(disagreements)),
                    "any_seed_state_change_pct": pct(mean(any_switch)),
                }
        out["paired"]["seed_instability"] = seed_instability

        forced = [r for r in mrows if r["protocol"] == "forced"]
        findex = {(r["video_id"], r["qid"], r["condition"], r.get("alpha"), r.get("noise_seed")): r
                  for r in forced}
        decision_gap = {}
        for condition, alpha, seed in (("original", None, None), ("evidence", 0.2, 17),
                                       ("evidence", 0.6, 17), ("non_evidence", 0.2, 17),
                                       ("non_evidence", 0.6, 17)):
            pairs = []
            for key, f in findex.items():
                vid, qid, c, a, s = key
                if (c, a, s) != (condition, alpha, seed):
                    continue
                sel = index.get((vid, qid, c, a, s))
                if sel:
                    pairs.append((sel["_state"], f["_state"]))
            label = "original" if condition == "original" else f"{condition}_a{alpha:.1f}"
            decision_gap[label] = {
                "n_pairs": len(pairs),
                "forced_minus_selective_accuracy_pp": round(100 * mean([(f == "C") - (s == "C") for s, f in pairs]), 2),
                "selective_refused_but_forced_correct_pct": pct(mean([s == "R" and f == "C" for s, f in pairs])),
            }
        out["paired"]["decision_gap"] = decision_gap
        result["models"][model] = out
    return result


def main():
    a = parse_args()
    summary = summarize(load_files(a.root))
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
