#!/usr/bin/env python3
"""Create paired diagnostics from qwen25_risk_ladder.py JSONL output."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


TAUS = (0.0, 0.2, 0.5, 0.8, 0.95)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    return p.parse_args()


def load_latest(path: Path) -> list[dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "ok":
                rows[str(row["key"])] = row
    return list(rows.values())


def ikey(row: dict) -> tuple[str, str]:
    return str(row["video_id"]), str(row["qid"])


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    a = args()
    rows = load_latest(a.input)
    forced = {(ikey(r), r["evidence_condition"]): r for r in rows if r.get("condition") == "forced"}
    risks = {(ikey(r), r["evidence_condition"], r["risk_id"], r["prompt_variant"]): r
             for r in rows if r.get("condition") == "risk"}

    forced_metrics = {}
    for evidence in ("full", "no_evidence"):
        values = [r for (key, ev), r in forced.items() if ev == evidence]
        confidences = [r["response"]["choice_probabilities"][r["response"]["decision"]] for r in values]
        outcomes = [r["response"]["decision"] == r["gold_answer"] for r in values]
        brier = mean([(p - int(y)) ** 2 for p, y in zip(confidences, outcomes)])
        bins = []
        for low, high in zip((0, .2, .4, .6, .8), (.2, .4, .6, .8, 1.0000001)):
            idx = [i for i, p in enumerate(confidences) if low <= p < high]
            if idx:
                bins.append({"range": [low, min(high, 1.0)], "n": len(idx),
                             "mean_confidence": mean([confidences[i] for i in idx]),
                             "accuracy": mean([int(outcomes[i]) for i in idx])})
        ece = sum(b["n"] / len(values) * abs(b["mean_confidence"] - b["accuracy"]) for b in bins)
        forced_metrics[evidence] = {
            "n": len(values), "accuracy": mean([int(v) for v in outcomes]),
            "mean_chosen_probability": mean(confidences), "brier": brier, "ece_5bin": ece,
            "reliability_bins": bins,
            "raw_threshold_coverage": {str(tau): mean([int(p >= tau) for p in confidences]) for tau in TAUS},
        }

    common = sorted(set(k for k, ev in forced if ev == "full") & set(k for k, ev in forced if ev == "no_evidence"))
    full_no = {
        "n": len(common),
        "answer_agreement": mean([int(forced[(k, "full")]["response"]["decision"] == forced[(k, "no_evidence")]["response"]["decision"]) for k in common]),
        "full_correct_no_evidence_wrong": sum(forced[(k, "full")]["response"]["decision"] == forced[(k, "full")]["gold_answer"] and forced[(k, "no_evidence")]["response"]["decision"] != forced[(k, "no_evidence")]["gold_answer"] for k in common),
        "full_wrong_no_evidence_correct": sum(forced[(k, "full")]["response"]["decision"] != forced[(k, "full")]["gold_answer"] and forced[(k, "no_evidence")]["response"]["decision"] == forced[(k, "no_evidence")]["gold_answer"] for k in common),
    }

    comparisons = []
    monotonicity = []
    for evidence in ("full", "no_evidence"):
        keys = sorted(k for k, ev in forced if ev == evidence)
        for risk_id, tau in zip(("tau000", "tau020", "tau050", "tau080", "tau095"), TAUS):
            r0 = [risks[(k, evidence, risk_id, 0)] for k in keys]
            r1 = [risks[(k, evidence, risk_id, 1)] for k in keys]
            a0 = [r["response"]["decision"] != "R" for r in r0]
            a1 = [r["response"]["decision"] != "R" for r in r1]
            raw = []
            for k in keys:
                f = forced[(k, evidence)]
                p = f["response"]["choice_probabilities"][f["response"]["decision"]]
                raw.append(p >= tau)
            for variant, observed in ((0, a0), (1, a1)):
                observed_rows = r0 if variant == 0 else r1
                forced_confidence = []
                for k in keys:
                    f = forced[(k, evidence)]
                    forced_confidence.append(f["response"]["choice_probabilities"][f["response"]["decision"]])
                answered_indices = [i for i, value in enumerate(observed) if value]
                refused_indices = [i for i, value in enumerate(observed) if not value]
                comparisons.append({
                    "evidence": evidence, "risk_id": risk_id, "tau": tau, "variant": variant,
                    "prompt_coverage": mean([int(x) for x in observed]),
                    "raw_threshold_coverage": mean([int(x) for x in raw]),
                    "decision_agreement_with_raw_threshold": mean([int(x == y) for x, y in zip(observed, raw)]),
                    "prompt_answers_below_raw_threshold": sum(x and not y for x, y in zip(observed, raw)),
                    "prompt_refuses_above_raw_threshold": sum(not x and y for x, y in zip(observed, raw)),
                    "mean_forced_confidence_when_prompt_answers": mean([forced_confidence[i] for i in answered_indices]),
                    "mean_forced_confidence_when_prompt_refuses": mean([forced_confidence[i] for i in refused_indices]),
                    "answered_choice_agreement_with_forced": mean([
                        int(observed_rows[i]["response"]["decision"] == forced[(keys[i], evidence)]["response"]["decision"])
                        for i in answered_indices
                    ]),
                    "forced_correct_but_prompt_refuses": sum(
                        forced[(keys[i], evidence)]["response"]["decision"] == forced[(keys[i], evidence)]["gold_answer"]
                        for i in refused_indices
                    ),
                    "forced_wrong_and_prompt_refuses": sum(
                        forced[(keys[i], evidence)]["response"]["decision"] != forced[(keys[i], evidence)]["gold_answer"]
                        for i in refused_indices
                    ),
                })
            comparisons.append({
                "evidence": evidence, "risk_id": risk_id, "tau": tau, "comparison": "variant_pair",
                "decision_agreement": mean([int(x == y) for x, y in zip(a0, a1)]),
                "v0_answer_v1_refuse": sum(x and not y for x, y in zip(a0, a1)),
                "v0_refuse_v1_answer": sum(not x and y for x, y in zip(a0, a1)),
            })
        for variant in (0, 1):
            sequences = []
            for k in keys:
                sequence = [int(risks[(k, evidence, risk_id, variant)]["response"]["decision"] != "R")
                            for risk_id in ("tau000", "tau020", "tau050", "tau080", "tau095")]
                sequences.append(sequence)
            monotonicity.append({
                "evidence": evidence, "variant": variant, "n": len(sequences),
                "monotone_nonincreasing": sum(all(seq[i] >= seq[i + 1] for i in range(4)) for seq in sequences),
                "nonmonotone": sum(not all(seq[i] >= seq[i + 1] for i in range(4)) for seq in sequences),
                "unique_sequences": {"".join(map(str, seq)): count for seq, count in sorted(Counter(map(tuple, sequences)).items())},
            })

    audits = [r for r in rows if r.get("condition") == "reasoning_audit"]
    audit_stats = {}
    audit_examples = []
    for evidence in ("full", "no_evidence"):
        values = [r for r in audits if r["evidence_condition"] == evidence]
        parsed = [r for r in values if isinstance(r["response"].get("parsed"), dict)]
        confidences = []
        for r in parsed:
            value = r["response"]["parsed"].get("confidence_percent")
            if isinstance(value, (int, float)):
                confidences.append(float(value))
        audit_stats[evidence] = {
            "n": len(values), "json_parsed": len(parsed), "mean_stated_confidence_percent": mean(confidences),
            "says_would_answer": sum(r["response"]["parsed"].get("would_answer_under_this_scoring") is True for r in parsed),
            "says_used_elimination": sum(r["response"]["parsed"].get("used_elimination") is True for r in parsed),
            "best_answer_correct": sum(str(r["response"]["parsed"].get("best_answer", "")).upper() == r["gold_answer"] for r in parsed),
        }
        for r in parsed[:3]:
            audit_examples.append({"video_id": r["video_id"], "qid": r["qid"], "evidence": evidence,
                                   "gold": r["gold_answer"], "audit_trigger": r["audit_trigger_risk_id"],
                                   "report": r["response"]["parsed"]})

    result = {
        "input": str(a.input), "ok_records": len(rows), "forced": forced_metrics,
        "full_vs_no_evidence": full_no, "risk_vs_raw_threshold": comparisons,
        "risk_monotonicity": monotonicity, "reasoning_audit": audit_stats,
        "audit_examples": audit_examples,
        "caveat": "Confidence diagnostics use a deliberately selected Qwen-refusal subset and raw constrained-choice probabilities; they are not population calibration estimates.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
