#!/usr/bin/env python3
"""Compute paired Sufficient/Insufficient and Partial C/W/R diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ALPHAS = (0.2, 0.4, 0.5, 0.6)
VALID = frozenset("ABCDER")


def rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm_answer(value):
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    return value if value in VALID else None


def unique_match(pattern, text):
    values = {value.upper() for value in re.findall(pattern, text, re.I)}
    return next(iter(values)) if len(values) == 1 else None


def loose_answer(row):
    if not row:
        return None
    parsed = row.get("model_parsed") or {}
    answer = norm_answer(parsed.get("answer"))
    if answer:
        return answer
    raw = str(row.get("model_response") or parsed.get("raw") or "").strip()
    if not raw:
        return None
    answer = unique_match(r'''["']answer["']\s*:\s*["']([A-ER])["']''', raw)
    if answer:
        return answer
    answer = unique_match(r"(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([A-ER])\)?\b", raw)
    if answer:
        return answer
    exact = re.fullmatch(r"\s*\(?([A-ER])\)?[.\s]*", raw, re.I)
    if exact:
        return exact.group(1).upper()
    label_answer = norm_answer(parsed.get("label"))
    if label_answer:
        return label_answer
    label = unique_match(r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE)["']''', raw)
    if str(parsed.get("label") or label or "").strip().upper() == "UNANSWERABLE":
        return "R"
    return None


def probability(row):
    if not row:
        return None
    value = ((row.get("model_parsed") or {}).get("decision_probabilities") or {}).get("ANSWERABLE")
    return float(value) if isinstance(value, (int, float)) and 0 <= float(value) <= 1 else None


def reason_value(row):
    if not row:
        return None
    return (row.get("model_parsed") or {}).get("reject_reason")


def correct_insufficient_reason(value):
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z]+", "_", value.strip().lower()).strip("_")
    return normalized in {"insufficient", "insufficient_evidence", "evidence_insufficient"}


def pct(n, d):
    return round(100 * n / d, 2) if d else None


def state(answer, gold):
    if answer == "R":
        return "R"
    if answer is not None and answer in "ABCDE":
        return "C" if answer in gold else "W"
    return "U"


def transition_summary(items, left, right):
    matrix = Counter((states[left][0], states[right][0]) for states in items)
    total = len(items)
    valid = sum(v for (a, b), v in matrix.items() if "U" not in (a, b))
    named = {f"{a}->{b}": {"n": matrix[a, b], "pct_all": pct(matrix[a, b], total)}
             for a in "CWRU" for b in "CWRU" if matrix[a, b]}
    r_to_a = matrix["R", "C"] + matrix["R", "W"]
    a_to_r = matrix["C", "R"] + matrix["W", "R"]
    prob_deltas = defaultdict(list)
    for states in items:
        a_state, a_prob = states[left]
        b_state, b_prob = states[right]
        if a_prob is not None and b_prob is not None:
            prob_deltas[f"{a_state}->{b_state}"].append(100 * (b_prob - a_prob))
    return {
        "n_all_pairs": total,
        "n_both_parsed": valid,
        "excluded_unparsed_pct": pct(total - valid, total),
        "matrix": named,
        "net_answer_change": round(pct(r_to_a, total) - pct(a_to_r, total), 2),
        "answer_status_switch_rate": pct(r_to_a + a_to_r, total),
        "any_cwr_state_change_rate": pct(sum(v for (a, b), v in matrix.items() if a != b and "U" not in (a, b)), total),
        "mean_delta_p_answerable_by_transition": {
            key: {"n": len(values), "mean_pp": round(sum(values) / len(values), 2)}
            for key, values in sorted(prob_deltas.items()) if values
        },
    }


def evaluate(manifest_path, prediction_path):
    partial = defaultdict(dict)
    paired = defaultdict(dict)
    wanted = set()
    for row in rows(manifest_path):
        condition = row.get("evidence_condition")
        record_id = str(row.get("record_id") or "")
        base = str(row.get("id") or "")
        gold = frozenset(row.get("valid_option_ids") or [])
        if condition in {"partial", "partially_sufficient"} and row.get("alpha") is not None:
            alpha = float(row["alpha"])
            if alpha in ALPHAS:
                partial[base][alpha] = (record_id, gold, row.get("evidence_intervals") or [])
                wanted.add(record_id)
        elif condition in {"sufficient", "insufficient"}:
            paired[base][condition] = (record_id, gold, row.get("evidence_intervals") or [])
            wanted.add(record_id)

    predictions = {}
    for row in rows(prediction_path):
        record_id = str(row.get("record_id") or "")
        if record_id in wanted and record_id not in predictions:
            predictions[record_id] = row

    partial_items = []
    sampling_checked = partial_grid_matched = sufficient_grid_matched = 0
    evidence_frame_counts = []
    for base, by_alpha in partial.items():
        if all(alpha in by_alpha for alpha in ALPHAS):
            states = {}
            for alpha in ALPHAS:
                record_id, gold, _ = by_alpha[alpha]
                pred = predictions.get(record_id)
                states[alpha] = (state(loose_answer(pred), gold), probability(pred))
            partial_items.append(states)
            partial_preds = [predictions.get(by_alpha[alpha][0]) for alpha in ALPHAS]
            sufficient_pred = None
            if base in paired and "sufficient" in paired[base]:
                sufficient_pred = predictions.get(paired[base]["sufficient"][0])
            all_preds = partial_preds + [sufficient_pred]
            if all(pred and isinstance(pred.get("sampled_seconds"), list) for pred in all_preds):
                sampling_checked += 1
                grids = [tuple(round(float(x), 4) for x in pred["sampled_seconds"]) for pred in all_preds]
                partial_grid_matched += len(set(grids[:4])) == 1
                sufficient_grid_matched += grids[0] == grids[4]
                intervals = by_alpha[ALPHAS[0]][2]
                count = sum(any(float(iv["start"]) <= t <= float(iv["end"]) for iv in intervals) for t in grids[0])
                evidence_frame_counts.append(count)

    adjacent = {
        f"{left:.1f}->{right:.1f}": transition_summary(partial_items, left, right)
        for left, right in zip(ALPHAS, ALPHAS[1:])
    }
    endpoint = transition_summary(partial_items, ALPHAS[0], ALPHAS[-1])
    trajectories = Counter("".join(states[a][0] for a in ALPHAS) for states in partial_items)

    pair_counts = Counter()
    reason_audit = Counter()
    sufficient_states = Counter()
    sufficient_probabilities = []
    pair_total = 0
    for by_condition in paired.values():
        if not all(c in by_condition for c in ("sufficient", "insufficient")):
            continue
        pair_total += 1
        s_id, s_gold, _ = by_condition["sufficient"]
        i_id, i_gold, _ = by_condition["insufficient"]
        s_pred, i_pred = predictions.get(s_id), predictions.get(i_id)
        s_answer, i_answer = loose_answer(s_pred), loose_answer(i_pred)
        s_state, i_state = state(s_answer, s_gold), state(i_answer, i_gold)
        sufficient_states[s_state] += 1
        s_probability = probability(s_pred)
        if s_probability is not None:
            sufficient_probabilities.append(s_probability)
        s_decision = "R" if s_state == "R" else ("A" if s_state in "CW" else "U")
        i_decision = "R" if i_state == "R" else ("A" if i_state in "CW" else "U")
        pair_counts[f"decision_{s_decision}->{i_decision}"] += 1
        pair_counts[f"state_{s_state}->{i_state}"] += 1
        if s_state == "C" and i_state == "R":
            pair_counts["joint_sufficient_correct_insufficient_reject"] += 1
            if correct_insufficient_reason(reason_value(i_pred)):
                pair_counts["joint_reason_aware_success"] += 1
        if i_state == "R":
            value = reason_value(i_pred)
            if value is None or (isinstance(value, str) and not value.strip()):
                reason_audit["missing_or_null"] += 1
            elif correct_insufficient_reason(value):
                reason_audit["canonical_correct"] += 1
            else:
                reason_audit[f"other::{value}"] += 1

    paired_output = {
        "n_pairs": pair_total,
        "sufficient_reference": {
            "coverage": pct(sufficient_states["C"] + sufficient_states["W"], pair_total),
            "qa_accuracy": pct(sufficient_states["C"], pair_total),
            "mean_p_answerable": round(100 * sum(sufficient_probabilities) / len(sufficient_probabilities), 2) if sufficient_probabilities else None,
        },
        "metrics": {key: {"n": value, "pct": pct(value, pair_total)} for key, value in sorted(pair_counts.items())},
        "insufficient_reject_reason_audit": {
            "denominator_rejected": sum(reason_audit.values()),
            "counts": dict(reason_audit.most_common()),
        },
    }
    return {
        "partial": {
            "n_qa": len(partial_items),
            "sampling_audit": {
                "n_checked": sampling_checked,
                "identical_grid_across_four_alpha_pct": pct(partial_grid_matched, sampling_checked),
                "sufficient_grid_matches_partial_pct": pct(sufficient_grid_matched, sampling_checked),
                "zero_sampled_evidence_frames_pct": pct(sum(x == 0 for x in evidence_frame_counts), len(evidence_frame_counts)),
                "mean_sampled_evidence_frames": round(sum(evidence_frame_counts) / len(evidence_frame_counts), 2) if evidence_frame_counts else None,
                "min_sampled_evidence_frames": min(evidence_frame_counts) if evidence_frame_counts else None,
            },
            "adjacent": adjacent,
            "endpoint_0.2_to_0.6": endpoint,
            "top_trajectories": [
                {"trajectory": key, "n": value, "pct": pct(value, len(partial_items))}
                for key, value in trajectories.most_common(12)
            ],
        },
        "sufficient_to_insufficient": paired_output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", action="append", default=[], help="NAME=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        result[name] = evaluate(args.manifest, path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
