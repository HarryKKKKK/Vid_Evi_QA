"""Paired v2.1/v2.2 prompt comparison with a pre-specified margin."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    indexed = {}
    for row in data:
        key = row.get("record_id")
        if not key and row.get("id"):
            key = f"{row.get('source_dataset', 'unknown')}::{row['id']}::conflict"
        if not key:
            raise KeyError(f"Cannot identify result row: {row.get('video_id')} / {row.get('qid')}")
        indexed[str(key)] = row
    return indexed


def condition(row):
    value = str(row.get("evidence_condition", "")).lower()
    return {"partially_sufficient": "partial", "conflicting": "conflict"}.get(value, value)


def decision(row):
    parsed = row.get("model_parsed") or {}
    return parsed.get("label"), parsed.get("answer"), parsed.get("reject_reason")


def correct(row):
    parsed = row.get("model_parsed") or {}
    expected = row.get("expected_behavior")
    if expected == "answer":
        return int(
            parsed.get("label") == "ANSWERABLE"
            and parsed.get("answer") in set(row.get("valid_option_ids") or [])
        )
    if expected == "reject":
        wanted = row.get("expected_reject_reason")
        return int(
            parsed.get("label") == "UNANSWERABLE"
            and (wanted is None or parsed.get("reject_reason") == wanted)
        )
    return None


def paired_ci(differences, seed, draws=20000):
    if not differences:
        return None
    rng = random.Random(seed)
    n = len(differences)
    samples = sorted(sum(rng.choice(differences) for _ in range(n)) / n for _ in range(draws))
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws)]]


def summarize(left, right, margin, seed):
    ids = sorted(set(left) & set(right))
    scored = [(left[key], right[key]) for key in ids if correct(left[key]) is not None and correct(right[key]) is not None]
    differences = [correct(b) - correct(a) for a, b in scored]
    ci = paired_ci(differences, seed)
    result = {
        "paired_records": len(ids),
        "v21_only": len(set(left) - set(right)),
        "v22_only": len(set(right) - set(left)),
        "decision_agreement": sum(decision(left[key]) == decision(right[key]) for key in ids) / len(ids) if ids else None,
        "label_agreement": sum(decision(left[key])[0] == decision(right[key])[0] for key in ids) / len(ids) if ids else None,
        "format_valid_v21": sum(not (left[key].get("schema_errors") or []) for key in ids) / len(ids) if ids else None,
        "format_valid_v22": sum(not (right[key].get("schema_errors") or []) for key in ids) / len(ids) if ids else None,
        "accuracy_v21": sum(correct(a) for a, _ in scored) / len(scored) if scored else None,
        "accuracy_v22": sum(correct(b) for _, b in scored) / len(scored) if scored else None,
        "accuracy_delta_v22_minus_v21": sum(differences) / len(differences) if differences else None,
        "paired_bootstrap_95_ci": ci,
        "equivalence_margin": margin,
        "equivalent_within_margin": bool(ci and ci[0] > -margin and ci[1] < margin),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("model")
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    left = load(args.root / args.model / "v2.1" / "merged.classify.json")
    right = load(args.root / args.model / "v2.2" / "merged.classify.json")
    report = {
        "model": args.model,
        "overall": summarize(left, right, args.margin, args.seed),
        "by_condition": {},
        "decision_changes": dict(Counter(
            f"{decision(left[key])}->{decision(right[key])}"
            for key in set(left) & set(right) if decision(left[key]) != decision(right[key])
        )),
    }
    for name in ("sufficient", "insufficient", "partial", "conflict"):
        left_part = {key: row for key, row in left.items() if condition(row) == name}
        right_part = {key: row for key, row in right.items() if condition(row) == name}
        report["by_condition"][name] = summarize(left_part, right_part, args.margin, args.seed)
    output = args.root / args.model / "prompt_equivalence.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
