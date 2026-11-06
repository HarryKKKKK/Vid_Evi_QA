#!/usr/bin/env python3
"""Estimate representative matched-control ESS with clustered intervals."""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260916)
    return p.parse_args()


def parse_response(row: dict) -> dict:
    text = str((row.get("response") or {}).get("text") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group())
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def answer_letter(value: object) -> str | None:
    match = re.search(r"\b([A-E])\b", str(value or "").upper())
    return match.group(1) if match else None


def state(row: dict) -> str:
    value = parse_response(row)
    label = str(value.get("label") or "").upper()
    if label == "UNANSWERABLE":
        return "R"
    answer = answer_letter(value.get("answer") or value.get("final_answer"))
    if label != "ANSWERABLE" or not answer:
        return "U"
    return "C" if answer == str(row.get("answer") or "").upper() else "W"


def weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    denominator = sum(weight for _, weight in pairs)
    return sum(value * weight for value, weight in pairs) / denominator if denominator else None


def evaluate(items: list[dict]) -> dict[str, float | None]:
    item_values = defaultdict(list)
    seed_values = defaultdict(list)
    for item in items:
        weight = item["weight"]
        original = item["original"]
        item_values["original_correct"].append((original == "C", weight))
        seed_ess = []
        seed_e_reject = []
        seed_n_correct = []
        for seed, (evidence, non_evidence) in item["seeds"].items():
            seed_values["evidence_reject"].append((evidence == "R", weight))
            seed_values["non_evidence_correct"].append((non_evidence == "C", weight))
            seed_values["ess"].append((original == "C" and evidence == "R" and non_evidence == "C", weight))
            seed_values["conditional_ess_num"].append((evidence == "R" and non_evidence == "C", weight if original == "C" else 0.0))
            seed_values["conditional_e_reject"].append((evidence == "R", weight if original == "C" else 0.0))
            seed_values["conditional_n_correct"].append((non_evidence == "C", weight if original == "C" else 0.0))
            seed_values["specificity_gap"].append(((evidence == "R") - (non_evidence == "R"), weight if original == "C" else 0.0))
            seed_ess.append(evidence == "R" and non_evidence == "C")
            seed_e_reject.append(evidence == "R")
            seed_n_correct.append(non_evidence == "C")
        majority = math.ceil(len(seed_ess) / 2)
        item_values["robust_ess"].append((original == "C" and sum(seed_ess) >= majority, weight))
        item_values["robust_conditional_ess_num"].append((sum(seed_ess) >= majority, weight if original == "C" else 0.0))
        item_values["majority_evidence_reject"].append((sum(seed_e_reject) >= majority, weight))
        item_values["majority_non_evidence_correct"].append((sum(seed_n_correct) >= majority, weight))
    result = {key: weighted_mean(values) for key, values in item_values.items()}
    result.update({key: weighted_mean(values) for key, values in seed_values.items()})
    result["conditional_ess"] = result.pop("conditional_ess_num")
    result["robust_conditional_ess"] = result.pop("robust_conditional_ess_num")
    return result


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def main() -> None:
    args = parse_args()
    paths = sorted({Path(path) for pattern in args.inputs for path in glob.glob(pattern)})
    raw_by_model = defaultdict(list)
    errors = defaultdict(int)
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model = str(row.get("model_tag") or path.stem)
            if row.get("status") != "ok":
                errors[model] += 1
            else:
                row["_state"] = state(row)
                raw_by_model[model].append(row)

    report = {"analysis": {"bootstrap_replicates": args.bootstrap_replicates,
                            "seed": args.seed,
                            "target": "matched-control-eligible Sufficient NExT-GQA population",
                            "primary_metric": "ESS=P(original=C,evidence-corrupted=R,matched-non-evidence-corrupted=C)"},
              "files": [str(path) for path in paths], "models": {}}
    metrics_for_ci = ("original_correct", "ess", "conditional_ess", "robust_ess",
                      "robust_conditional_ess", "conditional_e_reject",
                      "conditional_n_correct", "specificity_gap")
    for model, rows in sorted(raw_by_model.items()):
        index = defaultdict(dict)
        for row in rows:
            item = (str(row["video_id"]), str(row["qid"]))
            condition = row.get("condition")
            if condition == "original":
                index[item]["original"] = row
            elif condition in {"evidence", "non_evidence"} and float(row.get("alpha")) == 0.0:
                index[item][(condition, int(row["noise_seed"]))] = row
        items = []
        for (video_id, qid), values in index.items():
            seeds = sorted({key[1] for key in values if isinstance(key, tuple)})
            complete = [seed for seed in seeds if ("evidence", seed) in values and ("non_evidence", seed) in values]
            if "original" not in values or not complete:
                continue
            original_row = values["original"]
            items.append({
                "video_id": video_id, "qid": qid,
                "weight": float(original_row.get("sampling_weight") or 1.0),
                "original": original_row["_state"],
                "seeds": {seed: (values[("evidence", seed)]["_state"],
                                  values[("non_evidence", seed)]["_state"])
                          for seed in complete},
            })
        point = evaluate(items)
        clusters = defaultdict(list)
        for item in items:
            clusters[item["video_id"]].append(item)
        cluster_ids = sorted(clusters)
        draws = defaultdict(list)
        rng = random.Random(args.seed)
        for _ in range(args.bootstrap_replicates):
            sampled = []
            for _ in cluster_ids:
                sampled.extend(clusters[rng.choice(cluster_ids)])
            values = evaluate(sampled)
            for metric in metrics_for_ci:
                if values.get(metric) is not None:
                    draws[metric].append(values[metric])
        ci = {metric: [percentile(draws[metric], 0.025), percentile(draws[metric], 0.975)]
              for metric in metrics_for_ci if draws[metric]}

        seed_specific = {}
        for seed in sorted({s for item in items for s in item["seeds"]}):
            reduced = [{**item, "seeds": {seed: item["seeds"][seed]}}
                       for item in items if seed in item["seeds"]]
            seed_specific[str(seed)] = evaluate(reduced)
        report["models"][model] = {
            "complete_items": len(items), "source_video_clusters": len(cluster_ids),
            "error_records": errors[model], "point_estimates": point,
            "cluster_bootstrap_ci": ci, "seed_specific": seed_specific,
        }

    def percentize(value):
        if isinstance(value, float):
            return round(value * 100, 3)
        if isinstance(value, list):
            return [percentize(x) for x in value]
        if isinstance(value, dict):
            return {k: percentize(v) for k, v in value.items()}
        return value

    report["models_percent"] = percentize(report["models"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["models_percent"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
