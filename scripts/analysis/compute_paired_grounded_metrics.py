"""Compute paired evidence/grounding metrics with source-video clustered inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gc
import json
import math
from pathlib import Path
import re
from typing import Callable

import numpy as np


PRIMARY_SIGNIFICANCE = (
    "suff_accuracy", "suff_coverage", "insuff_reject", "insuff_reason",
    "delta_reject", "ecs", "ewr", "ecs_reason", "detect", "miop", "miou",
    "acc_gqa_iop_05", "g_ecs_iop_05", "g_ewr_iop_05",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(start, end) for start, end in out]


def duration(spans: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in spans)


def intersection(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def raw_spans(value) -> list[tuple[float, float]]:
    if isinstance(value, dict):
        if "start" in value and "end" in value:
            return [(value["start"], value["end"])]
        return []
    if not isinstance(value, list):
        return []
    if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        return [(value[0], value[1])]
    out = []
    for item in value:
        out.extend(raw_spans(item))
    return out


def clean_spans(value, video_duration: float) -> tuple[list[tuple[float, float]], dict]:
    counters = Counter(raw=len(raw_spans(value)))
    cleaned = []
    for start, end in raw_spans(value):
        try:
            start, end = float(start), float(end)
        except (TypeError, ValueError):
            counters["non_numeric"] += 1
            continue
        if not math.isfinite(start) or not math.isfinite(end):
            counters["non_finite"] += 1
            continue
        if start > end:
            start, end = end, start
            counters["reversed"] += 1
        bounded_start = min(max(start, 0.0), video_duration)
        bounded_end = min(max(end, 0.0), video_duration)
        if bounded_start != start or bounded_end != end:
            counters["clamped"] += 1
        if bounded_end <= bounded_start:
            counters["zero_or_empty"] += 1
            continue
        cleaned.append((bounded_start, bounded_end))
    return union(cleaned), dict(counters)


def predicted_spans(row: dict) -> tuple[list[tuple[float, float]], dict]:
    parsed = row.get("model_parsed") or {}
    video_duration = float(row.get("video_duration") or max(row.get("sampled_seconds") or [0]))
    value = parsed.get("evidence_seconds")
    if value is None:
        value = [entry.get("evidence_seconds") for entry in (parsed.get("evidence") or [])
                 if isinstance(entry, dict)]
    recovered = recover_raw_object(row)
    used_raw = False
    if not raw_spans(value or []) and recovered:
        value = recovered.get("evidence_seconds")
        if value is None:
            value = [entry.get("evidence_seconds") for entry in (recovered.get("evidence") or [])
                     if isinstance(entry, dict)]
        used_raw = bool(raw_spans(value or []))
    spans, audit = clean_spans(value or [], video_duration)
    if used_raw:
        audit["raw_json_recovered"] = 1
    return spans, audit


def reference_spans(row: dict) -> list[tuple[float, float]]:
    video_duration = float(row.get("video_duration") or max(row.get("sampled_seconds") or [0]))
    spans, _ = clean_spans(row.get("evidence_intervals") or [], video_duration)
    return spans


def recover_raw_object(row: dict) -> dict:
    parsed = row.get("model_parsed") or {}
    raw = str(parsed.get("raw") or row.get("model_response") or "").strip()
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except (TypeError, json.JSONDecodeError):
            pass
    return {}


def unique_match(pattern: str, text: str) -> str | None:
    values = {match.upper() for match in re.findall(pattern, text, re.I)}
    return next(iter(values)) if len(values) == 1 else None


def loose_decision(row: dict) -> str | None:
    """Frozen model-agnostic answer-only parser used by the v2.2 main table."""
    parsed = row.get("model_parsed") or {}
    answer = str(parsed.get("answer") or "").strip().upper()
    if answer in "ABCDER" and len(answer) == 1:
        return answer
    raw = str(parsed.get("raw") or row.get("model_response") or "")
    answer = unique_match(r'''["']answer["']\s*:\s*["']([A-ER])["']''', raw)
    if answer:
        return answer
    answer = unique_match(r"(?:final\s+answer|answer)\s*(?:is|:)\s*\(?([A-ER])\)?\b", raw)
    if answer:
        return answer
    exact = re.fullmatch(r"\s*\(?([A-ER])\)?[.\s]*", raw, re.I)
    if exact:
        return exact.group(1).upper()
    label = str(parsed.get("label") or "").strip().upper()
    if label in "ABCDER" and len(label) == 1:
        return label
    raw_label = unique_match(r'''["']label["']\s*:\s*["'](ANSWERABLE|UNANSWERABLE)["']''', raw)
    if label == "UNANSWERABLE" or raw_label == "UNANSWERABLE":
        return "R"
    return None


def state(row: dict) -> str:
    answer = loose_decision(row)
    if answer == "R":
        return "R"
    if answer is None or answer not in "ABCDE":
        return "U"
    valid = {str(x).strip().upper() for x in (row.get("valid_option_ids") or [])}
    return "C" if answer in valid else "W"


def load_model(name: str, path: Path) -> tuple[list[dict], list[dict], dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    relevant = [row for row in rows if row.get("source_dataset") == "nextgqa"
                and row.get("evidence_condition") in ("sufficient", "insufficient")]
    by_key: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in relevant:
        by_key[str(row["id"])][str(row["evidence_condition"])] = row
    records = []
    conflict_records = [{
        "id": str(row["id"]), "video_id": str(row["video_id"]),
        "detected": str((row.get("model_parsed") or {}).get("reject_reason") or "").upper()
                    == "CONFLICTING_EVIDENCE",
    } for row in rows if row.get("source_dataset") == "etbench"
       and row.get("evidence_condition") == "conflict"]
    audit = Counter(total_json_records=len(rows), relevant_records=len(relevant), unique_ids=len(by_key))
    parser_counts = Counter()
    for key in sorted(by_key):
        pair = by_key[key]
        if set(pair) != {"sufficient", "insufficient"}:
            audit["incomplete_pairs"] += 1
            continue
        suf, ins = pair["sufficient"], pair["insufficient"]
        ss, si = state(suf), state(ins)
        gold = reference_spans(suf)
        pred, parse_audit = predicted_spans(suf)
        parser_counts.update(parse_audit)
        overlap = intersection(pred, gold)
        pred_duration, gold_duration = duration(pred), duration(gold)
        iop = overlap / pred_duration if pred_duration else 0.0
        iog = overlap / gold_duration if gold_duration else 0.0
        denom = pred_duration + gold_duration - overlap
        iou = overlap / denom if denom else 0.0
        times = [float(x) for x in (suf.get("sampled_seconds") or [])]
        sampled_hit = any(any(start <= t <= end for start, end in gold) for t in times)
        transition = ("A" if ss in ("C", "W") else ss) + ("A" if si in ("C", "W") else si)
        records.append({
            "id": key, "video_id": str(suf["video_id"]), "qid": str(suf.get("qid")),
            "question_type": suf.get("question_type"), "suff_state": ss, "insuff_state": si,
            "transition": transition, "suff_correct": ss == "C", "suff_answer": ss in ("C", "W"),
            "suff_reject": ss == "R", "insuff_reject": si == "R",
            "insuff_reason_correct": si == "R" and str((ins.get("model_parsed") or {}).get("reject_reason") or "").upper() == "INSUFFICIENT_EVIDENCE",
            "iop": iop, "iog": iog, "iou": iou, "predicted_grounding": bool(pred),
            "sampled_evidence_hit": sampled_hit,
        })
    audit["complete_pairs"] = len(records)
    audit["unique_videos"] = len({r["video_id"] for r in records})
    audit["sampled_evidence_hit"] = sum(r["sampled_evidence_hit"] for r in records)
    audit["predicted_nonempty_all"] = sum(r["predicted_grounding"] for r in records)
    audit["predicted_nonempty_answered"] = sum(r["predicted_grounding"] and r["suff_answer"] for r in records)
    audit["suff_unusable"] = sum(r["suff_state"] == "U" for r in records)
    audit["insuff_unusable"] = sum(r["insuff_state"] == "U" for r in records)
    audit["conflict_records"] = len(conflict_records)
    audit["conflict_unique_videos"] = len({r["video_id"] for r in conflict_records})
    return records, conflict_records, {"model": name, **dict(audit), "interval_parser": dict(parser_counts)}


def indicator(records: list[dict], fn: Callable[[dict], float]) -> np.ndarray:
    return np.asarray([float(fn(row)) for row in records], dtype=np.float64)


def metric_components(records: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    one = np.ones(len(records), dtype=np.float64)
    c = indicator(records, lambda r: r["suff_correct"])
    sa = indicator(records, lambda r: r["suff_answer"])
    sr = indicator(records, lambda r: r["suff_reject"])
    ir = indicator(records, lambda r: r["insuff_reject"])
    reason = indicator(records, lambda r: r["insuff_reason_correct"])
    ecs = c * ir
    result = {
        "suff_accuracy": (c, one), "suff_coverage": (sa, one), "suff_reject": (sr, one),
        "insuff_reject": (ir, one), "insuff_reason": (reason, one),
        "delta_reject": (ir - sr, one), "ecs": (ecs, one), "ewr": (ecs, c),
        "ecs_reason": (c * reason, one),
        "miop": (indicator(records, lambda r: r["iop"]), one),
        "miog": (indicator(records, lambda r: r["iog"]), one),
        "miou": (indicator(records, lambda r: r["iou"]), one),
        "sampled_evidence_hit": (indicator(records, lambda r: r["sampled_evidence_hit"]), one),
    }
    for metric in ("iop", "iou"):
        values = indicator(records, lambda r, m=metric: r[m])
        for threshold in (0.3, 0.5, 0.7):
            suffix = str(threshold).replace(".", "")
            grounded = (values >= threshold).astype(float)
            result[f"{metric}_{suffix}"] = (grounded, one)
            result[f"acc_gqa_{metric}_{suffix}"] = (c * grounded, one)
            result[f"g_ecs_{metric}_{suffix}"] = (c * grounded * ir, one)
            result[f"g_ewr_{metric}_{suffix}"] = (c * grounded * ir, c * grounded)
    for transition in ("AA", "AR", "RA", "RR", "AU", "RU", "UA", "UR", "UU"):
        result[f"transition_{transition.lower()}"] = (
            indicator(records, lambda r, t=transition: r["transition"] == t), one)
    return result


def aggregate_by_cluster(records: list[dict], components: dict[str, tuple[np.ndarray, np.ndarray]],
                         clusters: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    index = {cluster: i for i, cluster in enumerate(clusters)}
    assignment = np.asarray([index[r["video_id"]] for r in records])
    out = {}
    for name, (numerator, denominator) in components.items():
        out[name] = (np.bincount(assignment, weights=numerator, minlength=len(clusters)),
                     np.bincount(assignment, weights=denominator, minlength=len(clusters)))
    return out


def ratio(numerator: np.ndarray | float, denominator: np.ndarray | float):
    return np.divide(numerator, denominator, out=np.full_like(np.asarray(numerator, dtype=float), np.nan),
                     where=np.asarray(denominator) != 0)


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, original_index in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[original_index])
        running = max(running, value)
        adjusted[original_index] = running
    return adjusted


def fmt(result: dict) -> str:
    return f"{100*result['estimate']:.2f} [{100*result['ci_low']:.2f}, {100*result['ci_high']:.2f}]"


def main() -> None:
    args = parse_args()
    models = {}
    conflict_models = {}
    audits = []
    for spec in args.model:
        name, raw_path = spec.split("=", 1)
        records, conflict_records, audit = load_model(name, Path(raw_path))
        models[name] = records
        conflict_models[name] = conflict_records
        audits.append(audit)
        gc.collect()
    key_sets = {name: {(r["video_id"], r["qid"]) for r in rows} for name, rows in models.items()}
    common = set.intersection(*key_sets.values())
    if not common:
        raise RuntimeError("models have no common paired questions")
    for name in models:
        models[name] = sorted((r for r in models[name] if (r["video_id"], r["qid"]) in common),
                              key=lambda r: (r["video_id"], r["qid"]))
    order = [(r["video_id"], r["qid"]) for r in next(iter(models.values()))]
    if any([(r["video_id"], r["qid"]) for r in rows] != order for rows in models.values()):
        raise RuntimeError("model item alignment failed")
    clusters = sorted({video_id for video_id, _ in order})
    aggregated = {name: aggregate_by_cluster(rows, metric_components(rows), clusters)
                  for name, rows in models.items()}

    conflict_key_sets = {name: {r["id"] for r in rows} for name, rows in conflict_models.items()}
    common_conflicts = set.intersection(*conflict_key_sets.values())
    for name in conflict_models:
        conflict_models[name] = sorted((r for r in conflict_models[name] if r["id"] in common_conflicts),
                                       key=lambda r: r["id"])
    conflict_order = [r["id"] for r in next(iter(conflict_models.values()))]
    if any([r["id"] for r in rows] != conflict_order for rows in conflict_models.values()):
        raise RuntimeError("conflict item alignment failed")
    conflict_clusters = sorted({r["video_id"] for r in next(iter(conflict_models.values()))})
    conflict_cluster_index = {cluster: i for i, cluster in enumerate(conflict_clusters)}
    conflict_aggregated = {}
    for name, rows in conflict_models.items():
        assignment = np.asarray([conflict_cluster_index[r["video_id"]] for r in rows])
        conflict_aggregated[name] = (
            np.bincount(assignment, weights=indicator(rows, lambda r: r["detected"]),
                        minlength=len(conflict_clusters)),
            np.bincount(assignment, weights=np.ones(len(rows)), minlength=len(conflict_clusters)),
        )

    point = {}
    for model, metrics in aggregated.items():
        point[model] = {metric: float(ratio(num.sum(), den.sum())) for metric, (num, den) in metrics.items()}
        cnum, cden = conflict_aggregated[model]
        point[model]["detect"] = float(ratio(cnum.sum(), cden.sum()))
    draws: dict[str, dict[str, np.ndarray]] = {name: {} for name in models}
    rng = np.random.default_rng(args.seed)
    batch_size = min(500, args.bootstrap)
    chunks: dict[str, dict[str, list[np.ndarray]]] = {
        name: {metric: [] for metric in metrics} for name, metrics in aggregated.items()}
    for start in range(0, args.bootstrap, batch_size):
        size = min(batch_size, args.bootstrap - start)
        weights = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=size)
        for model, metrics in aggregated.items():
            for metric, (num, den) in metrics.items():
                chunks[model][metric].append(ratio(weights @ num, weights @ den))
    conflict_chunks = {name: [] for name in models}
    conflict_rng = np.random.default_rng(args.seed + 1)
    for start in range(0, args.bootstrap, batch_size):
        size = min(batch_size, args.bootstrap - start)
        weights = conflict_rng.multinomial(
            len(conflict_clusters), np.full(len(conflict_clusters), 1 / len(conflict_clusters)), size=size)
        for model, (num, den) in conflict_aggregated.items():
            conflict_chunks[model].append(ratio(weights @ num, weights @ den))
    alpha_lo, alpha_hi = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
    summary = {"protocol": {"bootstrap": args.bootstrap, "seed": args.seed,
                             "cluster": "source video_id", "confidence_level": 1 - args.alpha,
                             "common_questions": len(common), "clusters": len(clusters),
                             "conflict_questions": len(common_conflicts),
                             "conflict_clusters": len(conflict_clusters),
                             "interval": "percentile clustered bootstrap"},
               "parser_audit": audits, "models": {}}
    for model in models:
        for metric in chunks[model]:
            draws[model][metric] = np.concatenate(chunks[model][metric])
        draws[model]["detect"] = np.concatenate(conflict_chunks[model])
        summary["models"][model] = {
            metric: {"estimate": point[model][metric],
                     "ci_low": float(np.nanpercentile(values, alpha_lo)),
                     "ci_high": float(np.nanpercentile(values, alpha_hi)),
                     "bootstrap_defined": int(np.isfinite(values).sum())}
            for metric, values in draws[model].items()
        }

    comparisons = []
    names = list(models)
    for metric in PRIMARY_SIGNIFICANCE:
        metric_rows = []
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                delta = draws[left][metric] - draws[right][metric]
                finite = delta[np.isfinite(delta)]
                raw_p = min(1.0, 2 * min((np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1),
                                         (np.count_nonzero(finite >= 0) + 1) / (len(finite) + 1)))
                metric_rows.append({
                    "metric": metric, "model_a": left, "model_b": right,
                    "difference": point[left][metric] - point[right][metric],
                    "ci_low": float(np.percentile(finite, alpha_lo)),
                    "ci_high": float(np.percentile(finite, alpha_hi)), "p_raw": raw_p,
                })
        adjusted = holm([row["p_raw"] for row in metric_rows])
        for row, p_holm in zip(metric_rows, adjusted):
            row["p_holm"] = p_holm
            row["significant_holm_005"] = p_holm < 0.05
            comparisons.append(row)
    summary["pairwise_comparisons"] = comparisons

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics_with_clustered_ci.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "pairwise_significance.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = (list(comparisons[0]) if comparisons else
                      ["metric", "model_a", "model_b", "difference", "ci_low", "ci_high",
                       "p_raw", "p_holm", "significant_holm_005"])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(comparisons)
    with (args.output_dir / "item_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for model, rows in models.items():
            for row in rows:
                stream.write(json.dumps({"model": model, **row}, ensure_ascii=False) + "\n")

    main_metrics = ("suff_accuracy", "suff_coverage", "insuff_reject", "insuff_reason",
                    "delta_reject", "ecs", "ewr", "ecs_reason", "detect")
    grounded_metrics = ("miop", "iop_05", "miou", "iou_05", "acc_gqa_iop_05",
                        "acc_gqa_iou_05", "g_ecs_iop_05", "g_ewr_iop_05")
    main_lines = [model + " & " + " & ".join(fmt(summary["models"][model][m]) for m in main_metrics) + r" \\"
                  for model in names]
    grounded_lines = [model + " & " + " & ".join(fmt(summary["models"][model][m]) for m in grounded_metrics) + r" \\"
                      for model in names]
    (args.output_dir / "main_table_rows.tex").write_text("\n".join(main_lines) + "\n", encoding="utf-8")
    (args.output_dir / "grounded_table_rows.tex").write_text("\n".join(grounded_lines) + "\n", encoding="utf-8")
    transition = {model: {cell: summary["models"][model][f"transition_{cell.lower()}"]
                          for cell in ("AA", "AR", "RA", "RR", "AU", "RU", "UA", "UR", "UU")}
                  for model in names}
    (args.output_dir / "transition_cells.json").write_text(
        json.dumps(transition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "models": names,
                      "questions": len(common), "clusters": len(clusters),
                      "conflict_questions": len(common_conflicts),
                      "conflict_clusters": len(conflict_clusters),
                      "bootstrap": args.bootstrap}, indent=2))


if __name__ == "__main__":
    main()
