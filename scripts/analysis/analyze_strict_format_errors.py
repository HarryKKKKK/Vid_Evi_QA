"""Summarize strict v2 format failures without repairing model output."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def primary_reason(row):
    parsed = row.get("model_parsed") or {}
    parse_error = parsed.get("parse_error")
    if parse_error:
        tokens = ((row.get("api_logprobs") or {}).get("tokens") or [])
        if len(tokens) == 768:
            return "json_truncated_at_768_token_cap"
        if not (row.get("model_response") or "").strip():
            return "empty_response"
        return "invalid_json_before_token_cap"
    errors = row.get("schema_errors") or []
    if not errors:
        return "validator_mismatch_or_unknown"
    return "schema:" + " + ".join(sorted(set(errors)))


def errors_without_option_prefix(row):
    """Counterfactual v2 check: descriptions may contain any non-empty text."""
    errors = [x for x in (row.get("schema_errors") or []) if x != "missing evidence description"]
    parsed = row.get("model_parsed") or {}
    evidence = parsed.get("evidence")
    if isinstance(evidence, list):
        for group in evidence:
            if not isinstance(group, dict):
                continue
            description = group.get("description")
            if not isinstance(description, str) or not description.strip():
                errors.append("missing or empty evidence description")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--examples-per-reason", type=int, default=3)
    args = parser.parse_args()
    root = args.result_root
    rows = []
    for path in sorted(root.glob("shard_*/merged.classify.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    invalid = [row for row in rows if row.get("schema_errors")]
    reasons = Counter(primary_reason(row) for row in invalid)
    secondary = Counter(error for row in invalid for error in row.get("schema_errors", []))
    bad_description_prefixes = Counter()
    for row in invalid:
        for group in (row.get("model_parsed") or {}).get("evidence", []):
            if not isinstance(group, dict):
                continue
            description = group.get("description")
            if (not isinstance(description, str) or not description.strip()
                    or not description.startswith(tuple(f"Option {x}: " for x in "ABCDE"))):
                prefix = description.split(":", 1)[0] if isinstance(description, str) else type(description).__name__
                bad_description_prefixes[prefix] += 1
    relaxed_invalid = [row for row in rows if errors_without_option_prefix(row)]
    relaxed_reasons = Counter()
    for row in relaxed_invalid:
        parsed = row.get("model_parsed") or {}
        tokens = ((row.get("api_logprobs") or {}).get("tokens") or [])
        if parsed.get("parse_error") and len(tokens) == 768:
            key = "json_truncated_at_768_token_cap"
        elif parsed.get("parse_error"):
            key = "invalid_json_before_token_cap"
        else:
            key = "schema:" + " + ".join(sorted(set(errors_without_option_prefix(row))))
        relaxed_reasons[key] += 1
    examples = defaultdict(list)
    for row in invalid:
        reason = primary_reason(row)
        if len(examples[reason]) >= args.examples_per_reason:
            continue
        raw = row.get("model_response") or ""
        tokens = ((row.get("api_logprobs") or {}).get("tokens") or [])
        examples[reason].append({
            "record_id": row.get("record_id"),
            "condition": row.get("evidence_condition"),
            "schema_errors": row.get("schema_errors"),
            "parse_error": (row.get("model_parsed") or {}).get("parse_error"),
            "output_tokens": len(tokens),
            "raw_characters": len(raw),
            "raw_tail": raw[-240:],
        })
    report = {
        "result_root": str(root.resolve()),
        "records_observed": len(rows),
        "strict_invalid": len(invalid),
        "strict_valid": len(rows) - len(invalid),
        "strict_format_valid_percent": round(100 * (len(rows) - len(invalid)) / len(rows), 4) if rows else None,
        "primary_reasons": dict(reasons),
        "validator_error_frequency": dict(secondary),
        "invalid_description_prefixes": dict(bad_description_prefixes),
        "counterfactual_without_option_prefix": {
            "strict_invalid": len(relaxed_invalid),
            "strict_valid": len(rows) - len(relaxed_invalid),
            "strict_format_valid_percent": round(100 * (len(rows) - len(relaxed_invalid)) / len(rows), 4) if rows else None,
            "primary_reasons": dict(relaxed_reasons),
        },
        "examples": dict(examples),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
