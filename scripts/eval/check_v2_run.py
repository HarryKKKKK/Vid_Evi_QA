"""Validate current (not hard-coded-size) manifests and report v2 completion."""
import argparse
from collections import Counter
import json
from pathlib import Path
from prompt_v2 import PROMPT_VERSION, PROMPT_SHA256, response_errors, record_identity


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [record_identity(row) for row in rows]
    if not rows or any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("empty manifest or missing/duplicate record IDs")
    allowed = {"sufficient", "insufficient", "partially_sufficient", "conflicting"}
    for row in rows:
        if row.get("evidence_condition") not in allowed:
            raise ValueError("unknown evidence condition")
        if len(row.get("choices", [])) != 5 or not all(isinstance(x, str) and x.strip() for x in row["choices"]):
            raise ValueError("manifest must have five nonempty choices per record")
        if row.get("expected_behavior") != ("answer" if row["evidence_condition"] == "sufficient" else "reject"):
            raise ValueError("inconsistent expected behavior")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    report = {"expected_records": len(manifest), "condition_counts": dict(Counter(x["evidence_condition"] for x in manifest)),
              "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256}
    failed = False
    if args.result:
        results = json.loads(args.result.read_text(encoding="utf-8")) if args.result.exists() else []
        wanted = {record_identity(x) for x in manifest}
        found = [record_identity(x) for x in results]
        invalid = [record_identity(x) for x in results if x.get("prompt_version") != PROMPT_VERSION
                   or x.get("prompt_sha256") != PROMPT_SHA256 or not x.get("input_fingerprint")
                   or response_errors(x.get("model_parsed"))]
        failed = set(found) != wanted or len(found) != len(wanted) or bool(invalid)
        report.update(status="incomplete_or_invalid" if failed else "complete", result_records=len(results),
                      invalid_records=len(invalid), invalid_ids=invalid, missing_ids=sorted(wanted-set(found)),
                      obsolete_ids=sorted(str(x) for x in set(found)-wanted),
                      unavailable_metrics=["option_probabilities", "reject_reason_probabilities", "evidence_quality", "overall_resolvability"])
        args.result.with_suffix(".validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit("v2 run incomplete/invalid; rerun same job to retry missing or schema-invalid records")


if __name__ == "__main__":
    main()
