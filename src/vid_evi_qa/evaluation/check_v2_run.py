"""Validate current (not hard-coded-size) manifests and report v2 completion."""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/src/vid_evi_qa/evaluation/check_v2_run.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/check_v2_run.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import argparse
from collections import Counter
import json
from pathlib import Path
from prompt_v2 import PROMPT_VERSION, PROMPT_SHA256, response_errors, record_identity
import prompt_v2 as _prompt_protocol


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
              "prompt_version": PROMPT_VERSION, "prompt_sha256": PROMPT_SHA256,
              "response_validation_version": getattr(_prompt_protocol, "RESPONSE_VALIDATION_VERSION", "legacy-unversioned")}
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
