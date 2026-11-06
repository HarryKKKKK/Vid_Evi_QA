"""Fail unless a completed smoke job passed every frozen-schema check."""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/src/vid_evi_qa/evaluation/check_smoke_gate.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/check_smoke_gate.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("kind")
    parser.add_argument("smoke_job_id")
    args = parser.parse_args()
    status_path = args.release.resolve() / "results" / f"{args.kind}_{args.smoke_job_id}" / "deployment_status.json"
    status = json.loads(status_path.read_text())
    validation = status.get("validation") or {}
    problems = []
    if status.get("status") != "passed":
        problems.append(f"deployment status is {status.get('status')!r}")
    if validation.get("result_records") != validation.get("expected_records"):
        problems.append("result count does not match expected count")
    if validation.get("invalid_records") != 0:
        problems.append(f"invalid_records={validation.get('invalid_records')}")
    if validation.get("missing_ids"):
        problems.append(f"missing_ids={len(validation['missing_ids'])}")
    if validation.get("obsolete_ids"):
        problems.append(f"obsolete_ids={len(validation['obsolete_ids'])}")
    if problems:
        raise SystemExit("SMOKE_GATE_FAILED: " + "; ".join(problems))
    print(f"SMOKE_GATE_PASSED kind={args.kind} job={args.smoke_job_id} records={validation['result_records']}")


if __name__ == "__main__":
    main()
