"""Fail unless a completed smoke job passed every frozen-schema check."""
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
