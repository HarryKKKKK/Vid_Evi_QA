#!/usr/bin/env bash
# Compatibility entry; also works inside a newly frozen code bundle.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$HERE/run_eval.sh"
[[ -f "$RUNNER" ]] || RUNNER="$HERE/../../scripts/jobs/lib/run_eval.sh"
exec bash "$RUNNER" merged "$@"
