#!/usr/bin/env bash
#
# Regenerate every table using archived-results defaults.
#
# Each script reads from archived_results/ and writes to
# reproduced_artifacts/tables/<category>/. Run from anywhere; this script
# cd's to the repo root first. Continues past a failing script and exits
# non-zero if any script failed.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

fail=0
for f in scripts/tables/make_*.py; do
  echo "=== $f ==="
  if ! uv run python "$f"; then
    echo "FAILED: $f" >&2
    fail=1
  fi
done

exit "$fail"
