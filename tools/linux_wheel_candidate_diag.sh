#!/usr/bin/env bash
# Main-only phase9 entry; caller owns lease and teardown.
set -euo pipefail
: "${OUT:?}" "${REPO:?}"
mkdir -p "$OUT/diag"
rc=0
bash "$REPO/tools/linux_wheel_candidate.sh" "$OUT/diag/candidate" > "$OUT/diag/candidate.log" 2>&1 || rc=$?
printf 'candidate_exit=%s\n' "$rc" > "$OUT/diag/SUMMARY.txt"
exit "$rc"
