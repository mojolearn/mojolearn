#!/usr/bin/env bash
# Run inside an activated IDENTICAL toolchain, or set MOJO to its mojo binary.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:?usage: check_embedding_plan_sort.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
out=$(cd "$out" && pwd)
mojo_bin=${MOJO:-mojo}
"$mojo_bin" build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I "$repo" "$repo/embedding/checks/embedding_check.mojo" -o "$out/embedding-check"
MOJOLEARN_EMB_CHECK_CLAUSE_D=1 "$out/embedding-check" > "$out/fixture.log" 2>&1
"$mojo_bin" build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I "$repo" "$repo/embedding/checks/embedding_sort_check.mojo" -o "$out/embedding-sort-check"
"$out/embedding-sort-check" > "$out/edges.log" 2>&1
"$mojo_bin" build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_EMB_SORT_NEGATIVE_CONTROL=1 -I "$repo" "$repo/embedding/checks/embedding_sort_check.mojo" -o "$out/embedding-sort-negative"
if "$out/embedding-sort-negative" > "$out/negative.log" 2>&1; then
  echo 'FAIL: reverse-tie device sort negative control passed' >&2
  exit 1
fi
if ! grep -q 'sabotage= SORT_NEGATIVE_CONTROL' "$out/negative.log"; then
  echo 'FAIL: device negative control was not registered' >&2
  exit 1
fi
if ! grep -q 'PLAN_SORT edge metadata mismatch' "$out/negative.log"; then
  echo 'FAIL: negative control failed for an unrelated reason' >&2
  exit 1
fi
printf '%s\n' 'PASS: production plan/geometry, edge cases, and device negative control' > "$out/verdict.txt"
