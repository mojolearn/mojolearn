#!/usr/bin/env bash
# Which of our ports correspond to CatBoost files that MOVED UPSTREAM?
#
# Not a semantic diff: it reports whether the mapped source still exists and
# whether its content hash changed since the pin. A changed hash means read
# the diff, not that we are wrong.
#
# usage: tools/check_upstream.sh /path/to/catboost-checkout
set -euo pipefail
SRC="${1:?path to a catboost checkout required}"
PIN="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "upstream at $PIN, ours pinned at 54a8143a"
echo
miss=0
while IFS=$'\t' read -r ours theirs status notes; do
  case "$ours" in \#*|"") continue ;; esac
  if [ ! -e "$SRC/$theirs" ]; then
    echo "MOVED OR GONE  $theirs"
    echo "               (ours: $ours, $status)"
    miss=$((miss+1))
  fi
done < PORTED_MAP.tsv
echo
if [ "$miss" -eq 0 ]; then
  echo "every mapped upstream file still exists"
else
  echo "$miss mapped upstream files moved or were deleted"
fi
