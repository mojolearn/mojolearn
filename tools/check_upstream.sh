#!/usr/bin/env bash
# Which of our ports correspond to UPSTREAM files that MOVED?
#
# Not a semantic diff: it reports whether the mapped source still exists and
# whether its content hash changed since the pin. A changed hash means read
# the diff, not that we are wrong.
#
# This tree now mirrors TWO upstreams. Each section owns a map and a pin:
#
#   boosting (root)   CatBoost   PORTED_MAP.tsv          pinned 54a8143a
#   cluster           cuVS       cluster/PORTED_MAP.tsv  pinned 2140532c
#
# usage: tools/check_upstream.sh /path/to/checkout [section]
#        section defaults to the root CatBoost map.
set -euo pipefail
SRC="${1:?path to an upstream checkout required}"
SECTION="${2:-}"

if [ -z "$SECTION" ]; then
  MAP="PORTED_MAP.tsv"
  OURS_PREFIX=""
  EXPECTED="54a8143a"
  NAME="CatBoost"
else
  MAP="$SECTION/PORTED_MAP.tsv"
  OURS_PREFIX="$SECTION/"
  EXPECTED="2140532c"
  NAME="cuVS"
fi

[ -f "$MAP" ] || { echo "no map at $MAP"; exit 1; }

PIN="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "$NAME upstream at $PIN, ours pinned at $EXPECTED"
echo
miss=0
gone_ours=0
while IFS=$'\t' read -r ours theirs status notes; do
  case "$ours" in \#*|"") continue ;; esac
  if [ ! -e "$SRC/$theirs" ]; then
    echo "MOVED OR GONE  $theirs"
    echo "               (ours: $OURS_PREFIX$ours, $status)"
    miss=$((miss+1))
  fi
  if [ ! -e "$OURS_PREFIX$ours" ]; then
    echo "OURS MISSING   $OURS_PREFIX$ours"
    gone_ours=$((gone_ours+1))
  fi
done < "$MAP"
echo
if [ "$miss" -eq 0 ]; then
  echo "every mapped upstream file still exists"
else
  echo "$miss mapped upstream files moved or were deleted"
fi
if [ "$gone_ours" -ne 0 ]; then
  echo "$gone_ours mapped files of OURS do not exist -- the map is lying"
  exit 1
fi
