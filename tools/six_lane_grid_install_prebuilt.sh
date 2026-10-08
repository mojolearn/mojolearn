#!/bin/bash
# tools/six_lane_grid_install_prebuilt.sh <tree> <nvidia|amd> <defines-csv|-> <bindings-csv> [store]
#
# Installs prebuilt IDENTICAL device bindings (tools/six_lane_grid_prebuild.py) into a branch worktree where
# bindings/build_<x>.sh would have written them (python/mojolearn/identical/_mojolearn_<x>.so, per the store's
# lookup.tsv `dest` column), instead of compiling them on the GPU box.
#   <defines-csv>  the line's MOJOLEARN_BUILD_DEFINES value (comma list, no spaces); "-" or "" = no defines (arm B)
#   <bindings-csv> the line's BUILDS= value (build,build_x_linear,...)
#   [store]        default $LQ_PREBUILT, else /root/grid-prebuilt; the store holds <vendor>/lookup.tsv + artifacts
# Exit 0: every listed binding was installed and its sha256 verified (one "PREBUILT installed ..." line each).
# Exit 2: cannot serve the line (no store, store built from another commit, a host binding, a binding or define
#         set the store lacks, a corrupt artifact): the caller builds as before. Nothing is left half-installed.
# Exit 1: usage.
set -u
[ $# -ge 4 ] || { echo "usage: $0 <tree> <nvidia|amd> <defines-csv|-> <bindings-csv> [store]" >&2; exit 1; }
T=$1; V=$2; DEFS=$3; BUILDS=$4; STORE=${5:-${LQ_PREBUILT:-/root/grid-prebuilt}}
case $V in nvidia|amd) ;; *) echo "vendor must be nvidia or amd" >&2; exit 1;; esac
sha256() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$@" | cut -d' ' -f1; else shasum -a 256 "$@" | cut -d' ' -f1; fi; }
L=$STORE/$V/lookup.tsv
[ -f "$L" ] || { echo "PREBUILT none: no $L"; exit 2; }
src=$(head -1 "$L" | sed -n 's/.*source_sha=\([0-9a-f]*\).*/\1/p')
head=$(git -C "$T" rev-parse HEAD 2>/dev/null)
[ -n "$src" ] && [ "$src" = "$head" ] || { echo "PREBUILT skip: store source=${src:-?} tree=${head:-?}"; exit 2; }
[ "$DEFS" = - ] && DEFS=
# the same normalization as six_lane_grid_prebuild.full_sha: sorted unique entries, one per line
dsha=$( { [ -n "$DEFS" ] && printf '%s' "$DEFS" | tr ',' '\n' | sed '/^$/d' | LC_ALL=C sort -u; } | sha256)
plan=""   # "<src>|<dest>|<sha>" per binding, resolved before anything is copied
for b in $(echo "$BUILDS" | tr ',' ' '); do
  case $b in
    build) stem=_mojolearn ;;
    *_host|build_host_family) echo "PREBUILT skip: $b is a host binding (not prebuilt)"; exit 2 ;;
    build_*) stem=_mojolearn_${b#build_} ;;
    *) echo "PREBUILT skip: unknown build script $b"; exit 2 ;;
  esac
  row=$(awk -F'\t' -v b="$stem" -v s="$dsha" '$1==b && $2==s {print; exit}' "$L")
  [ -n "$row" ] || { echo "PREBUILT skip: no artifact for $stem defines-sha=${dsha:0:16} in $L"; exit 2; }
  rel=$(printf '%s' "$row" | cut -f3); want=$(printf '%s' "$row" | cut -f4); dest=$(printf '%s' "$row" | cut -f5)
  [ -f "$STORE/$V/$rel" ] || { echo "PREBUILT skip: missing $STORE/$V/$rel"; exit 2; }
  plan="$plan $STORE/$V/$rel|$dest|$want|$stem"
done
n=0
for e in $plan; do
  IFS='|' read -r from dest want stem <<< "$e"
  mkdir -p "$T/$(dirname "$dest")" || exit 2
  cp "$from" "$T/$dest.prebuilt.tmp" || { echo "PREBUILT skip: copy failed for $stem"; rm -f "$T/$dest.prebuilt.tmp"; exit 2; }
  got=$(sha256 "$T/$dest.prebuilt.tmp")
  [ "$got" = "$want" ] || { echo "PREBUILT skip: sha mismatch for $stem ($got != $want)"; rm -f "$T/$dest.prebuilt.tmp"; exit 2; }
  mv -f "$T/$dest.prebuilt.tmp" "$T/$dest" || exit 2
  echo "PREBUILT installed $stem -> $dest sha=${got:0:16} defines-sha=${dsha:0:16}"
  n=$((n+1))
done
echo "PREBUILT ok $V n=$n source=${src:0:12}"
exit 0
