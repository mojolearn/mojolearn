#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# mcd_wide_b_only.sh <tag> <lane> <dataset> <quality_tag> "<defines B>"
#
# w2-mcd2: ONE board-shape FAST race of arm B only, for a row where arm A
# (main) has no time because it does not finish (min-cov-det /
# elliptic-envelope istella: FAST killed > 20 min on the board). Refuses
# unless ~/afc-def/<quality_tag>/PASS exists (tools/mcd_compat_ab.sh's
# capped fitted-state gate for the same lane and dataset). Uses a staged
# ~/afc-def/<tag>/B.so if present, else builds x_decomp with "<defines B>".
# The race line lands in ~/mq/out/race-<tag>/race.log (afc_ab.sh, env
# AFC_DEF_ARM=B). Run inside this branch's tree on the M3 serial queue.
set -u
TAG=$1 LANE=$2 DS=$3 QTAG=$4 DB=$5
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
q=$HOME/afc-def/$QTAG
[ -f "$q/PASS" ] || { echo "MCDW-REFUSE no quality PASS at $q"; exit 2; }
"$HOME/board-0834/cache/venv/bin/python" - "$q/A.npz" "$LANE" "$DS" <<'PY' || exit 2
import sys
import numpy as np
with np.load(sys.argv[1]) as z:
    assert str(z['lane']) == sys.argv[2], 'quality PASS is for another lane'
    assert str(z['dataset']) == sys.argv[3], 'quality PASS is for another dataset'
PY
so=python/mojolearn/_mojolearn_x_decomp.so
out=$HOME/afc-def/$TAG; mkdir -p "$out"
echo "MCDW $TAG head=$(git rev-parse --short HEAD) lane=$LANE ds=$DS B='$DB' gate=$QTAG"
if [ ! -f "$out/B.so" ]; then
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$DB" MOJOLEARN_SKIP_BUILD_GATE=1 \
    MOJOLEARN_COMPILE_JOBS=1 bash bindings/build_x_decomp.sh > "$out/build_B.log" 2>&1
  rc=$?; echo "MCDW-BUILD $TAG arm=B rc=$rc defines='$DB'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_B.log" | cut -c1-300; exit 1; }
  cp "$so" "$out/B.so"
fi
cp "$out/B.so" "$so.tmp" && mv -f "$so.tmp" "$so"
bash tools/afc_ab.sh "$TAG" "$LANE" "$DS" 1 1 "AFC_DEF_ARM=B" | sed "s/^AFC-AB /AFC-AB def=B-only /"
