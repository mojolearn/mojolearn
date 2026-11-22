#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# w2_clres_quality.sh <tag> <binding> <lane> <dataset> "<defines A>" "<defines B>" <mode>
#
# Quality job of lane apple-fast-w2-clres: arm A (main, defines A) and arm B
# (the candidate define) built exactly as tools/afc_ab_def.sh builds them,
# into ~/afc-def/<tag>/{A,B}.so (AFC_SKIP_BUILD=1 reuses prebuilt arms
# staged there), each installed in turn for one board-shape fit dumped by
# tools/w2_clres_quality.py, then compared (mode exact|labrg; the tolerances
# are in that file's docstring). Arm B's .so is left installed, as
# afc_ab_def.sh leaves it. One fit per arm; nothing is timed.
set -u
TAG=$1 BIND=$2 LANE=$3 DS=$4 DA=$5 DB=$6 MODE=$7
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
so=python/mojolearn/_mojolearn_$BIND.so; script=bindings/build_$BIND.sh
out=$HOME/afc-def/$TAG; mkdir -p "$out"
for b in board-0834 board-0833; do [ -x $HOME/$b/cache/venv/bin/python ] && { VP=$HOME/$b/cache/venv/bin/python; break; }; done
echo "W2CLRES-Q $TAG head=$(git rev-parse --short HEAD) bind=$BIND lane=$LANE ds=$DS mode=$MODE A='$DA' B='$DB'"
build() {  # $1 arm, $2 defines
  if [ "${AFC_SKIP_BUILD:-0}" = 1 ] && [ -f "$out/$1.so" ]; then return 0; fi
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$2" MOJOLEARN_SKIP_BUILD_GATE=1 \
    bash "$script" > "$out/build_$1.log" 2>&1
  rc=$?; echo "W2CLRES-Q-BUILD $TAG arm=$1 rc=$rc defines='$2'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_$1.log" | cut -c1-300; exit 1; }
  cp "$so" "$out/$1.so"
}
build A "$DA"; build B "$DB"
for a in A B; do
  cp "$out/$a.so" "$so.tmp" && mv -f "$so.tmp" "$so"
  rm -f "$out/q_$a.npz"
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python \
    "$VP" tools/w2_clres_quality.py dump "$LANE" "$DS" "$out/q_$a.npz" > "$out/q_$a.log" 2>&1
  rc=$?; grep -m 1 W2CLRES-DUMP "$out/q_$a.log" | sed "s/^/arm=$a /"
  [ $rc = 0 ] || { echo "W2CLRES-Q-FAIL arm=$a rc=$rc"; tail -n 8 "$out/q_$a.log" | cut -c1-300; exit 1; }
done
MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python \
  "$VP" tools/w2_clres_quality.py compare "$out/q_A.npz" "$out/q_B.npz" "$MODE" "$DS" | tee "$out/quality.txt"
exit ${PIPESTATUS[0]}
