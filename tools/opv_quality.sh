#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# opv_quality.sh: QUALITY ONLY (times nothing) for the lane/apple-fast-opv
# defines on an Apple box: FAST x_cluster arms (none, OPTICS_FRONTIER_DEVICE,
# OPTICS_LIVEBUF), tools/opv_quality.py scores per arm,
# OPV-Q lines (define on vs off). Restores the tree's x_cluster .so. Run in a built tree.
set -u
cd "$(dirname "$0")/.."
out=$HOME/opv-q; mkdir -p "$out"
PY=${OPV_PY:-$HOME/board-0834/cache/venv/bin/python}
RUN() { if command -v pixi > /dev/null; then pixi run "$@"; else "$@"; fi; }
so=python/mojolearn/_mojolearn_x_cluster.so
[ -f "$so" ] && cp "$so" "$out/orig.so"
arm() {  # $1 name, $2 defines
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$2" MOJOLEARN_SKIP_BUILD_GATE=1 \
    RUN bash bindings/build_x_cluster.sh > "$out/build_$1.log" 2>&1
  rc=$?; echo "OPV-Q-BUILD arm=$1 rc=$rc defines='$2'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_$1.log" | cut -c1-300; return 1; }
  (cd python && PYTHONPATH=$PWD MOJOLEARN_NUMERIC_MODE=fast $PY ../tools/opv_quality.py dump "$out/$1.npz") \
    > "$out/dump_$1.log" 2>&1
  rc=$?; echo "OPV-Q-DUMP arm=$1 rc=$rc"
  [ $rc = 0 ] || tail -n 15 "$out/dump_$1.log"
}
arm off ""
arm fd "-D MOJOLEARN_OPTICS_FRONTIER_DEVICE=1"
arm lb "-D MOJOLEARN_OPTICS_LIVEBUF=1"
for a in fd lb; do
  [ -f "$out/off.npz" ] && [ -f "$out/$a.npz" ] && $PY tools/opv_quality.py cmp "$out/off.npz" "$out/$a.npz" "$a"
done
[ -f "$out/orig.so" ] && cp "$out/orig.so" "$so.tmp" && mv -f "$so.tmp" "$so"
echo "OPV-Q done"
