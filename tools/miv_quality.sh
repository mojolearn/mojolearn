#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# miv_quality.sh: QUALITY ONLY (times nothing) for the lane/apple-fast-miv
# defines on an Apple box: FAST x_prep arms (none, MI_REG_TIES,
# MI_CLF_RANKMAJOR), tools/miv_quality.py scores per arm,
# MIV-Q lines (define on vs off). Restores the tree's x_prep .so. Run in a built tree.
set -u
cd "$(dirname "$0")/.."
out=$HOME/miv-q; mkdir -p "$out"
PY=${MIV_PY:-$HOME/board-0834/cache/venv/bin/python}
RUN() { if command -v pixi > /dev/null; then pixi run "$@"; else "$@"; fi; }
so=python/mojolearn/_mojolearn_x_prep.so
[ -f "$so" ] && cp "$so" "$out/orig.so"
arm() {  # $1 name, $2 defines
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$2" MOJOLEARN_SKIP_BUILD_GATE=1 \
    RUN bash bindings/build_x_prep.sh > "$out/build_$1.log" 2>&1
  rc=$?; echo "MIV-Q-BUILD arm=$1 rc=$rc defines='$2'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_$1.log" | cut -c1-300; return 1; }
  (cd python && PYTHONPATH=$PWD MOJOLEARN_NUMERIC_MODE=fast $PY ../tools/miv_quality.py dump "$out/$1.npz") \
    > "$out/dump_$1.log" 2>&1
  rc=$?; echo "MIV-Q-DUMP arm=$1 rc=$rc"
  [ $rc = 0 ] || tail -n 15 "$out/dump_$1.log"
}
arm off ""
arm ties "-D MOJOLEARN_MI_REG_TIES"
arm clfrank "-D MOJOLEARN_MI_CLF_RANKMAJOR"
for a in ties clfrank; do
  [ -f "$out/off.npz" ] && [ -f "$out/$a.npz" ] && $PY tools/miv_quality.py cmp "$out/off.npz" "$out/$a.npz" "$a"
done
[ -f "$out/orig.so" ] && cp "$out/orig.so" "$so.tmp" && mv -f "$so.tmp" "$so"
echo "MIV-Q done"
