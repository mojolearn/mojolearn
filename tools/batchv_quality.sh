#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# batchv_quality.sh: QUALITY ONLY (times nothing) for the lane/apple-fast-batchv
# defines on an Apple box. Builds FAST x_prep arms (none,
# SI_ONEPASS, PT no-spec set) and FAST estimators arms (none, KDE_DIMTILE),
# dumps tools/batchv_quality.py outputs with each arm's .so installed, then
# prints BATCHV-Q lines (define on vs off). The tree's original .so files are
# restored at the end. Run in a built tree.
set -u
cd "$(dirname "$0")/.."
out=$HOME/batchv-q; mkdir -p "$out"
PY=${BATCHV_PY:-$HOME/board-0834/cache/venv/bin/python}
[ -x "$PY" ] || PY="pixi run python"
RUN() { if command -v pixi > /dev/null; then pixi run "$@"; else "$@"; fi; }
for b in x_prep estimators; do
  so=python/mojolearn/_mojolearn_$b.so
  [ -f "$so" ] && cp "$so" "$out/orig_$b.so"
done
build() {  # $1 binding, $2 arm, $3 defines
  local so=python/mojolearn/_mojolearn_$1.so
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$3" MOJOLEARN_SKIP_BUILD_GATE=1 \
    RUN bash "bindings/build_$1.sh" > "$out/build_$1_$2.log" 2>&1
  rc=$?; echo "BATCHV-Q-BUILD $1 arm=$2 rc=$rc defines='$3'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_$1_$2.log" | cut -c1-300; return 1; }
  cp "$so" "$out/$1_$2.so"
}
dump() {  # $1 binding, $2 arm, $3 prep|kde
  local so=python/mojolearn/_mojolearn_$1.so
  cp "$out/$1_$2.so" "$so.tmp" && mv -f "$so.tmp" "$so"
  (cd python && PYTHONPATH=$PWD MOJOLEARN_NUMERIC_MODE=fast $PY ../tools/batchv_quality.py dump "$3" "$out/$3_$2.npz") \
    > "$out/dump_$3_$2.log" 2>&1
  rc=$?; echo "BATCHV-Q-DUMP $3 arm=$2 rc=$rc"
  [ $rc = 0 ] || tail -n 15 "$out/dump_$3_$2.log"
}
build x_prep off "" && dump x_prep off prep
build x_prep si "-D MOJOLEARN_SI_ONEPASS" && dump x_prep si prep
build x_prep ptns "-D MOJOLEARN_PT_COLBATCH -D MOJOLEARN_PT_FUSED_TRANSFORM -D MOJOLEARN_SI_ONEPASS" && dump x_prep ptns prep
build estimators off "" && dump estimators off kde
build estimators kdt "-D MOJOLEARN_KDE_DIMTILE" && dump estimators kdt kde
for a in si ptns; do
  [ -f "$out/prep_off.npz" ] && [ -f "$out/prep_$a.npz" ] && $PY tools/batchv_quality.py cmp "$out/prep_off.npz" "$out/prep_$a.npz" "$a"
done
[ -f "$out/kde_off.npz" ] && [ -f "$out/kde_kdt.npz" ] && $PY tools/batchv_quality.py cmp "$out/kde_off.npz" "$out/kde_kdt.npz" kdt
for b in x_prep estimators; do
  [ -f "$out/orig_$b.so" ] && cp "$out/orig_$b.so" "python/mojolearn/_mojolearn_$b.so.tmp" \
    && mv -f "python/mojolearn/_mojolearn_$b.so.tmp" "python/mojolearn/_mojolearn_$b.so"
done
echo "BATCHV-Q done"
