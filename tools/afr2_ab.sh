#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
# afr2_ab.sh <tag> <binding> <case> "<defines B>"
#
# Lane apple-fast-round2's off-board A/B (bench/apple_fast_round2_ab.py has
# why): two FAST builds of ONE binding as tools/afc_ab_def.sh makes them (arm
# A no defines, arm B "<defines B>"), then ONE run of <case> per arm (one run
# per arm, Andrew Oct 3). Prints AFR2-AB tag=.. arm=A|B ... lines. Arm B's .so
# is left installed. binding: the suffix of bindings/build_<binding>.sh.
TAG=$1 BIND=$2 CASE=$3 DB=$4
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
script="bindings/build_$BIND.sh"
so="python/mojolearn/_mojolearn_$BIND.so"
out="$HOME/afc-def/$TAG"; mkdir -p "$out"
# the board venv afc_ab.sh races with (numpy present), else python3
VP=python3
for b in board-0834 board-0833; do [ -x "$HOME/$b/cache/venv/bin/python" ] && { VP=$HOME/$b/cache/venv/bin/python; break; }; done
[ -f "$script" ] || { echo "AFR2-AB $TAG no build script $script"; exit 2; }
build() {  # $1 arm, $2 defines
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$2" MOJOLEARN_SKIP_BUILD_GATE=1 \
    bash "$script" > "$out/build_$1.log" 2>&1
  rc=$?; echo "AFR2-AB-BUILD $TAG arm=$1 rc=$rc defines='$2'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_$1.log" | cut -c1-300; exit 1; }
  cp "$so" "$out/$1.so"
}
build A ""; build B "$DB"
for a in A B; do
  cp "$out/$a.so" "$so.tmp" && mv -f "$so.tmp" "$so"
  line=$(env MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python "$VP" bench/apple_fast_round2_ab.py "$CASE" 2>"$out/run_$a.err"); rc=$?
  echo "AFR2-AB tag=$TAG arm=$a rc=$rc $(printf '%s\n' "$line" | tail -n 1) defines='$([ $a = B ] && echo "$DB")'"
done
