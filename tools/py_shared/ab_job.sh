#!/bin/bash
# tools/py_shared/ab_job.sh <out> [lanes]   (lane py-shared, one NVIDIA queue job)
#
# The lane's LIGHT proof, in one job on one pod (x86 CPU + one GPU):
#  1. BASE tree = this tree with tools/py_shared/base.patch reversed (the
#     lane's diff against its base 0a11b50c7; the patch travels with the sync).
#  2. tools/algos_lane_check.sh per lane in both trees (builds the stale
#     bindings, GPU arm, CPU arm, GPU == CPU), then ab_diff.py base vs head
#     per column: every hash SAME.
#  3. HEAD tests: the new ones and the families' own (pytest, .pixi/envs/test).
#  4. Timing on this machine: bench/py_shared_micro.py (CPU) and the x_prep /
#     x_metrics GPU boards with MOJOLEARN_ARENA_RANGES=0 (whole arena) and =1
#     (ranges), interleaved 0,1,1,0.
# PHASES (default lanes,tests,bench) picks the parts; SKIP_BASE=1 re-uses the
# base tree an earlier job made.
set -u
OUT=${1:-/root/ev-py-shared/ab-$(date -u +%m%dT%H%M)}; mkdir -p "$OUT"; echo "OUT $OUT"
PHASES=${PHASES:-lanes,tests,bench}
has() { case ",$PHASES," in *",$1,"*) return 0 ;; esac; return 1; }
H=$(cd "$(dirname "$0")/../.." && pwd); B=${H}-base
LANES=${2:-$(cat "$H/tools/py_shared/lanes.txt")}
export MOJOLEARN_BUILD_LOCK_HELD=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-6}
PIXI=$(command -v pixi || echo ~/.pixi/bin/pixi)
echo "$(date -u +%FT%TZ) $(hostname) head $H $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if [ "${SKIP_BASE:-0}" != 1 ]; then
  rm -rf "$B" && mkdir -p "$B"
  tar -C "$H" --exclude=./.pixi --exclude='*.so' --exclude='*.dylib' -cf - . | tar -C "$B" -xf -
  ( cd "$B" && git apply -R --whitespace=nowarn tools/py_shared/base.patch ) || { echo "BASE PATCH FAIL"; exit 1; }
  echo "base tree ready: $(cd "$B" && git status --short | wc -l) paths differ from its git HEAD"
fi

IFS=, read -ra L <<< "$LANES"
has lanes && for T in "$B" "$H"; do
  tag=$(basename "$T"); mkdir -p "$OUT/$tag"
  ( cd "$T"
    $PIXI install -e default > "$OUT/$tag/pixi.log" 2>&1 || { echo "PIXI FAIL $tag"; tail -3 "$OUT/$tag/pixi.log"; }
    for lane in "${L[@]}"; do
      PIXI=$PIXI sh tools/algos_lane_check.sh "$lane" --out "$OUT/$tag" 2>&1 | grep -E 'RESULT|CLEAN:' | sed "s/^/$tag $lane: /"
    done )
done
has lanes && python3 "$H/tools/py_shared/ab_diff.py" "$OUT/$(basename "$B")" "$OUT/$(basename "$H")"

cd "$H"
if has tests; then
$PIXI install -e test > "$OUT/pixi_test.log" 2>&1 || { echo "PIXI TEST FAIL"; tail -3 "$OUT/pixi_test.log"; }
T=python/mojolearn/tests
.pixi/envs/test/bin/python -m pytest -q -p no:cacheprovider \
  $T/test_arena_ranges.py $T/test_portable_math_fast.py $T/test_labels_native.py $T/test_hotpath_native.py \
  $T/test_x_metrics_repeat.py $T/test_x_metrics_sanity.py $T/test_x_prep_*.py $T/test_model_selection_numpy_free.py \
  2>&1 | tail -25 | sed 's/^/PYTEST /'
fi

if has bench; then
$PIXI run -e default python -u bench/py_shared_micro.py --n 1000000 --reps 3 2>&1 | grep -E 'PYSHARED|Error|error'
for arm in 0 1 1 0; do
  MOJOLEARN_ARENA_RANGES=$arm $PIXI run -e default python -u bench/x_prep_speed.py --rows 1000000 --reps 3 2>&1 \
    | grep -E '^XPSPEED|Error' | sed "s/^/RANGES=$arm /"
  MOJOLEARN_ARENA_RANGES=$arm $PIXI run -e default python -u bench/x_metrics_speed.py --rows 1000000 --reps 3 2>&1 \
    | grep -E '^XMSPEED' | sed "s/^/RANGES=$arm /"
done
fi
echo "JOB END $(date -u +%FT%TZ)"
