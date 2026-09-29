#!/bin/sh
# lane/sym-quality: the identity check of the weighted-fit stat snap on one
# Apple box, Metal column and CPU column, then the negative control.
#
#   sh bench/speed/sym_quality_identity.sh OUTDIR [metal|cuda|hip] [build]
#
# Expects bindings/build_gbdt.sh and bindings/build_gbdt_host.sh already run
# at this commit under MOJOLEARN_NUMERIC_MODE=identical (the steward's
# --builds), or pass `build` to run them here first (the NVIDIA and AMD
# queues; MOJOLEARN_GPU_ARCHS must name the box's arch there). Clean: the Metal and CPU columns must AGREE on every lane.
# Sabotage (-D MOJOLEARN_SNAP_SABOTAGE=1, device only): the weighted lanes
# the CPU column carries (gbdt-multiclass, gbdt-multiclass-defaults) must
# DISAGREE and every unit-weight lane must still AGREE.
set -u
OUT=${1:?outdir}
BK=${2:-metal}
mkdir -p "$OUT"
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$PWD/python"
if [ "${3:-}" = build ]; then
    pixi run -e default sh bindings/build_gbdt.sh > "$OUT/build.log" 2>&1 || { echo "build failed"; tail -40 "$OUT/build.log"; exit 1; }
    pixi run -e default sh bindings/build_gbdt_host.sh > "$OUT/build_host.log" 2>&1 || { echo "host build failed"; tail -40 "$OUT/build_host.log"; exit 1; }
fi
R="pixi run -e default python -u"
L=gbdt-class-weights,gbdt-multiclass,gbdt-multiclass-defaults,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide
HOST="$PWD/python/mojolearn/host"
rc=0
$R tools/identity_break.py --lanes $L --json "$OUT/$BK.json" --require-backend $BK --vendor $BK-symq || rc=1
MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR="$HOST" $R tools/identity_break.py --lanes $L --json "$OUT/cpu.json" \
    --require-cpu --require-backend cpu --vendor cpu-$BK-box-symq || rc=1
echo "== CLEAN DIFF ($BK vs cpu)"
$R tools/identity_break.py --diff "$OUT/$BK.json" "$OUT/cpu.json" --lanes $L --require-columns 0 || rc=1
echo "== SABOTAGE BUILD"
MOJOLEARN_EXTRA_DEFINES="-D MOJOLEARN_SNAP_SABOTAGE=1" pixi run -e default sh bindings/build_gbdt.sh > "$OUT/sabotage_build.log" 2>&1 || { echo "sabotage build failed"; tail -30 "$OUT/sabotage_build.log"; exit 1; }
$R tools/identity_break.py --lanes $L --json "$OUT/${BK}_sabotage.json" --require-backend $BK --vendor $BK-symq-sab
echo "== SABOTAGE DIFF ($BK sabotage vs cpu): the weighted lanes must DISAGREE"
$R tools/identity_break.py --diff "$OUT/${BK}_sabotage.json" "$OUT/cpu.json" --lanes $L --require-columns 0
echo "== SABOTAGE vs CLEAN $BK: only the weighted lanes may move"
$R tools/identity_break.py --diff "$OUT/${BK}_sabotage.json" "$OUT/$BK.json" --lanes $L --require-columns 0
echo "== restore the clean build"
pixi run -e default sh bindings/build_gbdt.sh > "$OUT/restore_build.log" 2>&1 || rc=1
exit $rc
