#!/bin/sh
# lane/sym-quality: the identity check of the weighted-fit stat snap on one
# Apple box, Metal column and CPU column, then the negative control.
#
#   sh bench/speed/sym_quality_identity.sh OUTDIR
#
# Expects bindings/build_gbdt.sh and bindings/build_gbdt_host.sh already run
# at this commit under MOJOLEARN_NUMERIC_MODE=identical (the steward's
# --builds). Clean: the Metal and CPU columns must AGREE on every lane.
# Sabotage (-D MOJOLEARN_SNAP_SABOTAGE=1, device only): the weighted lanes
# the CPU column carries (gbdt-multiclass, gbdt-multiclass-defaults) must
# DISAGREE and every unit-weight lane must still AGREE.
set -u
OUT=${1:?outdir}
mkdir -p "$OUT"
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$PWD/python"
R="pixi run -e default python -u"
L=gbdt-class-weights,gbdt-multiclass,gbdt-multiclass-defaults,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide
HOST="$PWD/python/mojolearn/host"
rc=0
$R tools/identity_break.py --lanes $L --json "$OUT/metal.json" --require-backend metal --vendor apple-m2pro-symq || rc=1
MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR="$HOST" $R tools/identity_break.py --lanes $L --json "$OUT/cpu.json" \
    --require-cpu --require-backend cpu --vendor cpu-m2pro-symq || rc=1
echo "== CLEAN DIFF (metal vs cpu)"
$R tools/identity_break.py --diff "$OUT/metal.json" "$OUT/cpu.json" --lanes $L --require-columns 0 || rc=1
echo "== SABOTAGE BUILD"
MOJOLEARN_EXTRA_DEFINES="-D MOJOLEARN_SNAP_SABOTAGE=1" pixi run -e default sh bindings/build_gbdt.sh > "$OUT/sabotage_build.log" 2>&1 || { echo "sabotage build failed"; tail -30 "$OUT/sabotage_build.log"; exit 1; }
$R tools/identity_break.py --lanes $L --json "$OUT/metal_sabotage.json" --require-backend metal --vendor apple-m2pro-symq-sab
echo "== SABOTAGE DIFF (metal_sabotage vs cpu): the weighted lanes must DISAGREE"
$R tools/identity_break.py --diff "$OUT/metal_sabotage.json" "$OUT/cpu.json" --lanes $L --require-columns 0
echo "== SABOTAGE vs CLEAN METAL: only the weighted lanes may move"
$R tools/identity_break.py --diff "$OUT/metal_sabotage.json" "$OUT/metal.json" --lanes $L --require-columns 0
echo "== restore the clean build"
pixi run -e default sh bindings/build_gbdt.sh > "$OUT/restore_build.log" 2>&1 || rc=1
exit $rc
