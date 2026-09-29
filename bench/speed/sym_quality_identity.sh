#!/bin/sh
# lane/sym-quality: the identity check of the weighted-fit stat snap on one
# Apple box, Metal column and CPU column, then the negative control.
#
#   sh bench/speed/sym_quality_identity.sh OUTDIR [metal|cuda|hip] [build]
#
# Expects bindings/build_gbdt.sh and bindings/build_gbdt_host.sh already run
# at this commit under MOJOLEARN_NUMERIC_MODE=identical (the steward's
# --builds, with bindings/build.sh and build_core_host.sh), or pass `build` to run them here first (the NVIDIA and AMD
# queues; MOJOLEARN_GPU_ARCHS must name the box's arch there). Clean: the Metal and CPU columns must AGREE on every lane.
# Every GBDT lane of tools/identity_break.py. Sabotage (-D
# MOJOLEARN_SNAP_SABOTAGE=1, device only, the gradient snap keyed one row
# off): every lane whose fit goes through the greedy searcher
# (run_tree_layout / fit_non_symmetric_tree) must DISAGREE with the CPU
# column; the pointwise and Ordered lanes, which the snap does not reach,
# must still AGREE.
set -u
OUT=${1:?outdir}
BK=${2:-metal}
mkdir -p "$OUT"
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$PWD/python"
HOSTD="$PWD/python/mojolearn/host"
if [ "${3:-}" = build ]; then
    pixi run -e default sh bindings/build.sh > "$OUT/build_base.log" 2>&1 || { echo "base build failed"; tail -40 "$OUT/build_base.log"; exit 1; }
    pixi run -e default sh bindings/build_gbdt.sh > "$OUT/build.log" 2>&1 || { echo "build failed"; tail -40 "$OUT/build.log"; exit 1; }
fi
if [ "${3:-}" = build ] || [ "${3:-}" = hostbuild ]; then
    # the metrics binding gbdt-adapter-score-weighted's score() reaches
    [ -f python/mojolearn/identical/_mojolearn_metrics.so ] || pixi run -e default sh bindings/build_metrics.sh > "$OUT/build_metrics.log" 2>&1 || echo "metrics build failed (gbdt-adapter-score-weighted will refuse)"
fi
if [ "${3:-}" = build ] || [ "${3:-}" = hostbuild ]; then
    # the host builds never overwrite: this tree's stale host objects go first
    rm -f "$HOSTD/_mojolearn_core_host.so" "$HOSTD/_mojolearn_gbdt_host.so" "$HOSTD/_mojolearn_forest_host.so" "$HOSTD/_mojolearn_metrics_host.so"
    env -u MOJOLEARN_GPU_ARCHS pixi run -e default sh bindings/build_core_host.sh > "$OUT/build_core_host.log" 2>&1 || { echo "core host build failed"; tail -40 "$OUT/build_core_host.log"; exit 1; }
    env -u MOJOLEARN_GPU_ARCHS pixi run -e default sh bindings/build_gbdt_host.sh > "$OUT/build_host.log" 2>&1 || { echo "host build failed"; tail -40 "$OUT/build_host.log"; exit 1; }
    # the CTR-table lanes' CPU route and the weighted adapter's score()
    env -u MOJOLEARN_GPU_ARCHS pixi run -e default sh bindings/build_forest_host.sh > "$OUT/build_forest_host.log" 2>&1 || echo "forest host build failed (the CTR-table lanes will refuse)"
    env -u MOJOLEARN_GPU_ARCHS pixi run -e default sh bindings/build_metrics_host.sh > "$OUT/build_metrics_host.log" 2>&1 || echo "metrics host build failed"
fi
R="pixi run -e default python -u"
# libMojolearnMath, which `_portable_math` dlopens on the CPU arm (the
# Bayesian bootstrap of gbdt-multiclass-defaults reaches it): the tree's own
# recipe, as tools/algos_lane_check.py ensure_portable_math builds it
case "$(uname)" in Darwin) PM=python/mojolearn/.dylibs/libMojolearnMath.dylib ;; *) PM=python/mojolearn/.libs/libMojolearnMath.so ;; esac
[ -f "$PM" ] || PYTHONPATH=packaging/portable_math $R -c "import pathlib, stage; stage.build(pathlib.Path('$PM'))" > "$OUT/portable_math.log" 2>&1 || { echo "portable math build failed"; tail -20 "$OUT/portable_math.log"; }
L=gbdt-symmetric,gbdt-symmetric-eval,gbdt-depthwise,gbdt-lossguide,gbdt-class-weights,gbdt-multiclass-offgrid,gbdt-rmse,gbdt-ordered,gbdt-ordered-bayesian-noise,gbdt-binary-columns,gbdt-catboost-defaults,gbdt-stochastic-arms,gbdt-bfa-quantile,gbdt-border-types,gbdt-ordered-rmse,gbdt-feature-freq,gbdt-multiclass,gbdt-onevsall,gbdt-multiclass-defaults,gbdt-parametric-losses,gbdt-lossguide-newtoncosine,gbdt-pointwise-l2-bayesian-eval,gbdt-exact-mae,gbdt-categorical-ctr,gbdt-categorical-ctr-tables,gbdt-tensor-ctr-tables,gbdt-nan-modes,gbdt-adapter-clf,gbdt-adapter-reg,gbdt-ranking-defaults,gbdt-query-rmse,gbdt-pair-logit,gbdt-yeti-rank,gbdt-adapter-score-weighted
HOST="$HOSTD"
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
