# Forest host gate recordings on a GPU box (docs/lanes/BRIEF_forest_host_inference_2026-09-13.md):
# build the IDENTICAL forest bindings for this box, then for each of the eight
# kinds fit the small model on THIS GPU (`make`) and record its predictions
# (`record`), one directory per kind, named by vendor and kind. The Mac's
# host binding and the seven CI CPUs then check these against the GPU bits.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/forest_host
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
run() { _n=$1; shift; _t0=$(date +%s); "$@" > "$OUT/logs/$_n.log" 2>&1; _e=$?; echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"; return "$_e"; }
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) vendor=${MOJOLEARN_TARGET_COLUMN:-unset} archs=${MOJOLEARN_GPU_ARCHS:-unset}"
BUILD_ENV="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}"
for b in build build_rf build_trees build_gbdt build_estimators; do run "$b" $BUILD_ENV sh bindings/$b.sh || say "BUILD FAILED $b"; done
V=${MOJOLEARN_TARGET_COLUMN:-gpu}
PY="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python"
for k in rf_classifier rf_regressor et_classifier et_regressor gbdt_symmetric gbdt_depthwise gbdt_lossguide gbdt_rmse; do
    d="$OUT/2026-09-13-$V-$k"
    run "make-$k" $PY tools/forest_host_gate.py make "$d" --kind "$k" || say "MAKE FAILED $k"
    run "record-$k" $PY tools/forest_host_gate.py record "$d" || say "RECORD FAILED $k"
    grep -h -E "^record " "$OUT/logs/record-$k.log" >> "$G" 2>/dev/null
done
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
