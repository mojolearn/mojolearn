#!/bin/sh
# Ready-to-run MI300X/MI325X A/B. The AMD default remains unchanged until this passes.
set -u
ROOT=/root/mojolearn
OUT=${MOJOLEARN_FOREST_AMD_OUT:-/root/trees_out/forest-ordered-amd}
MODELS=/root/forest-ordered-amd-models
PY="$ROOT/.pixi/envs/default/bin/python3"
OUTERS=${MOJOLEARN_FOREST_ORDERED_OUTERS:-3}
ROUNDS=${MOJOLEARN_FOREST_ORDERED_ROUNDS:-5}
ROWS=${MOJOLEARN_FOREST_ORDERED_ROWS:-1000000}
mkdir -p "$OUT" "$MODELS" /root/bins/baseline
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH" GBM_BENCH_DATA=/root/datasets/gbm-bench
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-4}"

die() { echo "FOREST_ORDERED_AMD FAIL: $*" >&2; exit 1; }
step() {
    _name=$1; _cap=$2; shift 2
    timeout -k 30 "$_cap" "$@" > "$OUT/$_name.log" 2>&1
    _rc=$?
    printf '%s\t%s\n' "$_name" "$_rc" >> "$OUT/status.tsv"
    [ "$_rc" -eq 0 ] || die "$_name rc=$_rc"
}

: > "$OUT/status.tsv"
rocm-smi --showproductname --showdriverversion > "$OUT/gpu.txt" 2>&1 || die rocm-smi
for f in "$GBM_BENCH_DATA/taxi/taxi_speed.npz" "$GBM_BENCH_DATA/istella/istella_speed.npz"; do
    [ -s "$f" ] || die "missing R2-staged $f"
done
[ -x "$PY" ] || die "missing pinned Python $PY"
if [ ! -s /root/bins/baseline/_mojolearn.so ]; then
    step build_base 1500 sh bindings/build.sh
    cp python/mojolearn/identical/_mojolearn.so /root/bins/baseline/
fi
step build_seq_rf 1500 sh tools/trees_identical_ab.sh build forestamdseq rf
step build_seq_et 1500 sh tools/trees_identical_ab.sh build forestamdseq trees
step build_ord_rf 1500 sh tools/trees_identical_ab.sh build forestamdordered rf -D MOJOLEARN_FOREST_ORDERED_RESIDENT=1
step build_ord_et 1500 sh tools/trees_identical_ab.sh build forestamdordered trees -D MOJOLEARN_FOREST_ORDERED_RESIDENT=1
step analytic_ordered 1500 pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    -D MOJOLEARN_FOREST_ORDERED_RESIDENT=1 checks/forest_inference_model.mojo
step use_prepare 60 sh tools/trees_identical_ab.sh use forestamdseq
step prepare 3600 env PYTHONPATH="$ROOT/python" "$PY" -u \
    bench/speed/forest_ordered_resident_ab.py prepare --out "$MODELS"

outer=1
while [ "$outer" -le "$OUTERS" ]; do
    if [ $((outer % 2)) -eq 1 ]; then order="sequential ordered"; else order="ordered sequential"; fi
    for arm in $order; do
        setname=forestamdseq; [ "$arm" = ordered ] && setname=forestamdordered
        step "use.$outer.$arm" 60 sh tools/trees_identical_ab.sh use "$setname"
        for dataset in taxi istella; do
            step "run.$outer.$arm.$dataset" 3600 env PYTHONPATH="$ROOT/python" "$PY" -u \
                bench/speed/forest_ordered_resident_ab.py run --dataset "$dataset" \
                --arm "$arm" --models "$MODELS" --outer "$outer" --rounds "$ROUNDS" \
                --rows "$ROWS" --json "$OUT/$dataset.$arm.o$outer.json"
        done
    done
    outer=$((outer + 1))
done
step summarize 300 env PYTHONPATH="$ROOT/python" "$PY" \
    bench/speed/forest_ordered_resident_ab.py summarize "$OUT"/'*.o*.json' \
    --outers "$OUTERS" --out "$OUT/verdict.json"
cat "$OUT/verdict.json"
