#!/bin/sh
# Ready-to-run guarded GPU A/B for the default-off RF bootstrap/gather fusion.
# The provider wrapper owns lease/watchdog/teardown; this body never rents.
set -u

ROOT=${MOJOLEARN_RF_FUSED_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_RF_FUSED_OUT:-/root/trees_out/rf-fused-bootstrap}
DATA=${GBM_BENCH_DATA:-/root/datasets/gbm-bench}
PY=${MOJOLEARN_RF_FUSED_PY:-$ROOT/.pixi/envs/default/bin/python3}
ROWS=${MOJOLEARN_RF_FUSED_ROWS:-1000000}
PREDICT_ROWS=${MOJOLEARN_RF_FUSED_PREDICT_ROWS:-100000}
TREES=${MOJOLEARN_RF_FUSED_TREES:-100}
DEPTH=${MOJOLEARN_RF_FUSED_DEPTH:-16}
REPEATS=${MOJOLEARN_RF_FUSED_REPEATS:-5}

die() { echo "RF_FUSED_BOOTSTRAP FAIL: $*" >&2; exit 1; }
step() {
    _name=$1; _cap=$2; shift 2
    timeout -k 30 "$_cap" "$@" > "$OUT/$_name.log" 2>&1
    _rc=$?
    printf '%s\t%s\n' "$_name" "$_rc" >> "$OUT/status.tsv"
    [ "$_rc" -eq 0 ] || die "$_name rc=$_rc"
}

[ "${MOJOLEARN_RF_FUSED_RUN_GUARD:-}" = "R2_TAXI_ISTELLA" ] || \
    die "set MOJOLEARN_RF_FUSED_RUN_GUARD=R2_TAXI_ISTELLA"
[ -d "$ROOT" ] || die "missing source tree $ROOT"
[ -x "$PY" ] || die "missing pinned Python $PY"
COMMIT=${MOJOLEARN_COMMIT:-$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)}
[ -n "$COMMIT" ] || COMMIT=$(cat "$ROOT/SHIPPED_COMMIT.txt" 2>/dev/null)
[ -n "$COMMIT" ] || die "missing commit witness"
for f in "$DATA/taxi/taxi_speed.npz" "$DATA/istella/istella_speed.npz"; do
    [ -s "$f" ] || die "missing R2-staged $f"
done
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU=nvidia
elif command -v rocm-smi >/dev/null 2>&1; then
    GPU=amd
else
    die "no NVIDIA or AMD GPU inventory command"
fi

mkdir -p "$OUT" /root/bins/baseline
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export GBM_BENCH_DATA="$DATA" MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-4}"
: > "$OUT/status.tsv"
printf '%s\n' "$COMMIT" > "$OUT/commit.txt"
if [ "$GPU" = nvidia ]; then
    nvidia-smi -L > "$OUT/gpu.txt" 2>&1 || die nvidia-smi
else
    rocm-smi --showproductname --showdriverversion > "$OUT/gpu.txt" 2>&1 || die rocm-smi
fi
sha256sum "$DATA/taxi/taxi_speed.npz" "$DATA/istella/istella_speed.npz" > "$OUT/r2-inputs.sha256"

if [ ! -s /root/bins/baseline/_mojolearn.so ]; then
    step build_base 1500 sh bindings/build.sh
    cp python/mojolearn/identical/_mojolearn.so /root/bins/baseline/
fi
step build_baseline 1500 sh tools/trees_identical_ab.sh build rffusedbase rf
step build_fused 1500 sh tools/trees_identical_ab.sh build rffusedcand rf \
    -D MOJOLEARN_RF_FUSED_BOOTSTRAP_GATHER=1

outer=0
while [ "$outer" -lt 3 ]; do
    if [ $((outer % 2)) -eq 0 ]; then order="baseline fused"; else order="fused baseline"; fi
    position=0
    for arm in $order; do
        setname=rffusedbase
        [ "$arm" = fused ] && setname=rffusedcand
        step "use.$outer.$arm" 60 sh tools/trees_identical_ab.sh use "$setname"
        for dataset in taxi istella; do
            step "run.$outer.$arm.$dataset" 7200 env PYTHONPATH="$ROOT/python" "$PY" -u \
                bench/speed/rf_fused_bootstrap_ab.py run \
                --dataset "$dataset" --arm "$arm" --outer "$outer" \
                --launch-position "$position" --rows "$ROWS" \
                --predict-rows "$PREDICT_ROWS" --trees "$TREES" \
                --depth "$DEPTH" --repeats "$REPEATS" \
                --json "$OUT/$dataset.$arm.o$outer.json"
        done
        position=$((position + 1))
    done
    outer=$((outer + 1))
done

step summarize 300 env PYTHONPATH="$ROOT/python" "$PY" \
    bench/speed/rf_fused_bootstrap_ab.py summarize "$OUT"/'*.o*.json' \
    --out "$OUT/verdict.json"
cat "$OUT/verdict.json"
