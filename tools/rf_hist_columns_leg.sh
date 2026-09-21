#!/bin/sh
# Local/provider-neutral exact A/B body for the existing default-off RF tile4.
set -u
ROOT=${MOJOLEARN_RF_COLUMNS_ROOT:-$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)}
OUT=${MOJOLEARN_RF_COLUMNS_OUT:-$ROOT/bench/results/rf_hist_columns_trial}
PY=${MOJOLEARN_RF_COLUMNS_PY:-$ROOT/.pixi/envs/default/bin/python}
ROWS=${MOJOLEARN_RF_COLUMNS_ROWS:-1000000}
REPEATS=${MOJOLEARN_RF_COLUMNS_REPEATS:-5}
MODULE=$ROOT/python/mojolearn/identical/_mojolearn_rf.so
if [ "$(uname -s)" = Darwin ] && [ -z "${MOJOLEARN_SLOT_TOKEN:-}" ]; then
    exec python3 "$ROOT/tools/mac_slot.py" --timeout 7200 --wait-timeout 3600 \
        metal sh "$ROOT/tools/rf_hist_columns_leg.sh"
fi
mkdir -p "$OUT/bin" "$OUT/logs"
cd "$ROOT" || exit 9

die() { echo "RF_HIST_COLUMNS FAIL: $*" >&2; exit 1; }
step() {
    name=$1 cap=$2; shift 2
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 30 "$cap" "$@" > "$OUT/logs/$name.log" 2>&1
    else
        "$@" > "$OUT/logs/$name.log" 2>&1
    fi
    rc=$?
    printf '%s\t%s\n' "$name" "$rc" >> "$OUT/status.tsv"
    [ "$rc" -eq 0 ] || die "$name rc=$rc"
}

[ "${MOJOLEARN_RF_COLUMNS_RUN_GUARD:-}" = R2_TAXI_ISTELLA ] || \
    die "set MOJOLEARN_RF_COLUMNS_RUN_GUARD=R2_TAXI_ISTELLA"
[ -n "${GBM_BENCH_DATA:-}" ] || die "GBM_BENCH_DATA must name the R2 staging root"
for file in "$GBM_BENCH_DATA/taxi/taxi_speed.npz" \
            "$GBM_BENCH_DATA/istella/istella_speed.npz"; do
    [ -s "$file" ] || die "missing R2-staged $file"
done
[ -x "$PY" ] || die "missing Python $PY"
[ -s "$MODULE" ] || die "missing installed RF binding"
: > "$OUT/status.tsv"
sha256sum "$GBM_BENCH_DATA/taxi/taxi_speed.npz" \
          "$GBM_BENCH_DATA/istella/istella_speed.npz" > "$OUT/r2-inputs.sha256"
commit=${MOJOLEARN_COMMIT:-$(git rev-parse HEAD 2>/dev/null)}
[ -n "$commit" ] || commit=$(cat "$ROOT/SHIPPED_COMMIT.txt" 2>/dev/null)
[ -n "$commit" ] || die "missing source commit witness"
printf '%s\n' "$commit" > "$OUT/commit.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L > "$OUT/gpu.txt"
elif command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showdriverversion > "$OUT/gpu.txt"
else
    system_profiler SPDisplaysDataType > "$OUT/gpu.txt" 2>&1 || die "missing GPU inventory"
fi
cp "$MODULE" "$OUT/bin/original.so"
restore() { cp "$OUT/bin/original.so" "$MODULE"; }
trap restore EXIT INT TERM
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4}
export PYTHONPATH="$ROOT/python"

MOJOLEARN_EXTRA_DEFINES='' step build_baseline 1500 sh bindings/build_rf.sh
cp "$MODULE" "$OUT/bin/baseline.so"
printf '%s\n' '(none)' > "$OUT/bin/baseline.defines"
MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_RF_HIST_COLUMNS4_WIDE=1' \
    step build_columns4 1500 sh bindings/build_rf.sh
cp "$MODULE" "$OUT/bin/columns4.so"
printf '%s\n' 'MOJOLEARN_RF_HIST_COLUMNS4_WIDE=1' > "$OUT/bin/columns4.defines"
restore
sha256sum "$OUT/bin/baseline.so" "$OUT/bin/columns4.so" > "$OUT/binaries.sha256"

for arm in baseline columns4; do
    : > "$OUT/$arm.launches"
    expected=''
    [ "$arm" = columns4 ] && expected=--expect-tile4
    # shellcheck disable=SC2086 -- expected is one optional flag.
    RF_LAUNCH_LOG="$OUT/$arm.launches" step "probe_$arm" 900 env \
        RF_LAUNCH_LOG="$OUT/$arm.launches" "$PY" bench/speed/rf_hist_columns_ab.py probe \
        --dataset istella --arm "$arm" --binding "$OUT/bin/$arm.so" $expected
done

: > "$OUT/columns4.taxi-fallback.launches"
RF_LAUNCH_LOG="$OUT/columns4.taxi-fallback.launches" step probe_columns4_taxi_fallback 900 env \
    RF_LAUNCH_LOG="$OUT/columns4.taxi-fallback.launches" "$PY" \
    bench/speed/rf_hist_columns_ab.py probe --dataset taxi --arm columns4 \
    --binding "$OUT/bin/columns4.so"

# Effective negative control: a baseline binary falsely labelled tile4 must be
# rejected by the same compile/route witness used above.
: > "$OUT/sabotage.launches"
RF_LAUNCH_LOG="$OUT/sabotage.launches" "$PY" bench/speed/rf_hist_columns_ab.py probe \
    --dataset istella --arm baseline --binding "$OUT/bin/baseline.so" \
    --expect-tile4 > "$OUT/logs/probe_sabotage.log" 2>&1
rc=$?
[ "$rc" -ne 0 ] || die "selection sabotage was accepted"
printf 'probe_sabotage_rejected\t0\n' >> "$OUT/status.tsv"

outer=0
while [ "$outer" -lt 3 ]; do
    if [ $((outer % 2)) -eq 0 ]; then order="baseline columns4"; else order="columns4 baseline"; fi
    position=0
    for arm in $order; do
        for dataset in taxi istella; do
            step "run_${outer}_${arm}_${dataset}" 7200 env "$PY" \
                bench/speed/rf_hist_columns_ab.py run --dataset "$dataset" \
                --arm "$arm" --binding "$OUT/bin/$arm.so" --outer "$outer" \
                --launch-position "$position" --rows "$ROWS" --repeats "$REPEATS" \
                --json "$OUT/$dataset.$arm.o$outer.json"
        done
        position=$((position + 1))
    done
    outer=$((outer + 1))
done
step summarize 300 "$PY" bench/speed/rf_hist_columns_ab.py summarize \
    "$OUT"/'*.o*.json' --wide-only --out "$OUT/verdict.json"
cat "$OUT/verdict.json"
