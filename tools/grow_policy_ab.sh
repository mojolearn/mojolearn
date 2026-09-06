#!/bin/sh
# THE LOSSGUIDE RUNG DEVIATION 1902 ASKED FOR AND NO HARNESS PROVIDED.
#
# `ridx_only_splits_for`'s docstring names its own gate: "the orchestrator's
# A/B (byte-compare of the FAST model pre/post routing, plus the 1M/2M
# LOSSGUIDE rungs) is where the default dies if the gather costs more than
# the reorder saved". `tools/fast_replication_ab.sh` runs lanes `gbdt rf et`
# out of `bench/lanes_price_main.mojo`, which has no lossguide lane and no
# depthwise lane, so the half of the gate that could VINDICATE the row has
# never been runnable by the harness that keeps ruling on it.
#
# That asymmetry matters here because the row's saving is explicitly
# proportional to "the `max_leaves - 1` sequential splits a LOSSGUIDE tree
# runs". Measuring only symmetric lanes and turning the row off would be
# deciding a case on the evidence that happens to exist -- the symmetric
# lanes collect none of the benefit by construction.
#
# So this drives `bench/speed/forest_speed_arm.py --ours-only` across the
# GROW POLICIES, one process per (policy, arm), and prints the medians beside
# each other. Ours only: no opponent's number changes with our build define.
#
#   MOJOLEARN_GP_DEFINES   defines to test, space separated (each its own build)
#   MOJOLEARN_GP_LANES     default "gbdt-lossguide gbdt-depthwise"
#   MOJOLEARN_GP_ROWS      training rows (default 1000000, floor 1000000)
set -u

OUT="${MOJOLEARN_GP_OUT:-bench/results/grow_policy_ab/$(date +%Y-%m-%d_%H%M%S)}"
DEFINES="${MOJOLEARN_GP_DEFINES:-MOJOLEARN_2044_FAST_NO_RIDX_ONLY MOJOLEARN_2045_FAST_NO_QUANT_HIST}"
LANES="${MOJOLEARN_GP_LANES:-gbdt-lossguide gbdt-depthwise}"
ROWS="${MOJOLEARN_GP_ROWS:-1000000}"
PY="${MOJOLEARN_GP_PYTHON:-pixi run -e gbmbench python}"
# shellcheck disable=SC2086  # $PY is deliberately word-split
py() { $PY "$@"; }
MOJOLEARN_PYTHON="$PY"; export MOJOLEARN_PYTHON

if [ "$ROWS" -lt 1000000 ]; then
    echo "!! MOJOLEARN_GP_ROWS=$ROWS is below the 1,000,000-row tree floor; refusing" >&2
    exit 2
fi

mkdir -p "$OUT/logs" "$OUT/so"
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host $(hostname)"; echo "lanes $LANES"; echo "rows $ROWS"
  echo "defines $DEFINES"; echo "store ${GBM_BENCH_DATA:-unset}"
  command -v nvidia-smi >/dev/null 2>&1 && echo "device $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
} > "$OUT/env.txt" 2>&1
cat "$OUT/env.txt"

build_fail=0
echo "== build BASE (FAST as shipped) =="
bash bindings/build_estimators.sh > "$OUT/logs/build.estimators.log" 2>&1 \
    || { echo "!! estimators build failed"; build_fail=1; }
bash bindings/build_gbdt.sh > "$OUT/logs/build.base.log" 2>&1 \
    || { echo "!! BASE build failed"; tail -15 "$OUT/logs/build.base.log"; build_fail=1; }
cp python/mojolearn/_mojolearn_gbdt.so "$OUT/so/base.so" 2>/dev/null || build_fail=1

for D in $DEFINES; do
    echo "== build $D =="
    MOJOLEARN_EXTRA_DEFINES="-D $D=1" bash bindings/build_gbdt.sh \
        > "$OUT/logs/build.$D.log" 2>&1 \
        || { echo "!! $D build failed"; tail -15 "$OUT/logs/build.$D.log"; build_fail=1; }
    cp python/mojolearn/_mojolearn_gbdt.so "$OUT/so/$D.so" 2>/dev/null || build_fail=1
done
# Leave the checkout holding BASE: a diagnostic binary under the shipped name
# is how the next run in this checkout silently measures the wrong arm.
cp "$OUT/so/base.so" python/mojolearn/_mojolearn_gbdt.so 2>/dev/null || true

run_one() {  # run_one <so> <lane> <tag>
    cp "$1" python/mojolearn/_mojolearn_gbdt.so 2>/dev/null || return 1
    MOJOLEARN_NUMERIC_MODE=fast py bench/speed/forest_speed_arm.py \
        --lane "$2" --dataset higgs --rows "$ROWS" --devices gpu --ours-only \
        > "$OUT/logs/$3.log" 2>&1
}
med() { grep "^FSPEED " "$1" | grep "arm=ours " \
    | awk '{for(i=1;i<=NF;i++) if(index($i,"ms=")==1) print substr($i,4)}' \
    | sort -n | awk '{v[n++]=$1} END{if(n==0)exit;
        if(n%2)printf "%.1f",v[(n-1)/2]; else printf "%.1f",(v[n/2-1]+v[n/2])/2}'; }
hsh() { grep "^FSPEED " "$1" \
    | awk '{for(i=1;i<=NF;i++) if(index($i,"hash=")==1) print substr($i,6)}' \
    | sort -u | tr '\n' ',' | sed 's/,$//'; }
shp() { grep "^FSPEED " "$1" | head -1 \
    | awk '{for(i=1;i<=NF;i++) if(index($i,"shape=")==1) printf "%s", substr($i,7)}'; }

echo
for L in $LANES; do
    echo "== lane $L =="
    run_one "$OUT/so/base.so" "$L" "base.$L"
    BM=$(med "$OUT/logs/base.$L.log"); BH=$(hsh "$OUT/logs/base.$L.log")
    S=$(shp "$OUT/logs/base.$L.log")
    echo "GPAB lane=$L arm=base shape=${S:-none} med_ms=${BM:--} hash=${BH:--}"
    for D in $DEFINES; do
        run_one "$OUT/so/$D.so" "$L" "$D.$L"
        PM=$(med "$OUT/logs/$D.$L.log"); PH=$(hsh "$OUT/logs/$D.$L.log")
        if [ -n "$BM" ] && [ -n "$PM" ]; then
            R=$(awk -v a="$PM" -v b="$BM" 'BEGIN{if(b>0)printf "%.3f", a/b; else printf "-"}')
        else R="-"; fi
        V="bits equal"; [ "$BH" != "$PH" ] && V="BITS MOVED -- not a speed result"
        echo "GPAB lane=$L arm=$D med_ms=${PM:--} ratio=$R $V"
    done
done
cp "$OUT/so/base.so" python/mojolearn/_mojolearn_gbdt.so 2>/dev/null || true

echo
echo "ratio = define / base. Below 1.0 means the define made FAST faster."
echo "A ratio is a speed result only on a row whose hashes are EQUAL."
echo "results in $OUT"
rc=0
[ "$build_fail" != "0" ] && rc=2
# THE SHAPE TAG AND THE MISSING-ROUNDS CASE ARE DIFFERENT FAILURES AND THIS
# CONFLATED THEM. `shp` reads timed `FSPEED ` lines only, so an arm whose
# rounds were all SKIPPED leaves it empty and the old test then reported
# "NOT HIGGS: a synthetic fixture" over a run whose fixture was higgs. That
# happened on the first run: gbdt-lossguide's WARM-UP FIT ALONE took 338.7 s
# against the harness's 300 s per-arm budget, every timed round was skipped,
# and the leg blamed the dataset. A wrong diagnosis is worse than no
# diagnosis; it sends the next person to fix the fetch.
#
# So: read the shape from the WARM-UP too, and separate the two verdicts.
for L in $LANES; do
    LOG="$OUT/logs/base.$L.log"
    S=$(shp "$LOG")
    [ -z "$S" ] && S=$(grep "^FSPEED-WARMUP " "$LOG" 2>/dev/null | head -1 \
        | awk '{for(i=1;i<=NF;i++) if(index($i,"shape=")==1) printf "%s", substr($i,7)}')
    case "${S:-none}" in
        higgs-*) ;;
        none) echo "!! lane $L produced NO fixture at all -- the arm never ran"; rc=4 ;;
        *) echo "!! lane $L ran on $S, not higgs: a synthetic fixture answers nothing here"; rc=4 ;;
    esac
    if [ -z "$(med "$LOG")" ]; then
        echo "!! lane $L has NO TIMED ROUNDS. Check for a budget refusal:"
        grep "^FSPEED-REFUSED " "$LOG" 2>/dev/null | sed 's/^/     /' | head -2
        echo "     This is a BUDGET result, not a dataset result. Raise"
        echo "     MOJOLEARN_SPEED_BUDGET (or the lane's rounds) and re-run."
        rc=5
    fi
done
[ "$rc" != "0" ] && echo "!! grow_policy_ab FAILED (rc=$rc)"
exit $rc
