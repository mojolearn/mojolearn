#!/bin/sh
# RUNS ON THE POD, AFTER the sweep. DEVIATION 2634 on criteo, made ATTRIBUTABLE.
#
#   sh bench/results/baseline_sweep_2026-09-12/criteo_2634_ab.sh
#
# WHY THIS EXISTS. `bench/OPPONENT_REFERENCE.md:2274-2320` records a 2634 A/B
# that measured a clean 2.1% separation and then REFUSED TO CLAIM IT, because
# the run never observed whether the branch under test executed. 2634 gates on
# `len(dependent_configs) > 0 and ctr_prep_wanted`, and nothing in those logs
# recorded `dependent_configs`; if that list is empty on criteo then NEITHER
# build built the prep and the 2.1% belongs to something else entirely. That is
# the reached-but-inert trap, and a number whose mechanism was never witnessed
# is not a result.
#
# The marker now exists (`gbdt/train.mojo`, under MOJOLEARN_CTR_TRACE=1):
#
#   [ctr-2634] simple_ctr_configs=N independent=N dependent=N cat_columns=N \
#     ctr_prep_wanted=True prep=ran|skipped gate_2634=on|off permutations=N \
#     target_classes=N binarized_target_rows=N ctr_orders=N
#
# so this re-runs the A/B WITH the marker in the log. `dependent=` is the term
# whose emptiness would invalidate the whole comparison; `prep=` is the branch
# itself; `gate_2634=` says which side of the switch was compiled.
#
# THE SWITCH IS COMPILE-TIME AND INVERTED:
#   `comptime CTR_TARGET_PREP_NEEDS_CAT_2634 = not is_defined["MOJOLEARN_2634_CTR_PREP_OFF"]`
# so NO define is the shipped ON side (skip the prep when no column is
# categorical) and `-D MOJOLEARN_2634_CTR_PREP_OFF=1` restores the
# unconditional build. Both arms are OURS: this is ours-vs-ours and is NEVER
# quoted as an opponent ratio.
#
# criteo is NOT a section 9 gating dataset. It exists to reach the categorical
# and CTR code that taxi and Istella-S never touch, so a win here alone is a
# one-kind win and moves no default. THIS SCRIPT DECIDES NOTHING.
set -u
ROOT=/root/mojolearn
OUT=/root/trees_out
BINS=/root/bins
AB="sh tools/trees_identical_ab.sh"
ROWS="${CRITEO_ROWS:-1000000}"
ROUNDS="${CRITEO_ROUNDS:-3}"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90a
export GBM_BENCH_DATA=/root/datasets/gbm-bench
export MOJOLEARN_CTR_TRACE=1
say() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/logs/criteo2634.txt"; }

# cmd_build SEEDS A NEW SET FROM $BINS/baseline so that a set is always a
# COMPLETE tier. This sweep built straight into the tier and created
# /root/bins/all, so without this the A/B sets would hold _mojolearn_gbdt.so
# ALONE and every run would fail on the missing base extension.
if [ ! -d "$BINS/baseline" ]; then
    mkdir -p "$BINS/baseline"
    cp "$ROOT"/python/mojolearn/identical/*.so "$BINS/baseline"/ || exit 3
    say "seeded $BINS/baseline with $(ls "$BINS"/baseline/*.so | wc -l | tr -d ' ') extensions"
fi

# criteo is NOT in the R2 store, so this pays a ~291 MB fetch plus a decode.
if [ ! -s /root/datasets/gbm-bench/criteo/criteo_speed.npz ]; then
    say "downloading + decoding criteo (about 291 MB; not in the R2 store)"
    timeout -k 30 3600 python3 tools/speed_gbdt_arm.py --download criteo \
        > "$OUT/logs/criteo_download.log" 2>&1
    say "download rc=$?"
fi

# Two builds of ONE binding from ONE source tree; everything else equal.
$AB build ctr_on  gbdt                                  || exit 4
$AB build ctr_off gbdt -D MOJOLEARN_2634_CTR_PREP_OFF=1 || exit 4

# THE TWO BUILDS MUST DIFFER. If the define did not reach the compiler the
# .so are byte-identical and the whole A/B is measuring one binary twice --
# which is exactly the class of silent failure this lane keeps finding.
ON=$(sha256sum "$BINS/ctr_on/_mojolearn_gbdt.so"  | cut -d' ' -f1)
OFF=$(sha256sum "$BINS/ctr_off/_mojolearn_gbdt.so" | cut -d' ' -f1)
say "ctr_on  _mojolearn_gbdt.so $ON"
say "ctr_off _mojolearn_gbdt.so $OFF"
if [ "$ON" = "$OFF" ]; then
    say "REFUSING: the two builds are byte-identical, so -D MOJOLEARN_2634_CTR_PREP_OFF=1"
    say "  never reached the compiler and this A/B would measure one binary twice."
    exit 5
fi

# Ours only: 2634 is ours-vs-ours. Opponents add nothing and cost lease.
for set in ctr_on ctr_off; do
    MOJOLEARN_SPEED_TAG="$set" $AB speed "$set" gbdt-symmetric criteo "$ROWS" "$ROUNDS" ours
done

say "=== THE MARKER, which is what makes this attributable ==="
for set in ctr_on ctr_off; do
    _log=$(ls -t "$OUT"/speed/"$set".gbdt-symmetric.criteo.*."$set".log 2>/dev/null | head -1)
    say "--- $set  ($_log)"
    grep -m1 "ctr-2634" "$_log" 2>/dev/null | tee -a "$OUT/logs/criteo2634.txt" \
        || say "  NO [ctr-2634] MARKER -- the branch was never witnessed; report UNKNOWN"
    grep -E "^FSPEED lane" "$_log" 2>/dev/null | tail -3
done

say "=== READ IT LIKE THIS ==="
say "  gate_2634=on  + prep=ran     on ctr_on   -> the shipped branch built the prep"
say "  gate_2634=off + prep=ran     on ctr_off  -> the unconditional build did too"
say "  dependent=0 on BOTH          -> NEITHER built it and any time gap is NOT 2634"
say "A time difference without dependent>0 and prep=ran is the reached-but-inert"
say "trap, and stays UNKNOWN no matter how clean the separation looks."
