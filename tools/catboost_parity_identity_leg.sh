#!/bin/sh
# tools/catboost_parity_identity_leg.sh -- the OWED NVIDIA and AMD legs of
# lane/catboost-parity (2026-09-19): Ordered boosting (boosting_type='Ordered')
# and the six non-default feature_border_types, whose Apple M4 Metal column and
# CPU host column agree bit for bit on this Mac and have no NVIDIA or AMD
# column yet. docs/lanes/LANE_STATUS_catboost-parity.md carries the Apple
# hashes this leg's JSON is diffed against at home.
#
# WHAT IT ASKS, on one box, one source:
#
#   ARM clean     every IDENTICAL binding, then identity_break on the three new
#                 lanes (gbdt-ordered, gbdt-ordered-bayesian-noise,
#                 gbdt-border-types) and six guard lanes that must still read
#                 their shipped references (gbdt-symmetric, gbdt-rmse,
#                 gbdt-ordered-rmse, gbdt-pointwise-l2-bayesian-eval,
#                 gbdt-depthwise, gbdt-lossguide), all nine fixtures.
#   ARM sabotage  the gbdt binding rebuilt with MOJOLEARN_ORDERED_SABOTAGE and
#                 MOJOLEARN_BORDER_TYPES_SABOTAGE; the three new lanes must
#                 read DIVERGENT from the clean arm and the six guards
#                 IDENTICAL (the two arms act only on the new branches).
#   ARM par       with two or more GPUs visible, par-ordered,
#                 par-ordered-rmse, par-border-types, par-boosting and
#                 par-boosting-reg at MOJOLEARN_PAR_DEVICES=0,1: fit_boosting's
#                 partitioned histograms must equal the one-device fit (each
#                 lane asserts it), and the two-device column must hash equal
#                 to the one-device column. par-boosting and par-boosting-reg
#                 ride along because the logical-shard diagnostic on the M4
#                 read intermittent two-shard divergence on the greedy
#                 (Plain) partition (docs/lanes/LANE_STATUS_catboost-parity.md);
#                 the par arm runs three repeats for that reason.
#   host check    `check-border-types` (294 CatBoost border cases, host code)
#                 and its sabotage, which must fail.
#
# THE VERDICT IS COMPUTED AT HOME on the Mac:
#   PYTHONPATH=python python3 tools/identity_break.py --diff \
#       bench/results/identity_break/2026-09-19_catboost-parity/apple-m4.json \
#       bench/results/identity_break/2026-09-19_catboost-parity/cpu-apple-m4.json \
#       <leg out>/remote/catboost_parity/identity_break.<label>.json
#   (three columns IDENTICAL on every cell of the three new lanes is the pass)
#
# HOW IT RUNS: as the MOJOLEARN_GEMM_LEG_EXTRA body of tools/do_extra_leg.sh
# (DigitalOcean AMD MI325X gfx942), tools/hotaisle_leg.sh (Hot Aisle MI300X) or
# tools/gemm_remote_leg.sh (RunPod NVIDIA H100 sm_90a), which ship the pinned
# commit to /root/mojolearn, run `pixi install` and copy this file to
# /root/gemm_leg_extra.sh. NOT RUN on 2026-09-19: the lane's brief forbade
# renting. POSIX sh only (dash on the images).
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/catboost_parity
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "commit=$(cat /root/mojolearn/COMMIT 2>/dev/null || echo unknown)"
(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; rocminfo 2>/dev/null | grep -m1 -oE "gfx[0-9a-z]+") > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
        case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;; 8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    elif command -v rocminfo >/dev/null 2>&1; then
        MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    fi
    export MOJOLEARN_GPU_ARCHS
fi
[ -n "${MOJOLEARN_TARGET_COLUMN:-}" ] || { case "${MOJOLEARN_GPU_ARCHS:-}" in sm_*) MOJOLEARN_TARGET_COLUMN=nvidia ;; gfx*) MOJOLEARN_TARGET_COLUMN=amd ;; esac; export MOJOLEARN_TARGET_COLUMN; }
say "gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset} column=${MOJOLEARN_TARGET_COLUMN:-unset}"

BUILD_ENV="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}"
NEW=gbdt-ordered,gbdt-ordered-bayesian-noise,gbdt-border-types,gbdt-bfa-quantile,gbdt-catboost-defaults
GUARDS=gbdt-symmetric,gbdt-rmse,gbdt-ordered-rmse,gbdt-pointwise-l2-bayesian-eval,gbdt-depthwise,gbdt-lossguide
LABEL=${MOJOLEARN_IDENTITY_VENDOR_LABEL:-${MOJOLEARN_TARGET_COLUMN:-box}-${MOJOLEARN_GPU_ARCHS:-arch}}
IB="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py"

built=0; failed=""
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh)
    case "$n" in build_byte_lm|build_byte_lm_host|build_forest_host|*_host) continue ;; esac
    if run "$n" $BUILD_ENV sh "$s"; then built=$((built + 1)); else failed="$failed $n"; fi
done
say "bindings_built=$built failed=${failed:-none}"
if [ -n "$failed" ]; then say "verdict=BUILD-FAILED"; exit 1; fi

# ARM clean
run ib_clean $IB --lanes "$NEW,$GUARDS" --vendor "$LABEL" --json "$OUT/identity_break.$LABEL.json"
say "ib_clean_exit=$(awk -F'\t' '$1=="ib_clean"{print $2}' "$OUT/status.tsv")"

# ARM par (two devices)
_ndev=0
if command -v nvidia-smi >/dev/null 2>&1; then _ndev=$(nvidia-smi -L 2>/dev/null | wc -l); fi
if [ "$_ndev" -lt 1 ] && command -v rocm-smi >/dev/null 2>&1; then _ndev=$(rocm-smi --showid 2>/dev/null | grep -c 'GPU\['); fi
say "devices=$_ndev"
if [ "$_ndev" -ge 2 ]; then
    run ib_par env MOJOLEARN_PAR_DEVICES=0,1 $IB --repeats 3 --lanes par-ordered,par-ordered-rmse,par-border-types,par-boosting,par-boosting-reg --vendor "$LABEL-par2" --json "$OUT/identity_break.$LABEL.par2.json"
    say "ib_par_exit=$(awk -F'\t' '$1=="ib_par"{print $2}' "$OUT/status.tsv") (expected 0; the lanes assert partitioned == one-device)"
    run ib_par1 env MOJOLEARN_PAR_DEVICES=0 $IB --repeats 3 --lanes par-ordered,par-ordered-rmse,par-border-types,par-boosting,par-boosting-reg --vendor "$LABEL-par1" --json "$OUT/identity_break.$LABEL.par1.json"
    run diff_par $IB --diff "$OUT/identity_break.$LABEL.par1.json" "$OUT/identity_break.$LABEL.par2.json"
    say "diff_par_exit=$(awk -F'\t' '$1=="diff_par"{print $2}' "$OUT/status.tsv") (expected 0: one device IDENTICAL to two)"
else
    say "par=SKIPPED (one device; the par-* lanes need MOJOLEARN_PAR_DEVICES=0,1 on a two-GPU box)"
fi

# ARM sabotage: the two new branches' own defines, gbdt binding only
run build_gbdt_sab $BUILD_ENV env MOJOLEARN_EXTRA_DEFINES="-D MOJOLEARN_ORDERED_SABOTAGE=1 -D MOJOLEARN_BORDER_TYPES_SABOTAGE=1 -D MOJOLEARN_SAMPLE_QUANTILE_SABOTAGE=1" sh bindings/build_gbdt.sh
run ib_sab $IB --lanes "$NEW,$GUARDS" --vendor "$LABEL-sab" --json "$OUT/identity_break.$LABEL.sab.json"
run diff_sab $IB --diff "$OUT/identity_break.$LABEL.json" "$OUT/identity_break.$LABEL.sab.json"
say "diff_sab_exit=$(awk -F'\t' '$1=="diff_sab"{print $2}' "$OUT/status.tsv") (expected non-zero: DIVERGENT on the three new lanes only)"
grep -E "^summary|DIVERGENT" "$OUT/logs/diff_sab.log" 2>/dev/null | head -60 >> "$G"
run build_gbdt_clean_again $BUILD_ENV sh bindings/build_gbdt.sh

# ARM defaults-sabotage: the Python-side negative control of the defaults
# lane (the auto learning rate reads the CPU rows); its train cell must move
run ib_defsab env MOJOLEARN_CATBOOST_DEFAULTS_SABOTAGE=1 MOJOLEARN_HOST_ALLOW_SABOTAGE=1 $IB --lanes gbdt-catboost-defaults --vendor "$LABEL-defsab" --json "$OUT/identity_break.$LABEL.defsab.json"
run diff_defsab $IB --diff "$OUT/identity_break.$LABEL.json" "$OUT/identity_break.$LABEL.defsab.json"
say "diff_defsab_exit=$(awk -F'\t' '$1=="diff_defsab"{print $2}' "$OUT/status.tsv") (expected non-zero: gbdt-catboost-defaults train DIVERGENT)"

# host checks
run border_types pixi run check-border-types
say "check_border_types_exit=$(awk -F'\t' '$1=="border_types"{print $2}' "$OUT/status.tsv") (expected 0)"
run border_types_sab pixi run check-border-types-sabotage
say "check_border_types_sabotage_exit=$(awk -F'\t' '$1=="border_types_sab"{print $2}' "$OUT/status.tsv") (expected non-zero)"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
