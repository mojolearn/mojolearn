#!/bin/sh
# tools/feature_freq_identity_leg.sh -- DEVIATION 2710, the owed AMD (and
# NVIDIA) leg of the feature frequency divergence lane.
#
# WHAT IT SETTLES. identity_break's `gbdt-feature-freq` lane gave three
# answers on three vendors on 2026-09-13 while every other GBDT lane was
# IDENTICAL. The cause is the synchronized tensor driver constructing its
# workspace with a literal `acc_live=False`, which under NUMERIC_IDENTICAL
# left the fixed-point histogram accumulator at ONE cell, unzeroed, while
# the histogram kernels wrote `hist_cells` cells into it. On the M4 the
# out-of-bounds cells fell into private, zero page slack and the tree was
# the host oracle's argmax on all nine fixtures; on the other two vendors
# the neighbours were live buffers. So the Mac's column is the reference
# and this leg asks the other vendor two questions, on one box, one source:
#
#   ARM fixed  the shipped source (DEVIATION 2710): the six GBDT lanes,
#              expected to equal the Apple column CELL FOR CELL, including
#              gbdt-feature-freq (Apple train hashes 2026-09-13: base
#              7d9c56b51213cb42, ties 57ff9964d1af1d4a, hashed
#              9d97a55431b6f8e0, wide c6d7fccba2483372, denormal
#              e7f1da14d1a9794b, denormal_ftz e7f1da14d1a9794b, dupes
#              e8c61a9407503755, odd c481331145bfb998, negative
#              75028913a14bb938).
#   ARM dead   the same gbdt binding rebuilt with
#              -D MOJOLEARN_2710_TENSOR_ACC_DEAD=1, which restores the
#              literal `False`; expected to reproduce this vendor's own
#              2026-09-13 gbdt-feature-freq hashes (AMD: base
#              459e5ba5892267e5 ... negative 47777b8ece73b417) and to leave
#              the five other lanes untouched. THIS IS THE SWITCH FLIPPING:
#              a define that changes only the divergent lane on the box
#              where it diverged.
#   ARM fixed again  the shipped source rebuilt and rerun after the dead
#              arm, the allocator-order bracket.
#
# THE VERDICT IS COMPUTED AT HOME on the Mac:
#   PYTHONPATH=python python3 tools/identity_break.py --diff \
#       bench/results/identity_break/2026-09-13_46-lanes/apple-m4.json \
#       <leg out>/remote/feature_freq/identity_break.<label>.2710.json
# and the on-box gate.txt carries the two in-box diffs (fixed vs dead must
# be DIVERGENT on gbdt-feature-freq only; fixed vs fixed-again IDENTICAL).
#
# HOW IT RUNS. As the MOJOLEARN_GEMM_LEG_EXTRA body of tools/do_extra_leg.sh
# (DigitalOcean, AMD MI325X gfx942) or tools/gemm_remote_leg.sh (RunPod,
# NVIDIA H100 sm_90a). Both ship /root/mojolearn as an archive of the pinned
# commit, run `pixi install`, and copy this file to /root/gemm_leg_extra.sh:
#
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_DO_EXTRA_ENV="MOJOLEARN_IDENTITY_VENDOR_LABEL=amd-mi325x-gfx942" \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/feature_freq_identity_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-do-feature-freq-2710 \
#   bash tools/do_extra_leg.sh amd --minutes 60
#
# NOT RUN on 2026-09-13: the lane's resource cap forbade renting. POSIX sh
# only (dash on the images).
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/feature_freq
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
say "vendor=${MOJOLEARN_TARGET_COLUMN:-unset} archs=${MOJOLEARN_GPU_ARCHS:-unset}"
say "commit=$(cat /root/mojolearn/COMMIT 2>/dev/null || echo unknown)"
(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; rocminfo 2>/dev/null | grep -m1 -oE "gfx[0-9a-z]+") > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"

BUILD_ENV="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}"
LANES=gbdt-feature-freq,gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse,gbdt-ordered-rmse
LABEL=${MOJOLEARN_IDENTITY_VENDOR_LABEL:-${MOJOLEARN_TARGET_COLUMN:-box}-${MOJOLEARN_GPU_ARCHS:-arch}}
IB="env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes $LANES"

# every IDENTICAL binding once: identity_break imports the whole package
# The RunPod runner passes no environment to the body and bindings/build_byte_lm.sh
# refuses without an explicit architecture, which cost this leg its second NVIDIA
# attempt on 2026-09-13; resolve the architecture from the device when unset, and
# skip the byte LM bindings, which no GBDT lane reads.
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
say "gpu_archs_resolved=${MOJOLEARN_GPU_ARCHS:-unset} column=${MOJOLEARN_TARGET_COLUMN:-unset}"
built=0; failed=""
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh)
    # CPU-only host bindings take no MOJOLEARN_GPU_ARCHS and no GBDT lane reads them.
    case "$n" in build_byte_lm|build_byte_lm_host|build_forest_host) continue ;; esac
    if run "$n" $BUILD_ENV sh "$s"; then built=$((built + 1)); else failed="$failed $n"; fi
done
say "bindings_built=$built failed=${failed:-none}"
if [ -n "$failed" ]; then say "verdict=BUILD-FAILED"; exit 1; fi

# ARM fixed: the shipped source
say "gbdt_fixed_sha=$(sha256sum python/mojolearn/identical/_mojolearn_gbdt.so | cut -c1-16)"
run ib_fixed $IB --vendor "$LABEL-2710" --json "$OUT/identity_break.$LABEL.2710.json"
say "ib_fixed_exit=$(awk -F'\t' '$1=="ib_fixed"{print $2}' "$OUT/status.tsv")"

# ARM dead: the literal False restored by its define, gbdt binding only
run build_gbdt_dead $BUILD_ENV env MOJOLEARN_EXTRA_DEFINES="-D MOJOLEARN_2710_TENSOR_ACC_DEAD=1" sh bindings/build_gbdt.sh
say "gbdt_dead_sha=$(sha256sum python/mojolearn/identical/_mojolearn_gbdt.so | cut -c1-16)"
run ib_dead $IB --vendor "$LABEL-2710dead" --json "$OUT/identity_break.$LABEL.2710dead.json"
say "ib_dead_exit=$(awk -F'\t' '$1=="ib_dead"{print $2}' "$OUT/status.tsv")"

# ARM fixed again: the bracket
run build_gbdt_fixed_again $BUILD_ENV sh bindings/build_gbdt.sh
say "gbdt_fixed_again_sha=$(sha256sum python/mojolearn/identical/_mojolearn_gbdt.so | cut -c1-16)"
run ib_fixed_again $IB --vendor "$LABEL-2710b" --json "$OUT/identity_break.$LABEL.2710b.json"
say "ib_fixed_again_exit=$(awk -F'\t' '$1=="ib_fixed_again"{print $2}' "$OUT/status.tsv")"

# in-box diffs; the cross-vendor diff against the Apple column runs at home
DIFF="env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --diff"
run diff_fixed_vs_dead $DIFF "$OUT/identity_break.$LABEL.2710.json" "$OUT/identity_break.$LABEL.2710dead.json"
say "diff_fixed_vs_dead_exit=$(awk -F'\t' '$1=="diff_fixed_vs_dead"{print $2}' "$OUT/status.tsv") (expected non-zero: DIVERGENT on gbdt-feature-freq only)"
grep -E "^summary|DIVERGENT|MOVED" "$OUT/logs/diff_fixed_vs_dead.log" 2>/dev/null | head -40 >> "$G"
run diff_fixed_vs_fixed_again $DIFF "$OUT/identity_break.$LABEL.2710.json" "$OUT/identity_break.$LABEL.2710b.json"
say "diff_fixed_vs_fixed_again_exit=$(awk -F'\t' '$1=="diff_fixed_vs_fixed_again"{print $2}' "$OUT/status.tsv") (expected 0: IDENTICAL)"
grep -E "^summary|DIVERGENT|MOVED" "$OUT/logs/diff_fixed_vs_fixed_again.log" 2>/dev/null | head -40 >> "$G"
for a in fixed dead fixed_again; do
    grep -E "^\| gbdt-feature-freq " "$OUT/logs/ib_$a.log" 2>/dev/null | head -3 | sed "s/^/$a /" >> "$G"
done
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
