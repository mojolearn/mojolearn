#!/bin/sh
# tools/identity_three_columns_leg.sh: the on-box body of the three-vendor identity legs
# (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh, tools/do_extra_leg.sh and
# tools/hotaisle_leg.sh). Evidence of 2026-09-13: bench/results/identity_break/.

# Three-vendor identity at the shipped default (claims audit, 2026-09-13):
# build every IDENTICAL binding for this box's own architecture, then run
# tools/identity_break.py on EVERY lane with the training, inference and
# model columns, twice in one process, and bring the JSON home. The Apple
# column is the same script on the M4; the diff runs on the Mac.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/identity
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
say "commit=$(cat /root/mojolearn/COMMIT 2>/dev/null || git -C /root/mojolearn rev-parse HEAD 2>/dev/null || echo unknown)"
(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; rocminfo 2>/dev/null | grep -m1 -oE "gfx[0-9a-z]+") > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
# The RunPod runner passes no environment to the body, and bindings/build_byte_lm.sh
# refuses without an explicit architecture ("one explicit sm_NN or gfxNNN target
# required", which cost the byte-lm lane its nine cells on 2026-09-13). Derive it
# from the device when unset.
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
        case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;; 8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;; 12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    elif command -v rocminfo >/dev/null 2>&1; then
        MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    fi
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs_resolved=${MOJOLEARN_GPU_ARCHS:-unset}"
BUILD_ENV="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8}"
built=0; failed=""
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh)
    # build_byte_lm_host takes no MOJOLEARN_GPU_ARCHS (a CPU build); the two byte-lm-host lanes need it.
    if [ "$n" = build_byte_lm_host ]; then run "$n" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-8} sh "$s" && built=$((built + 1)) || failed="$failed $n"; continue; fi
    if run "$n" $BUILD_ENV sh "$s"; then built=$((built + 1)); else failed="$failed $n"; fi
done
say "bindings_built=$built failed=${failed:-none}"
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null
# The label names the box from its own devices, never from an env default
# that a runner may not pass (the RunPod runner passes none).
if [ -n "${MOJOLEARN_IDENTITY_VENDOR_LABEL:-}" ]; then VENDOR_LABEL=$MOJOLEARN_IDENTITY_VENDOR_LABEL
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR_LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-${MOJOLEARN_GPU_ARCHS:-sm}"
elif command -v rocminfo >/dev/null 2>&1; then
    VENDOR_LABEL="amd-$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')"
else VENDOR_LABEL="unknown-box"; fi
run identity_break env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --vendor "$VENDOR_LABEL" --json "$OUT/identity_break.$VENDOR_LABEL.json"
say "identity_break_exit=$(awk -F'\t' '$1=="identity_break"{print $2}' "$OUT/status.tsv")"
grep -E "^cells=|^summary|MOVED|DIVERGENT|REFUSED|RELOAD" "$OUT/logs/identity_break.log" 2>/dev/null | head -40 >> "$G"
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
