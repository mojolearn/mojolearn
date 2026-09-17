#!/bin/sh
# tools/lowbit_mma_leg.sh -- DEVIATION 2910, the int8 matrix-unit lane's
# on-box work: the three low-bit gate tasks plus the forced-flat run, on a
# box that HAS an integer matrix unit. VENDOR-AGNOSTIC: runs ON THE BOX as
# the MOJOLEARN_GEMM_LEG_EXTRA body of tools/gemm_remote_leg.sh (RunPod,
# NVIDIA) or of tools/do_extra_leg.sh (DigitalOcean, AMD), from
# /root/mojolearn with pixi on PATH; everything it writes under
# /root/gemm_leg_out/lowbit-mma/ comes home with the leg's fetch to
# MOJOLEARN_GEMM_LEG_OUT, which the two invocations below point OUTSIDE the
# checkout ($HOME/mojolearn-evidence, tools/gemm_remote_leg.sh's own
# default root), so no evidence lands in the tree.
#
# NVIDIA, an H100 on RunPod (the body derives sm_90a from nvidia-smi):
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/lowbit_mma_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-h100-lowbit-mma \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# AMD, an MI325X on DigitalOcean (gfx942; the runner must name the arch):
#
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/lowbit_mma_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi325x-lowbit-mma \
#   bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
#
# Without --rent (RunPod) or with --dry-run (DigitalOcean) the runners rent
# nothing; this file has NOT been run anywhere as of 2026-09-17 (RUN OWED,
# both boxes). It has been checked with `sh -n` only.
#
# WHAT IT RUNS, each with its own exit code and EXPECTED verdict in
# status.tsv; a later phase runs even when an earlier one fails, because a
# red phase is a finding:
#
#   lowbit            pixi run check-gemm-lowbit
#                     The dispatcher: on this box `identical_gemm_int8_into`
#                     takes the MMA plan (lib_int8_matrix_unit_for is True),
#                     so check_int8_device_matches_oracle is the unit against
#                     the host oracle, and check_int8_mma_matches_flat runs
#                     both plans on every shape. EXPECTED exit 0.
#   lowbit-force-flat pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1
#                       -D MOJOLEARN_INT8_FORCE_FLAT=1 -I .
#                       gemm/checks/gemm_lowbit_check.mojo
#                     The same file with the dispatcher pinned to the flat
#                     plan; the direct mma-vs-flat gate still runs.
#                     EXPECTED exit 0.
#   lowbit-sabotage   pixi run check-gemm-lowbit-sabotage
#                     The device value arm reaches BOTH int8 plans and the
#                     bf16 fused plan. EXPECTED non-zero, and the log must
#                     name check_int8_device_matches_oracle and
#                     check_int8_mma_matches_flat among the failed gates.
#   lowbit-host-sabotage  pixi run check-gemm-lowbit-host-sabotage
#                     The host value arm. EXPECTED non-zero, with the two
#                     oracle gates and the mma gate's oracle comparison
#                     failed and check_bf16_plans_agree passing.
#
# The verdict line at the end of gate.txt reads GREEN only when every
# EXPECTED outcome held; any other combination reads RED with the phase
# named. A RED here is a finding about the fragment layouts in
# gemm/checks/gemm_int8_mma.mojo (unverified until this runs) or about the
# intrinsic names, and the logs say which.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lowbit-mma
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"

# THE VENDOR. tools/do_extra_leg.sh exports MOJOLEARN_TARGET_COLUMN (amd or
# nvidia); tools/gemm_remote_leg.sh exports nothing, so the box is read: a
# working nvidia-smi is NVIDIA, /dev/kfd or an AMD tool is AMD (/dev/dri alone
# is not AMD evidence). Recorded only; the gate file reads the column from
# the accelerator the compiler detects.
case "${MOJOLEARN_TARGET_COLUMN:-}" in
    amd|nvidia) VENDOR=$MOJOLEARN_TARGET_COLUMN ;;
    *) if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
           VENDOR=nvidia
       elif [ -e /dev/kfd ] || command -v rocm-smi > /dev/null 2>&1 || command -v amd-smi > /dev/null 2>&1; then
           VENDOR=amd
       else
           VENDOR=unknown
       fi ;;
esac
if [ "$VENDOR" = unknown ]; then
    echo "vendor=unknown: no working nvidia-smi, no /dev/kfd, no rocm-smi or amd-smi; nothing run" > "$OUT/gate.txt"
    exit 9
fi

gpu_snapshot() {  # <file>
    if [ "$VENDOR" = nvidia ]; then
        nvidia-smi --query-gpu=name,driver_version,uuid,clocks.sm,temperature.gpu --format=csv > "$1" 2>&1
    else
        { rocm-smi --showproductname --showdriverversion --showuse --showmemuse --showtemp 2>&1 \
            || echo "rocm-smi did not answer"; } > "$1"
    fi
}

if [ "$VENDOR" = nvidia ]; then
    # MAX's bundled CUDA 13 assembler needs driver 580. Older-driver pods use
    # their installed assembler at BOTH build and runtime
    # (tools/attention_step_leg.sh carries the same lines).
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in
        ''|*[!0-9]*) ;;
        *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
               export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}
           fi ;;
    esac
fi

{
    echo "deviation=2910"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT"
    echo "vendor=$VENDOR gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset} column=${MOJOLEARN_TARGET_COLUMN:-unset}"
    [ -f /root/gemm_leg_out/leg.txt ] && grep -E '^(commit|vendor|provider|size)=' /root/gemm_leg_out/leg.txt
} > "$OUT/gate.txt"
gpu_snapshot "$OUT/gpu_before.txt"

red=0
run() {
    # run <name> <expected: pass|fail> <command...>: log to $OUT/<name>.log,
    # record the exit code and whether the EXPECTED outcome held.
    _name=$1
    _want=$2
    shift 2
    _t0=$(date +%s)
    "$@" > "$OUT/$_name.log" 2>&1
    _code=$?
    _held=no
    if [ "$_want" = pass ] && [ "$_code" -eq 0 ]; then _held=yes; fi
    if [ "$_want" = fail ] && [ "$_code" -ne 0 ]; then _held=yes; fi
    printf '%s\t%s\t%s\texpected=%s\theld=%s\n' "$_name" "$_code" "$(( $(date +%s) - _t0 ))s" "$_want" "$_held" >> "$OUT/status.tsv"
    [ "$_held" = yes ] || red=1
    return "$_code"
}
must_name() {
    # must_name <log name> <gate>: the sabotage log must list the gate as
    # FAILED; a sabotage that does not reach a gate is a finding.
    if ! grep -q "GATE FAILED: $2" "$OUT/$1.log" 2>/dev/null; then
        echo "reach: $1 did not fail $2" >> "$OUT/gate.txt"
        red=1
    fi
}

run lowbit pass pixi run check-gemm-lowbit
run lowbit-force-flat pass pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_INT8_FORCE_FLAT=1 -I . gemm/checks/gemm_lowbit_check.mojo
run lowbit-sabotage fail pixi run check-gemm-lowbit-sabotage
must_name lowbit-sabotage check_int8_device_matches_oracle
must_name lowbit-sabotage check_int8_mma_matches_flat
must_name lowbit-sabotage check_bf16_device_matches_oracle
run lowbit-host-sabotage fail pixi run check-gemm-lowbit-host-sabotage
must_name lowbit-host-sabotage check_int8_device_matches_oracle
must_name lowbit-host-sabotage check_int8_mma_matches_flat
must_name lowbit-host-sabotage check_bf16_device_matches_oracle

# The plan the dispatcher ran, from the gate banner, so the record says
# which plan the oracle comparison covered.
grep -h "int8 dispatch:" "$OUT/lowbit.log" "$OUT/lowbit-force-flat.log" 2>/dev/null >> "$OUT/gate.txt"

gpu_snapshot "$OUT/gpu_after.txt"
{
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [ "$red" -eq 0 ]; then
        echo "verdict=GREEN every expected outcome held (see status.tsv)"
    else
        echo "verdict=RED an expected outcome did not hold (see status.tsv and the reach lines above)"
    fi
} >> "$OUT/gate.txt"
exit "$red"
