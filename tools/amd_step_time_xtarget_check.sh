#!/bin/bash
# tools/amd_step_time_xtarget_check.sh -- lane/amd-step-time (2026-09-24):
# the lane's one shared-source change that reaches non-AMD code is the GEMM
# kernels' launch-bound decorator (`GEMM_LAUNCH_BOUND`). This compiles, on a
# CPU box (tools/runpod_cpu_leg.sh --cmd-file), what the other columns compile:
#   1. the NVIDIA column: the GEMM A/B harness and the byte LM binding for
#      sm_90a (build only), and the PTX of the four decorated GEMM kernels
#      with the directive lines they carry;
#   2. the launch-bound test kernel for the Apple targets (compile only).
# $LEG_OUT receives the logs.
set -u
OUT=${LEG_OUT:-/tmp/xtarget}
mkdir -p "$OUT"
export MOJOLEARN_NUMERIC_MODE=identical
t0=$(date +%s)
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_NVIDIA --target-accelerator sm_90a -I . \
    bench/gemm_excp_ab_main.mojo -o /tmp/ab_nv > "$OUT/ab_sm90a.log" 2>&1
echo "ab sm_90a build exit=$? secs=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
t0=$(date +%s)
MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_BYTE_LM_OUTDIR=/tmp/blm_nv \
    bash -c 'mkdir -p /tmp/blm_nv && sh bindings/build_byte_lm.sh' > "$OUT/byte_lm_sm90a.log" 2>&1
echo "byte_lm sm_90a build exit=$? secs=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
for t in sm_90a apple_m4 metal; do
    pixi run mojo build tools/amd_codegen/lb_target_$t.mojo -o /tmp/lbt_$t > "$OUT/lb_target_$t.log" 2>&1 && /tmp/lbt_$t >> "$OUT/lb_target_$t.log" 2>&1
    echo "lb_target $t exit=$?: $(grep -iE 'maxntid|max_total|flat-work|error' "$OUT/lb_target_$t.log" | head -3 | tr '\n' ' ' | cut -c1-240)" | tee -a "$OUT/summary.txt"
done
