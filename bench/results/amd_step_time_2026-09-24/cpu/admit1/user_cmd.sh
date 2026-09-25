#!/bin/bash
# tools/amd_step_time2_cpu_check.sh -- lane/amd-step-time-2 (2026-09-25):
# compile checks and gfx942 assembly on a CPU box (tools/runpod_cpu_leg.sh
# --cmd-file), so no rented GPU minute is spent on a compile error:
#   1. the matrix-core GEMM kernel's asm (tools/amd_codegen/probe_mfma_gemm.mojo)
#      and its instruction census;
#   2. the GEMM A/B harness for gfx942 (the AMD column) and sm_90a (build only);
#   3. optional extra files named in $AMD2_EXTRA_MOJO (asm probes), built and run.
# $LEG_OUT receives the logs.
set -u
OUT=${LEG_OUT:-/tmp/amd2cpu}
mkdir -p "$OUT"
export MOJOLEARN_NUMERIC_MODE=identical
t0=$(date +%s)
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD -I . \
    tools/amd_codegen/probe_mfma_gemm.mojo -o /tmp/probe_mfma > "$OUT/probe_mfma.build.log" 2>&1 && /tmp/probe_mfma > "$OUT/mfma.s" 2>&1
echo "probe_mfma exit=$? secs=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
python3 tools/amd_codegen/mfma_census.py "$OUT/mfma.s" > "$OUT/census.txt" 2>&1; cat "$OUT/census.txt" >> "$OUT/summary.txt"
gzip -9 -f "$OUT/mfma.s"
for col in "AMD gfx942" "NVIDIA sm_90a"; do
    set -- $col
    t0=$(date +%s)
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_$1 --target-accelerator $2 -I . \
        bench/gemm_excp_ab_main.mojo -o /tmp/ab_$2 > "$OUT/ab_$2.log" 2>&1
    echo "ab $2 build exit=$? secs=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
done
for f in ${AMD2_EXTRA_MOJO:-}; do
    b=$(basename "$f" .mojo)
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD -I . "$f" -o /tmp/x_$b > "$OUT/$b.build.log" 2>&1 && /tmp/x_$b > "$OUT/$b.out" 2>&1
    echo "extra $b exit=$?" | tee -a "$OUT/summary.txt"
done
