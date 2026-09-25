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
# the attention dq kernels' asm
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD -I . \
    tools/amd_codegen/probe_attn_dq.mojo -o /tmp/probe_dq > "$OUT/probe_dq.build.log" 2>&1 && /tmp/probe_dq > "$OUT/dq.s" 2>&1
echo "probe_dq exit=$?" | tee -a "$OUT/summary.txt"
python3 tools/amd_codegen/mfma_census.py "$OUT/dq.s" >> "$OUT/summary.txt" 2>&1
gzip -9 -f "$OUT/dq.s"
# device programs: build only (no GPU here)
for f in gemm/checks/amd_mfma_probe3.mojo tools/amd_codegen/stall_probe.mojo; do
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator gfx942 -I . "$f" -o /tmp/dev_$(basename "$f" .mojo) > "$OUT/$(basename "$f" .mojo).build.log" 2>&1
    echo "build $f exit=$?" | tee -a "$OUT/summary.txt"
done
# the byte LM binding for both GPU columns (build only)
for col in "amd gfx942" "nvidia sm_90a"; do
    set -- $col
    t0=$(date +%s)
    mkdir -p /tmp/blm_$1
    MOJOLEARN_GPU_ARCHS=$2 MOJOLEARN_TARGET_COLUMN=$1 MOJOLEARN_BYTE_LM_OUTDIR=/tmp/blm_$1 sh bindings/build_byte_lm.sh > "$OUT/byte_lm_$1.log" 2>&1
    echo "byte_lm $1 build exit=$? secs=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
done
for f in ${AMD2_EXTRA_MOJO:-}; do
    b=$(basename "$f" .mojo)
    pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD -I . "$f" -o /tmp/x_$b > "$OUT/$b.build.log" 2>&1 && /tmp/x_$b > "$OUT/$b.out" 2>&1
    echo "extra $b exit=$?" | tee -a "$OUT/summary.txt"
done
