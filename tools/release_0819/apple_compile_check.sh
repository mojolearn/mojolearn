#!/bin/bash
# tools/release_0819/apple_compile_check.sh -- lane/release-0819 (2026-09-25):
# on a RunPod CPU pod (tools/runpod_cpu_leg.sh --cmd-file), compile the merged
# source's changed kernels for the Apple Metal target (LLVM/AIR for apple-m4,
# compile only; no Metal library, no GPU), and try the GEMM A/B harness for
# the Apple column with --target-accelerator apple-m4 (build only).
# $LEG_OUT receives the logs.
set -u
OUT=${LEG_OUT:-/tmp/apple819}
mkdir -p "$OUT"
export MOJOLEARN_NUMERIC_MODE=identical
t0=$(date +%s)
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_APPLE -I . \
    tools/release_0819/apple_compile_probe.mojo -o /tmp/apple_probe > "$OUT/probe.build.log" 2>&1 && /tmp/apple_probe > "$OUT/probe.out" 2>&1
echo "apple probe exit=$? secs=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
grep -E '^###|max_work_group_size|target triple|APPLE_COMPILE_PROBE' "$OUT/probe.out" 2>/dev/null | cut -c1-300 >> "$OUT/summary.txt"
t0=$(date +%s)
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_APPLE --target-accelerator apple-m4 -I . \
    bench/gemm_excp_ab_main.mojo -o /tmp/ab_apple > "$OUT/ab_apple_m4.build.log" 2>&1
echo "ab apple-m4 build exit=$? secs=$(( $(date +%s) - t0 ))" | tee -a "$OUT/summary.txt"
tail -5 "$OUT/ab_apple_m4.build.log" >> "$OUT/summary.txt"
