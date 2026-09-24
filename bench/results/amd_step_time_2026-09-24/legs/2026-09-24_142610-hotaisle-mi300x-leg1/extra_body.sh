#!/bin/sh
# tools/amd_step_time_leg1.sh -- lane/amd-step-time (2026-09-24), the first
# AMD leg, SCRIPTED (a Hot Aisle leg is 60 minutes at most and runs in a
# container). In priority order, each item logged to session.txt, a failure
# is a result and the next item still runs:
#   1. the box facts; the base binding
#   2. gemm/checks/amd_excp_probe.mojo: the device facts the EXCP seam rests on
#   3. gfx942 asm of the step GEMM kernels, shipped seam and EXCP seam
#   4. the T3-shape GEMM A/B (bench/gemm_excp_ab_main.mojo): shipped seam
#      first, then the EXCP seam; hashes must match line for line
#   5. the B4 itemization: timers binding on the shipped seam, then on the
#      EXCP seam, tools/lm_step_memory_probe.py at the T3 shard shape
# Then it holds (for a lane ssh session through `docker exec`) until
# /root/amd_step_done or the leg's bound.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
S="sh tools/amd_step_time_session.sh"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg1 started=$(date -u +%Y-%m-%dT%H:%M:%SZ) archs=$MOJOLEARN_GPU_ARCHS"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
rocm-smi --showproductname --showuniqueid --showclocks > "$OUT/gpu.txt" 2>&1
rocminfo 2>/dev/null | grep -E 'Marketing Name|Name: +gfx|Compute Unit|Max Clock' > "$OUT/rocminfo.txt" 2>&1
{ cat /opt/rocm/.info/version 2>/dev/null; ls -d /opt/rocm* 2>/dev/null; command -v rocprofv3; command -v rocprof; } > "$OUT/rocm.txt" 2>&1
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1

t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
say "build base exit=$? secs=$(( $(date +%s) - t0 ))"
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

BIN=/root/amd_bin; mkdir -p "$BIN"
t0=$(date +%s)
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
    -I . gemm/checks/amd_excp_probe.mojo -o "$BIN/amd_excp_probe" > "$OUT/probe_build.log" 2>&1; rc=$?
say "probe build exit=$rc secs=$(( $(date +%s) - t0 ))"
[ "$rc" -eq 0 ] && { "$BIN/amd_excp_probe" > "$OUT/probe.log" 2>&1; say "probe exit=$?: $(grep EXCP_CHAIN "$OUT/probe.log" | tr '\n' ' ' | cut -c1-400)"; }

$S asm shipped MOJOLEARN_GEMM_NO_EXCP_FAST=1 > "$OUT/asm_shipped.out" 2>&1
$S asm excp > "$OUT/asm_excp.out" 2>&1
$S ab shipped MOJOLEARN_GEMM_NO_EXCP_FAST=1 > "$OUT/ab_shipped.out" 2>&1
$S ab excp > "$OUT/ab_excp.out" 2>&1
if [ -s "$OUT/ab/shipped.log" ] && [ -s "$OUT/ab/excp.log" ]; then
    grep '^EXCP_AB call' "$OUT/ab/shipped.log" | sed 's/ ms=.*//' > "$OUT/ab/shipped.hashes"
    grep '^EXCP_AB call' "$OUT/ab/excp.log" | sed 's/ ms=.*//' > "$OUT/ab/excp.hashes"
    if cmp -s "$OUT/ab/shipped.hashes" "$OUT/ab/excp.hashes"; then say "AB HASHES IDENTICAL ($(wc -l < "$OUT/ab/excp.hashes") lines)"; else say "AB HASHES DIFFER"; diff "$OUT/ab/shipped.hashes" "$OUT/ab/excp.hashes" > "$OUT/ab/hash.diff"; fi
fi

# ---- the B4 itemization (random-init weights, T3 shard shape) ----
item() {  # $1 tag, rest: defines
    tag=$1; shift
    $S bind "$tag" MOJOLEARN_STEP_PHASE_TIMERS=1 MOJOLEARN_ATTN_PHASE_TIMERS=1 "$@" >> "$OUT/bind.out" 2>&1
    [ -s "$BIN/byte_lm.$tag.so" ] || return 1
    $S use "$tag" >> "$OUT/bind.out" 2>&1
    t0=$(date +%s)
    PYTHONPATH="$ROOT/python:$ROOT" pixi run python tools/lm_step_memory_probe.py --out "$OUT/item-$tag" \
        --shape 4 2048 768 12 12 64 2048 12 50257 --steps 1 --resident-lean \
        --component-timing --component-timing-steps 2 --budget-seconds 900 > "$OUT/item-$tag.log" 2>&1
    say "itemization $tag exit=$? secs=$(( $(date +%s) - t0 ))"
    grep -h '^timing ' "$OUT/item-$tag.log" "$OUT/item-$tag"/*.log 2>/dev/null > "$OUT/item-$tag.timing.txt"
    python3 tools/amd_step_timing_summary.py "$OUT/item-$tag.timing.txt" --skip-shards 1 --tsv "$OUT/item-$tag.summary.tsv" > /dev/null 2>&1
    say "itemization $tag: $(tail -1 "$OUT/item-$tag.summary.tsv" 2>/dev/null)"
}
item timers-shipped MOJOLEARN_GEMM_NO_EXCP_FAST=1
item timers-excp

touch /root/amd_step_ready
say "leg1 scripted part done; holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
