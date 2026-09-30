#!/bin/sh
# tools/amd_step_time_leg8.sh -- lane/amd-step-time (2026-09-24): where the
# time goes AFTER the matrix-core GEMM, before the next kernel is touched:
#   1. the 16x16x1 MFMA layout and the MODE facts again (amd_mfma_probe2)
#   2. rocprofv3 kernel trace + stats of one lean B4 step on the branch head
#   3. counters on the matrix-core GEMM for three calls (VALU, MFMA busy,
#      LDS waits) through bench/gemm_excp_ab_main.mojo
# then hold for the lane's session until /root/amd_step_done.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds" "$OUT/prof" "$OUT/ab"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
BIN=/root/amd_bin; mkdir -p "$BIN"
S="sh tools/amd_step_time_session.sh"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg8 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
rocm-smi --showproductname --showuniqueid > "$OUT/gpu.txt" 2>&1
( apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libdw1 elfutils ) > "$OUT/apt.log" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator gfx942 -I . \
    gemm/checks/amd_mfma_probe2.mojo -o "$BIN/mfma_probe2" > "$OUT/mfma2_build.log" 2>&1 && "$BIN/mfma_probe2" > "$OUT/mfma_probe2.log" 2>&1
say "probe2 exit=$?"
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "base exit=$?"
( $S bind branch > "$OUT/bind.out" 2>&1 ) &
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator gfx942 -I . \
    bench/gemm_excp_ab_main.mojo -o "$BIN/ab_branch" > "$OUT/ab/branch.build.log" 2>&1
for call in proj_fwd gateup_dB head_fwd; do
    for set in "SQ_WAVES SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_MFMA" "SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_LDS SQ_WAIT_INST_LDS SQ_WAVE_CYCLES"; do
        tag=$(echo "$set" | cut -d' ' -f1)
        MOJOLEARN_EXCP_AB_CALLS=$call MOJOLEARN_EXCP_AB_KINDS=ordinary MOJOLEARN_EXCP_AB_ROUNDS=1 timeout 300 \
            rocprofv3 --pmc $set -d "$OUT/prof/pmc-$call-$tag" -o pmc -- "$BIN/ab_branch" > "$OUT/prof/pmc-$call-$tag.log" 2>&1
        say "pmc $call $tag exit=$?"
    done
done
wait
$S use branch > /dev/null
timeout 600 rocprofv3 --kernel-trace --stats -d "$OUT/prof/trace" -o lean -- /root/mojolearn/.pixi/envs/default/bin/python \
    tools/lm_step_memory_probe.py --out "$OUT/lean-traced" --shape 4 2048 768 12 12 64 2048 12 50257 --steps 2 \
    --resident-lean --budget-seconds 500 > "$OUT/prof/trace.log" 2>&1
say "kernel trace exit=$?"
find "$OUT/prof/trace" -name '*stats*' -exec cp {} "$OUT/prof/" \; 2>/dev/null
find "$OUT/prof/trace" -name '*kernel_trace*.csv' -exec gzip -9 {} \; 2>/dev/null
touch /root/amd_step_ready
say "leg8 holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
