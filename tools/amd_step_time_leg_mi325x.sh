#!/bin/sh
# tools/amd_step_time_leg5.sh -- lane/amd-step-time (2026-09-24): the
# MI325X confirmation on the T3 run's own GPU model (DigitalOcean
# gpu-mi325x1-256gb through tools/do_extra_leg.sh). Needs /root/urls and
# /root/amd_in pushed by the lane after the droplet is up.
#   1. base binding; byte LM: baseline (-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1
#      -D MOJOLEARN_GEMM_NO_LEAF_SPLIT=1, main's GEMM dispatch) and the branch
#      head; the branch's timers build
#   2. replays held to the H100 chain: baseline steps 101..102 from ckpt 100,
#      branch steps 101..103 from ckpt 100 and 1999..2000 from ckpt 1998
#   3. lean B4 step and the timed B4 itemization on the branch
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/amd-step-time
mkdir -p "$OUT/builds"
cd "$ROOT" || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="$ROOT/python:$ROOT"
BIN=/root/amd_bin; mkdir -p "$BIN"
S="sh tools/amd_step_time_session.sh"
ST="$OUT/session.txt"
say() { echo "$(date -u +%H:%M:%S) $*" >> "$ST"; }
say "leg5 started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; free -g; } > "$OUT/host.txt" 2>&1
rocm-smi --showproductname --showuniqueid --showclocks > "$OUT/gpu.txt" 2>&1
{ cat /opt/rocm/.info/version 2>/dev/null; ls -d /opt/rocm* 2>/dev/null; } > "$OUT/rocm.txt" 2>&1
pixi run python -c 'import numpy' > /dev/null 2>&1 || pixi run python -m pip install numpy > "$OUT/numpy.log" 2>&1

blm() {  # tag, defines
    mkdir -p "$BIN/out_$1"; rm -f "$BIN/out_$1/_mojolearn_byte_lm.so"
    t0=$(date +%s)
    MOJOLEARN_BYTE_LM_OUTDIR="$BIN/out_$1" MOJOLEARN_BUILD_EXTRA_DEFINES="$2" sh bindings/build_byte_lm.sh > "$OUT/builds/byte_lm.$1.log" 2>&1
    say "byte_lm $1 exit=$? secs=$(( $(date +%s) - t0 ))"
    cp "$BIN/out_$1/_mojolearn_byte_lm.so" "$BIN/byte_lm.$1.so"
}
t0=$(date +%s)
MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/builds/base.log" 2>&1
say "base exit=$? secs=$(( $(date +%s) - t0 ))"
( blm baseline "-D MOJOLEARN_GEMM_NO_LAUNCH_BOUND=1 -D MOJOLEARN_GEMM_NO_LEAF_SPLIT=1" ) &
( blm branch "" ) &
( blm timers "-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1" ) &
while [ ! -s /root/urls/tokens.json ] || [ ! -s /root/amd_in/A-1.chain.partial.jsonl ]; do sleep 10; done
$S fetch > "$OUT/fetch.out" 2>&1
say "fetch exit=$?"
wait

$S replay baseline ckpt_00000100.blm A-1.chain.partial.jsonl 2 > /dev/null 2>&1
$S replay branch ckpt_00000100.blm A-1.chain.partial.jsonl 3 > /dev/null 2>&1
$S replay branch ckpt_00001998.blm A-2.chain.jsonl 2 > /dev/null 2>&1
$S use branch > /dev/null
pixi run python tools/lm_step_memory_probe.py --out "$OUT/lean-branch" --shape 4 2048 768 12 12 64 2048 12 50257 \
    --steps 3 --resident-lean --witness-every-step --budget-seconds 600 > "$OUT/lean-branch.log" 2>&1
say "lean branch: $(grep -o '"steady_median_seconds": [0-9.]*' "$OUT/lean-branch/result.json")"
$S use timers > /dev/null
pixi run python tools/lm_step_memory_probe.py --out "$OUT/item-timers" --shape 4 2048 768 12 12 64 2048 12 50257 \
    --steps 1 --resident-lean --component-timing --component-timing-steps 2 --budget-seconds 600 > "$OUT/item-timers.log" 2>&1
grep -h '^timing ' "$OUT/item-timers.log" "$OUT/item-timers"/*.log 2>/dev/null > "$OUT/item-timers.timing.txt"
python3 tools/amd_step_timing_summary.py "$OUT/item-timers.timing.txt" --skip-shards 1 --tsv "$OUT/item-timers.summary.tsv" > /dev/null 2>&1
say "item timers: $(tail -1 "$OUT/item-timers.summary.tsv")"

touch /root/amd_step_ready
say "leg5 scripted part done; holding"
while [ ! -e /root/amd_step_done ]; do sleep 20; done
exit 0
