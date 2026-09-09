#!/bin/sh
# tools/gemm_identical_leg_extra.sh -- the GEMM lane's timing and probe work,
# run ON THE POD by tools/gemm_remote_leg.sh's gemm payload when it is named
# in MOJOLEARN_GEMM_LEG_EXTRA. It runs after the payload's own device check
# and card (both IDENTICAL), from /root/mojolearn, with pixi on PATH.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_identical_leg_extra.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3" \
#       --local-card bench/results/e1/2026-08-28_131651-runpod-nvidia/lanes/gemm.identical.card
#
# Everything here is OUR IDENTICAL arm only (-D MOJOLEARN_NUMERIC_IDENTICAL=1
# through tools/with_identical_mode.sh). No FAST or DETERMINISTIC arm is
# built, timed or mentioned, and no opponent runs: the cuBLAS rows come from
# bench/OPPONENT_REFERENCE.md. What it produces, under
# <leg out>/remote/identical/:
#
#   status.txt              one line per build and run: exit code, seconds
#   probe.log               gemm_tuned_probe: dispatcher vs untuned plan,
#                           bits and time, every shape (must end 0 MOVED)
#   probe.plan<N>.log       the same probe with MOJOLEARN_GEMM_PLAN=N forced
#                           on the second arm, N over every plan id: the
#                           all-plans time table (refusals are shapes the
#                           plan does not admit)
#   speed_v1.log            gemm_speed_main, arm ours-v1-identical, every row
#   speed_core.log          gemm_speed_main, arm ours-core-identical
#                           (the estimators' route), lm_head.t512 capped out
#   gemm_identity.log       core/gemm_identity_check.mojo under IDENTICAL
#   gpu_before.txt / gpu_after.txt   clocks and temperature around the work
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/identical
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
ROUNDS="${MOJOLEARN_SPEED_ROUNDS:-5}"
PLANS="${MOJOLEARN_GEMM_PROBE_PLANS:-0 1 2 3 4 5 6 7 8 9 10}"
PATH="$HOME/.pixi/bin:$PATH"
export PATH

smi() {
    nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu \
        --format=csv,noheader > "$1" 2>&1 || echo "no nvidia-smi" > "$1"
}
bld() {   # <name> <source.mojo> [mojo build args...]
    _n=$1; _src=$2; shift 2
    _t0=$(date +%s)
    tools/with_identical_mode.sh pixi run mojo build -I . "$@" -o "/root/bin_$_n" "$_src" \
        > "$OUT/build.$_n.log" 2>&1
    echo "build $_n exit=$? secs=$(( $(date +%s) - _t0 ))" >> "$ST"
}
run() {   # <log name> <binary> [VAR=value ...]
    _n=$1; _b=$2; shift 2
    _t0=$(date +%s)
    env "$@" "$_b" > "$OUT/$_n.log" 2>&1
    echo "run $_n exit=$? secs=$(( $(date +%s) - _t0 ))" >> "$ST"
}

echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) rounds=$ROUNDS plans=$PLANS" > "$ST"
smi "$OUT/gpu_before.txt"

bld probe gemm/checks/gemm_tuned_probe.mojo
bld speed bench/speed/gemm_speed_main.mojo
bld gemm_identity core/gemm_identity_check.mojo

run probe /root/bin_probe
for _p in $PLANS; do
    run "probe.plan$_p" /root/bin_probe MOJOLEARN_GEMM_PLAN="$_p"
done
run speed_v1 /root/bin_speed MOJOLEARN_SPEED_GEMM_ARMS=v1 MOJOLEARN_SPEED_ROUNDS="$ROUNDS"
run speed_core /root/bin_speed MOJOLEARN_SPEED_GEMM_ARMS=core MOJOLEARN_SPEED_ROUNDS="$ROUNDS" \
    MOJOLEARN_SPEED_MAX_GMACS=40
run gemm_identity /root/bin_gemm_identity

smi "$OUT/gpu_after.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
exit 0
