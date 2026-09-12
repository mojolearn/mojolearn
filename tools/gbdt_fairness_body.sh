#!/bin/sh
# lane gbdt-fairness: every cell that attacks the 0.437x claim, RUN ON THE POD
# from /root/mojolearn, serialized on one GPU with an nvidia-smi sample beside
# each timed cell.
#
#   nohup sh tools/gbdt_fairness_body.sh > /root/fair_out/body_console.log 2>&1 &
#
# SERIALIZED IS NOT A STYLE CHOICE. One H100 runs every arm here, so two timed
# cells that overlap would each measure the other. Each cell also writes an
# `.smi` sample at 5 Hz, which is what answers "is CatBoost actually on the
# GPU" without taking its word for it.
set -u
R=/root/mojolearn
OUT=/root/fair_out
C="$OUT/cells"
mkdir -p "$C" "$OUT/logs"
cd "$R" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONPATH="$R/python"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda
export MOJOLEARN_SPEED_SIZE=shipped
export MOJOLEARN_SPEED_ROUNDS=5
export MOJOLEARN_SPEED_BUDGET_S=1800
export MOJOLEARN_SPEED_DEADLINE_S=3600
export GBM_BENCH_DATA=/root/datasets/gbm-bench
PY=python3
mark() { echo "$* $(date -u +%T)" | tee -a "$OUT/progress.txt"; }

# THE FAST TIER IS BUILT BEFORE ANY TIMED CELL, never between them: a mojo
# build takes every core, and a timed cell running beside it would be measuring
# the compiler. (It exists for the last question: if IDENTICAL costs us nothing
# against our own FAST tier, that is itself suspicious and belongs in the
# report rather than being left out of it.)
mark fast_build_start
MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 \
    timeout -k 30 1500 bash bindings/build_gbdt.sh > "$OUT/logs/build_gbdt_fast.log" 2>&1
mark "fast_build=$?"

smi_start() {
    ( while :; do
        printf 't=%s gpu=%s apps=%s\n' "$(date -u +%T)" \
          "$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader | tr '\n' ' ')" \
          "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
        sleep 0.2
      done ) > "$1" 2>&1 &
    echo $!
}

cell() {
    _n="$1"; shift
    if [ -f "$C/$_n.log" ]; then mark "$_n=SKIPPED_ALREADY_RUN"; return 0; fi
    _s=$(smi_start "$C/$_n.smi")
    mark "$_n start"
    timeout -k 30 2400 "$@" > "$C/$_n.log" 2>&1
    _rc=$?
    kill "$_s" 2>/dev/null
    mark "$_n=$_rc"
}

# 1. REPRODUCE THE CLAIM first, on this box and this build, exactly as the
# harness runs it: our IDENTICAL arm against CatBoost GPU, 5 interleaved
# rounds. Nothing below is worth reading unless this lands near 310 / 709.
cell control_taxi $PY -u bench/speed/forest_speed_arm.py \
    --lane gbdt-symmetric --dataset taxi --rows 1000000 --devices gpu --arms catboost-gpu

# 2. THE LEADING HYPOTHESIS: a third arm that is ours plus a REAL device
# drain, interleaved with the other two in one process and one heat window.
cell race_taxi $PY -u tools/gbdt_fairness_probe.py race --dataset taxi --rounds 5
cell race_istella $PY -u tools/gbdt_fairness_probe.py race --dataset istella --rounds 5

# 3. ARE THE TWO MODELS THE SAME SIZE? The check the harness has never made.
cell models_taxi $PY -u tools/gbdt_fairness_probe.py models --dataset taxi
cell models_istella $PY -u tools/gbdt_fairness_probe.py models --dataset istella

# 4. Startup against boosting, and whether a pinned flag is a slow path for
# CatBoost that a CatBoost user would never take.
cell decompose_taxi $PY -u tools/gbdt_fairness_probe.py decompose --dataset taxi --reps 3 --trees 1,10,100
cell flags_taxi $PY -u tools/gbdt_fairness_probe.py flags --dataset taxi --reps 3
cell b2b_taxi $PY -u tools/gbdt_fairness_probe.py b2b --dataset taxi --reps 5

# 5. IDENTICAL against our own FAST tier, ours only, interleaved in one
# process by the harness's own --ours-ab arm.
cell fastab_taxi $PY -u bench/speed/forest_speed_arm.py \
    --lane gbdt-symmetric --dataset taxi --rows 1000000 --ours-only --ours-ab "numeric_mode='fast'"

# 6. The other taxi cell the claim names (depthwise, 0.518x), so the verdict
# is not read off one growth policy.
cell race_taxi_depthwise $PY -u tools/gbdt_fairness_probe.py \
    race --dataset taxi --lane gbdt-depthwise --rounds 5

mark body_done
: > "$OUT/body.done"
