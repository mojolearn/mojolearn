#!/bin/sh
# tools/lm_shakedown_long_body.sh -- lane/lm-training-shakedown, leg 2.
#
# Leg 1 asked whether a long run survives and found that it does not survive
# WELL. At the 162,147,840-parameter target shape, batch 1, one resident
# session, 2,000 consecutive steps:
#
#   * device memory grew from 16.949 GB to 34.397 GB between roughly step 210
#     and step 476, then plateaued;
#   * step time kept climbing past that plateau, 0.207 s to over 0.44 s by
#     step 650, while GPU clocks sat at the 1980 MHz maximum, no throttle
#     reason was active, the card was at 42 C, host RSS was flat at 8.19 GB
#     and host CPU pressure `full` was exactly 0.
#
# Neither is visible in a 3-step or 4-step run, and every LM run on record in
# this repository is 3 or 4 steps.
#
# This leg does three things, in the order that spends the lease best:
#
#   1. THE RECYCLE ARM, first because it is the decisive one and, if it
#      works, the cheapest. Same run, but every --recycle-every steps the
#      resident session is torn down and rebuilt from export_state().
#      tools/lm_shakedown_resume.py proved on leg 1's box that this continues
#      bit for bit at this shape, with a missing-moments control that
#      separated, so the recycle is lossless and the only question is whether
#      it resets the clock and the memory.
#   2. THE STRAIGHT ARM, which is the REPLICATION of leg 1's observation. One
#      box seeing drift once is not a result.
#   3. The LOGICAL SHARDS probe: single-device gradient accumulation at the
#      target shape, the only path past the 999,999-step ceiling, and never
#      run above a toy shape.
#
# 10,000 CONSECUTIVE STEPS IS NOT ATTEMPTED HERE and that is itself a
# finding: at leg 1's observed degradation 10,000 straight steps do not fit
# inside a 60-minute lease. Whether they become feasible is what arm 1 asks.
#
# POSIX sh. Never `set -e`.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-shakedown-long
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"
export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_TARGET_COLUMN="${MOJOLEARN_TARGET_COLUMN:-nvidia}"
export PYTHONPATH="$ROOT/python:$ROOT"

STEPS="${MOJOLEARN_LM_LONG_STEPS:-2500}"
RECYCLE_EVERY="${MOJOLEARN_LM_RECYCLE_EVERY:-250}"
RECYCLE_BUDGET="${MOJOLEARN_LM_RECYCLE_BUDGET:-1000}"
STRAIGHT_BUDGET="${MOJOLEARN_LM_STRAIGHT_BUDGET:-1300}"
SHARDS="${MOJOLEARN_LM_SHARDS:-1 4 16 64}"
SHARD_STEPS="${MOJOLEARN_LM_SHARD_STEPS:-3}"
CORPUS=training/corpus/enwik8/input.txt

smi() {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,clocks.max.sm,temperature.gpu \
        --format=csv,noheader > "$1" 2>&1 || echo "no nvidia-smi" > "$1"
}

# The drift diagnostics leg 1 had to collect by hand over ssh. Sampled by the
# body this time, so they arrive with the evidence instead of beside it.
diag() {
    {
        echo "sample=$1 date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total --format=csv,noheader
        nvidia-smi -q -d PERFORMANCE 2>/dev/null | sed -n '/Clocks Event Reasons/,/Counters/p'
        echo "nproc=$(nproc) cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
        echo "uptime: $(uptime)"
        echo "PSI cpu: $(tr '\n' ' ' < /proc/pressure/cpu 2>/dev/null)"
        echo "PSI memory: $(tr '\n' ' ' < /proc/pressure/memory 2>/dev/null)"
        free -g | head -2
        echo
    } >> "$OUT/drift_diagnostics.txt" 2>&1
}

echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) steps=$STEPS recycle_every=$RECYCLE_EVERY shards='$SHARDS'" > "$ST"
smi "$OUT/gpu_before.txt"

if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    cap=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    case "$cap" in
        9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;;
        [0-9].[0-9]) MOJOLEARN_GPU_ARCHS="sm_$(echo "$cap" | tr -d .)" ;;
        *) echo "arch unknown (compute_cap='$cap'); set MOJOLEARN_GPU_ARCHS" >> "$ST"; exit 2 ;;
    esac
fi
export MOJOLEARN_GPU_ARCHS
echo "gpu_archs=$MOJOLEARN_GPU_ARCHS" >> "$ST"

if [ -f "$CORPUS" ]; then
    echo "corpus=$CORPUS bytes=$(wc -c < "$CORPUS")" >> "$ST"
else
    echo "corpus MISSING at $CORPUS; arms will run on synthetic tokens" >> "$ST"
    CORPUS=
fi

rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
_t0=$(date +%s)
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
_rc=$?
echo "build base exit=$_rc secs=$(( $(date +%s) - _t0 ))" >> "$ST"
[ "$_rc" -eq 0 ] || { echo "base build failed; nothing run" >> "$ST"; exit 1; }
_t0=$(date +%s)
sh bindings/build_byte_lm.sh > "$OUT/build_byte_lm.log" 2>&1
_rc=$?
echo "build byte_lm exit=$_rc secs=$(( $(date +%s) - _t0 ))" >> "$ST"
[ "$_rc" -eq 0 ] || { echo "byte_lm build failed; nothing run" >> "$ST"; exit 1; }

if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
    echo "numpy pip exit=$?" >> "$ST"
fi

corpus_args() { [ -n "$CORPUS" ] && echo "--corpus $CORPUS"; }

# 1. THE RECYCLE ARM.
echo "--- recycle arm: $STEPS steps, session rebuilt every $RECYCLE_EVERY ---" >> "$ST"
diag "before-recycle"
_t0=$(date +%s)
# shellcheck disable=SC2046
pixi run python tools/lm_recycle_probe.py --out "$OUT/recycle" --target \
    --steps "$STEPS" --recycle-every "$RECYCLE_EVERY" \
    --budget-seconds "$RECYCLE_BUDGET" $(corpus_args) > "$OUT/recycle.log" 2>&1
_rc=$?
echo "recycle exit=$_rc secs=$(( $(date +%s) - _t0 )) (2 = budget limitation recorded)" >> "$ST"
diag "after-recycle"

# 2. THE STRAIGHT ARM: the replication of leg 1.
echo "--- straight arm: $STEPS steps, one session (replicates leg 1) ---" >> "$ST"
_t0=$(date +%s)
# shellcheck disable=SC2046
pixi run python tools/lm_recycle_probe.py --out "$OUT/straight" --target \
    --steps "$STEPS" --recycle-every 0 \
    --budget-seconds "$STRAIGHT_BUDGET" $(corpus_args) > "$OUT/straight.log" 2>&1
_rc=$?
echo "straight exit=$_rc secs=$(( $(date +%s) - _t0 )) (2 = budget limitation recorded)" >> "$ST"
diag "after-straight"

# 3. LOGICAL SHARDS: the path past the 999,999 step ceiling.
echo "--- logical shards at the target shape: K = $SHARDS ---" >> "$ST"
_t0=$(date +%s)
# shellcheck disable=SC2046
pixi run python tools/lm_shards_probe.py --out "$OUT/shards" --target \
    --shards $SHARDS --steps "$SHARD_STEPS" $(corpus_args) \
    > "$OUT/shards.log" 2>&1
_rc=$?
echo "shards exit=$_rc secs=$(( $(date +%s) - _t0 ))" >> "$ST"
diag "after-shards"

smi "$OUT/gpu_after.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
exit 0
