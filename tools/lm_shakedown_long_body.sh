#!/bin/sh
# tools/lm_shakedown_long_body.sh -- lane/lm-training-shakedown, leg 2.
#
# Leg 1 (tools/lm_shakedown_body.sh) answered the batch sweep and resume.
# This leg answers the one question a short leg cannot: does a long run
# survive? Nothing in this repository has ever run more than 128 consecutive
# optimizer steps at any size, and at the 162,147,840-parameter target shape
# the record is FOUR.
#
#   1. build the base and byte-LM bindings under IDENTICAL for this arch;
#   2. THE LONG RUN: MOJOLEARN_LM_LONG_STEPS consecutive steps at the target
#      shape, resident + lean, enwik8, NO per-step witnesses (an export costs
#      about 16 s of per-element Python at this size and would dominate).
#      events.jsonl carries per-step wall seconds, loss, host RSS and the
#      polled device window, which is what memory growth and time drift are
#      read from;
#   3. the LOGICAL SHARDS probe: single-device gradient accumulation at the
#      target shape, K = 1, 4, 16, 64. This is the only path past the 999,999
#      step ceiling and it has never run above a toy shape. It goes last so a
#      lease that runs short costs the bonus arm, not the long run.
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

LONG_STEPS="${MOJOLEARN_LM_LONG_STEPS:-10000}"
LONG_BUDGET="${MOJOLEARN_LM_LONG_BUDGET:-2700}"
SHARDS="${MOJOLEARN_LM_SHARDS:-1 4 16 64}"
SHARD_STEPS="${MOJOLEARN_LM_SHARD_STEPS:-3}"
CORPUS=training/corpus/enwik8/input.txt

smi() {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,clocks.max.sm,temperature.gpu \
        --format=csv,noheader > "$1" 2>&1 || echo "no nvidia-smi" > "$1"
}

echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) long_steps=$LONG_STEPS shards='$SHARDS'" > "$ST"
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

# 2. THE LONG RUN.
echo "--- long run: $LONG_STEPS consecutive steps, target shape, batch 1, budget $LONG_BUDGET s ---" >> "$ST"
_t0=$(date +%s)
# shellcheck disable=SC2046
pixi run python tools/lm_step_memory_probe.py --out "$OUT/long" --target \
    --steps "$LONG_STEPS" --budget-seconds "$LONG_BUDGET" \
    --resident-lean $(corpus_args) > "$OUT/long.log" 2>&1
_rc=$?
echo "long exit=$_rc secs=$(( $(date +%s) - _t0 )) (2 = budget limitation recorded)" >> "$ST"
smi "$OUT/gpu_after_long.txt"

# 3. LOGICAL SHARDS: the path past the 999,999 step ceiling, first run above
# a toy shape. A refusal at any K is recorded by name inside result.json.
echo "--- logical shards at the target shape: K = $SHARDS ---" >> "$ST"
_t0=$(date +%s)
# shellcheck disable=SC2046
pixi run python tools/lm_shards_probe.py --out "$OUT/shards" --target \
    --shards $SHARDS --steps "$SHARD_STEPS" $(corpus_args) \
    > "$OUT/shards.log" 2>&1
_rc=$?
echo "shards exit=$_rc secs=$(( $(date +%s) - _t0 ))" >> "$ST"

smi "$OUT/gpu_after.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
exit 0
