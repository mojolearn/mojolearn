#!/bin/sh
# DEVIATION 3111: does the missing `dk`/`dv` corner guard explain the whole
# 2.13x? The 2026-09-18 H100 leg measured, at this exact shape, that the
# GUARDED forward refused 0 times out of 8,400 observations and the UNGUARDED
# backward refused 3,697 times out of 8,400, and that the discarded fused
# launch the brief blamed is worth 0.00237 s a step, 0.5%. So the cost is the
# eager path itself, and the only way to stop paying it is to stop refusing.
#
# ARMS:
#   unguarded  the default kernel (the guard is OPT IN). It
#              must REPRODUCE 3,697 backward refusals and a 0.457 s tail; if
#              it does not, the box or the data changed and nothing else here
#              is comparable.
#   guarded    `-D MOJOLEARN_ATTN_KV_CORNER_GUARD=1`. Predicted 0 refusals,
#              `eager_bytes` still 432 at
#              step 699, tail = head.
#   BITS       every step's loss and 8 full state anchors, both ways. THE
#              GUARD IS A CHANGE TO A BIT-EQUALITY TEST, so a single differing
#              step means the refusals were REAL and the guard is a DEFECT.
#   batch4     2,000 steps under the guarded build. This is the deliverable's
#              "what batch now survives 2,000 steps".
set -eu
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-guard
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"
export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_TARGET_COLUMN="${MOJOLEARN_TARGET_COLUMN:-nvidia}"
export PYTHONPATH="$ROOT/python:$ROOT"
WITNESS_STEPS="${MOJOLEARN_LM_WITNESS_STEPS:-700}"
BATCH_STEPS="${MOJOLEARN_LM_BATCH_STEPS:-2000}"
CORPUS=training/corpus/enwik8/input.txt
TARGET="1 2048 768 12 12 64 2048 12 50257"
TARGET_B4="4 2048 768 12 12 64 2048 12 50257"
CONTROL="1 32 256 4 2 64 64 2 256"

smi() {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,temperature.gpu \
        --format=csv,noheader > "$1" 2>&1 || echo "no nvidia-smi" > "$1"
}

echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) witness_steps=$WITNESS_STEPS" > "$ST"
echo "commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)" >> "$ST"
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
echo "gpu_archs=$MOJOLEARN_GPU_ARCHS column=$MOJOLEARN_TARGET_COLUMN" >> "$ST"

# The staging call in tools/gemm_remote_leg.sh ends in `|| true`, so the FILE
# is the check and the exit code is not.
test "$(wc -c < "$CORPUS" | tr -d ' ')" = 100000000
printf '%s  %s\n' 2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8 "$CORPUS" | sha256sum -c - > "$OUT/corpus_verified.log"
CORPUS_ARG="--corpus $CORPUS"

rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
_t0=$(date +%s)
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
_rc=$?
echo "build base exit=$_rc secs=$(( $(date +%s) - _t0 ))" >> "$ST"
[ "$_rc" -eq 0 ] || { echo "base build failed; nothing run" >> "$ST"; exit 1; }

build_byte_lm() {
    rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
    _t0=$(date +%s)
    MOJOLEARN_BUILD_EXTRA_DEFINES="${2:-}" sh bindings/build_byte_lm.sh \
        > "$OUT/build_byte_lm_$1.log" 2>&1
    _rc=$?
    echo "build byte_lm $1 exit=$_rc secs=$(( $(date +%s) - _t0 )) defines='${2:-}'" >> "$ST"
    sha256sum python/mojolearn/identical/_mojolearn_byte_lm.so >> "$ST" 2>&1
    return $_rc
}

probe() {
    _n=$1; shift
    _t0=$(date +%s)
    pixi run python tools/lm_ce_alias_probe.py --out "$OUT/$_n" "$@" > "$OUT/$_n.log" 2>&1
    _rc=$?
    echo "probe $_n exit=$_rc secs=$(( $(date +%s) - _t0 ))" >> "$ST"
    return $_rc
}

if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
fi

build_byte_lm unguarded ""
probe unguarded --shape $TARGET --steps "$WITNESS_STEPS" --tail 0 \
    --smi-every 10 --witness-every 100 $CORPUS_ARG
probe unguarded-forced-fused --shape $CONTROL --steps 1 --tail 0 --attention-path fused || true

build_byte_lm guarded "-D MOJOLEARN_ATTN_KV_CORNER_GUARD=1"
probe guarded --shape $TARGET --steps "$WITNESS_STEPS" --tail 0 \
    --smi-every 10 --witness-every 100 $CORPUS_ARG
probe guarded-forced-eager --shape $CONTROL --steps 1 --tail 0 --attention-path eager || true
probe guarded-forced-fused --shape $CONTROL --steps 1 --tail 0 --attention-path fused || true

smi "$OUT/gpu_mid.txt"
pixi run python tools/lm_attention_guard_verdict.py "$OUT" > "$OUT/verdict.log" 2>&1 || true
cat "$OUT/verdict.log"
echo "verdict_1_written=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"

# Batch 4, the deliverable's last question, under the guarded build and LAST
# so an open-ended run cannot take the rest of the leg down inside the lease.
_t0=$(date +%s)
_rc=0
timeout "${MOJOLEARN_LM_BATCH_TIMEOUT:-900}" \
    pixi run python tools/lm_ce_alias_probe.py --out "$OUT/batch4" \
    --shape $TARGET_B4 --steps "$BATCH_STEPS" --tail 0 \
    --smi-every 25 --witness-every 0 $CORPUS_ARG > "$OUT/batch4.log" 2>&1 || _rc=$?
# 124 is `timeout`, i.e. the lease, NOT an OOM.
echo "batch4 exit=$_rc secs=$(( $(date +%s) - _t0 )) lines=$(wc -l < "$OUT/batch4.log")" >> "$ST"
tail -3 "$OUT/batch4.log" >> "$ST" 2>&1 || true

smi "$OUT/gpu_after.txt"
pixi run python tools/lm_attention_guard_verdict.py "$OUT" > "$OUT/verdict.log" 2>&1 || true
cat "$OUT/verdict.log"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
