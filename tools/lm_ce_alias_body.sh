#!/bin/sh
# tools/lm_ce_alias_body.sh -- DEVIATIONS 3010, 3011 and the binary
# checkpoint, ON THE POD, run by tools/gemm_remote_leg.sh's gemm payload
# when named in MOJOLEARN_GEMM_LEG_EXTRA. It runs after the payload's own
# device check and card, from /root/mojolearn, with pixi on PATH.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_ce_alias_body.sh \
#   sh tools/gemm_remote_leg.sh nvidia --payload gemm --rent --minutes 75 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# WHAT IT DECIDES, in order:
#
#  A. the CE ALIASING A/B (DEVIATION 3011). The shipped build overlays
#     `ce_shift` on `logits` and `ce_weights`/`ce_dlogits` on `ce_expo`,
#     five [M, V] buffers becoming two. The claim is that no bit moves, so
#     the binding is built TWICE -- clean, then with
#     -D MOJOLEARN_BYTE_LM_CE_UNALIASED=1 -- and the same five steps are
#     run against each at the 162,147,840-parameter shape. Every step's
#     loss, gradient, parameter, m, v and flags hash must be equal, AND the
#     two runs must report DIFFERENT `ce_aliased`, or the comparison was of
#     one arm with itself. The device peaks are the measurement.
#
#  B. the EAGER-FALLBACK WITNESS (DEVIATION 3010). `attention_stage_report()`
#     must read zero grown layers on the fused path and every layer grown
#     under MOJOLEARN_TRANSFORMER_ATTN_PATH=eager. A witness that reads zero
#     in both arms is not a witness. This is the falsifier for the step
#     change lane/lm-training-shakedown measured on 2026-09-17 (device
#     memory 16.949 -> 34.397 GB, seconds 0.207 -> 0.44, between steps 210
#     and 480, cause not established): if that was the eager fallback, a
#     long run's report moves at exactly that step.
#
#  C. the BINARY CHECKPOINT at 162M across a PROCESS BOUNDARY. Three steps,
#     save, exit; a second process loads and runs two more; its witnesses
#     must equal the uninterrupted run's. The --drop-moments control zeroes
#     m and v after the load and MUST separate, otherwise the comparison is
#     not reading the moments.
#
# Everything is OUR IDENTICAL arm; no opponent runs and none is installed
# (lane/r2-opponent-hygiene made that opt-in behind --opponents).
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-ce-alias
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"
export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_TARGET_COLUMN="${MOJOLEARN_TARGET_COLUMN:-nvidia}"
export PYTHONPATH="$ROOT/python:$ROOT"
STEPS="${MOJOLEARN_LM_ALIAS_STEPS:-3}"
TAIL="${MOJOLEARN_LM_ALIAS_TAIL:-2}"
TARGET="1 2048 768 12 12 64 2048 12 50257"
CONTROL="1 2048 384 6 6 64 1024 8 8192"

smi() {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,temperature.gpu \
        --format=csv,noheader > "$1" 2>&1 || echo "no nvidia-smi" > "$1"
}

echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) steps=$STEPS tail=$TAIL" > "$ST"
echo "commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)" >> "$ST"
smi "$OUT/gpu_before.txt"

# One mojo build is one GPU arch; 9.0 is spelled sm_90a (DEVIATION 2293).
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

# The NumPy-free Python layer needs the IDENTICAL base binding for its host
# helpers; build it first. The byte LM build refuses to overwrite, so a stale
# box copy (never the repo's) is removed before each arm.
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
_t0=$(date +%s)
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
_rc=$?
echo "build base exit=$_rc secs=$(( $(date +%s) - _t0 ))" >> "$ST"
[ "$_rc" -eq 0 ] || { echo "base build failed; nothing run" >> "$ST"; exit 1; }

build_byte_lm() {   # <arm-name> [extra defines]
    rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
    _t0=$(date +%s)
    MOJOLEARN_BUILD_EXTRA_DEFINES="${2:-}" sh bindings/build_byte_lm.sh \
        > "$OUT/build_byte_lm_$1.log" 2>&1
    _rc=$?
    echo "build byte_lm $1 exit=$_rc secs=$(( $(date +%s) - _t0 )) defines='${2:-}'" >> "$ST"
    # A .so digest never proves a define (see the memory rule); the witness
    # is byte_lm_ce_aliased() read from INSIDE the process that loaded it,
    # which every result.json carries.
    sha256sum python/mojolearn/identical/_mojolearn_byte_lm.so >> "$ST" 2>&1
    return $_rc
}

probe() {   # <dir> [probe args...]
    _n=$1; shift
    _t0=$(date +%s)
    pixi run python tools/lm_ce_alias_probe.py --out "$OUT/$_n" "$@" > "$OUT/$_n.log" 2>&1
    _rc=$?
    echo "probe $_n exit=$_rc secs=$(( $(date +%s) - _t0 ))" >> "$ST"
    return $_rc
}

if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
    echo "numpy pip exit=$?" >> "$ST"
fi

# ---- ARM A: the shipped, ALIASED build -----------------------------------
build_byte_lm aliased "" || { echo "aliased build failed" >> "$ST"; exit 1; }

# A1. the A/B arm at the target shape.
probe aliased --shape $TARGET --steps "$STEPS" --tail "$TAIL"
smi "$OUT/gpu_after_aliased.txt"

# B. the eager-fallback witness, at the CONTROL shape so the eager arm's
# quadratic arrays fit comfortably beside everything else. Fused first.
probe eager-off --shape $CONTROL --steps 1 --tail 0 --attention-path fused
probe eager-on  --shape $CONTROL --steps 1 --tail 0 --attention-path eager

# C. the binary checkpoint across a process boundary, at the target shape.
CKPT="$OUT/target-162m.byte-lm.bin"
probe ckpt-save --shape $TARGET --steps "$STEPS" --tail 0 --mode checkpoint --checkpoint "$CKPT"
if [ -f "$CKPT" ]; then
    probe ckpt-resume  --shape $TARGET --tail "$TAIL" --mode checkpoint --checkpoint "$CKPT" --resume
    probe ckpt-control --shape $TARGET --tail "$TAIL" --mode checkpoint --checkpoint "$CKPT" --resume --drop-moments
    ls -l "$CKPT" >> "$ST"
    # 1.95 GB of evidence does not come home; the digest and the result do.
    sha256sum "$CKPT" >> "$ST" 2>&1
    rm -f "$CKPT"
fi

# ---- ARM B: the UNALIASED build, five separate [M, V] buffers -------------
build_byte_lm unaliased "-D MOJOLEARN_BYTE_LM_CE_UNALIASED=1" \
    || { echo "unaliased build failed" >> "$ST"; exit 1; }
probe unaliased --shape $TARGET --steps "$STEPS" --tail "$TAIL"
smi "$OUT/gpu_after_unaliased.txt"

# ---- the verdict ---------------------------------------------------------
pixi run python tools/lm_ce_alias_compare.py --out "$OUT" > "$OUT/verdict.log" 2>&1
echo "compare exit=$? " >> "$ST"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
smi "$OUT/gpu_after.txt"
cat "$OUT/verdict.log" >> "$ST"
exit 0
