#!/bin/sh
# DEVIATION 3110, the eager attention fallback: name the trigger, then
# measure the sticky latch against today's launch-then-discard behavior.
#
# ARMS, in the order they run and the order they must be read:
#   before     `-D MOJOLEARN_ATTN_NO_STICKY=1`, 700 steps at the target
#              shape. THIS IS THE DIAGNOSIS and it runs FIRST. Its per-step
#              per-layer forward/backward statuses name the trigger. If it
#              shows zero refusals in 700 steps, NOTHING BELOW MEANS
#              ANYTHING and the fix is unmotivated; read this arm first.
#   eager-ref  the same build, 700 steps, MOJOLEARN_TRANSFORMER_ATTN_PATH
#              =eager. The fused kernels never run. Two jobs: it prices the
#              eager path alone (the `before` step minus this one is the
#              discarded fused launch), and its loss series is the reference
#              the fused path claims to equal bit for bit.
#   after      the default build (the latch), 700 steps at the target shape.
#   batch4     the latch at batch 4 for 2,000 steps. The latch removes a
#              LAUNCH, not a BUFFER, so this is expected to OOM at the same
#              step it always did; running it is how that stops being a
#              prediction.
#   forced-*   head_dim 64 controls. At head_dim 8 the fused kernel is not
#              instantiated and BOTH arms fall back, which is a pass-shaped
#              result that means nothing.
set -eu
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-fallback
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

# THE CORPUS. The leg stages corpus/enwik8/input.txt from R2 before this body
# runs, and that call is wrapped in `|| true` in tools/gemm_remote_leg.sh, so
# the FILE is the check and the staging exit code is not. Drive the leg with
# MOJOLEARN_STAGE_STRICT=1 so a failure is at least legible in stage.log.
test "$(wc -c < "$CORPUS" | tr -d ' ')" = 100000000
printf '%s  %s\n' 2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8 "$CORPUS" | sha256sum -c - > "$OUT/corpus_verified.log"
CORPUS_ARG="--corpus $CORPUS"

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
    # A .so digest never proves a define. The witness that the two arms are
    # two arms is FUSED_SKIPPED_STICKY (status 3) appearing in one result and
    # in neither of the others, read from inside the process that ran.
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

# ---- ARM SET 1: today's behavior. THE DIAGNOSIS RUNS FIRST. ----------------
build_byte_lm nosticky "-D MOJOLEARN_ATTN_NO_STICKY=1"
probe before --shape $TARGET --steps "$WITNESS_STEPS" --tail 0 \
    --smi-every 10 --witness-every 100 $CORPUS_ARG
probe eager-ref --shape $TARGET --steps "$WITNESS_STEPS" --tail 0 \
    --smi-every 10 --witness-every 100 --attention-path eager $CORPUS_ARG
probe nosticky-forced-eager --shape $CONTROL --steps 1 --tail 0 --attention-path eager
probe nosticky-forced-fused --shape $CONTROL --steps 1 --tail 0 --attention-path fused

# ---- ARM SET 2: the latch. ------------------------------------------------
build_byte_lm sticky ""
probe after --shape $TARGET --steps "$WITNESS_STEPS" --tail 0 \
    --smi-every 10 --witness-every 100 $CORPUS_ARG
probe forced-eager --shape $CONTROL --steps 1 --tail 0 --attention-path eager
probe forced-fused --shape $CONTROL --steps 1 --tail 0 --attention-path fused
# Best effort, and an OOM here is a RESULT: the latch removes a launch, not a
# buffer. `|| true` so the verdict below still runs and files what did land.
probe batch4 --shape $TARGET_B4 --steps "$BATCH_STEPS" --tail 0 \
    --smi-every 25 --witness-every 0 $CORPUS_ARG || true

smi "$OUT/gpu_after.txt"
pixi run python tools/lm_attention_fallback_verdict.py "$OUT" > "$OUT/verdict.log" 2>&1 || true
cat "$OUT/verdict.log"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
