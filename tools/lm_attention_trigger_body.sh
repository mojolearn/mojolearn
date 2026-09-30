#!/bin/sh
# Instrumentation only; no numerical or dispatch changes.
set -eu
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-attention-trigger
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
# 700 puts the run well past the 210-to-480 window the other lane's
# transition lived in, at about 0.2 to 0.45 s a step: four minutes of GPU.
WITNESS_STEPS="${MOJOLEARN_LM_WITNESS_STEPS:-700}"
CORPUS=training/corpus/enwik8/input.txt
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

# THE CORPUS, and a loud line when it is not there. The leg stages
# corpus/enwik8/input.txt from R2 before this body runs; that call is wrapped
# in `|| true` in tools/gemm_remote_leg.sh, so the FILE is the check, not the
# staging exit code. Drive the leg with MOJOLEARN_STAGE_STRICT=1 so the
# failure is at least legible in stage.log.
test "$(wc -c < "$CORPUS" | tr -d ' ')" = 100000000
printf '%s  %s\n' 2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8 "$CORPUS" | sha256sum -c - > "$OUT/corpus_verified.log"
CORPUS_ARG="--corpus $CORPUS"
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


build_byte_lm instrumented "-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1"
probe baseline --shape $TARGET --steps 700 --tail 0 --smi-every 10 --witness-every 0 $CORPUS_ARG
probe forced-eager --shape 1 32 256 4 2 64 64 2 256 --steps 1 --tail 0 --attention-path eager
probe forced-fused --shape 1 32 256 4 2 64 64 2 256 --steps 1 --tail 0 --attention-path fused
pixi run python tools/lm_attention_trigger_check.py "$OUT" > "$OUT/verdict.log" 2>&1
cat "$OUT/verdict.log"
