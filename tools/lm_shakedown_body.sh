#!/bin/sh
# tools/lm_shakedown_body.sh -- lane/lm-training-shakedown, leg 1.
#
# Runs on the pod as tools/gemm_remote_leg.sh's MOJOLEARN_GEMM_LEG_EXTRA, from
# /root/mojolearn, after the payload's device check and card, with enwik8
# already staged from R2 into training/corpus/enwik8/input.txt.
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_shakedown_body.sh \
#   MOJOLEARN_GEMM_LEG_GPU_NVIDIA="NVIDIA H100 80GB HBM3" \
#   sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# In order:
#   1. build the base and byte-LM bindings under IDENTICAL for this box's arch;
#   2. BATCH SWEEP at the 162,147,840-parameter target shape, L2048: batch 1,
#      2, 4 and 8, three complete steps each, resident + lean, enwik8. Records
#      tokens/s and the polled device peak per batch, or the refusal. Every
#      LM run on record before this one is batch 1, so a failure here is a
#      result, not an error, and the sweep continues to the next batch.
#   3. RESUME at the target shape through export_state()/load_state_dict(),
#      the path export_checkpoint() refuses above 87,381 parameters. The
#      missing-moments control runs BEFORE the resume arm and must fail.
#   4. LONG-RUN SHAKEDOWN: MOJOLEARN_LM_SHAKEDOWN_STEPS consecutive steps at
#      batch 1, no per-step witnesses, per-step wall time / loss / host RSS /
#      device peak in events.jsonl.
#
# POSIX sh (RunPod images link /bin/sh to dash). Never `set -e`: a red arm is
# a result that has to come home.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-shakedown
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"
export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_TARGET_COLUMN="${MOJOLEARN_TARGET_COLUMN:-nvidia}"
export PYTHONPATH="$ROOT/python:$ROOT"

SWEEP_STEPS="${MOJOLEARN_LM_SWEEP_STEPS:-3}"
SWEEP_BUDGET="${MOJOLEARN_LM_SWEEP_BUDGET:-900}"
LONG_STEPS="${MOJOLEARN_LM_SHAKEDOWN_STEPS:-2000}"
LONG_BUDGET="${MOJOLEARN_LM_SHAKEDOWN_BUDGET:-1500}"
RESUME_WARMUP="${MOJOLEARN_LM_RESUME_WARMUP:-8}"
RESUME_TAIL="${MOJOLEARN_LM_RESUME_TAIL:-4}"
CORPUS=training/corpus/enwik8/input.txt

smi() {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,clocks.max.sm,temperature.gpu \
        --format=csv,noheader > "$1" 2>&1 || echo "no nvidia-smi" > "$1"
}

echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) sweep_steps=$SWEEP_STEPS long_steps=$LONG_STEPS" > "$ST"
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

# The corpus must be the R2-staged one; say so either way rather than
# silently falling back to synthetic tokens.
if [ -f "$CORPUS" ]; then
    echo "corpus=$CORPUS bytes=$(wc -c < "$CORPUS")" >> "$ST"
else
    echo "corpus MISSING at $CORPUS; arms will run on synthetic tokens" >> "$ST"
    CORPUS=
fi

# 1. BUILD. The NumPy-free host layer needs the IDENTICAL base binding.
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

# 2. BATCH SWEEP at the target dims: B 2048 768 12 12 64 2048 12 50257.
echo "--- batch sweep: target dims, L2048, resident+lean, $SWEEP_STEPS steps ---" >> "$ST"
for B in 1 2 4 8; do
    smi "$OUT/gpu_before_b$B.txt"
    _t0=$(date +%s)
    # shellcheck disable=SC2046
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/batch-$B" \
        --shape "$B" 2048 768 12 12 64 2048 12 50257 \
        --steps "$SWEEP_STEPS" --budget-seconds "$SWEEP_BUDGET" \
        --resident-lean $(corpus_args) > "$OUT/batch-$B.log" 2>&1
    _rc=$?
    echo "batch=$B exit=$_rc secs=$(( $(date +%s) - _t0 )) (0 ok, 2 budget limitation, other = refusal/OOM)" >> "$ST"
    smi "$OUT/gpu_after_b$B.txt"
done

# 3. RESUME. A cheap CONTROL-SHAPE smoke first (20,453,376 parameters, also
# far above the 87,381 the checkpoint file accepts, so it is the same code
# path): if the harness itself is wrong, it says so in seconds rather than
# after the target's minute-long exports.
echo "--- resume smoke at the control shape ---" >> "$ST"
_t0=$(date +%s)
# shellcheck disable=SC2046
pixi run python tools/lm_shakedown_resume.py --out "$OUT/resume-control" \
    --state /root/lm_shakedown_state_control \
    --shape 1 2048 384 6 6 64 1024 8 8192 \
    --warmup 4 --tail 3 $(corpus_args) > "$OUT/resume-control.log" 2>&1
_rc=$?
echo "resume-control exit=$_rc secs=$(( $(date +%s) - _t0 )) (0 = control differed AND resume matched)" >> "$ST"
rm -rf /root/lm_shakedown_state_control

echo "--- resume at the target shape (warmup $RESUME_WARMUP, tail $RESUME_TAIL) ---" >> "$ST"
_t0=$(date +%s)
# shellcheck disable=SC2046
pixi run python tools/lm_shakedown_resume.py --out "$OUT/resume" \
    --state /root/lm_shakedown_state --target \
    --warmup "$RESUME_WARMUP" --tail "$RESUME_TAIL" $(corpus_args) \
    > "$OUT/resume.log" 2>&1
_rc=$?
echo "resume exit=$_rc secs=$(( $(date +%s) - _t0 )) (0 = control differed AND resume matched)" >> "$ST"
rm -rf /root/lm_shakedown_state

# 4. LONG-RUN SHAKEDOWN at batch 1, no per-step witnesses.
echo "--- long run: $LONG_STEPS consecutive steps, batch 1, budget $LONG_BUDGET s ---" >> "$ST"
smi "$OUT/gpu_before_long.txt"
_t0=$(date +%s)
# shellcheck disable=SC2046
pixi run python tools/lm_step_memory_probe.py --out "$OUT/long" --target \
    --steps "$LONG_STEPS" --budget-seconds "$LONG_BUDGET" \
    --resident-lean $(corpus_args) > "$OUT/long.log" 2>&1
_rc=$?
echo "long exit=$_rc secs=$(( $(date +%s) - _t0 )) (2 = budget limitation recorded)" >> "$ST"
smi "$OUT/gpu_after.txt"

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
exit 0
