#!/bin/sh
# tools/lm_step_memory_probe.sh -- DEVIATION 2495: complete IDENTICAL LM
# training steps at the control and target shapes, ON THE POD, run by
# tools/gemm_remote_leg.sh's gemm payload when named in
# MOJOLEARN_GEMM_LEG_EXTRA. It runs after the payload's own device check and
# card, from /root/mojolearn, with pixi on PATH (see the header of
# tools/gemm_remote_leg.sh at MOJOLEARN_GEMM_LEG_EXTRA).
#
#   MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key \
#   MOJOLEARN_GPU_ARCHS=sm_90a \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/lm_step_memory_probe.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 \
#       --gpu "NVIDIA H100 80GB HBM3"
#
# What it does, in order, every step bounded and recorded in status.txt:
#   1. builds ONLY the byte LM binding under IDENTICAL for this box's GPU
#      architecture (bindings/build_byte_lm.sh; MOJOLEARN_GPU_ARCHS from the
#      leg, else derived from nvidia-smi compute_cap);
#   2. runs tools/lm_step_memory_probe.py at the CONTROL shape
#      (20,453,376 parameters, B1 L2048 V8192), 3 complete steps, 300 s budget;
#   3. attempts the TARGET shape (162,147,840 parameters, 12 layers DM768
#      FF2048 V50257 untied, B1 L2048), 3 complete steps, 300 s budget. A
#      budget miss is exit 2 and a recorded limitation, never a smaller model;
#   4. if the target completed, one extra target step with the phase
#      printer (MOJOLEARN_TRANSFORMER_TIMING=1) in its own directory; the
#      probe sums its `timing` lines (blocks, and since DEVIATION 2499 the
#      `step.*` phases outside the blocks plus the Python wrapper) into
#      result.json as component_timing_ms / component_bytes with the
#      covered fraction of step_seconds. Not a timing sample.
#   5. DEVIATION 2514 (design step 8, gate G5): the same control and target
#      probes again with --resident-lean --witness-every-step, into
#      control-lean/ and target-lean/, same step count and budget; the
#      trainer then runs step_result='lean' (train_step returns
#      loss/step/flags, nothing else crosses) and the per-step witnesses
#      come from export_gradients()/export_state() after each step, outside
#      the timed boundary, so they compare hash for hash with the full
#      runs' step_end records on the same GPU;
#   6. if the lean target completed, one lean target step with the phase
#      printer into target-lean-timing/ (G5 reads its component_timing_ms
#      for the phases that remain and the ones that are gone).
# Full first, then lean, so the full numbers are the same boxes' baseline.
# Outputs land under /root/gemm_leg_out/lm-step-memory/ and come home with
# the leg's fetch. Everything is OUR IDENTICAL arm; no opponent runs.
#
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/lm-step-memory
mkdir -p "$OUT"
cd "$ROOT" || exit 9
ST="$OUT/status.txt"
PATH="$HOME/.pixi/bin:$PATH"
export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_TARGET_COLUMN="${MOJOLEARN_TARGET_COLUMN:-nvidia}"
export PYTHONPATH="$ROOT/python:$ROOT"
BUDGET="${MOJOLEARN_LM_PROBE_BUDGET:-300}"
STEPS="${MOJOLEARN_LM_PROBE_STEPS:-3}"

smi() {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.sm,clocks.max.sm,temperature.gpu \
        --format=csv,noheader > "$1" 2>&1 || echo "no nvidia-smi" > "$1"
}

echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ) budget=$BUDGET steps=$STEPS" > "$ST"
smi "$OUT/gpu_before.txt"

# THE ARCHITECTURE. One mojo build is one GPU arch; read it from the leg's
# environment first, else from the device. 9.0 is spelled sm_90a here because
# that is the only spelling the compiler produces on an H100 (DEVIATION 2293).
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

# 1. BUILD ONLY THE BYTE LM BINDING. The build script refuses to overwrite,
# so a stale box copy (never the repo's) is removed first.
# The NumPy-free Python layer needs the IDENTICAL base binding for its host
# helpers (all_finite_f32, converters); build it first (first H100 run,
# 2026-09-11 03:06Z, failed at import without it).
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
[ "$_rc" -eq 0 ] || { echo "build failed; nothing run" >> "$ST"; exit 1; }

# numpy is the probe's only non-stdlib import (weights and token batches).
if ! pixi run python -c 'import numpy' > "$OUT/numpy.log" 2>&1; then
    pixi run python -m pip install numpy >> "$OUT/numpy.log" 2>&1
    echo "numpy pip exit=$?" >> "$ST"
fi

probe() {   # <name> [probe args...]
    _n=$1; shift
    _t0=$(date +%s)
    pixi run python tools/lm_step_memory_probe.py --out "$OUT/$_n" \
        --steps "$STEPS" --budget-seconds "$BUDGET" "$@" > "$OUT/$_n.log" 2>&1
    _rc=$?
    echo "probe $_n exit=$_rc secs=$(( $(date +%s) - _t0 )) (2 = budget limitation recorded)" >> "$ST"
    return $_rc
}

# 2. CONTROL: 20.45M parameters, B1 L2048 V8192 (the Apple pilot's shape).
probe control --shape 1 2048 384 6 6 64 1024 8 8192

# 3. TARGET: 162,147,840 parameters, B1 L2048 V50257. Attempted regardless of
# the control's outcome; a miss is a limitation, not a reason to shrink it.
smi "$OUT/gpu_between.txt"
if probe target --target; then
    # 4. Component time fractions: one target step with the phase printer.
    # Its `timing <phase> <ms>` lines are in target-timing/worker.log and
    # summed by name in target-timing/result.json (DEVIATION 2499).
    # argparse keeps the last --steps, so this overrides the default count.
    probe target-timing --target --component-timing --steps 1
fi

# 5. LEAN (DEVIATION 2514): the same two shapes with step_result='lean',
# witnesses exported after every step so they line up with 2. and 3.
smi "$OUT/gpu_between_lean.txt"
echo "lean arm: --resident-lean --witness-every-step (control-lean, target-lean, target-lean-timing)" >> "$ST"
probe control-lean --shape 1 2048 384 6 6 64 1024 8 8192 --resident-lean --witness-every-step
if probe target-lean --target --resident-lean --witness-every-step; then
    # 6. G5's timed step: which phases remain on the lean path.
    probe target-lean-timing --target --resident-lean --component-timing --steps 1
fi

smi "$OUT/gpu_after.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$ST"
exit 0
