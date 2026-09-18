#!/bin/sh
# One corpus, both source arms, 700 steps each, with separate pure/timer builds.
# The corpus argument is fixed by the payload wrapper, never inferred from data.
set -eu
CORPUS=${1:?enwik8 or pile_github}
case "$CORPUS" in enwik8|pile_github) ;; *) exit 9 ;; esac
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
cd "$ROOT"
OUT=/root/gemm_leg_out/class-training-$CORPUS
mkdir -p "$OUT/bin" "$OUT/build"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT/python:$ROOT"
unset MOJOLEARN_TRANSFORMER_TIMING MOJOLEARN_GEMM_ARM MOJOLEARN_ATTN_ARM
if [ -e /dev/kfd ]; then
    export MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942
    COLUMN=MOJOLEARN_COLUMN_AMD
    rocm-smi > "$OUT/device.txt" 2>&1
else
    export MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
    COLUMN=MOJOLEARN_COLUMN_NVIDIA
    nvidia-smi > "$OUT/device.txt" 2>&1
    [ ! -x /usr/local/cuda/bin/ptxas ] || export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
fi
run() {
    name=$1; shift
    start=$(date +%s)
    if "$@" > "$OUT/$name.log" 2>&1; then code=0; else code=$?; fi
    printf '%s\t%s\t%s\n' "$name" "$code" "$(( $(date +%s)-start ))" >> "$OUT/status.tsv"
    return "$code"
}
# Check-only: missing staged bytes cannot trigger an origin download.
run corpus sh "tools/fetch_corpus_$CORPUS.sh" --check
run build-base env MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=2 sh bindings/build.sh
for arm in base class; do
    for kind in pure timers; do
        EXTRA=
        [ "$arm" != class ] || EXTRA='-D MOJOLEARN_GEMM_CLASS_FLUSH=1'
        [ "$kind" != timers ] || EXTRA="$EXTRA -D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1"
        # All builds have the same source directory and compiler output path.
        run "build-$arm-$kind" pixi run mojo build -j 2 --emit shared-lib \
            --target-accelerator "$MOJOLEARN_GPU_ARCHS" --target-cpu x86-64-v3 \
            -D "$COLUMN" -D MOJOLEARN_NUMERIC_IDENTICAL=1 $EXTRA -I . -I bindings \
            bindings/_mojolearn_byte_lm.mojo -o "$OUT/build/_mojolearn_byte_lm.so"
        cp "$OUT/build/_mojolearn_byte_lm.so" "$OUT/bin/$arm-$kind.so"
        python3 tools/gemm_class_sections.py "$OUT/build/_mojolearn_byte_lm.so" > "$OUT/sections-$arm-$kind.json"
    done
done
for kind in pure timers; do
    for arm in base class; do
        cp "$OUT/bin/$arm-$kind.so" python/mojolearn/identical/_mojolearn_byte_lm.so
        EXTRA=
        [ "$kind" != timers ] || EXTRA='--component-timing --component-timing-steps 10'
        run "$arm-$kind" pixi run python tools/lm_step_memory_probe.py \
            --out "$OUT/$arm-$kind" --target --resident-lean --steps 700 \
            --seed 93261 --budget-seconds 1500 --corpus "training/corpus/$CORPUS/input.txt" $EXTRA
    done
done
run summary python3 tools/gemm_class_training_summary.py "$OUT"
rm -rf "$OUT/bin" "$OUT/build"
echo 'COMPLETE 700 steps per arm/build plus ten instrumented steps after step 700'
