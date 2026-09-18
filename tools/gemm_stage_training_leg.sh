#!/bin/sh
# One pinned R2 corpus, 700 full steps per arm; no component timing.
set -eu
CORPUS=${1:?enwik8 or pile_github}
case "$CORPUS" in enwik8|pile_github) ;; *) exit 9 ;; esac
KIND=${2:-stage}
case "$KIND" in stage|workspace) ;; *) exit 9 ;; esac
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
OUT=/root/gemm_leg_out/gemm-$KIND-training
mkdir -p "$OUT/$CORPUS"
cd "$ROOT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT/python:$ROOT"
unset MOJOLEARN_GEMM_ARM MOJOLEARN_ATTN_ARM MOJOLEARN_TRANSFORMER_TIMING
if [ -e /dev/kfd ]; then
    export MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942
else
    export MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
fi
run() {
    name=$1; shift
    if "$@" > "$OUT/$CORPUS/$name.log" 2>&1; then code=0; else code=$?; fi
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/$CORPUS/status.tsv"
    return "$code"
}
# Check-only: a missing or unpinned corpus fails, never origin-downloads.
run corpus sh "tools/fetch_corpus_$CORPUS.sh" --check
if [ ! -f python/mojolearn/identical/_mojolearn.so ]; then
    run build-core env MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh
fi
for arm in base stage; do
    EXTRA='-D MOJOLEARN_GEMM_LEGACY_STAGE_FTZ=1'
    [ "$arm" = base ] || EXTRA='-D MOJOLEARN_GEMM_STAGE_FTZ=1'
    if [ "$KIND" = workspace ]; then
        EXTRA='-D MOJOLEARN_GEMM_LEGACY_REUSE_GROUP_WS=1'
        [ "$arm" = base ] || EXTRA='-D MOJOLEARN_GEMM_REUSE_GROUP_WS=1'
    fi
    BINROOT=/root/gemm-$KIND-binaries/$arm
    if [ ! -f "$BINROOT/_mojolearn_byte_lm.so" ]; then
        run "build-$arm" env MOJOLEARN_BUILD_EXTRA_DEFINES="$EXTRA" \
            MOJOLEARN_BYTE_LM_OUTDIR="$BINROOT" sh bindings/build_byte_lm.sh
    else
        printf 'reuse-build-%s\t0\n' "$arm" >> "$OUT/$CORPUS/status.tsv"
    fi
    cp "$BINROOT/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
    run "$arm" pixi run python tools/lm_ce_alias_probe.py \
        --out "$OUT/$CORPUS/$arm" --steps 700 --tail 0 --seed 20260917 \
        --corpus "training/corpus/$CORPUS/input.txt" --smi-every 10 --witness-every 699
done
run compare python3 tools/gemm_training_compare.py "$OUT" "$CORPUS" --kind "$KIND"
echo "COMPLETE $CORPUS 700 steps per arm"
