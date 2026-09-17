#!/bin/sh
# tools/model_leg/run_local_m4.sh: THE APPLE COLUMNS OF THE MODEL LEG on this
# M4: the Metal column through the identical GPU builds and the CPU column
# through the host bindings, one prompt file, then the incumbent on the same
# box through torch's mps device. FOR THE ORCHESTRATOR TO RUN LATER; it
# compiles and runs on the Mac and this lane never executed it.
#
#   sh tools/model_leg/run_local_m4.sh                  everything, into $MOJOLEARN_EVIDENCE_ROOT/model-leg/<stamp>-apple-m4/
#   sh tools/model_leg/run_local_m4.sh --skip-build     the bindings are already built in this worktree
#   sh tools/model_leg/run_local_m4.sh --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
#
# It runs tools/model_leg/leg_body.sh with ROOT = this worktree, OUT outside
# the checkout, CPU_COLUMN_ALSO=1, and the builds through
# tools/macos_serial_guard.py where it exists (bindings/build_host_family.sh
# asks for it on the Mac). Nothing is written under bench/results.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd) || exit 2
cd "$REPO" || exit 2
[ "$(uname -s)" = Darwin ] || { echo "this wrapper is the Apple column; run it on the M4" >&2; exit 2; }
MODEL=${MOJOLEARN_MODEL_LEG_MODEL:-HuggingFaceTB/SmolLM2-360M}; SKIP_BUILD=${MOJOLEARN_MODEL_LEG_SKIP_BUILD:-0}
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-build) SKIP_BUILD=1 ;;
        --model) shift; MODEL=${1:-} ;;
        -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
done
case "$MODEL" in *[!A-Za-z0-9_./-]*|'') echo "--model '$MODEL': letters, digits and _./- only" >&2; exit 2 ;; esac
STAMP=$(date -u +%Y-%m-%d_%H%M%S)
EVIDENCE=${MOJOLEARN_EVIDENCE_ROOT:-$HOME/mojolearn-evidence}
OUT=$EVIDENCE/model-leg/$STAMP-apple-m4
mkdir -p "$OUT" || exit 2
echo "$(git rev-parse HEAD)" > "$OUT/commit.txt"
if [ -n "$(git status --porcelain -- bindings python/mojolearn tools/model_leg bench/model gemm transformer core 2>/dev/null)" ]; then
    echo "REFUSED: the worktree is dirty under a path the column compiles or runs; commit or stash first (one variable is the device)" >&2
    git status --porcelain -- bindings python/mojolearn tools/model_leg bench/model gemm transformer core >&2
    exit 3
fi
SLUG=$(printf '%s' "$MODEL" | sed 's|/|__|g')
export MOJOLEARN_COMMIT="$(cat "$OUT/commit.txt")"
export MOJOLEARN_MODEL_LEG_ROOT="$REPO" MOJOLEARN_MODEL_LEG_OUT="$OUT/model-leg"
export MOJOLEARN_MODEL_LEG_MODEL="$MODEL" MOJOLEARN_MODEL_LEG_MODEL_DIR="$EVIDENCE/model-leg/models/$SLUG"
export MOJOLEARN_MODEL_LEG_VENV="$EVIDENCE/model-leg/.venv-model-leg-m4"
export MOJOLEARN_MODEL_LEG_CPU_COLUMN_ALSO=1 MOJOLEARN_MODEL_LEG_SKIP_BUILD="$SKIP_BUILD"
export MOJOLEARN_MODEL_LEG_LABEL="${MOJOLEARN_MODEL_LEG_LABEL:-apple-m4-metal}"
export MOJOLEARN_COMPILE_JOBS="${MOJOLEARN_COMPILE_JOBS:-2}"
# the Mac's builds go through the serial guard when it exists, one at a time, two compiler workers
if [ "$SKIP_BUILD" != 1 ] && [ -f tools/macos_serial_guard.py ]; then
    export MOJOLEARN_MODEL_LEG_PYTHON="pixi run python"
    for b in ${MOJOLEARN_MODEL_LEG_BUILDS:-build build_linalg build_transformer build_training}; do
        MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 python3 tools/macos_serial_guard.py sh "bindings/$b.sh" \
            > "$OUT/build-$b.log" 2>&1 || echo "build-$b FAILED (see $OUT/build-$b.log)"
    done
    for b in ${MOJOLEARN_MODEL_LEG_HOST_BUILDS:-build_core_host build_linalg_host build_transformer_host build_tokenizer_host}; do
        env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
            python3 tools/macos_serial_guard.py sh "bindings/$b.sh" > "$OUT/build-$b.log" 2>&1 || echo "build-$b FAILED (see $OUT/build-$b.log)"
    done
    export MOJOLEARN_MODEL_LEG_SKIP_BUILD=1
fi
sh tools/model_leg/leg_body.sh
rc=$?
echo "records: $OUT/model-leg/ (ours.apple-m4-metal.json, ours.apple-m4-metal-cpu.json, torch.apple-m4-metal.json, ratio.txt, status.tsv)"
echo "then, on this Mac (no compile, no GPU):"
echo "  python3 bench/model/diff.py --diff $OUT/model-leg/ours.apple-m4-metal.json $OUT/model-leg/ours.apple-m4-metal-cpu.json --require-columns 2"
exit $rc
