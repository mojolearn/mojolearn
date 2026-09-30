#!/bin/sh
# lane/attention-replay-vendors: paired 700-step legacy/default training on
# ONE corpus on this box's own column (AMD or NVIDIA), the same probe, seed,
# shape and arms as the NVIDIA flip (tools/lm_attention_default_body.sh).
#
#   sh tools/lm_attention_vendor_train_body.sh <enwik8|pile_github> <third arm>
#
# Arms, in this order (the owed pair first):
#   legacy           -D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1
#   repaired         no defines: the column's shipped default after the flip
#   released_legacy  -D MOJOLEARN_ATTN_LEGACY_CORNER=1 (release-only control)
#   guarded          legacy + -D MOJOLEARN_ATTN_EXACT_TAIL_GUARD=1 (dk/dv attribution)
# The runners pass no environment, so the corpus and third arm are arguments.
set -eu
CORPUS=${1:?enwik8 or pile_github}
THIRD=${2:-none}
case "$CORPUS" in enwik8|pile_github) ;; *) echo "bad corpus $CORPUS"; exit 9 ;; esac
case "$THIRD" in released_legacy|guarded|none) ;; *) echo "bad arm $THIRD"; exit 9 ;; esac
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out/attn-replay-$CORPUS
mkdir -p "$OUT"
cd "$ROOT"
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python:$ROOT"
if [ -e /dev/kfd ]; then
    export MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_GPU_ARCHS=gfx942
    rocm-smi > "$OUT/device.txt" 2>&1 || true
else
    export MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_GPU_ARCHS=sm_90a
    nvidia-smi > "$OUT/device.txt" 2>&1 || true
    [ ! -x /usr/local/cuda/bin/ptxas ] || export MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas
fi
echo "column=$MOJOLEARN_TARGET_COLUMN arch=$MOJOLEARN_GPU_ARCHS corpus=$CORPUS third=$THIRD" > "$OUT/leg.txt"
run() {
    name=$1; shift
    start=$(date +%s)
    if "$@" > "$OUT/$name.log" 2>&1; then code=0; else code=$?; fi
    printf '%s\t%s\t%s\n' "$name" "$code" "$(( $(date +%s)-start ))" >> "$OUT/status.tsv"
    return "$code"
}
# --check is read-only: absent or wrong bytes fail here, never download.
run corpus sh "tools/fetch_corpus_$CORPUS.sh" --check
rm -f python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_byte_lm.so
run build-base env MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=4 sh bindings/build.sh
ARMS="legacy repaired"
[ "$THIRD" = none ] || ARMS="$ARMS $THIRD"
for arm in $ARMS; do
    rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
    defines=""
    case "$arm" in
        legacy) defines="-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1" ;;
        released_legacy) defines="-D MOJOLEARN_ATTN_LEGACY_CORNER=1" ;;
        guarded) defines="-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_ATTN_EXACT_TAIL_GUARD=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1" ;;
    esac
    run "build-$arm" env MOJOLEARN_BUILD_EXTRA_DEFINES="$defines" MOJOLEARN_COMPILE_JOBS=4 sh bindings/build_byte_lm.sh
    sha256sum python/mojolearn/identical/_mojolearn_byte_lm.so >> "$OUT/binding_sha256.txt"
    run "train-$arm" pixi run python tools/lm_ce_alias_probe.py --out "$OUT/$arm" --steps 700 --tail 0 \
        --corpus "training/corpus/$CORPUS/input.txt" --smi-every 10 --witness-every 699
done
# Same-box pairwise verdicts. The cross-vendor comparison against the NVIDIA
# record runs on the host, where bench/results exists.
run verdict pixi run python tools/lm_attention_repair_compare.py "$OUT" --default || true
echo COMPLETE >> "$OUT/status.tsv"
