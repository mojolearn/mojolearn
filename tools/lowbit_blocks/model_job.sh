#!/bin/bash
# lane/lowbit-blocks: SmolLM2-360M under a numeric profile on ONE box.
#   env: LB_BOX (label), LB_MODEL (staged dir, default /root/models/SmolLM2-360M),
#        LB_BUILD (1: rebuild the GPU and host bindings this lane touches or needs),
#        LB_RUNS  (space separated "profile:device:phases" items),
#        LB_OUT   (record dir)
# Every binding that imports a changed module is rebuilt (transformer, byte_lm,
# neural_host, transformer_host), plus those the model path needs (base,
# linalg, training; core_host, linalg_host, tokenizer_host).
set -u
cd "$(dirname "$0")/../.." || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
BOX=${LB_BOX:-$(hostname -s)}
MODEL=${LB_MODEL:-/root/models/SmolLM2-360M}
OUT=${LB_OUT:-$PWD/bench/results/lowbit_blocks/$BOX/model}
RUNS=${LB_RUNS:-"fixed15_v1:auto:identity,decode,batch fp32_v1:auto:identity"}
mkdir -p "$OUT"
[ -f .lowbit_blocks_commit ] && export MOJOLEARN_COMMIT=$(cat .lowbit_blocks_commit)
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    case "$driver_major" in ''|*[!0-9]*) ;; *) if [ "$driver_major" -lt 580 ] && [ -x /usr/local/cuda/bin/ptxas ]; then
        export MODULAR_NVPTX_COMPILER_PATH=${MODULAR_NVPTX_COMPILER_PATH:-/usr/local/cuda/bin/ptxas}; fi ;; esac
fi
PX="pixi run -e default"
if [ "${LB_BUILD:-1}" = 1 ]; then
    for b in ${LB_GPU_BUILDS-build build_linalg build_training build_transformer build_byte_lm}; do
        echo "== build $b"; env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 $PX sh bindings/$b.sh > "$OUT/build_$b.log" 2>&1; echo "build $b exit $?"; tail -2 "$OUT/build_$b.log"
    done
    for b in ${LB_HOST_BUILDS-build_core_host build_linalg_host build_neural_host build_transformer_host build_tokenizer_host}; do
        echo "== build $b"; env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 $PX sh bindings/$b.sh > "$OUT/build_$b.log" 2>&1; echo "build $b exit $?"; tail -2 "$OUT/build_$b.log"
    done
fi
(find python/mojolearn -name '*.so' 2>/dev/null | LC_ALL=C sort | while read -r f; do sha256sum "$f" 2>/dev/null || shasum -a 256 "$f"; done) > "$OUT/binaries.sha256"
for r in $RUNS; do
    prof=${r%%:*}; rest=${r#*:}; dev=${rest%%:*}; ph=${rest#*:}
    echo "== run $prof $dev $ph"
    $PX python tools/lowbit_blocks/model_logits.py --model "$MODEL" --profile "$prof" --device "$dev" --phases "$ph" --box "$BOX" --out "$OUT" 2>&1 | grep -v tcmalloc | tee -a "$OUT/runs.log"
done
echo "out $OUT"
