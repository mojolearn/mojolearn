#!/bin/bash
# lane/lowbit-default: SmolLM2-360M under the flipped default on ONE box, ALONE
# on it (submit with every GPU slot): the hashes and the escape hatches, then
# generate from a 512-token prompt, the default (fixed15_v1, per-layer route)
# against numeric_profile="fp32_v1" (its resident session), alternated.
# Builds the bindings the model path and the resident session need first.
set -u
cd "$(dirname "$0")/../.." || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
MODEL=""
for d in "${LB_MODEL:-}" "$HOME/models/SmolLM2-360M" /root/models/SmolLM2-360M; do
    [ -n "$d" ] && [ -f "$d/model.safetensors" ] && { MODEL=$d; break; }
done
[ -n "$MODEL" ] || { echo "no staged SmolLM2-360M"; exit 2; }
BOX=${LB_BOX:-$(hostname -s)}
OUT=$PWD/bench/results/lowbit_default/$BOX/generate
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>/dev/null
if [ "${LB_BUILD:-1}" = 1 ]; then
    for b in build build_linalg build_training build_transformer; do
        pixi run -e default sh bindings/$b.sh > "$OUT/$b.log" 2>&1; echo "build $b exit $?"
    done
fi
pixi run -e default python tools/lowbit_default/default_gate.py --model "$MODEL" --phases hash,generate \
    --new ${LB_NEW:-32} --rounds ${LB_ROUNDS:-3} --box "$BOX" --out "$OUT" 2>&1 | grep -v tcmalloc
