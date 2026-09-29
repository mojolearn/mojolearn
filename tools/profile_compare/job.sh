#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.pixi/bin:/opt/rocm/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_HOTPATH=native
unset MOJOLEARN_NUMERIC_PROFILE
export PYTHONPATH="$PWD/python"
export MOJOLEARN_COMMIT=$(git rev-parse HEAD)
MODEL=${LB_MODEL:-/root/models/SmolLM2-360M}
test -f "$MODEL/model.safetensors"
OUT=$(mktemp -d "$PWD/bench/results/profile-compare-XXXXXXXX")
exec > >(tee "$OUT/job.log") 2>&1
echo "RESULT_DIRECTORY=$OUT"
echo "source=$MOJOLEARN_COMMIT"
source tools/lowbit_mma_speed/box_env.sh
box_describe
pixi install -e default
for binding in build build_linalg build_training build_transformer; do
    sh "bindings/$binding.sh" > "$OUT/$binding.log" 2>&1 || {
        tail -40 "$OUT/$binding.log"; exit 1;
    }
done
find python/mojolearn -name '*.so' -exec sha256sum {} \; > "$OUT/binaries.sha256"
extra=()
if command -v rocminfo >/dev/null 2>&1; then extra=(--experimental-hip); fi
pixi run -e default python tools/lowbit_default/resident_gate.py \
    --model "$MODEL" --new 8 "${extra[@]}" > "$OUT/resident_gate.log" 2>&1 || {
    tail -50 "$OUT/resident_gate.log"; exit 1;
}
pixi run -e default python tools/profile_compare/run.py \
    --model "$MODEL" --rounds 7 --out "$OUT/comparison.json" "${extra[@]}"
cat "$OUT/resident_gate.log"
cat "$OUT/comparison.json"
