#!/bin/bash
# Queued only on existing resources. No provisioning, no default changes.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_NUMERIC_PROFILE=fp32_v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/python"
source tools/lowbit_mma_speed/box_env.sh
OUT=$(mktemp -d "$PWD/bench/results/lowbit-e2e-XXXXXXXX")
exec > >(tee "$OUT/job.log") 2>&1
if [ -f .lowbit_e2e_commit ]; then
    export MOJOLEARN_COMMIT=$(< .lowbit_e2e_commit)
else
    export MOJOLEARN_COMMIT=$(git rev-parse HEAD)
fi
printf 'source_commit=%s\n' "$MOJOLEARN_COMMIT"
box_describe
pixi install -e default
pixi install -e test
sh bindings/build_linalg.sh
# This is a new public-call diagnostic, not a rerun of the lane's gate suite.
pixi run -e test python tools/lowbit_e2e/public_probe.py --out "$OUT/public_api.json"
printf 'RESULT_DIRECTORY=%s\n' "$OUT"
cat "$OUT/public_api.json"
