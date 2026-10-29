#!/bin/bash
# Focused identical CPU compile only; never execute either artifact.
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
EV=${1:?evidence directory}
ROOT=${2:?compile_native or compile_boundary}
ATTEMPT=${3:?attempt number}
MAIN=/Users/andrewhendel/CascadeProjects/mojolearn
mkdir -p "$EV"
cd "$WT" || exit 2
bash tools/compile_slot.sh pixi run --manifest-path "$MAIN/pixi.toml" --locked --no-install --executable "$MAIN/.pixi/envs/default/bin/mojo" build -j 8 --emit shared-lib --optimization-level 0 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I "$WT" -I "$WT/bindings" "experiments/resident_generation_lifecycle_20261028/$ROOT.mojo" -o "$EV/$ROOT-$ATTEMPT.dylib" > "$EV/$ROOT-$ATTEMPT.log" 2>&1
result=$?
printf '%s\n' "$result" > "$EV/$ROOT-$ATTEMPT.exit"
exit "$result"
