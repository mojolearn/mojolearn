#!/bin/sh
# Focused local builds and public smoke. Does not qualify cross-vendor identity.
set -eu
cd "$(dirname "$0")/.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-0}" != 1 ]; then
    exec tools/with_build_lock.sh sh "$0" "$@"
fi
out=${1:-bench/results/standard_scaler_local}
mkdir -p "$out"
MOJOLEARN_PYTHON=${MOJOLEARN_PYTHON:-"$PWD/checks/pipeline_python.sh"}
export MOJOLEARN_PYTHON
for mode in fast deterministic identical; do
    MOJOLEARN_NUMERIC_MODE=$mode sh bindings/build_preprocessing.sh > "$out/binding-$mode.build.log" 2>&1
done
PYTHONPATH=python checks/pipeline_python.sh checks/standard_scaler_smoke.py > "$out/public.smoke.log" 2>&1
cat "$out/public.smoke.log"
