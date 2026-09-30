#!/bin/sh
# Focused local sampling build/smoke; no cross-vendor or throughput claim.
set -eu
cd "$(dirname "$0")/.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-0}" != 1 ]; then
    exec tools/with_build_lock.sh sh "$0" "$@"
fi
out=${1:-bench/results/feature_fraction_local}
mkdir -p "$out"
MOJOLEARN_PYTHON=${MOJOLEARN_PYTHON:-"$PWD/checks/pipeline_python.sh"}
export MOJOLEARN_PYTHON
for mode in fast deterministic identical; do
    MOJOLEARN_NUMERIC_MODE=$mode sh bindings/build_gbdt.sh > "$out/binding-$mode.build.log" 2>&1
done
pixi run mojo run -I . gbdt/checks/feature_sampling_check.mojo > "$out/native-fast.log" 2>&1
PYTHONPATH=python checks/pipeline_python.sh checks/gbdt_feature_fraction_binding.py > "$out/public.smoke.log" 2>&1
cat "$out/public.smoke.log"
