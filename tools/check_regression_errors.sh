#!/bin/sh
# Native and public regression errors in every numeric mode, one device at a time.
set -eu
cd "$(dirname "$0")/.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]; then
    exec tools/with_build_lock.sh sh tools/check_regression_errors.sh "$@"
fi
MOJOLEARN_PIPELINE_PYTHON=${MOJOLEARN_PYTHON:-python3}
export MOJOLEARN_PIPELINE_PYTHON
MOJOLEARN_PYTHON="$PWD/checks/pipeline_python.sh"
export MOJOLEARN_PYTHON
out=${1:-bench/results/regression_errors_local}
mkdir -p "$out" build/regression_errors
pixi run mojo --version > "$out/toolchain.txt"
for mode in fast deterministic identical; do
    set --
    if [ "$mode" = deterministic ]; then set -- -D MOJOLEARN_NUMERIC_DETERMINISTIC=1; fi
    if [ "$mode" = identical ]; then set -- -D MOJOLEARN_NUMERIC_IDENTICAL=1; fi
    echo "BUILD native $mode"
    pixi run mojo build -I . "$@" metrics/checks/regression_errors_check.mojo \
        -o "build/regression_errors/$mode" > "$out/$mode.build.log" 2>&1
    "build/regression_errors/$mode" > "$out/$mode.run.log" 2>&1
    echo "PASS native $mode"
    MOJOLEARN_NUMERIC_MODE=$mode sh bindings/build_metrics.sh > "$out/binding-$mode.build.log" 2>&1
    echo "BUILT public $mode"
done
PYTHONPATH=python "$MOJOLEARN_PYTHON" checks/regression_errors_binding.py > "$out/public.run.log" 2>&1
echo "PASS public interleaved modes"
