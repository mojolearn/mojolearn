#!/bin/sh
# Compile and exercise native + installed public GPU entry points in all modes.
main() {
set -eu
cd "$(dirname "$0")/.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]; then
    exec tools/with_build_lock.sh sh tools/check_min_child_hessian.sh "$@"
fi
MOJOLEARN_HESSIAN_PYTHON=${MOJOLEARN_PYTHON:-python3}
export MOJOLEARN_HESSIAN_PYTHON
MOJOLEARN_PYTHON="$PWD/checks/min_child_hessian_python.sh"
export MOJOLEARN_PYTHON
out=${1:-bench/results/min_child_hessian_2026-09-10}
mkdir -p "$out" build/min_child_hessian
pixi run mojo --version > "$out/toolchain.txt"
for mode in fast deterministic identical; do
    set --
    if [ "$mode" = deterministic ]; then set -- -D MOJOLEARN_NUMERIC_DETERMINISTIC=1; fi
    if [ "$mode" = identical ]; then set -- -D MOJOLEARN_NUMERIC_IDENTICAL=1; fi
    echo "BUILD native $mode"
    pixi run mojo build -I . "$@" checks/min_child_hessian_check.mojo \
        -o "build/min_child_hessian/$mode" > "$out/$mode.build.log" 2>&1
    "build/min_child_hessian/$mode" > "$out/$mode.run.log" 2>&1
    echo "PASS native $mode"
    MOJOLEARN_NUMERIC_MODE=$mode sh bindings/build_gbdt.sh > "$out/binding-$mode.build.log" 2>&1
    PYTHONPATH=python "$MOJOLEARN_PYTHON" checks/min_child_hessian_binding.py --mode "$mode" > "$out/binding-$mode.run.log" 2>&1
    echo "PASS public $mode"
done
}
main "$@"
