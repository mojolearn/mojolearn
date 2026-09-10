#!/usr/bin/env bash
# Compare full model bits with the original 32-wide kernel in both modes.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]]; then
    exec tools/with_build_lock.sh bash "$0" "$@"
fi
out=$(mktemp -d "${TMPDIR:-/tmp}/et-acc-dispatch.XXXXXX")
trap 'rm -rf "$out"' EXIT
for mode in fast identical; do
    defines=()
    expected_mode=FAST
    if [[ "$mode" == identical ]]; then
        defines+=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
        expected_mode=IDENTICAL
    fi
    for arm in baseline dispatch; do
        arm_defines=()
        if [[ "$arm" == baseline ]]; then
            arm_defines+=(-D MOJOLEARN_ET_MAX_ACC_32=1)
        fi
        pixi run mojo build -I . "${defines[@]}" "${arm_defines[@]}" \
            extratrees/checks/accumulator_dispatch_fingerprint.mojo \
            -o "$out/$mode-$arm"
        "$out/$mode-$arm" > "$out/$mode-$arm.txt"
        grep -Fx "numeric_mode $expected_mode" "$out/$mode-$arm.txt"
    done
    diff -u "$out/$mode-baseline.txt" "$out/$mode-dispatch.txt"
    echo "PASS Extra Trees accumulator dispatch $mode: 18 full-forest fingerprints match width 32"
done
