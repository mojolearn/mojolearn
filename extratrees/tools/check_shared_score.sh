#!/usr/bin/env bash
# Full-forest and actual probability-bit A/B in all numeric modes.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]]; then
    exec tools/with_build_lock.sh bash "$0" "$@"
fi
out=${1:-bench/results/et_shared_score}
mkdir -p "$out"
for mode in fast deterministic identical; do
    defines=()
    expected=FAST
    if [[ "$mode" == deterministic ]]; then
        defines+=(-D MOJOLEARN_NUMERIC_DETERMINISTIC=1)
        expected=DETERMINISTIC
    elif [[ "$mode" == identical ]]; then
        defines+=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
        expected=IDENTICAL
    fi
    for arm in baseline candidate; do
        arm_defines=()
        shared=0
        arm_defines+=(-D MOJOLEARN_ET_NO_SHARED_CLASS_COUNTS=1)
        if [[ "$arm" == candidate ]]; then
            arm_defines=(-D MOJOLEARN_ET_SHARED_CLASS_COUNTS=1)
            shared=15
        fi
        binary="/tmp/et-shared-$mode-$arm"
        pixi run mojo build -I . "${defines[@]}" "${arm_defines[@]}" \
            extratrees/checks/shared_score_fingerprint.mojo -o "$binary" \
            > "$out/$mode-$arm.build.log" 2>&1
        "$binary" > "$out/$mode-$arm.log"
        grep -Fx "numeric_mode $expected" "$out/$mode-$arm.log"
        grep -Fx "shared_class_counts_mask $shared" "$out/$mode-$arm.log"
        grep '^fingerprint ' "$out/$mode-$arm.log" > "$out/$mode-$arm.fingerprints"
        test "$(wc -l < "$out/$mode-$arm.fingerprints")" -eq 27
    done
    diff -u "$out/$mode-baseline.fingerprints" "$out/$mode-candidate.fingerprints"
    pixi run mojo build -I . "${defines[@]}" -D MOJOLEARN_ET_SHARED_CLASS_COUNTS=1 \
        extratrees/checks/score_kernel_check.mojo -o "/tmp/et-shared-$mode-score" \
        > "$out/$mode-score.build.log" 2>&1
    "/tmp/et-shared-$mode-score" > "$out/$mode-score.log"
    echo "PASS shared integer score $mode:27 complete forests+probabilities and210-cell kernel oracle/sabotages"
done
