#!/usr/bin/env bash
# Main-operator-only serial correctness payload. Parent owns lease/teardown.
set -uo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
out=${MOJOLEARN_CONTINUED_OUT:?set a fresh artifact directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
[ ! -e "$out/status.tsv" ] || { echo 'refusing to overwrite evidence'; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$ > "$out/affinity.log" 2>&1 || exit 2
printf '%s\n' "${MOJOLEARN_COMMIT:?exact frozen source required}" > "$out/commit.txt"
deadline=$(( $(date +%s) + ${MOJOLEARN_CONTINUED_SECONDS:-1500} ))
rc=0
run() {
    local name=$1 remaining code=0
    shift
    remaining=$((deadline - $(date +%s)))
    if (( remaining < 1 )); then code=124
    else timeout -k 10 "$remaining" "$@" > "$out/$name.log" 2>&1 || code=$?; fi
    printf '%s\t%s\n' "$name" "$code" >> "$out/status.tsv"
    if (( code != 0 )); then rc=1; fi
    return "$code"
}
if run ordered-build pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/ordered_rmse_check.mojo -o "$out/ordered-check"; then
    run ordered-check env MOJOLEARN_IDENTITY_TRACE="$out/ordered.card" "$out/ordered-check"
fi
if run ctr-build pixi run mojo build -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/tree_ctr_slice_check.mojo -o "$out/ctr-check"; then
    run ctr-check "$out/ctr-check"
fi
for arm in baseline selector transpose both; do
    flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
    # Since 2026-09-09 the two rows are kernel-matrix defaults (on for
    # NVIDIA/AMD, off for Apple), so every arm names BOTH rows explicitly
    # and the same four binaries mean the same thing on every column.
    case "$arm" in
        selector|both) flags+=(-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1);;
        *) flags+=(-D MOJOLEARN_KNN_IDENTICAL_LEGACY_SELECT=1);;
    esac
    case "$arm" in
        transpose|both) flags+=(-D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1);;
        *) flags+=(-D MOJOLEARN_KNN_IDENTICAL_LEGACY_LAYOUT=1);;
    esac
    if run "knn-build-$arm" pixi run mojo build -j 2 -I . "${flags[@]}" bench/knn_layout_adversarial_check.mojo -o "$out/knn-$arm"; then
        run "knn-$arm" "$out/knn-$arm"
    fi
done
rm -f "$out/ordered-check" "$out/ctr-check" "$out/knn-baseline" "$out/knn-selector" "$out/knn-transpose" "$out/knn-both"
printf 'continued_exit=%s\n' "$rc" > "$out/completion.txt"
exit "$rc"
