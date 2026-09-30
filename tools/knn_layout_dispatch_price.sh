#!/usr/bin/env bash
# Main-only serial correctness and four-arm native-request measurement.
# Parent owns provisioning, the overall lease, collection and destruction.
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT=${MOJOLEARN_LAYOUT_PRICE_OUT:?set a fresh evidence directory}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
[ ! -e "$OUT/status.tsv" ] || { echo 'refusing to overwrite evidence'; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MOJOLEARN_NUMERIC_MODE=identical
deadline=$(( $(date +%s) + 1200 ))
run() {
    local name=$1 remaining code=0
    shift
    remaining=$((deadline - $(date +%s)))
    if ((remaining < 1)); then printf '%s\t124\n' "$name" >> "$OUT/status.tsv"; return 124; fi
    if command -v timeout >/dev/null 2>&1; then
        timeout -k 10 "$remaining" "$@" > "$OUT/$name.log" 2>&1 || code=$?
    else  # macOS ships no timeout(1); the deadline check above still bounds the run
        "$@" > "$OUT/$name.log" 2>&1 || code=$?
    fi
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/status.tsv"
    return "$code"
}
: > "$OUT/status.tsv"
arms=(baseline selector transpose both)
drivers=(check price)
if [[ ${MOJOLEARN_KNN_CHECKS_ONLY:-0} == 1 ]]; then drivers=(check); fi
for arm in "${arms[@]}"; do
    flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
    if [[ -n ${MOJOLEARN_KNN_DEFINES:-} ]]; then
        read -r -a extra_flags <<< "$MOJOLEARN_KNN_DEFINES"
        flags+=("${extra_flags[@]}")
    fi
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
    for driver in "${drivers[@]}"; do
        run "build-$driver-$arm" pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-4}" -I . \
            "${flags[@]}" "bench/knn_layout_dispatch_$driver.mojo" -o "$OUT/$driver-$arm"
    done
    run "check-$arm" "$OUT/check-$arm"
done
sha256sum "$OUT"/check-baseline "$OUT"/check-selector "$OUT"/check-transpose "$OUT"/check-both \
    bench/knn_layout_dispatch_check.mojo bench/knn_layout_dispatch_price.mojo \
    bench/knn_smallk_dispatch_check.mojo bench/knn_smallk_dispatch_price.mojo \
    bench/knn_smallk_dispatch_fixture.mojo bench/knn_smallk_price_fixture.mojo \
    neighbors/impl/detail/knn_brute_force.mojo \
    neighbors/checks/transposed_index_distance_candidate.mojo \
    neighbors/checks/select_smallk_identical_candidate.mojo > "$OUT/SHA256SUMS"
if [[ ${MOJOLEARN_KNN_CHECKS_ONLY:-0} != 1 ]]; then
sha256sum "$OUT"/price-baseline "$OUT"/price-selector "$OUT"/price-transpose "$OUT"/price-both >> "$OUT/SHA256SUMS"
for queries in 32 128 1000; do
    for round in 0 1 2 3 4 5 6 7 8; do
        for offset in 0 1 2 3; do
            arm=${arms[$(((round + offset) % 4))]}
            run "q${queries}-r${round}-${arm}" env MOJOLEARN_SMALLK_PRICE_QUERIES="$queries" "$OUT/price-$arm"
        done
    done
done
fi
for arm in "${arms[@]}"; do
    for driver in "${drivers[@]}"; do rm "$OUT/$driver-$arm"; done
done
printf 'COMPLETE\n' > "$OUT/completion.txt"
