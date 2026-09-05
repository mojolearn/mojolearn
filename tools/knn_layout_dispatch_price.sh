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
    timeout -k 10 "$remaining" "$@" > "$OUT/$name.log" 2>&1 || code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/status.tsv"
    return "$code"
}
: > "$OUT/status.tsv"
arms=(baseline selector transpose both)
for arm in "${arms[@]}"; do
    flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
    case "$arm" in selector|both) flags+=(-D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1);; esac
    case "$arm" in transpose|both) flags+=(-D MOJOLEARN_EXPERIMENTAL_KNN_TRANSPOSE_IDENTICAL=1);; esac
    for driver in check price; do
        run "build-$driver-$arm" pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-4}" -I . \
            "${flags[@]}" "bench/knn_layout_dispatch_$driver.mojo" -o "$OUT/$driver-$arm"
    done
    run "check-$arm" "$OUT/check-$arm"
done
sha256sum "$OUT"/check-baseline "$OUT"/check-selector "$OUT"/check-transpose "$OUT"/check-both \
    "$OUT"/price-baseline "$OUT"/price-selector "$OUT"/price-transpose "$OUT"/price-both \
    bench/knn_layout_dispatch_check.mojo bench/knn_layout_dispatch_price.mojo \
    bench/knn_smallk_dispatch_check.mojo bench/knn_smallk_dispatch_price.mojo \
    neighbors/impl/neighbors/detail/knn_brute_force.mojo \
    neighbors/checks/transposed_index_distance_candidate.mojo \
    neighbors/checks/select_smallk_identical_candidate.mojo > "$OUT/SHA256SUMS"
for queries in 32 128 1000; do
    for round in 0 1 2 3 4 5 6 7 8; do
        for offset in 0 1 2 3; do
            arm=${arms[$(((round + offset) % 4))]}
            run "q${queries}-r${round}-${arm}" env MOJOLEARN_SMALLK_PRICE_QUERIES="$queries" "$OUT/price-$arm"
        done
    done
done
for arm in "${arms[@]}"; do rm "$OUT/check-$arm" "$OUT/price-$arm"; done
printf 'COMPLETE\n' > "$OUT/completion.txt"
