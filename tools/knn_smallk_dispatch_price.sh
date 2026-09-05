#!/usr/bin/env bash
# Main-only serial benchmark. Parent owns rental, overall timeout and fetching.
set -euo pipefail
cd "${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT=${MOJOLEARN_SMALLK_PRICE_OUT:?set output directory}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MOJOLEARN_NUMERIC_MODE=identical
started=$(date +%s)
deadline=$((started + 480))
run() {
    local name=$1 remaining
    shift
    remaining=$((deadline - $(date +%s)))
    if ((remaining < 1)); then printf '%s\t124\n' "$name" >> "$OUT/status.tsv"; return 124; fi
    local code=0
    timeout -k 10 "$remaining" "$@" > "$OUT/$name.log" 2>&1 || code=$?
    printf '%s\t%s\n' "$name" "$code" >> "$OUT/status.tsv"
    return "$code"
}
: > "$OUT/status.tsv"
run build-legacy pixi run mojo build -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_smallk_dispatch_price.mojo -o "$OUT/legacy"
run build-experimental pixi run mojo build -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1 -I . bench/knn_smallk_dispatch_price.mojo -o "$OUT/experimental"
sha256sum "$OUT/legacy" "$OUT/experimental" bench/knn_smallk_dispatch_price.mojo \
    bench/knn_smallk_price_fixture.mojo \
    neighbors/impl/neighbors/detail/knn_brute_force.mojo \
    neighbors/checks/select_smallk_identical_candidate.mojo > "$OUT/SHA256SUMS"
for queries in 32 128 1000; do
    for round in 0 1 2 3 4 5 6 7 8; do
        arms='legacy experimental'
        if ((round % 2)); then arms='experimental legacy'; fi
        for arm in $arms; do
            run "q${queries}-r${round}-${arm}" env MOJOLEARN_SMALLK_PRICE_QUERIES="$queries" "$OUT/$arm"
        done
    done
done
# Retain hashes, not machine-specific executables, in the fetched evidence.
rm "$OUT/legacy" "$OUT/experimental"
printf 'COMPLETE\n' > "$OUT/completion.txt"
