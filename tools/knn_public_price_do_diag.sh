#!/usr/bin/env bash
# Main-controller phase-9 payload only; no lease creation, bootstrap or fetch.
# Requires a pinned source checkout/archive, installed default pixi environment,
# Linux taskset/timeout/python3/sha256sum, and a working AMD Mojo GPU runtime.
# e2_remote_leg.sh knobs: MOJOLEARN_E1_PHASES=9,
# MOJOLEARN_P9_ONLY_DIAG=1,
# MOJOLEARN_P9_DIAG=tools/knn_public_price_do_diag.sh,
# MOJOLEARN_P9_DIAG_TIMEOUT=510. The child pricing budget is 480 seconds,
# including its two builds; timeout may use another 10 seconds to kill it.
# The bounded archive profile must explicitly include THIS new script.
set -euo pipefail
cd "${REPO:?controller must provide pinned REPO}"
REPO=$(pwd)
export REPO
mkdir -p "${OUT:?controller must provide artifact OUT}/diag"
diag=$(cd "$OUT/diag" && pwd)
finish() {
    local code=$?
    printf 'knn-public-price\t%s\n' "$code" > "$diag/knn-public-price-status.tsv"
    printf '%s\n' "$code" > "$diag/knn-public-price-exit-code"
}
trap finish EXIT

# Preserve prior partial evidence: a rerun must use a fresh controller OUT.
export MOJOLEARN_SMALLK_PRICE_OUT="$diag/knn-public-price"
mkdir "$MOJOLEARN_SMALLK_PRICE_OUT"
commit=${MOJOLEARN_COMMIT:?controller must provide exact source commit}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'invalid source commit'; exit 2; }
if [ -d .git ]; then
    actual=$(git rev-parse HEAD)
    [ "$actual" = "$commit" ] || { echo 'source HEAD mismatch'; exit 2; }
    git diff --quiet HEAD -- || { echo 'tracked source modifications'; exit 2; }
else
    # The controller verifies the archive hash before unpacking and writes
    # commit.txt; the archive itself intentionally contains no .git directory.
    [ "$(cat commit.txt)" = "$commit" ] || { echo 'archive commit mismatch'; exit 2; }
fi
printf '%s\n' "$commit" > "$MOJOLEARN_SMALLK_PRICE_OUT/commit.txt"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 MOJOLEARN_NUMERIC_MODE=identical
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')
[ -n "$cores" ] || { echo 'no allowed CPU affinity'; exit 2; }
taskset -pc "$cores" $$ > "$diag/knn-public-price-affinity.log" 2>&1
printf 'commit=%s\ncpu_affinity=%s\nprice_budget_seconds=480\nscope=native-public-upload-search-download-synchronize\n' \
    "$commit" "$cores" > "$MOJOLEARN_SMALLK_PRICE_OUT/controller-metadata.txt"
sha256sum tools/knn_public_price_do_diag.sh tools/knn_smallk_dispatch_price.sh \
    bench/knn_smallk_dispatch_price.mojo bench/knn_smallk_price_fixture.mojo pixi.toml pixi.lock \
    > "$MOJOLEARN_SMALLK_PRICE_OUT/payload-SHA256SUMS"

# Serial child owns both builds and all 54 rotating invocations. Retain its
# original status.tsv, logs, hashes and completion marker without rewriting.
# Do not call the broader optimization, Mamba, UMAP or transformer payloads.
timeout -k 10 480 bash tools/knn_smallk_dispatch_price.sh \
    > "$diag/knn-public-price.log" 2>&1
