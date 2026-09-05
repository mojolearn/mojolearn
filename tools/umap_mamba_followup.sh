#!/usr/bin/env bash
# Serial source/API qualification after the native backward certificate.
# No rental or wheel-publication actions. Parent must bound this whole job.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${MOJOLEARN_FOLLOWUP_OUT:?set an artifact directory}
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
export MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH="$ROOT/python"
rc=0
run() {
    local name=$1
    shift
    "$@" > "$OUT/$name.log" 2>&1
    local status=$?
    printf '%s\t%s\n' "$name" "$status" | tee -a "$OUT/results.tsv"
    [ "$status" = 0 ] || rc=1
    return "$status"
}
: > "$OUT/results.tsv"
run umap-broader pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . umap/checks/identity_broader_check.mojo
if run build-metrics bash bindings/build_metrics.sh; then
    run umap-api pixi run -e skgpu python python/mojolearn/tests/test_umap_surface.py
    run umap-transform-api pixi run -e skgpu python python/mojolearn/tests/test_umap_transform.py
    run umap-transform-quality pixi run -e skgpu python tools/umap_transform_quality_check.py --mode identical --device "${MOJOLEARN_MAMBA_CERT_VENDOR:-remote}" --output "$OUT/transform-quality.json"
    run umap-quality pixi run -e skgpu python tools/umap_quality_check.py --mode identical --device "${MOJOLEARN_MAMBA_CERT_VENDOR:-remote}" --output "$OUT/quality.json"
fi
run umap-transform-native pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . umap/checks/transform_check.mojo
run umap-sparse-graph pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . umap/checks/sparse_graph_check.mojo
run umap-sparse-fit pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . umap/checks/sparse_estimator_check.mojo
run umap-sparse-fast env MOJOLEARN_UMAP_SPARSE_FAST_LARGE=1 pixi run mojo run -j 4 -I . umap/checks/sparse_estimator_check.mojo
run umap-sparse-deterministic pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_DETERMINISTIC=1 -I . umap/checks/sparse_estimator_check.mojo
run knn-dispatch-legacy pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_smallk_dispatch_check.mojo
run knn-dispatch-experimental pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_EXPERIMENTAL_SMALLK_IDENTICAL=1 -I . bench/knn_smallk_dispatch_check.mojo
# Qualify the complete public UMAP fit/transform surface in every numeric mode.
# IDENTICAL was built and checked above; the other builds run one at a time.
for mode in fast deterministic; do
    if run "build-metrics-$mode" env MOJOLEARN_NUMERIC_MODE="$mode" bash bindings/build_metrics.sh; then
        run "umap-api-$mode" env MOJOLEARN_NUMERIC_MODE="$mode" pixi run -e skgpu python -m unittest discover -s python/mojolearn/tests -p 'test_umap*.py'
        run "umap-transform-quality-$mode" env MOJOLEARN_NUMERIC_MODE="$mode" pixi run -e skgpu python tools/umap_transform_quality_check.py --mode "$mode" --device "${MOJOLEARN_MAMBA_CERT_VENDOR:-remote}" --output "$OUT/transform-quality-$mode.json"
    fi
done
# The independent corpus generator is shipped; fixture bytes are generated
# on the remote host rather than silently borrowed from an unrelated build.
if run corpus pixi run -e skgpu python mamba/corpus/gen_corpus.py --family all; then
    if run build-mamba bash bindings/build_mamba.sh; then
        run mamba-api pixi run -e skgpu python python/mojolearn/tests/test_mamba_surface.py
    fi
fi
run knn-public-price env MOJOLEARN_SMALLK_PRICE_OUT="$OUT/knn-public-price" bash tools/knn_smallk_dispatch_price.sh
printf '%s\n' "$rc" > "$OUT/exit_code"
exit "$rc"
