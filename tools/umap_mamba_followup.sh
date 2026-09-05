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
    run umap-quality pixi run -e skgpu python tools/umap_quality_check.py --mode identical --device "${MOJOLEARN_MAMBA_CERT_VENDOR:-remote}" --output "$OUT/quality.json"
fi
# The independent corpus generator is shipped; fixture bytes are generated
# on the remote host rather than silently borrowed from an unrelated build.
if run corpus pixi run -e skgpu python mamba/corpus/gen_corpus.py --family all; then
    if run build-mamba bash bindings/build_mamba.sh; then
        run mamba-api pixi run -e skgpu python python/mojolearn/tests/test_mamba_surface.py
    fi
fi
printf '%s\n' "$rc" > "$OUT/exit_code"
exit "$rc"
