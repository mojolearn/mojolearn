#!/usr/bin/env bash
# Main-lane serial AMD qualification; parent owns lease, timeout and fetch.
set -uo pipefail
cd "${REPO:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 MOJOLEARN_NUMERIC_MODE=identical
export PYTHONPATH="$REPO/python"
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$
rc=0
run() {
    local name=$1 cap=$2 status
    shift 2
    timeout -k 10 "$cap" "$@" > "$OUT/diag/$name.log" 2>&1
    status=$?
    printf '%s\t%s\n' "$name" "$status" | tee -a "$OUT/diag/optimization-status.tsv"
    [ "$status" = 0 ] || rc=1
    return "$status"
}
: > "$OUT/diag/optimization-status.tsv"
run umap-mamba 1200 bash tools/umap_mamba_do_diag.sh
run transformer-forward 300 env MOJOLEARN_IDENTITY_TRACE="$OUT/diag/transformer.forward.card" \
    pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_check.mojo
run transformer-backward 300 env MOJOLEARN_IDENTITY_TRACE="$OUT/diag/transformer.backward.card" \
    pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_backward_check.mojo
if run build-transformer 240 bash bindings/build_transformer.sh; then
    run transformer-api 180 pixi run -e skgpu python python/mojolearn/tests/test_transformer_surface.py
fi
run gemv-candidate 300 env MOJOLEARN_GEMV_LAYOUT_LARGE=1 \
    pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/gemv_serial_layout_main.mojo
run knn-candidate 300 env MOJOLEARN_KNN_LAYOUT_LARGE=1 \
    pixi run mojo run -j 4 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_index_layout_main.mojo
printf '%s\n' "$rc" > "$OUT/diag/optimization-exit-code"
exit "$rc"
