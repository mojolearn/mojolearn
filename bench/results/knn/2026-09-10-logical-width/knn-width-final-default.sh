#!/bin/bash
set -euo pipefail
cd /root/knn-width
outdir=/root/knn-width-results
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mojo build -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_reference_price_main.mojo -o "$outdir/bin/final-default" > "$outdir/build-final-default.log" 2>&1
for source in neighbors/checks/zero_fma_candidate_check.mojo bench/knn_index_layout_main.mojo; do
 name=$(basename "$source" .mojo)
 mojo run -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 "$source" > "$outdir/final-default-$name.log" 2>&1
done
for k in 10 15; do
 env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=7 "$outdir/bin/final-default" > "$outdir/final-default-k$k.log" 2>&1
done
echo done > "$outdir/final-default-done"
