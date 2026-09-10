#!/bin/bash
set -u
cd /root/knn-residual
out=/root/knn-residual-results
mojo=/root/mojolearn/.pixi/envs/default/bin/mojo
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for k in 10 15; do
 env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=7 "$out/bin/pristine-identical" > "$out/pristine-k$k.log" 2>&1
done
"$mojo" build -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_reference_price_main.mojo -o "$out/bin/final" > "$out/final-build.log" 2>&1 || exit 1
for k in 10 15; do
 env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=7 "$out/bin/final" > "$out/final-k$k.log" 2>&1
done
for src in zero_fma_candidate_check pinned_distance_layout_check knn_selector_long_rows_check; do
 "$mojo" run -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 neighbors/checks/$src.mojo > "$out/final-$src.log" 2>&1
 echo "$src $?" >> "$out/final-status.tsv"
done
echo done > "$out/final-done"
