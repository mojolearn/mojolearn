#!/bin/bash
set -u
cd /root/knn-residual
out=/root/knn-residual-results
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/root/mojolearn/.pixi/envs/default/bin/mojo run -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/knn_index_layout_main.mojo > "$out/final-layout.log" 2>&1
echo "layout $?" >> "$out/final-status.tsv"
for n in 100000 400000; do
 for q in 32 128 1000 4000; do
  for k in 10 15; do
   env MOJOLEARN_KNN_REF_INDEX=$n MOJOLEARN_KNN_REF_QUERIES=$q MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=7 "$out/bin/final" > "$out/grid-$n-$q-$k.log" 2>&1
   echo "grid-$n-$q-$k $?" >> "$out/final-status.tsv"
  done
 done
done
echo done > "$out/grid-done"
