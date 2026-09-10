#!/bin/bash
set -u
outdir=/root/knn-width-results
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for d in 8 32 128; do
 for q in 32 1000; do
  for arm in candidate rows8; do
   env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=$q MOJOLEARN_KNN_REF_FEATURES=$d MOJOLEARN_KNN_REF_K=15 MOJOLEARN_KNN_REF_ROUNDS=5 "$outdir/bin/$arm" > "$outdir/coverage-$d-$q-$arm.log" 2>&1
  done
 done
done
for n in 100000 400000; do
 for q in 32 128 1000 4000; do
  for k in 10 15; do
   env MOJOLEARN_KNN_REF_INDEX=$n MOJOLEARN_KNN_REF_QUERIES=$q MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=7 "$outdir/bin/rows8" > "$outdir/grid-$n-$q-$k.log" 2>&1
  done
 done
done
echo done > "$outdir/coverage-done"
