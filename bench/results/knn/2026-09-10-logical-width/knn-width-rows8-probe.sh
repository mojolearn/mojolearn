#!/bin/bash
set -u
cd /root/knn-width
outdir=/root/knn-width-results
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python3 tools/knn_zero_fma_oracle.py > "$outdir/rows8-oracle-gen.log" 2>&1
for source in neighbors/checks/zero_fma_candidate_check.mojo bench/knn_index_layout_main.mojo; do
 name=$(basename "$source" .mojo)
 mojo run -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_IDENTICAL_ROWS8=1 "$source" > "$outdir/rows8-$name.log" 2>&1
 printf '%s %s\n' "$name" "$?" >> "$outdir/rows8-status.txt"
done
mojo build -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_IDENTICAL_ROWS8=1 bench/knn_reference_price_main.mojo -o "$outdir/bin/rows8" > "$outdir/build-rows8.log" 2>&1 || exit 1
for k in 10 15; do
 for arm in candidate rows8; do
  env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=7 "$outdir/bin/$arm" > "$outdir/rows-price-$arm-k$k.log" 2>&1
 done
done
for arm in rows4 rows8; do
 flags=()
 if [[ $arm == rows8 ]]; then flags=(-D MOJOLEARN_KNN_IDENTICAL_ROWS8=1); fi
 mojo build -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_PHASE_TIMERS=1 "${flags[@]}" bench/knn_reference_price_main.mojo -o "$outdir/bin/phase-$arm" > "$outdir/build-phase-$arm.log" 2>&1 || continue
 env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_K=10 MOJOLEARN_KNN_REF_ROUNDS=3 "$outdir/bin/phase-$arm" > "$outdir/phase-$arm.log" 2>&1
done
printf 'done\n' > "$outdir/rows8-done"
