#!/bin/bash
set -euo pipefail
# Invoke under the pod's activated Pixi environment. Own source and immutable
# baseline source must be staged at the following independent paths.
cd /root/knn-width
outdir=/root/knn-width-results
mkdir -p "$outdir/bin"
mojo=mojo
"$mojo" --version > "$outdir/mojo-version.log" 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$outdir/gpu.csv"
for gate in lane_minimum_check knn_selector_long_rows_check; do
 "$mojo" run -j4 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 "neighbors/checks/$gate.mojo" > "$outdir/$gate.log" 2>&1
done
for arm in baseline candidate; do
 src=/root/knn-width
 if [[ $arm == baseline ]]; then src=/root/knn-width-baseline; fi
 "$mojo" build -j4 -I "$src" -D MOJOLEARN_NUMERIC_IDENTICAL=1 "$src/bench/knn_reference_price_main.mojo" -o "$outdir/bin/$arm" > "$outdir/build-$arm.log" 2>&1
done
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for pass in 1 2; do
 arms=(baseline candidate)
 if [[ $pass == 2 ]]; then arms=(candidate baseline); fi
 for arm in "${arms[@]}"; do
  for k in 10 15; do
   env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=7 "$outdir/bin/$arm" > "$outdir/$pass-$arm-k$k.log" 2>&1
  done
 done
done
printf 'PASS\n' > "$outdir/status.txt"
