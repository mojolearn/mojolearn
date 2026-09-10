#!/bin/bash
set -u
cd /root/knn-residual
out=/root/knn-residual-results
mkdir -p "$out/bin"
mojo=/root/mojolearn/.pixi/envs/default/bin/mojo
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/nvidia.csv"
"$mojo" --version > "$out/mojo-version.log" 2>&1
cp commit.txt "$out/commit.txt"
for arm in baseline redux flush both decode; do
 flags=(-D MOJOLEARN_NUMERIC_IDENTICAL=1)
 if [[ $arm == redux || $arm == both || $arm == decode ]]; then flags+=(-D MOJOLEARN_KNN_IDENTICAL_REDUX=1); fi
 if [[ $arm == flush || $arm == both || $arm == decode ]]; then flags+=(-D MOJOLEARN_KNN_IDENTICAL_HARDWARE_FLUSH=1); fi
 if [[ $arm == decode ]]; then flags+=(-D MOJOLEARN_KNN_IDENTICAL_DECODE_VALUE=1); fi
 "$mojo" build -j 4 -I . "${flags[@]}" bench/knn_reference_price_main.mojo -o "$out/bin/$arm" > "$out/build-$arm.log" 2>&1
 rc=$?; echo "build-$arm $rc" >> "$out/status.tsv"
 if ((rc)); then continue; fi
 for k in 10 15; do
  env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=3 "$out/bin/$arm" > "$out/$arm-k$k.log" 2>&1
  echo "price-$arm-k$k $?" >> "$out/status.tsv"
 done
 if [[ $arm == decode ]]; then
  "$mojo" run -I . "${flags[@]}" neighbors/checks/knn_selector_long_rows_check.mojo > "$out/long-rows.log" 2>&1
  echo "long-rows $?" >> "$out/status.tsv"
  python3 tools/knn_zero_fma_oracle.py > "$out/oracle-gen.log" 2>&1
  "$mojo" run -I . "${flags[@]}" neighbors/checks/zero_fma_candidate_check.mojo > "$out/oracle.log" 2>&1
  echo "oracle $?" >> "$out/status.tsv"
 fi
done
echo done > "$out/done"
