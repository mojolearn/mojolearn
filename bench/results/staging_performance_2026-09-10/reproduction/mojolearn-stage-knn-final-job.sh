#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/knnfinal
while [ ! -f /root/jobs/final-stage.rc ]; do sleep 3; done
if [ "${KNN_FINAL_ACTIVE:-0}" != 1 ]; then
 export KNN_FINAL_ACTIVE=1
 exec pixi run bash /root/jobs/knn-final.sh
fi
out=/root/evidence/knn-final
mkdir -p "$out"
sha256sum neighbors/estimator.mojo neighbors/checks/pinned_distance_tile.mojo > "$out/source-sha256.txt"
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . neighbors/checks/query_batch_check.mojo > "$out/gate.log" 2>&1
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_layout_dispatch_check.mojo > "$out/public.log" 2>&1
awk '/^(DISPATCH_CELL|LAYOUT_CELL)/' "$out/public.log" > "$out/public.cells"
cmp "$out/public.cells" /root/evidence/knn-batch/baseline-public.cells
mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_reference_price_main.mojo -o "$out/price" > "$out/build-price.log" 2>&1
for k in 10 15; do
 MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=4000 MOJOLEARN_KNN_REF_FEATURES=32 MOJOLEARN_KNN_REF_K=$k MOJOLEARN_KNN_REF_ROUNDS=7 MOJOLEARN_KNN_REF_DUMP_FULL="$out/k$k.bin" "$out/price" > "$out/k$k.log" 2>&1
 cmp "$out/k$k.bin" "/root/evidence/knn-batch/n400000-q4000-d32-k$k-p0-baseline.bin"
done
printf 'PASS\n' > "$out/verdict.txt"
