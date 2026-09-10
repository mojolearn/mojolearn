#!/bin/bash
set -euo pipefail
out=/tmp/mojolearn-knn-apple-admission-coverage
mkdir -p "$out"
for pass in 1 2; do
 if [[ $pass == 1 ]]; then arms=(baseline preflight); else arms=(preflight baseline); fi
 for shape in 8:32 8:1000 32:32; do
  d=${shape%:*}; q=${shape#*:}
  for arm in "${arms[@]}"; do
   if [[ $arm == baseline ]]; then binary=/tmp/mojolearn-knn-chunk4-apple/baseline-price; else binary=/tmp/mojolearn-knn-preflight-apple/preflight-price; fi
   env MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES="$q" MOJOLEARN_KNN_REF_FEATURES="$d" MOJOLEARN_KNN_REF_K=15 MOJOLEARN_KNN_REF_ROUNDS=5 "$binary" > "$out/pass$pass-d$d-q$q-$arm.log" 2>&1
  done
 done
done
