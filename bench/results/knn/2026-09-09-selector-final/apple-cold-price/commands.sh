#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
out=/tmp/mojolearn-knn-cold-apple
mkdir -p "$out"
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . neighbors/checks/knn_distance_fma_boundary_check.mojo > "$out/boundary.log" 2>&1
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . neighbors/checks/zero_fma_candidate_check.mojo > "$out/oracle.log" 2>&1
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/knn_reference_price_main.mojo -o "$out/ref" > "$out/build.log" 2>&1
MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=1000 MOJOLEARN_KNN_REF_K=15 MOJOLEARN_KNN_REF_ROUNDS=3 "$out/ref" > "$out/default.log" 2>&1
MOJOLEARN_KNN_REF_INDEX=400000 MOJOLEARN_KNN_REF_QUERIES=1000 MOJOLEARN_KNN_REF_K=15 MOJOLEARN_KNN_REF_ROUNDS=3 /tmp/mojolearn-knn-repaired-apple/ref-before > "$out/before.log" 2>&1
