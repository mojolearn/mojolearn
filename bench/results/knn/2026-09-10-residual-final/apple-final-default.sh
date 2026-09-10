#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
out=/tmp/mojolearn-knn-final-default-apple
mkdir -p "$out"
for entry in neighbors/checks/zero_fma_candidate_check.mojo neighbors/checks/knn_distance_fma_boundary_check.mojo bench/knn_index_layout_main.mojo; do
 name=$(basename "$entry" .mojo)
 pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . "$entry" > "$out/$name.log" 2>&1
done
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build.sh > "$out/binding-build.log" 2>&1
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python pixi run python bench/results/knn/2026-09-09-selector-final/apple-public-binding/smoke.py > "$out/public-smoke.log" 2>&1
