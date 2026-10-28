#!/bin/bash
# Compile-only: the actual neighbors public owner root (neighbors_operation),
# covering neighbors_native/semi/graph/knn/ann and the new feature-name provider.
set -u
WT=/Users/andrewhendel/mojolearn-wt/purge-neighbors-20261028
OUT=${1:-/Users/andrewhendel/mojolearn-evidence/purge-neighbors-20261028/attempt1}
MOJO=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo
mkdir -p "$OUT"
cd "$WT" || exit 2
git rev-parse HEAD > "$OUT/source-commit.txt"
git status --short > "$OUT/source-status.txt"
"$MOJO" --version > "$OUT/compiler.txt" 2>&1
shasum -a 256 "$MOJO" > "$OUT/compiler.sha256"
start=$(date +%s)
pixi run --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml --environment default --locked --no-install \
  --executable "$MOJO" build --optimization-level 0 -j 10 --emit shared-lib --target-cpu apple-m1 \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I "$WT" -I "$WT/bindings" \
  "$WT/experiments/provider_radius_20261027/compile_radius.mojo" -o "$OUT/neighbors_public.so" > "$OUT/compile.log" 2>&1
rc=$?
echo "exit=$rc seconds=$(( $(date +%s) - start ))" > "$OUT/result.txt"
[ -f "$OUT/neighbors_public.so" ] && shasum -a 256 "$OUT/neighbors_public.so" > "$OUT/binary.sha256"
exit $rc
