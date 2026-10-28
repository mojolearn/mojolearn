#!/bin/bash
# Compile-only (no runtime): one entry of this lane's affected roots.
# usage: compile-command.sh <entry.mojo> <out-dir> <name>
set -u
WT=/Users/andrewhendel/mojolearn-wt/lifecycle-neighbors-cluster-20261028
ENTRY=$1
OUT=$2
NAME=$3
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
  "$WT/$ENTRY" -o "$OUT/$NAME.so" > "$OUT/compile.log" 2>&1
rc=$?
echo "exit=$rc seconds=$(( $(date +%s) - start ))" > "$OUT/result.txt"
[ -f "$OUT/$NAME.so" ] && shasum -a 256 "$OUT/$NAME.so" > "$OUT/binary.sha256"
exit $rc
