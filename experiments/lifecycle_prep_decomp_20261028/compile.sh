#!/bin/bash
# Usage: compile.sh <root.mojo> <name> [evidence-subdir]
# Compiles one root of this worktree on the Mac through the compile slot (CPU column, identical, O0, -j 10)
# with the main checkout's pixi env; writes log, binary and a one-line JSON receipt line.
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=$1; NAME=$2; SUB=${3:-compile}
E=$HOME/mojolearn-evidence/lifecycle-prep-decomp-20261028/$SUB
mkdir -p "$E"
cd "$WT" || exit 2
SRC=$(git rev-parse HEAD); DIRTY=$(git status --porcelain -- '*.mojo' | wc -l | tr -d ' ')
MOJO=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo
START=$(date +%s)
bash "$HOME/mojolearn-evidence/compile_slot.sh" pixi run --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml \
  --environment default --locked --no-install --executable "$MOJO" build --optimization-level 0 -j 10 --emit shared-lib \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I "$WT" -I "$WT/bindings" "$ROOT" -o "$E/$NAME.so" > "$E/$NAME.log" 2>&1
RC=$?
SECS=$(( $(date +%s) - START ))
SHA=$( [ -f "$E/$NAME.so" ] && shasum -a 256 "$E/$NAME.so" | cut -d' ' -f1 || echo none)
VER=$("$MOJO" --version 2>/dev/null | head -1)
printf '{"name":"%s","entry":"%s","source_commit":"%s","dirty_mojo_files":%s,"exit":%s,"seconds":%s,"jobs":10,"compiler":"%s","binary_sha256":"%s","log":"%s","command":"compile_slot pixi mojo build -O0 -j 10 --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings %s"}\n' \
  "$NAME" "$ROOT" "$SRC" "$DIRTY" "$RC" "$SECS" "$VER" "$SHA" "$E/$NAME.log" "$ROOT" | tee -a "$E/receipts.jsonl"
exit $RC
