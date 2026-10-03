#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# pca_quality_ab.sh "<defines B>": FAST estimators builds A (none) and B, a
# PCA fit with each (tools/pca_quality_ab.py), then the comparison. No timing.
set -u
DB=$1
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
PY=
for p in .pixi/envs/test/bin/python .pixi/envs/default/bin/python python3; do
  if [ -x "$p" ] || command -v "$p" >/dev/null 2>&1; then
    "$p" -c 'import numpy' 2>/dev/null && { PY=$p; break; }
  fi
done
[ -n "$PY" ] || { echo "PCAQ no python with numpy"; exit 1; }
out=$HOME/afc-def/pcaq; mkdir -p "$out"
for arm in A B; do
  D=""; [ $arm = B ] && D=$DB
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$D" MOJOLEARN_SKIP_BUILD_GATE=1 \
    bash bindings/build_estimators.sh > "$out/build_$arm.log" 2>&1
  rc=$?; echo "PCAQ-BUILD arm=$arm rc=$rc defines='$D'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_$arm.log" | cut -c1-300; exit 1; }
  MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python "$PY" tools/pca_quality_ab.py fit "$out/$arm.npz" || exit 1
done
"$PY" tools/pca_quality_ab.py compare "$out/A.npz" "$out/B.npz"
