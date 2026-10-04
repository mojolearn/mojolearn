#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# kpca_quality_ab.sh "<defines B>": FAST x_decomp builds A (none) and B, a
# KernelPCA fit with each (tools/kpca_quality_ab.py), then the comparison. No timing.
set -u
DB=${1:--D MOJOLEARN_KPCA_FAST_LANCZOS_DEV}
here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
PY=
for p in .pixi/envs/test/bin/python .pixi/envs/default/bin/python python3; do
  if [ -x "$p" ] || command -v "$p" >/dev/null 2>&1; then
    "$p" -c 'import numpy' 2>/dev/null && { PY=$p; break; }
  fi
done
[ -n "$PY" ] || { echo "KPCAQ no python with numpy"; exit 1; }
out=$HOME/afc-def/kpcaq; mkdir -p "$out"
# KernelPCA's kernel matrix comes from the x_neighbors binding: one FAST build
# of it (no defines) serves both arms
if [ ! -f python/mojolearn/_mojolearn_x_neighbors.so ]; then
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 bash bindings/build_x_neighbors.sh > "$out/build_xn.log" 2>&1
  rc=$?; echo "KPCAQ-BUILD x_neighbors rc=$rc"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_xn.log" | cut -c1-300; exit 1; }
fi
for arm in A B; do
  D=""; [ $arm = B ] && D=$DB
  MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS="$D" MOJOLEARN_SKIP_BUILD_GATE=1 \
    bash bindings/build_x_decomp.sh > "$out/build_$arm.log" 2>&1
  rc=$?; echo "KPCAQ-BUILD arm=$arm rc=$rc defines='$D'"
  [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 -i error "$out/build_$arm.log" | cut -c1-300; exit 1; }
  MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python "$PY" tools/kpca_quality_ab.py fit "$out/$arm.npz" || exit 1
done
"$PY" tools/kpca_quality_ab.py compare "$out/A.npz" "$out/B.npz"
