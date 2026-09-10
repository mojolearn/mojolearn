#!/bin/bash
set -euo pipefail
cd /tmp/mojolearn-mamba3-next
out=/tmp/mojolearn-m3-increment-apple
mkdir -p "$out" python/mojolearn/identical
pixi_root=/Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/tmp/mojolearn-mamba3-next/python
for arm in baseline tiled; do
  extra=(-D MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE=1)
  if [ "$arm" = tiled ]; then extra=(-D MOJOLEARN_MAMBA3_TILED_INCREMENT=1); fi
  pixi run --manifest-path "$pixi_root" mojo build -j 2 -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . mamba/checks/mamba3_check.mojo -o "$out/native-$arm" > "$out/build-native-$arm.log" 2>&1
  MOJOLEARN_IDENTITY_TRACE="$out/default-$arm.trace" "$out/native-$arm" > "$out/default-$arm.log" 2>&1
  MOJOLEARN_IDENTITY_TRACE="$out/long-$arm.trace" MOJOLEARN_MAMBA3_CHECK_B=2 MOJOLEARN_MAMBA3_CHECK_L=65 MOJOLEARN_MAMBA3_CHECK_DM=64 "$out/native-$arm" > "$out/long-$arm.log" 2>&1
  for gate in decode-cross continuation refusal; do "$out/native-$arm" "$gate" > "$out/$gate-$arm.log" 2>&1; done
done
cmp "$out/default-baseline.trace" "$out/default-tiled.trace"
cmp "$out/long-baseline.trace" "$out/long-tiled.trace"
pixi run --manifest-path "$pixi_root" mojo build -j 2 --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_MAMBA3_TILED_INCREMENT=1 -I . -I bindings bindings/_mojolearn_mamba.mojo -o python/mojolearn/identical/_mojolearn_mamba.so > "$out/build-binding-tiled.log" 2>&1
pixi run --manifest-path "$pixi_root" python tools/mamba3_fresh_prefill_check.py > "$out/fresh-tiled.log" 2>&1
pixi run --manifest-path "$pixi_root" python python/mojolearn/tests/test_mamba_surface.py > "$out/surface-tiled.log" 2>&1
