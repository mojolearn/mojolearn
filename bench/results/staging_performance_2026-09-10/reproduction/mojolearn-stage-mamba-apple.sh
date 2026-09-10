#!/bin/bash
set -euo pipefail
out=/tmp/mamba-stage-apple
mkdir -p "$out"
mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba3_check.mojo -o "$out/baseline" > "$out/build-baseline.log" 2>&1
for arm in baseline poison; do
 bin="$out/baseline"
 if [ "$arm" = poison ]; then bin=/tmp/mamba-scratch-poison-apple; fi
 MOJOLEARN_IDENTITY_TRACE="$out/default-$arm.trace" "$bin" > "$out/default-$arm.log" 2>&1
 MOJOLEARN_IDENTITY_TRACE="$out/long-$arm.trace" MOJOLEARN_MAMBA3_CHECK_B=2 MOJOLEARN_MAMBA3_CHECK_L=65 MOJOLEARN_MAMBA3_CHECK_DM=64 "$bin" > "$out/long-$arm.log" 2>&1
 for gate in decode-cross continuation refusal; do "$bin" "$gate" > "$out/$gate-$arm.log" 2>&1; done
done
cmp "$out/default-baseline.trace" "$out/default-poison.trace"
cmp "$out/long-baseline.trace" "$out/long-poison.trace"
