#!/bin/bash
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mamba3
mkdir -p /root/jobs/m3-results
pixi install > /root/jobs/m3-results/install.log 2>&1
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba3_check.mojo -o /root/jobs/m3-check > /root/jobs/m3-results/build-check.log 2>&1
for gate in default decode-cross continuation; do
  if [ "$gate" = default ]; then /root/jobs/m3-check > /root/jobs/m3-results/$gate.log 2>&1; else /root/jobs/m3-check "$gate" > /root/jobs/m3-results/$gate.log 2>&1; fi
done
for arm in baseline optimized; do
  extra=()
  if [ "$arm" = baseline ]; then extra=(-D MOJOLEARN_MAMBA3_LEGACY_STATEPASS=1); fi
  pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . mamba/checks/mamba3_statepass_price.mojo -o /root/jobs/m3-$arm > /root/jobs/m3-results/build-$arm.log 2>&1
  for length in 64 512 2048; do /root/jobs/m3-$arm "$length" > /root/jobs/m3-results/price-$arm-$length.log 2>&1; done
done
