#!/bin/bash
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd "${MOJOLEARN_MAMBA3_REPO:-/root/mamba3}"
mkdir -p /root/jobs/m3-results
# Native d_model=64 block proxy only, not the public September 7 grid.
# Exact public-grid reproduction commands and source snapshots are archived
# in bench/results/mamba3/2026-09-09-statepass/.
# Only our IDENTICAL arm is built; baseline is its previous execution plan.
pixi install > /root/jobs/m3-results/install.log 2>&1
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . mamba/checks/mamba3_check.mojo -o /root/jobs/m3-check > /root/jobs/m3-results/build-check.log 2>&1
for gate in default decode-cross continuation; do
  if [ "$gate" = default ]; then /root/jobs/m3-check > /root/jobs/m3-results/$gate.log 2>&1; else /root/jobs/m3-check "$gate" > /root/jobs/m3-results/$gate.log 2>&1; fi
done
for arm in baseline optimized; do
  extra=()
  if [ "$arm" = baseline ]; then extra=(-D MOJOLEARN_MAMBA3_LEGACY_STATEPASS=1 -D MOJOLEARN_MAMBA3_LEGACY_HOST_COPY=1 -D MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT=1 -D MOJOLEARN_MAMBA3_LEGACY_TRACE_SLICES=1 -D MOJOLEARN_MAMBA3_LEGACY_REFUSAL=1 -D MOJOLEARN_MAMBA3_LEGACY_YSTATE=1 -D MOJOLEARN_MAMBA3_LEGACY_QKS=1 -D MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V=1); fi
  pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . mamba/checks/mamba3_statepass_price.mojo -o /root/jobs/m3-$arm > /root/jobs/m3-results/build-$arm.log 2>&1
  for length in 64 512 2048; do /root/jobs/m3-$arm "$length" > /root/jobs/m3-results/price-$arm-$length.log 2>&1; done
done
