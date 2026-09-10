#!/bin/bash
set -euo pipefail
cd /root/gemm-tile-proof
out=/root/jobs/gemm-tile-proof
mkdir -p "$out"
for probe in gemm_tuned_probe gemm_device_check; do
 /root/.pixi/bin/pixi run --manifest-path /root/mojolearn/pixi.toml mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_TILE_PROOF=1 -I /root/gemm-tile-proof -I /root/mojolearn "gemm/checks/$probe.mojo" -o "$out/$probe" > "$out/$probe-build.log" 2>&1
done
