#!/bin/bash
set -euo pipefail
cd /root/mamba3
out=/root/gemm-corrected-final
mkdir -p "$out"
nvidia-smi --query-gpu=name,driver_version --format=csv > "$out/gpu.txt"
/root/.pixi/bin/pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_tuned_probe.mojo -o "$out/probe" > "$out/build-probe.log" 2>&1
MOJOLEARN_GEMM_BASELINE_PLAN=-2 MOJOLEARN_SPEED_ROUNDS=7 MOJOLEARN_SPEED_SHAPES=gram.32x32x1M,gram.128sq.x100003,llama8b.qkv.t512,llama8b.mlp_up.t512,llama8b.mlp_down.t512 "$out/probe" > "$out/price.log" 2>&1
/root/.pixi/bin/pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/attention_fma_boundary_check.mojo -o "$out/boundary" > "$out/build-boundary.log" 2>&1
"$out/boundary" > "$out/boundary.log" 2>&1
