#!/bin/bash
set -euo pipefail
out=/root/jobs/gemm-fragment
"$out/boundary" > "$out/boundary.log" 2>&1
"$out/device" > "$out/device.log" 2>&1
export MOJOLEARN_GEMM_BASELINE_PLAN=-2 MOJOLEARN_SPEED_ROUNDS=7
export MOJOLEARN_SPEED_SHAPES=gram.32x32x1M,gram.128sq.x100003,llama8b.qkv.t512,llama8b.mlp_up.t512,llama8b.mlp_down.t512,pca.transform.wide.8192x64x128
for arm in baseline admitted; do "$out/$arm" > "$out/$arm-price.log" 2>&1; done
