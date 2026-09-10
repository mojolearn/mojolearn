#!/bin/bash
set -euo pipefail
out=/root/jobs/gemm-tile-proof
"$out/gemm_device_check" > "$out/device.log" 2>&1
export MOJOLEARN_GEMM_BASELINE_PLAN=-2 MOJOLEARN_SPEED_ROUNDS=7 MOJOLEARN_SPEED_SHAPES=llama8b.qkv.t512,llama8b.mlp_up.t512,llama8b.mlp_down.t512
/root/jobs/gemm-residual/full > "$out/baseline-price.log" 2>&1
"$out/gemm_tuned_probe" > "$out/candidate-price.log" 2>&1
