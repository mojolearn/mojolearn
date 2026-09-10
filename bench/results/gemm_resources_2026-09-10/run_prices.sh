#!/bin/bash
set -euo pipefail
cd /root/performance
export MOJOLEARN_SPEED_SHAPES=llama8b.qkv.t512,llama8b.mlp_up.t512,llama8b.mlp_down.t512
export MOJOLEARN_SPEED_ROUNDS=7
export MOJOLEARN_GEMM_BASELINE_PLAN=-2
mkdir -p /root/jobs/gemm-scalar-prices
nvidia-smi -q > /root/jobs/gemm-scalar-prices/gpu-before.txt
sha256sum /root/jobs/gemm-baseline /root/jobs/gemm-scalar gemm/checks/gemm_identical.mojo gemm/checks/gemm_tuned_probe.mojo > /root/jobs/gemm-scalar-prices/sha256.txt
for arm in baseline scalar scalar baseline; do
    seq=${seq:-0}
    timeout 180 /root/jobs/gemm-$arm > /root/jobs/gemm-scalar-prices/$seq-$arm.log 2>&1
    seq=$((seq + 1))
done
nvidia-smi -q > /root/jobs/gemm-scalar-prices/gpu-after.txt
