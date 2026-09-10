#!/bin/bash
set -euo pipefail
cd /root/mojolearn
out=/root/jobs/gemm-residual
mkdir -p "$out"
export MOJOLEARN_SPEED_ROUNDS=7 MOJOLEARN_GEMM_BASELINE_PLAN=-2
export MOJOLEARN_SPEED_SHAPES=gram.32x32x1M,gram.128sq.x100003,llama8b.qkv.t512,llama8b.mlp_up.t512,llama8b.mlp_down.t512,pca.transform.wide.8192x64x128
for arm in full small; do
 flags=()
 if [ "$arm" = full ]; then flags=(-D MOJOLEARN_GEMM_FULL_FOLD_STACK=1); fi
 /root/.pixi/bin/pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${flags[@]}" -I . gemm/checks/gemm_tuned_probe.mojo -o "$out/$arm" > "$out/$arm-build.log" 2>&1
 "$out/$arm" > "$out/$arm-price.log" 2>&1
done
/root/.pixi/bin/pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo > "$out/device.log" 2>&1
