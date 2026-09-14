#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=cvisjmryfcmz6g MOJOLEARN_NUMERIC_MODE=identical
cd /root/gradient-pool
mkdir -p /root/layer-pool-out
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > /root/layer-pool-out/hardware.csv
pixi run mojo run --target-accelerator sm_120 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/byte_lm_layer_pool_check.mojo > /root/layer-pool-out/native.log 2>&1
