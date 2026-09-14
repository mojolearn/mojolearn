#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/pointwise_parallel_check.mojo > /root/pointwise-out/native.log 2>&1
