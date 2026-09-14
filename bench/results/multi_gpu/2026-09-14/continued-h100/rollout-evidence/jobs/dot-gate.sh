#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/dot_parallel_check.mojo > /root/solver-out/native-bitwise.log 2>&1
