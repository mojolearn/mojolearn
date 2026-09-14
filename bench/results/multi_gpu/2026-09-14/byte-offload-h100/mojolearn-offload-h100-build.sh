#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=6ewmfb4taf9u1q MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
mkdir -p /root/mojolearn /root/offload-out
cd /root/mojolearn
tar xzf /root/offload-source.tgz
cp /root/offload-source.tgz /root/offload-out/source-initial.tgz
sha256sum /root/offload-source.tgz > /root/offload-out/source-initial.sha256
export PYTHONPATH="$PWD/python"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > /root/offload-out/hardware.csv
sh bindings/build.sh > /root/offload-out/build-base.log 2>&1
sh bindings/build_byte_lm.sh > /root/offload-out/build-byte-initial.log 2>&1
pixi run mojo run --target-accelerator sm_90 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/byte_lm_offload_check.mojo > /root/offload-out/native-initial.log 2>&1
