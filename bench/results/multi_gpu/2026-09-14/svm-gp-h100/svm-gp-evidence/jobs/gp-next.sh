#!/bin/sh
set -eu
export RUNPOD_POD_ID=ok7m278wjgyo04 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/gp-out
tar -xzf /root/gp-overlay.tgz
sha256sum /root/gp-overlay.tgz > /root/gp-out/source.sha256
bash bindings/build_gp.sh > /root/gp-out/build.log 2>&1
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/gp_parallel_check.mojo > /root/gp-out/native-bitwise.log 2>&1
pixi run python tools/parallel_gp_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/gp-out/gp.json > /root/gp-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_gp.so > /root/gp-out/binaries.sha256
