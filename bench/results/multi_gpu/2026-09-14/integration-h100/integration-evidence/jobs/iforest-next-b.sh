#!/bin/sh
set -eu
export RUNPOD_POD_ID=nlfngsvejhgiq5 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/iforest-out
mv /root/iforest-out/build.log /root/iforest-out/build-move-failed.log
sh bindings/build_svm.sh > /root/iforest-out/build.log 2>&1
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/iforest_parallel_check.mojo > /root/iforest-out/native-bitwise.log 2>&1
pixi run python tools/parallel_iforest_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/iforest-out/iforest.json > /root/iforest-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_svm.so > /root/iforest-out/binaries.sha256
