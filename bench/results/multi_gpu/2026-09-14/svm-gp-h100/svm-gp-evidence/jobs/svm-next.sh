#!/bin/sh
set -eu
export RUNPOD_POD_ID=ok7m278wjgyo04 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/svm-out
pixi install > /root/svm-out/install.log 2>&1
sh bindings/build.sh > /root/svm-out/build-base.log 2>&1
sh bindings/build_svm.sh > /root/svm-out/build.log 2>&1
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/svm_parallel_check.mojo > /root/svm-out/native-bitwise.log 2>&1
pixi run python tools/parallel_svm_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/svm-out/svm.json > /root/svm-out/gate.log 2>&1
cp commit.txt /root/svm-out/source-commit.txt
sha256sum python/mojolearn/identical/_mojolearn_svm.so python/mojolearn/identical/_mojolearn.so > /root/svm-out/binaries.sha256
sha256sum training/corpus/enwik8/input.txt > /root/svm-out/corpus.sha256
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /root/svm-out/gpu.txt
pixi run mojo --version > /root/svm-out/toolchain.txt 2>&1
