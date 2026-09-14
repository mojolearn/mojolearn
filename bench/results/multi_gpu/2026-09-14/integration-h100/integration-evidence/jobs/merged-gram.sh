#!/bin/sh
set -eu
export RUNPOD_POD_ID=nlfngsvejhgiq5 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/merge-out
pixi install > /root/merge-out/install.log 2>&1
sh bindings/build_estimators.sh > /root/merge-out/build.log 2>&1
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/gram_parallel_check.mojo > /root/merge-out/native-bitwise.log 2>&1
pixi run python tools/parallel_gram_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/merge-out/gram.json > /root/merge-out/gram.log 2>&1
pixi run python tools/parallel_logistic_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/merge-out/logistic.json > /root/merge-out/logistic.log 2>&1
cp commit.txt /root/merge-out/source-commit.txt
sha256sum python/mojolearn/identical/_mojolearn_estimators.so core/gram_splitk.mojo > /root/merge-out/hashes.sha256
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /root/merge-out/gpu.txt
