#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/iforest-pool-out
cp /root/iforest-pool-overlay.tgz /root/iforest-pool-out/
sha256sum /root/iforest-pool-overlay.tgz > /root/iforest-pool-out/source.sha256
tar xzf /root/iforest-pool-overlay.tgz
cp /root/neural-clip-out/hardware.csv /root/iforest-pool-out/
cp /root/neural-clip-out/corpus.sha256 /root/iforest-pool-out/
sh bindings/build_svm.sh > /root/iforest-pool-out/build.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_svm.so > /root/iforest-pool-out/binaries.sha256
while [ ! -f /root/jobs/samba-checkpoint.done ]; do sleep 5; done

pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/iforest_parallel_check.mojo > /root/iforest-pool-out/native.log 2>&1
pixi run python tools/parallel_iforest_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/iforest-pool-out/iforest.json > /root/iforest-pool-out/iforest.log 2>&1
pixi run python tools/parallel_svm_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/iforest-pool-out/svm.json > /root/iforest-pool-out/svm.log 2>&1
