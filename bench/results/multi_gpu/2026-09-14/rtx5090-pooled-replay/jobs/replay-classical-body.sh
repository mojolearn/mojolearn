#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=cvisjmryfcmz6g MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_120 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
for name in gbdt estimators; do
    sh bindings/build_${name}.sh > /root/replay-out/build-${name}.log 2>&1
done
sha256sum python/mojolearn/identical/*.so > /root/replay-out/binaries.sha256
pixi run mojo run --target-accelerator sm_120 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/pointwise_parallel_check.mojo > /root/replay-out/native-pointwise.log 2>&1
pixi run mojo run --target-accelerator sm_120 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/gram_outputs_parallel_check.mojo > /root/replay-out/native-gram.log 2>&1
pixi run mojo run --target-accelerator sm_120 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/qr_parallel_check.mojo > /root/replay-out/native-qr.log 2>&1
for name in pointwise ordered_rmse feature_freq boosting_adapters boosting gram_wide gram pca_full; do
    mkdir -p /root/replay-out/$name
    pixi run python tools/parallel_${name}_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/replay-out/$name/report.json > /root/replay-out/$name/gate.log 2>&1
done
