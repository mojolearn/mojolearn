#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/gram-out
sh bindings/build_estimators.sh > /root/gram-out/build.log 2>&1
MOJOLEARN_GRAM_DEVICE_COUNT=2 pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . checks/gram_splitk_check.mojo > /root/gram-out/native.log 2>&1
pixi run python tools/parallel_gram_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/gram-out/gram.json > /root/gram-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_estimators.so > /root/gram-out/binaries.sha256
