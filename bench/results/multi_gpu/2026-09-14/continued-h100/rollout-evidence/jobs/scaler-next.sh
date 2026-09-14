#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/scaler-out
sh bindings/build_preprocessing.sh > /root/scaler-out/build.log 2>&1
pixi run python tools/parallel_preprocessing_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/scaler-out/preprocessing.json > /root/scaler-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_preprocessing.so > /root/scaler-out/binaries.sha256
