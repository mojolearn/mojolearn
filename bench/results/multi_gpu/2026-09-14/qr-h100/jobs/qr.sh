#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
mkdir -p /root/qr-out
tar -xzf /root/qr-source.tgz
cp /root/qr-source.tgz /root/qr-out/
sh bindings/build_estimators.sh > /root/qr-out/build.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_estimators.so python/mojolearn/identical/_mojolearn.so > /root/qr-out/binaries.sha256
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_pca_full_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/qr-out/report.json > /root/qr-out/public.log 2>&1
