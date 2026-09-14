#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
sh bindings/build_gbdt.sh > /root/pointwise-out/gbdt-build.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn.so python/mojolearn/identical/_mojolearn_gbdt.so > /root/pointwise-out/binaries.sha256
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_pointwise_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/pointwise-out/report.json > /root/pointwise-out/public.log 2>&1
