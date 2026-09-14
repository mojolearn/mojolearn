#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
sh bindings/build_training.sh > /root/optimizer-pool-out/build-retained.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_training.so > /root/optimizer-pool-out/retained-binary.sha256
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_optimizer_check.py --cloud --report /root/optimizer-pool-out/report-retained.json > /root/optimizer-pool-out/public-retained.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane mlp --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/mlp.json > /root/optimizer-pool-out/mlp.log 2>&1
