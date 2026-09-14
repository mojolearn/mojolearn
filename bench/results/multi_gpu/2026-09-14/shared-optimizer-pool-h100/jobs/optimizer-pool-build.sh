#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
sh bindings/build_training.sh > /root/optimizer-pool-out/build.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_training.so > /root/optimizer-pool-out/training-binary.sha256
