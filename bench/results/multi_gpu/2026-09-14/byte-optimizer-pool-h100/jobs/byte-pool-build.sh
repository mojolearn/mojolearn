#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_BYTE_LM_OUTDIR=/root/byte-pool-out/production
cd /root/mojolearn
sh bindings/build_byte_lm.sh > /root/byte-pool-out/build.log 2>&1
cp /root/byte-pool-out/production/_mojolearn_byte_lm.so python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum python/mojolearn/identical/_mojolearn_byte_lm.so > /root/byte-pool-out/binaries.sha256
