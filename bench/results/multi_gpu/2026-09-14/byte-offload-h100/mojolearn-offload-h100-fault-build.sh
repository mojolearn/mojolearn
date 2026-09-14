#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=6ewmfb4taf9u1q MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export MOJOLEARN_BYTE_LM_OUTDIR=/root/offload-fault
export MOJOLEARN_BUILD_EXTRA_DEFINES='-D MOJOLEARN_BYTE_POOL_FAULT_INJECT=1'
sh bindings/build_byte_lm.sh > /root/offload-out/build-fault.log 2>&1
sha256sum /root/offload-fault/_mojolearn_byte_lm.so > /root/offload-out/binaries-fault.sha256
