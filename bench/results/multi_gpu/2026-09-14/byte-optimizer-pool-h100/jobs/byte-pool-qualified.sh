#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
export MOJOLEARN_BYTE_LM_OUTDIR=/root/byte-pool-out/qualified
sh bindings/build_byte_lm.sh > /root/byte-pool-out/build-qualified.log 2>&1
cp "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" > /root/byte-pool-out/qualified-binary.sha256
pixi run python tools/byte_lm_optimizer_pool_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/byte-pool-out/report.json > /root/byte-pool-out/public.log 2>&1
pixi run python tools/byte_lm_parallel_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/byte-pool-out/previous.json > /root/byte-pool-out/previous.log 2>&1
cmp /root/byte-pool-out/previous-initial.json /root/byte-pool-out/previous.json
export MOJOLEARN_BYTE_LM_OUTDIR=/root/byte-pool-out/fault
export MOJOLEARN_BUILD_EXTRA_DEFINES='-D MOJOLEARN_BYTE_POOL_FAULT_INJECT=1'
sh bindings/build_byte_lm.sh > /root/byte-pool-out/build-fault.log 2>&1
cp "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" > /root/byte-pool-out/fault-binary.sha256
pixi run python tools/byte_lm_optimizer_pool_check.py --cloud --faults --corpus training/corpus/enwik8/input.txt --report /root/byte-pool-out/fault-report.json > /root/byte-pool-out/fault.log 2>&1
cp /root/byte-pool-out/qualified/_mojolearn_byte_lm.so python/mojolearn/identical/_mojolearn_byte_lm.so
