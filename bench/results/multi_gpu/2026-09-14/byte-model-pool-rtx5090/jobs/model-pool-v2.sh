#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=cvisjmryfcmz6g MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_120 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/gradient-pool
export PYTHONPATH="$PWD/python"
mkdir -p /root/model-pool-v2-out
export MOJOLEARN_BYTE_LM_OUTDIR=/root/model-pool-v2-out/production
sh bindings/build_byte_lm.sh > /root/model-pool-v2-out/build.log 2>&1
while [ ! -f /root/jobs/model-capacity-2.done ]; do sleep 10; done
cp "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" > /root/model-pool-v2-out/production.sha256
for shards in 1 3 5; do
    pixi run python tools/byte_lm_model_pool_check.py --cloud --logical-shards "$shards" --corpus training/corpus/enwik8/input.txt --report /root/model-pool-v2-out/pool-${shards}.json > /root/model-pool-v2-out/pool-${shards}.log 2>&1
done
pixi run python tools/byte_lm_parallel_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/model-pool-v2-out/byte-lm.json > /root/model-pool-v2-out/byte-lm.log 2>&1
pixi run mojo run --target-accelerator sm_120 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/byte_lm_model_pool_check.mojo > /root/model-pool-v2-out/native.log 2>&1
export MOJOLEARN_BYTE_LM_OUTDIR=/root/model-pool-v2-out/fault
export MOJOLEARN_BUILD_EXTRA_DEFINES='-D MOJOLEARN_BYTE_POOL_FAULT_INJECT=1'
sh bindings/build_byte_lm.sh > /root/model-pool-v2-out/build-fault.log 2>&1
cp "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" python/mojolearn/identical/_mojolearn_byte_lm.so
sha256sum "$MOJOLEARN_BYTE_LM_OUTDIR/_mojolearn_byte_lm.so" > /root/model-pool-v2-out/fault.sha256
pixi run python tools/byte_lm_model_pool_check.py --cloud --faults --corpus training/corpus/enwik8/input.txt --report /root/model-pool-v2-out/fault.json > /root/model-pool-v2-out/fault.log 2>&1
cp /root/model-pool-v2-out/production/_mojolearn_byte_lm.so python/mojolearn/identical/_mojolearn_byte_lm.so
pixi run python tools/byte_lm_model_pool_capacity_check.py --cloud --mode pooled --corpus training/corpus/enwik8/input.txt --report /root/model-pool-v2-out/capacity-pooled.json > /root/model-pool-v2-out/capacity-pooled.log 2>&1
