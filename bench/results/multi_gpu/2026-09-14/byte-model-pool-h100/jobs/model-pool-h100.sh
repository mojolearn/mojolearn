#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=lgie1o62m251xp MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
while [ ! -f /root/model-pool-out/source.sha256 ]; do sleep 2; done
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > /root/model-pool-out/hardware.csv
sh bindings/build.sh > /root/model-pool-out/build-base.log 2>&1
sh bindings/build_byte_lm.sh > /root/model-pool-out/build-byte-lm.log 2>&1
sha256sum python/mojolearn/identical/*.so > /root/model-pool-out/binaries.sha256
sha256sum training/corpus/enwik8/input.txt > /root/model-pool-out/corpus.sha256
for shards in 1 3 5; do
    pixi run python tools/byte_lm_model_pool_check.py --cloud --logical-shards "$shards" --corpus training/corpus/enwik8/input.txt --report /root/model-pool-out/pool-${shards}.json > /root/model-pool-out/pool-${shards}.log 2>&1
done
pixi run mojo run --target-accelerator sm_90 -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/byte_lm_model_pool_check.mojo > /root/model-pool-out/native.log 2>&1
