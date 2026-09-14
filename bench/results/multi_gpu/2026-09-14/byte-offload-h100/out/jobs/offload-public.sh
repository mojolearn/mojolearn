#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=6ewmfb4taf9u1q MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
sha256sum python/mojolearn/identical/*.so > /root/offload-out/binaries-production.sha256
sha256sum training/corpus/enwik8/input.txt > /root/offload-out/corpus.sha256
for shards in 1 3 5; do
    pixi run python tools/byte_lm_offload_check.py --cloud --logical-shards "$shards" --corpus training/corpus/enwik8/input.txt --report /root/offload-out/public-${shards}.json > /root/offload-out/public-${shards}.log 2>&1
done
for mode in pooled offloaded; do
    pixi run python tools/byte_lm_offload_capacity_check.py --cloud --mode "$mode" --corpus training/corpus/enwik8/input.txt --report /root/offload-out/capacity-${mode}.json > /root/offload-out/capacity-${mode}.log 2>&1
done
