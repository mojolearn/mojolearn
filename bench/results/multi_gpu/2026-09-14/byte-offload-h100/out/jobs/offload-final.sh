#!/bin/sh
set -eu
for job in offload-public offload-fault-build; do
    while [ ! -f /root/jobs/$job.done ]; do sleep 5; done
    test "$(cat /root/jobs/$job.rc)" = 0
done
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=6ewmfb4taf9u1q MOJOLEARN_NUMERIC_MODE=identical
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
# The first pooled process had already read the earlier thread-based sampler
# when its replacement arrived. Preserve it and repeat with process sampling.
mv /root/offload-out/capacity-pooled.json /root/offload-out/capacity-pooled-thread-sampler.json
mv /root/offload-out/capacity-pooled.log /root/offload-out/capacity-pooled-thread-sampler.log
pixi run python tools/byte_lm_offload_capacity_check.py --cloud --mode pooled --corpus training/corpus/enwik8/input.txt --report /root/offload-out/capacity-pooled.json > /root/offload-out/capacity-pooled.log 2>&1
mkdir -p /root/offload-production
cp python/mojolearn/identical/_mojolearn_byte_lm.so /root/offload-production/
cp /root/offload-fault/_mojolearn_byte_lm.so python/mojolearn/identical/_mojolearn_byte_lm.so
pixi run python tools/byte_lm_offload_check.py --cloud --faults --corpus training/corpus/enwik8/input.txt --report /root/offload-out/fault.json > /root/offload-out/fault.log 2>&1
cp /root/offload-production/_mojolearn_byte_lm.so python/mojolearn/identical/_mojolearn_byte_lm.so
pixi run python tools/byte_lm_parallel_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/offload-out/original-byte.json > /root/offload-out/original-byte.log 2>&1
python3 /root/offload-out/comparison.py > /root/offload-out/comparison.log 2>&1
