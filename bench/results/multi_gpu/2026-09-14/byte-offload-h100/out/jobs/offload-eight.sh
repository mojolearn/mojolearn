#!/bin/sh
set -eu
while [ ! -f /root/jobs/offload-final.done ]; do sleep 5; done
test "$(cat /root/jobs/offload-final.rc)" = 0
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=6ewmfb4taf9u1q MOJOLEARN_NUMERIC_MODE=identical
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run python tools/byte_lm_offload_check.py --cloud --logical-shards 8 --corpus training/corpus/enwik8/input.txt --report /root/offload-out/public-8.json > /root/offload-out/public-8.log 2>&1
