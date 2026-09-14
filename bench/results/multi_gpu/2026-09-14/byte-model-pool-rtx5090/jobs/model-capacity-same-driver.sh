#!/bin/sh
set -eu
while [ ! -f /root/jobs/model-pool-v2.done ]; do sleep 10; done
test "$(cat /root/jobs/model-pool-v2.rc)" = 0
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=cvisjmryfcmz6g MOJOLEARN_NUMERIC_MODE=identical
cd /root/gradient-pool
export PYTHONPATH="$PWD/python"
pixi run python tools/byte_lm_model_pool_capacity_check.py --cloud --mode pooled-one --corpus training/corpus/enwik8/input.txt --report /root/model-pool-v2-out/capacity-pooled-one.json > /root/model-pool-v2-out/capacity-pooled-one.log 2>&1
