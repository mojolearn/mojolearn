#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run python tools/byte_lm_optimizer_pool_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/byte-pool-out/report-initial.json > /root/byte-pool-out/public-initial.log 2>&1
pixi run python tools/byte_lm_parallel_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/byte-pool-out/previous-initial.json > /root/byte-pool-out/previous-initial.log 2>&1
