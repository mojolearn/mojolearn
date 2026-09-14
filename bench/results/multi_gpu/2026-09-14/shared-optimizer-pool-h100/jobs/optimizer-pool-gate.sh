#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_optimizer_check.py --cloud --report /root/optimizer-pool-out/report.json > /root/optimizer-pool-out/public.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane mlp --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/mlp.json > /root/optimizer-pool-out/mlp.log 2>&1
