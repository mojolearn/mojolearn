#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run python tools/parallel_ordered_rmse_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/pointwise-out/ordered-report.json > /root/pointwise-out/ordered.log 2>&1
pixi run python tools/parallel_feature_freq_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/pointwise-out/feature-freq-report.json > /root/pointwise-out/feature-freq.log 2>&1
pixi run python tools/parallel_boosting_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/pointwise-out/greedy-regression-report.json > /root/pointwise-out/greedy-regression.log 2>&1
