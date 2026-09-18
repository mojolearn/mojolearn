#!/bin/bash
# usage: run_arm.sh <host_dir> <out.json> [env assignments...]; one core, nice 19
set -u
cd ~/mojolearn-wt/tokenized-corpus
HD=$1; OUT=$2; shift 2
env "$@" MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$PWD/python MOJOLEARN_HOST_DIR=$HD OMP_NUM_THREADS=1 MOJOLEARN_NUM_THREADS=1 \
  nice -n 19 pixi run python tools/identity_break.py --lanes bpe-vocabulary,tokenized-corpus --repeats 2 --require-cpu --json "$OUT"
