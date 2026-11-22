#!/bin/bash
# usage: run_arm.sh <source_tree> <host_dir> <out.json> [env assignments...]; one core, nice 19
set -u
SRC=$1; HD=$2; OUT=$3; shift 3
cd "$SRC"
env "$@" MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=$SRC/python MOJOLEARN_HOST_DIR=$HD OMP_NUM_THREADS=1 MOJOLEARN_NUM_THREADS=1 \
  nice -n 19 pixi run --manifest-path $HOME/mojolearn-wt/bpe-builder-native/pixi.toml python tools/identity_break.py \
  --lanes tokenizer,bpe-trainer,bpe-vocabulary,tokenized-corpus --repeats 2 --require-cpu --json "$OUT"
