#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
sh bindings/build_linalg.sh > /root/optimizer-pool-out/build-linalg.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_linalg.so > /root/optimizer-pool-out/linalg-binary.sha256
pixi run python tools/parallel_training_check.py --cloud --lane mlp --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/mlp-final.json > /root/optimizer-pool-out/mlp-final.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/samba-final.json > /root/optimizer-pool-out/samba-final.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/samba-attention-final.json > /root/optimizer-pool-out/samba-attention-final.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/samba-attention-clip.json > /root/optimizer-pool-out/samba-attention-clip.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --max-norm .01 --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/samba-clip.json > /root/optimizer-pool-out/samba-clip.log 2>&1
