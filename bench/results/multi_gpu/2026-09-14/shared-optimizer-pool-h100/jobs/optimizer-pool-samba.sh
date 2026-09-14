#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=rbtojh7e0esekh MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
sh bindings/build_mamba.sh > /root/optimizer-pool-out/build-mamba.log 2>&1
sh bindings/build_transformer.sh > /root/optimizer-pool-out/build-transformer.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_mamba.so python/mojolearn/identical/_mojolearn_transformer.so > /root/optimizer-pool-out/samba-binaries.sha256
pixi run python tools/parallel_training_check.py --cloud --lane samba --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/samba.json > /root/optimizer-pool-out/samba.log 2>&1
pixi run python tools/parallel_training_check.py --cloud --lane samba --attention-dropout --corpus training/corpus/enwik8/input.txt --report /root/optimizer-pool-out/samba-attention.json > /root/optimizer-pool-out/samba-attention.log 2>&1
