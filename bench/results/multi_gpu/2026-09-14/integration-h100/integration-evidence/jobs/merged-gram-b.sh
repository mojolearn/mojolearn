#!/bin/sh
set -eu
export RUNPOD_POD_ID=nlfngsvejhgiq5 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mv /root/merge-out/gram.log /root/merge-out/gram-missing-base.log
sh bindings/build.sh > /root/merge-out/build-base.log 2>&1
pixi run python tools/parallel_gram_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/merge-out/gram.json > /root/merge-out/gram.log 2>&1
pixi run python tools/parallel_logistic_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/merge-out/logistic.json > /root/merge-out/logistic.log 2>&1
cp commit.txt /root/merge-out/source-commit.txt
sha256sum python/mojolearn/identical/_mojolearn_estimators.so python/mojolearn/identical/_mojolearn.so core/gram_splitk.mojo > /root/merge-out/hashes.sha256
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /root/merge-out/gpu.txt
