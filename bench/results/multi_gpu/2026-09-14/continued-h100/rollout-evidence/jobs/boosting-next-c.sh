#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/boost-out
while [ ! -f /root/jobs/arima-next.done ]; do sleep 5; done
sh bindings/build_gbdt.sh > /root/boost-out/build.log 2>&1
pixi run python tools/parallel_boosting_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/boost-out/boosting.json > /root/boost-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_gbdt.so > /root/boost-out/binaries.sha256
