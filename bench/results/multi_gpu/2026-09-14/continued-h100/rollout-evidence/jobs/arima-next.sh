#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/arima-out
while [ ! -f /root/jobs/boosting-next-b.done ]; do sleep 5; done
sh bindings/build_arima.sh > /root/arima-out/build.log 2>&1
pixi run python tools/parallel_arima_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/arima-out/arima.json > /root/arima-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_arima.so > /root/arima-out/binaries.sha256
