#!/bin/sh
set -eu
export RUNPOD_POD_ID=kpqg64gsa9lib0
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/holt-out
while [ ! -f /root/jobs/gram-gate-b.done ]; do sleep 5; done
sh bindings/build_tsa.sh > /root/holt-out/build.log 2>&1
pixi run python tools/parallel_holtwinters_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/holt-out/holtwinters.json > /root/holt-out/gate.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_tsa.so > /root/holt-out/binaries.sha256
