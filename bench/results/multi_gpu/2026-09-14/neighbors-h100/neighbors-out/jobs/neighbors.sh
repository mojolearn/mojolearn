#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=4ra98lfm0pqum0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/neighbors-out
cp commit.txt /root/neighbors-out/
nvidia-smi > /root/neighbors-out/gpus.txt
pixi install > /root/neighbors-out/pixi.log 2>&1
sh bindings/build.sh > /root/neighbors-out/base-build.log 2>&1
sh bindings/build_estimators.sh > /root/neighbors-out/estimators-build.log 2>&1
sha256sum python/mojolearn/parallel_neighbors.py python/mojolearn/_parallel_worker.py tools/parallel_neighbors_check.py > /root/neighbors-out/source.sha256
find python -name '*.so' -exec sha256sum {} \; > /root/neighbors-out/binaries.sha256
pixi run python tools/parallel_neighbors_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/neighbors-out/report.json > /root/neighbors-out/gate.log 2>&1
