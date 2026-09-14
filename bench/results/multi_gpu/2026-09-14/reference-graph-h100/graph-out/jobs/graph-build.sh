#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=4ra98lfm0pqum0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
mkdir -p /root/graph-out
tar -xzf /root/graph-source.tgz
cp /root/graph-source.tgz /root/graph-out/
sh bindings/build_metrics.sh > /root/graph-out/metrics-build.log 2>&1
sh bindings/build_solver.sh > /root/graph-out/solver-build.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_metrics.so python/mojolearn/identical/_mojolearn_solver.so > /root/graph-out/binaries.sha256
