#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=4ra98lfm0pqum0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
mkdir -p /root/dbscan-out
tar -xzf /root/dbscan-source.tgz
cp /root/dbscan-source.tgz /root/dbscan-out/
rm python/mojolearn/identical/_mojolearn_estimators.so
sh bindings/build_estimators.sh > /root/dbscan-out/build.log 2>&1
sha256sum python/mojolearn/identical/_mojolearn_estimators.so > /root/dbscan-out/binary.sha256
