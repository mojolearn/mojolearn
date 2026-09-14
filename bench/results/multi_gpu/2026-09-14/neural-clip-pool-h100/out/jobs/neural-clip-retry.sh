#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
mkdir -p /root/neural-clip-out/initial
mv /root/neural-clip-out/clip.log /root/neural-clip-out/initial/
cp /root/neural-clip-out/build.log /root/neural-clip-out/initial/
cp /root/neural-clip-out/production.sha256 /root/neural-clip-out/initial/
cp training/clip_multi_gpu.mojo /root/neural-clip-out/clip-fixed.mojo
sha256sum training/clip_multi_gpu.mojo > /root/neural-clip-out/clip-fixed.sha256
sh bindings/build_training_clip.sh > /root/neural-clip-out/build.log 2>&1
sha256sum /root/neural-clip-production/_mojolearn_training.so > /root/neural-clip-out/production.sha256
sh /root/jobs/neural-clip-check.sh
