#!/bin/bash
set -euo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_SKIP_BUILD_GATE=1
unset MOJOLEARN_EXTRA_DEFINES
trap 'echo "$?" > /root/forest_out/build-default.exit' EXIT
pixi run mojo build --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/forest_inference_gpu.mojo -o /root/forest_out/grove-default > /root/forest_out/grove-default.build.log 2>&1
/root/forest_out/grove-default > /root/forest_out/grove-default.run.log 2>&1
bash bindings/build_rf.sh > /root/forest_out/rf-default-identical.build.log 2>&1
bash bindings/build_trees.sh > /root/forest_out/et-default-identical.build.log 2>&1
