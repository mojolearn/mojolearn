#!/bin/bash
set -euo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_FOREST_VECTOR_GROVES=1'
mkdir -p /root/forest_out
trap 'echo "$?" > /root/forest_out/build-all.exit' EXIT
tar -xzf /root/forest-resident-overlay.tar.gz -C /root/mojolearn
pixi run mojo build --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 checks/forest_inference_gpu.mojo -o /root/forest_out/grove-scalar > /root/forest_out/grove-scalar.build.log 2>&1
pixi run mojo build --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_FOREST_VECTOR_GROVES=1 checks/forest_inference_gpu.mojo -o /root/forest_out/grove-vector > /root/forest_out/grove-vector.build.log 2>&1
pixi run mojo build --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_FOREST_VECTOR_GROVES=1 checks/forest_inference_model.mojo -o /root/forest_out/resident-check > /root/forest_out/resident-check.build.log 2>&1
pixi run mojo build --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 bench/speed/forest_grove_kernel.mojo -o /root/forest_out/grove-bench > /root/forest_out/grove-bench.build.log 2>&1
bash bindings/build_rf.sh > /root/forest_out/rf-identical.build.log 2>&1
bash bindings/build_trees.sh > /root/forest_out/et-identical.build.log 2>&1
