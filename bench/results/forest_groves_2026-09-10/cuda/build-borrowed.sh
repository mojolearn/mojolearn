#!/bin/bash
set -euo pipefail
cd /root/mojolearn
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=sm_90 MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_FOREST_VECTOR_GROVES=1'
trap 'echo "$?" > /root/forest_out/build-borrowed.exit' EXIT
tar -xzf /root/forest-borrowed-overlay.tar.gz -C /root/mojolearn
pixi run mojo build --target-accelerator sm_90 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_FOREST_VECTOR_GROVES=1 checks/forest_inference_model.mojo -o /root/forest_out/resident-into-check > /root/forest_out/resident-into-check.build.log 2>&1
bash bindings/build_rf.sh > /root/forest_out/rf-borrowed-identical.build.log 2>&1
bash bindings/build_trees.sh > /root/forest_out/et-borrowed-identical.build.log 2>&1
/root/forest_out/resident-into-check > /root/forest_out/resident-into-check.run.log 2>&1
