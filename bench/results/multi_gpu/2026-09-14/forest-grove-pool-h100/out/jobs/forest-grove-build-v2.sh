#!/bin/sh
set -eu
mkdir -p /root/forest-grove-out/initial
cp /root/forest-grove-out/source.sha256 /root/forest-grove-out/initial/source.sha256
mv /root/forest-grove-out/native-separate.log /root/forest-grove-out/initial/native-separate.log
#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90a MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_SKIP_BUILD_GATE=1
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mkdir -p /root/forest-grove-out /root/forest-grove-separate /root/forest-grove-packed
cp /root/forest-grove-final-source.tgz /root/forest-grove-out/
sha256sum /root/forest-grove-final-source.tgz > /root/forest-grove-out/source.sha256
tar xzf /root/forest-grove-final-source.tgz
cp /root/neural-clip-out/hardware.csv /root/forest-grove-out/
cp /root/neural-clip-out/corpus.sha256 /root/forest-grove-out/
export MOJOLEARN_EXTRA_DEFINES=''
pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/forest_pool_check.mojo > /root/forest-grove-out/native-separate.log 2>&1

sh bindings/build_rf.sh > /root/forest-grove-out/build-rf-separate.log 2>&1
sh bindings/build_trees.sh > /root/forest-grove-out/build-et-separate.log 2>&1
cp python/mojolearn/identical/_mojolearn_rf.so python/mojolearn/identical/_mojolearn_trees.so /root/forest-grove-separate/
export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_FOREST_PACKED_NODES=1'
sh bindings/build_rf.sh > /root/forest-grove-out/build-rf-packed.log 2>&1
sh bindings/build_trees.sh > /root/forest-grove-out/build-et-packed.log 2>&1
cp python/mojolearn/identical/_mojolearn_rf.so python/mojolearn/identical/_mojolearn_trees.so /root/forest-grove-packed/
sha256sum /root/forest-grove-separate/*.so /root/forest-grove-packed/*.so > /root/forest-grove-out/binaries.sha256
