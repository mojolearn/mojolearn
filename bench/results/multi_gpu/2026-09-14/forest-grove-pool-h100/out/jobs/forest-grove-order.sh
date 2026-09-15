#!/bin/sh
set -eu
export PATH="$HOME/.pixi/bin:$PATH"
export RUNPOD_POD_ID=smqlvlvt7exixd MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_FOREST_ORDER_ONLY=1
cd /root/mojolearn
cp /root/forest_pool_order_check.mojo training/checks/forest_pool_check.mojo
cp /root/forest_pool_order_check.mojo /root/forest-grove-out/order-check.mojo
sha256sum training/checks/forest_pool_check.mojo > /root/forest-grove-out/order-check.sha256
for layout in separate packed; do
    extra=''
    if [ "$layout" = packed ]; then extra='-D MOJOLEARN_FOREST_PACKED_NODES=1'; fi
    pixi run mojo run --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 $extra -I . training/checks/forest_pool_check.mojo > /root/forest-grove-out/order-$layout.log 2>&1
done
