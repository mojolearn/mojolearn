#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/mojolearn
mkdir -p /root/bin
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_fused_check.mojo -o /root/bin/fused_check
/root/bin/fused_check
