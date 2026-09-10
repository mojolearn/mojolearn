#!/bin/bash
set -euo pipefail
if [ "${M3_ENV_ACTIVE:-0}" != 1 ]; then
 export M3_ENV_ACTIVE=1
 exec /root/.pixi/bin/pixi run --manifest-path /root/mojolearn/pixi.toml bash "$0"
fi
cd /root/mamba3-residual
out=/root/jobs/m3-fresh-final
mkdir -p "$out"
sha256sum bindings/_mojolearn_mamba.mojo python/mojolearn/_mamba_impl.py gemm/checks/gemm_identical.mojo checks/kernel_matrix.mojo > "$out/source-sha256.txt"
mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -I bindings bindings/_mojolearn_mamba.mojo -o "$out/default.so" > "$out/build-default.log" 2>&1
