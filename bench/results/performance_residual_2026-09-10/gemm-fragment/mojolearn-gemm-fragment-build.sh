#!/bin/bash
set -euo pipefail
cd /root/mojolearn
out=/root/jobs/gemm-fragment
mkdir -p "$out"
for arm in baseline admitted; do
 flags=()
 if [ "$arm" = admitted ]; then flags=(-D MOJOLEARN_GEMM_FRAGMENT_ADMISSION=1); fi
 /root/.pixi/bin/pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${flags[@]}" -I . gemm/checks/gemm_tuned_probe.mojo -o "$out/$arm" > "$out/$arm-build.log" 2>&1
done
/root/.pixi/bin/pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_FRAGMENT_ADMISSION=1 -I . gemm/checks/gemm_fragment_boundary_check.mojo -o "$out/boundary" > "$out/boundary-build.log" 2>&1
/root/.pixi/bin/pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_FRAGMENT_ADMISSION=1 -I . gemm/checks/gemm_device_check.mojo -o "$out/device" > "$out/device-build.log" 2>&1
