#!/bin/bash
# GEMM ceiling on one NVIDIA box: the int15 price harness (fp32.v1 and every
# fixed15 arm, GPU-resident, conversions counted) then PyTorch at the same shapes.
set -uo pipefail
cd "$(dirname "$0")/../.."
O=/root/gemm-ceiling; mkdir -p $O
export PATH=/root/.pixi/bin:$PATH
ARCH=${MOJOLEARN_GPU_ARCHS:-sm_89}
git rev-parse HEAD > $O/head.txt; git diff --binary HEAD > $O/source.patch
nvidia-smi -q > $O/device.log 2>&1
pixi install > $O/pixi.log 2>&1 || { echo "pixi failed" > $O/FAILED; exit 1; }
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 --target-accelerator $ARCH -I . \
    bench/gemm_int15_price_main.mojo -o $O/int15_price > $O/build.log 2>&1 || { echo "build failed" > $O/FAILED; exit 1; }
sha256sum $O/int15_price > $O/binary.sha256
MOJOLEARN_INT15_PRICE_IDENTITY_ONLY=1 $O/int15_price > $O/identity.log 2>&1; echo "identity rc=$?" >> $O/rc.txt
$O/int15_price > $O/timed.log 2>&1; echo "timed rc=$?" >> $O/rc.txt
base=$(pixi run python3 -c 'import sys;print(sys.executable)' | tail -1)
[ -x $O/venv/bin/python ] || $base -m venv $O/venv
$O/venv/bin/pip -q install torch==2.13.0 > $O/torch-install.log 2>&1
$O/venv/bin/pip freeze > $O/packages.txt
$O/venv/bin/python tools/gemm_ceiling/torch_gemm.py $O/timed.log $O/torch.json > $O/torch.log 2>&1; echo "torch rc=$?" >> $O/rc.txt
echo done > $O/done
