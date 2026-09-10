#!/bin/bash
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mamba3
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mamba3/python
mkdir -p /root/jobs/m3-public python/mojolearn/identical
for arm in baseline optimized; do
  extra=()
  if [ "$arm" = baseline ]; then extra=(-D MOJOLEARN_MAMBA3_LEGACY_STATEPASS=1 -D MOJOLEARN_MAMBA3_LEGACY_HOST_COPY=1); fi
  pixi run mojo build -j 2 --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${extra[@]}" -I . -I bindings bindings/_mojolearn_mamba.mojo -o /root/jobs/m3-public/$arm.so > /root/jobs/m3-public/build-$arm.log 2>&1
  cp /root/jobs/m3-public/$arm.so python/mojolearn/identical/_mojolearn_mamba.so
  python3 bench/speed/seq_py_speed_arm.py --lane mamba3 --rounds 3 > /root/jobs/m3-public/price-$arm.log 2>&1
  PYTHONPATH=python python3 -m unittest mojolearn.tests.test_mamba_surface > /root/jobs/m3-public/surface-$arm.log 2>&1
done
