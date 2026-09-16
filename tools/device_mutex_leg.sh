#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Small GPU primitive gate, launched through the existing guarded provider leg.
set -euo pipefail
cd /root/mojolearn
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 MOJOLEARN_CPU_THREADS=1
export MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BUILD_JOBS=1
mkdir -p /root/gemm_leg_out
exec > >(tee /root/gemm_leg_out/device_mutex.log) 2>&1
date -u
printf 'GPU architecture: %s\n' "$MOJOLEARN_GPU_ARCHS"
sha256sum core/device_mutex.mojo core/device_mutex_check.mojo
nice -n 19 python3 tools/check_mutex_handoff_model.py
nice -n 19 pixi run mojo build -j 1 --target-accelerator "$MOJOLEARN_GPU_ARCHS" \
  -I . core/device_mutex_check.mojo -o /tmp/device_mutex_check
nice -n 19 /tmp/device_mutex_check
