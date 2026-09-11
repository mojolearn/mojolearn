#!/bin/sh
# AMD GEMM ksplit leg body for tools/gemm_remote_leg.sh (RunPod MI300X).
# That runner does not export MOJOLEARN_GPU_ARCHS or MOJOLEARN_TARGET_COLUMN
# into the extra body, and tools/gemm_step_leg.sh refuses AMD without the
# arch (first RunPod AMD try, e1g/2026-09-11_162854: "gpu_archs=MISSING").
# The MI300X build target is gfx942 (rocminfo on the Hot Aisle smoke).
set -u
cd /root/mojolearn || exit 9
MOJOLEARN_GPU_ARCHS=${MOJOLEARN_GPU_ARCHS:-gfx942} \
MOJOLEARN_TARGET_COLUMN=amd \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/gemm_longk_leg.sh
