#!/bin/sh
# tools/gemm_seam_probe_column_leg.sh -- DEVIATION 2701. A four-line wrapper
# whose whole job is the thing tools/gemm_remote_leg.sh does not do: name the
# COLUMN before the build.
#
# WHY THIS EXISTS. tools/gemm_seam_probe_leg.sh builds
# gemm/checks/gemm_seam_probe.mojo and prints its lanes. The probe's fifth lane
# (the AMD wave-mode flush, 2026-09-17) is `comptime MODE_LANE = TARGET_COLUMN
# == COLUMN_AMD`, and TARGET_COLUMN comes from MOJOLEARN_TARGET_COLUMN at BUILD
# time. tools/gemm_remote_leg.sh exports MOJOLEARN_GPU_ARCHS to the box (its
# remote body substitutes @GPUARCHS@) but it does NOT export the column, and
# tools/hotaisle_leg.sh DOES -- which is why the 2026-09-13 Hot Aisle probe read
# `column=amd` and a RunPod AMD run of the same body would not have. A build
# without the column is not the shipped column's build, and here it would
# silently compile the mode lane OUT and print a lane that looks like an answer.
#
# So: derive the vendor from the box the way tools/step_breakdown_leg.sh does
# (a working nvidia-smi, else /dev/kfd or an AMD tool; /dev/dri alone is NOT AMD
# evidence), export it, and hand over. Everything else is the existing body.
set -u
case "${MOJOLEARN_TARGET_COLUMN:-}" in
    amd|nvidia) VENDOR=$MOJOLEARN_TARGET_COLUMN ;;
    *) if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
           VENDOR=nvidia
       elif [ -e /dev/kfd ] || command -v rocm-smi > /dev/null 2>&1 || command -v amd-smi > /dev/null 2>&1; then
           VENDOR=amd
       else
           VENDOR=unknown
       fi ;;
esac
OUT=${MOJOLEARN_GEMM_STEP_LEG_OUT:-/root/gemm_leg_out/seam-probe}
mkdir -p "$OUT"
if [ "$VENDOR" = unknown ]; then
    echo "vendor=unknown: no working nvidia-smi, no /dev/kfd, no rocm-smi or amd-smi; nothing built" > "$OUT/column.txt"
    exit 9
fi
export MOJOLEARN_TARGET_COLUMN="$VENDOR"
{
    echo "column=$VENDOR (derived on the box, exported before the build)"
    echo "gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset}"
} > "$OUT/column.txt"
exec sh "$(dirname "$0")/gemm_seam_probe_leg.sh"
