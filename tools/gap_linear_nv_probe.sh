#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# lane/gap-linear-nv stage profile, run by `lq add nv|amd CMD` in the branch
# tree (bindings already built by the job):
#   1. the per-step kernels at the board's shapes (bench/apple_identical_steps_main.mojo),
#      built with the lane's kernels (new) and with -D MOJOLEARN_NV_AMD_STEPS_OFF=1 (off);
#      a `gemv.` hash must equal its `pgemv.` hash, `co.xty.` its `xty.`
#   2. LinearSVR / LogisticRegression / KNeighborsClassifier fits at the board's
#      shapes (tools/gap_linear_nv_fit_probe.py): ms, n_iter, ms per iteration
#   3. a kernel-level profile of one Istella-sized LinearSVR fit when nsys
#      (NVIDIA) or rocprofv3 (AMD) is on the box
set -u
cd "$(dirname "$0")/.."
mkdir -p build
ACC=""; COL=""
[ -n "${MOJOLEARN_GPU_ARCHS:-}" ] && ACC="--target-accelerator $MOJOLEARN_GPU_ARCHS"
[ -n "${MOJOLEARN_TARGET_COLUMN:-}" ] && COL="-D MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')"
for arm in new off; do
  D=""; [ $arm = off ] && D="-D MOJOLEARN_NV_AMD_STEPS_OFF=1"
  if pixi run mojo build -j 4 $ACC $COL -D MOJOLEARN_NUMERIC_IDENTICAL=1 $D -I . \
      bench/apple_identical_steps_main.mojo -o build/steps_$arm > build/steps_$arm.build.log 2>&1; then
    MOJOLEARN_STEPS_NO_TINY=1 MOJOLEARN_STEPS_REPS=5 MOJOLEARN_STEPS_BATCH=5 ./build/steps_$arm 2>&1 \
      | sed "s/^STEP/STEP $arm/"
  else
    echo "STEPBUILD FAIL $arm"; grep -m 5 -A 4 error build/steps_$arm.build.log
  fi
done
python3 tools/gap_linear_nv_fit_probe.py
if command -v nsys > /dev/null 2>&1; then
  GAP_PROBE_REPS=1 nsys profile -o build/qn_prof --force-overwrite true --stats=false \
    python3 tools/gap_linear_nv_fit_probe.py qn > /dev/null 2>&1
  nsys stats -r cuda_gpu_kern_sum -f csv build/qn_prof.nsys-rep 2>/dev/null | head -25 | sed 's/^/KERN /'
elif command -v rocprofv3 > /dev/null 2>&1; then
  GAP_PROBE_REPS=1 rocprofv3 --kernel-trace --stats -d build/qn_prof -o qn \
    -- python3 tools/gap_linear_nv_fit_probe.py qn > /dev/null 2>&1
  f=$(find build/qn_prof -name '*kernel_stats.csv' | head -1)
  [ -n "$f" ] && sort -t, -k3 -g -r "$f" | head -25 | sed 's/^/KERN /'
else
  echo "KERN no profiler on this box"
fi
echo "PROBE done"
