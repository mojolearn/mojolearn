#!/bin/sh
# Regenerate the *_shim.h copies of cuML's headers from the pin.
#
#   ensemble/tools/cuml_oracle/regen_shims.sh
#
# The shims are their headers with the CUDA #includes stripped and NOTHING
# ELSE changed -- no transcription, no paraphrase. `shims.h` supplies the
# CUDA-shaped names (HDI/DI, raft::log, atomicAdd, threadIdx, ...) so the
# rest compiles on a host with no CUDA toolkit.
#
# Run this when the pin moves, then READ THE DIFF, because that diff IS the
# behaviour change.
set -eu
C="${CUML_DIR:-$HOME/CascadeProjects/upstream/cuml-v26.08.00}/cpp/src/decisiontree/batched-levelalgo"
HERE=$(cd "$(dirname "$0")" && pwd)
[ -d "$C" ] || { echo "no cuML checkout at $C" >&2; exit 1; }

grep -v '#include <raft/util/cuda_utils.cuh>\|#include <cuda/std/array>\|#pragma once' \
  "$C/bins.cuh" > "$HERE/bins_shim.h"
grep -v '#include <raft/util/cuda_utils.cuh>\|#pragma once' \
  "$C/dataset.h" > "$HERE/dataset_shim.h"
# split.cuh minus the raft::linalg-dependent tail (initSplit / printSplits),
# which are launch helpers rather than arithmetic.
awk '/^ \* @brief Initialize the split array/ {exit} {print}' "$C/split.cuh" \
  | grep -v '#include <raft/linalg/unary_op.cuh>\|#include <raft/util/cuda_utils.cuh>\|#include "bins.cuh"\|#pragma once' \
  | sed '$ { /^\/\*\*$/d; }' > "$HERE/split_shim.h"
printf '}  // namespace DT\n}  // namespace ML\n' >> "$HERE/split_shim.h"
grep -v '#include "bins.cuh"\|#include "dataset.h"\|#include "split.cuh"\|#include <cuml/tree/algo_helper.h>\|#pragma once' \
  "$C/objectives.cuh" > "$HERE/objectives_shim.h"
awk '/^\/\/ Returns the lowest index in `array`/,/^}$/' \
  "$C/kernels/builder_kernels.cuh" > "$HERE/lower_bound_shim.h"
echo "regenerated shims from $C"
