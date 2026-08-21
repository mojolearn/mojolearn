#!/bin/sh
# Build and run the cuML decision-tree oracle.
#
#   ensemble/tools/cuml_oracle/build.sh > ensemble/bench/cuml_oracle.txt
#
# NO CUDA TOOLKIT AND NO GPU. cuML's numeric core -- Split::update, all six
# gain functions, every bins.cuh operator, lower_bound -- is HDI/DI
# (host-device inline) and compiles with a plain c++ compiler. The four
# __global__ kernels are the only CUDA in the learner and none of the
# arithmetic lives there.
#
# The *_shim.h files are their headers with the CUDA #includes stripped and
# nothing else changed; shims.h supplies the CUDA-shaped names. Regenerate
# the shims with regen_shims.sh if the pin moves, and READ THE DIFF.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
c++ -std=c++17 -O1 -I "$HERE" -o "$OUT/oracle" "$HERE/oracle.cpp"
"$OUT/oracle"
