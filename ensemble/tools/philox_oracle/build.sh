#!/bin/sh
# Build and run the RAFT/cuRAND Philox oracle, and write the reference table
# this repository commits.
#
#     ensemble/tools/philox_oracle/build.sh > ensemble/bench/philox_oracle.txt
#
# THE PIN. cuML v26.08.00 (265b9da6a0e75dbef071a3168398b993a5ff6f0e) resolves
# RAFT through cpp/cmake/thirdparty/get_raft.cmake, which asks for
#   VERSION    = ${CUML_VERSION_MAJOR}.${CUML_VERSION_MINOR}.00 = 26.08.00
#   PINNED_TAG = ${rapids-cmake-checkout-tag}
# and cmake/rapids_config.cmake + VERSION + RAPIDS_BRANCH set rapids-cmake to
# release/26.08.  That resolves to rapidsai/raft v26.08.00, i.e. commit
# ebf92684b8a15addcddb43f442a382d528e8bd77 (VERSION file says 26.08.00),
# checked out read-only at ~/CascadeProjects/upstream/raft-v26.08.00.
#
# RECORDED, NOT SMOOTHED OVER: the RNG headers are byte-identical between RAFT
# 25.08.00 and 26.08.00 apart from the SPDX license-header rewrite, one added
# `#include <raft/core/detail/macros.hpp>`, and cub/cub.cuh being split into
# three narrower cub includes.  No constant, no shift and no control flow moved
# across that range.  Older versions were not checked.
#
# WHAT IS COMPILED VS WHAT IS TRANSCRIBED, because the split matters.
#
#   * The Philox-4x32-10 rounds, the counter-increment helpers and the cuRAND
#     state machine (curand_init / skipahead / skipahead_sequence / curand)
#     are NVIDIA'S OWN BYTES.  They are fetched at build time and compiled
#     unmodified.  They are NOT committed to this repository: curand's headers
#     carry a proprietary NVIDIA licence ("reproduction or disclosure of the
#     Licensed Deliverables to any third party ... is prohibited"), unlike
#     RAFT (Apache-2.0) and CCCL (Apache-2.0).  Only the NUMBERS they produce
#     are committed, in ensemble/bench/philox_oracle.txt.
#   * RAFT's own layer -- PhiloxGenerator, custom_next's uint32 arm, the
#     UniformIntDistParams struct and rngKernel's index mapping -- is Apache-2.0
#     and IS transcribed into oracle.cpp, cited line by line.  RAFT's RNG entry
#     points are `DI` (__device__ __forceinline__) and pull in cuda_runtime, so
#     they cannot be host-included; cuRAND's Philox, by contrast, is
#     host-and-device and compiles on a Mac with three small shims.
#
# NO CUDA TOOLKIT AND NO GPU ARE REQUIRED.
#
# The cuRAND headers come from the nvidia-curand-cu12 wheel, which ships
# include/ as well as lib/.  Point CURAND_INC at any other copy (a CUDA
# toolkit's include directory works) to build against that instead; the
# version actually used is printed into the table header.
set -eu

CURAND_INC="${CURAND_INC:-}"
CACHE="${CURAND_CACHE:-$HOME/CascadeProjects/upstream/curand-headers}"
WHEEL_URL="${CURAND_WHEEL_URL:-https://files.pythonhosted.org/packages/31/44/193a0e171750ca9f8320626e8a1f2381e4077a65e69e2fb9708bd479e34a/nvidia_curand_cu12-10.3.10.19-py3-none-manylinux_2_27_x86_64.whl}"

if [ -z "$CURAND_INC" ]; then
    if [ ! -f "$CACHE/nvidia/curand/include/curand_kernel.h" ]; then
        echo "fetching cuRAND headers into $CACHE (not committed)" >&2
        mkdir -p "$CACHE"
        curl -sL -o "$CACHE/curand.whl" "$WHEEL_URL"
        (cd "$CACHE" && unzip -o -q curand.whl 'nvidia/curand/include/*')
    fi
    CURAND_INC="$CACHE/nvidia/curand/include"
fi

if [ ! -f "$CURAND_INC/curand_philox4x32_x.h" ]; then
    echo "error: no curand_philox4x32_x.h under $CURAND_INC" >&2
    exit 1
fi

CURAND_VER=$(awk '/#define CURAND_VER_MAJOR/{a=$3}
                  /#define CURAND_VER_MINOR/{b=$3}
                  /#define CURAND_VER_PATCH/{c=$3}
                  /#define CURAND_VER_BUILD/{d=$3}
                  END{if (a != "") printf "%s.%s.%s.%s", a, b, c, d}' \
             "$CURAND_INC/curand.h")
CURAND_VER="${CURAND_VER:-unknown}"

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

# Shim 1: <nv/target>.  curand_philox4x32_x.h uses NV_IF_ELSE_TARGET to pick
# between a 64-bit host multiply and __umulhi on the device.  We want the host
# arm, and the header itself documents them as computing the same product.
mkdir -p "$OUT/shim/nv"
cat > "$OUT/shim/nv/target" <<'EOF'
#pragma once
#define NV_IS_HOST 1
#define NV_IF_ELSE_TARGET(cond, host_code, device_code) host_code
EOF

# Shim 2: the Philox half of curand_kernel.h, sliced out of NVIDIA's own file
# at build time rather than copied into this repository.  The slice runs from
# the Philox overload of curand() to the start of the MRG32k3a section, and
# contains exactly five functions: curand, curand4, skipahead,
# skipahead_sequence, curand_init.  If NVIDIA renames any of those the slice
# comes up empty and this script fails loudly instead of silently dumping a
# table built from nothing.
awk '/^QUALIFIERS unsigned int curand\(curandStatePhilox4_32_10_t \*state\)$/{p=1}
     /MRG32k3a RNG/{p=0}
     p' "$CURAND_INC/curand_kernel.h" > "$OUT/curand_philox_state.inc"
for sym in curand_init skipahead skipahead_sequence; do
    grep -q "QUALIFIERS.*$sym" "$OUT/curand_philox_state.inc" || {
        echo "error: slice of curand_kernel.h is missing $sym" >&2
        echo "  (NVIDIA moved or renamed it; fix the awk markers in build.sh)" >&2
        exit 1
    }
done

# -fwrapv: RAFT's `OutType(m >> 32) + params.start` is a signed int addition
# that CAN overflow, and does, for the adversarial `start = -2` case in the
# table below.  cuML's own call site (start = 0, end = n_rows <= INT_MAX)
# never reaches it.  -fwrapv makes the wrap defined and equal to what the
# hardware does, so the table records the machine's answer rather than an
# optimiser's licence to assume it cannot happen.
c++ -std=c++17 -O2 -fwrapv \
    -I "$CURAND_INC" -I "$OUT/shim" -I "$OUT" \
    -DORACLE_CURAND_VERSION="\"$CURAND_VER\"" \
    -o "$OUT/oracle" "$HERE/oracle.cpp"
"$OUT/oracle"
