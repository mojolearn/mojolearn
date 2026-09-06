#!/bin/sh
# Build and run the CCCL shuffle_iterator oracle, and write the reference
# table this repository commits.
#
#     ensemble/tools/shuffle_oracle/build.sh > ensemble/bench/shuffle_oracle.txt
#
# THE PIN. cuML does not pin CCCL directly: cpp/cmake/thirdparty/get_cccl.cmake
# calls rapids_cpm_cccl(), which resolves through rapids-cmake. cuML v26.08.00
# sets rapids-cmake-version 26.08 (cmake/rapids_config.cmake + VERSION), and
# rapids-cmake v26.08.00's rapids-cmake/cpm/versions.json names CCCL 3.4.3 at
# commit 9d65c77f9763cfec20452e4071128d3f0bd2625b. That is what this builds
# against.
#
# CAVEAT, recorded rather than smoothed over: rapids_cpm_cccl goes through
# rapids_cpm_find, which will accept a pre-installed CCCL (a cuda-cccl conda
# package, or a CUDA toolkit copy) instead of the pinned tarball unless
# rapids_cmake_always_download is set. So this is the version cuML RESOLVES,
# and a given distributed binary may have been built against another. The four
# load-bearing headers are byte-identical between 3.4.3 and CCCL main (3.6.0)
# apart from a _CCCL_API -> _CCCL_HOST_DEVICE_API macro rename, so the
# arithmetic has not moved across that range; versions older than 3.4.3 were
# not checked.
#
# No CUDA toolkit and no GPU are required. cuda::shuffle_iterator and
# cuda::std::minstd_rand are _CCCL_HOST_DEVICE, so this is a host compile.
set -eu

CCCL_DIR="${CCCL_DIR:-$HOME/CascadeProjects/upstream/cccl-3.4.3}"
CCCL_COMMIT=9d65c77f9763cfec20452e4071128d3f0bd2625b

if [ ! -d "$CCCL_DIR/libcudacxx/include" ]; then
    echo "error: no CCCL checkout at $CCCL_DIR" >&2
    echo "  git init $CCCL_DIR && cd $CCCL_DIR \\" >&2
    echo "    && git fetch --depth 1 --filter=blob:none \\" >&2
    echo "         https://github.com/NVIDIA/cccl.git $CCCL_COMMIT \\" >&2
    echo "    && git checkout FETCH_HEAD" >&2
    exit 1
fi

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT

c++ -std=c++17 -O1 -I "$CCCL_DIR/libcudacxx/include" \
    -o "$OUT/oracle" "$HERE/oracle.cpp"
"$OUT/oracle"
