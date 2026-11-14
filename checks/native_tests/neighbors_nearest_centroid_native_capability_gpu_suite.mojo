# SPDX-License-Identifier: Apache-2.0
"""NearestCentroid softmax-capability fixtures on the one physical GPU.

The descriptor's graphs share one DeviceContext, so the fixtures are
submitted in order (as `nearest_centroid_numeric_gpu_suite.mojo` does); the
kernels keep their own parallelism.
"""
from x_neighbors.native_descriptor_device import xn_native_v1
from checks.native_tests.neighbors_nearest_centroid_native_capability_cases import (
    NC_CAPABILITY_FIXTURES, fixture, require,
)


def run_checks() raises:
    var failed = 0
    for unit in range(NC_CAPABILITY_FIXTURES):
        try:
            fixture(xn_native_v1, unit)
        except error:
            failed += 1
            print("FAIL neighbors_nearest_centroid_native_capability_gpu fixture", unit, String(error))
    require(failed == 0, "neighbors_nearest_centroid_native_capability_gpu failed " + String(failed) + " fixtures")
    print("PASS neighbors_nearest_centroid_native_capability_gpu:", NC_CAPABILITY_FIXTURES, "softmax fixtures")
