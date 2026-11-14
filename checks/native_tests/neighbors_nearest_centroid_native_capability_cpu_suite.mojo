# SPDX-License-Identifier: Apache-2.0
"""NearestCentroid softmax-capability fixtures; independent fixtures run across host workers."""
from core.host_parallel import host_parallelize
from x_neighbors.native_descriptor_host import xn_native_v1
from checks.native_tests.neighbors_nearest_centroid_native_capability_cases import (
    NC_CAPABILITY_FIXTURES, fixture, require,
)


def run_checks() raises:
    var failures = List[Int](length=NC_CAPABILITY_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(xn_native_v1, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_nearest_centroid_native_capability_cpu fixture", unit, String(error))
    host_parallelize(task, NC_CAPABILITY_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "neighbors_nearest_centroid_native_capability_cpu failed " + String(failed) + " fixtures")
    print("PASS neighbors_nearest_centroid_native_capability_cpu:", NC_CAPABILITY_FIXTURES, "softmax fixtures")
