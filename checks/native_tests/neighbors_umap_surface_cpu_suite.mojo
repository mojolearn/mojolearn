# SPDX-License-Identifier: Apache-2.0
"""UMAP fit/transform numerics on the CPU column; independent fixtures run across host workers."""
from core.host_parallel import host_parallelize
from umap.ops_entry_host import umap_ops_host_v1
from checks.native_tests.neighbors_umap_surface_cases import UMAP_SURFACE_FIXTURES, fixture, require


def run_checks() raises:
    var failures = List[Int](length=UMAP_SURFACE_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            fixture(umap_ops_host_v1, unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL neighbors_umap_surface_cpu fixture", unit, String(error))
    host_parallelize(task, UMAP_SURFACE_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "neighbors_umap_surface_cpu failed " + String(failed) + " fixtures")
    print("PASS neighbors_umap_surface_cpu:", UMAP_SURFACE_FIXTURES, "UMAP fit/transform fixtures")
