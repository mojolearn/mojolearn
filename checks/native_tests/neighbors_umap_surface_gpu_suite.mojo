# SPDX-License-Identifier: Apache-2.0
"""UMAP fit/transform numerics on the device column.

The entries share the metrics image's device context, so the fixtures submit
in order (as `umap_ops_gpu_suite.mojo` does); the kernels keep their native
parallelism.
"""
from umap.ops_entry_device import umap_ops_device_v1
from checks.native_tests.neighbors_umap_surface_cases import UMAP_SURFACE_FIXTURES, fixture, require


def run_checks() raises:
    var failed = 0
    for unit in range(UMAP_SURFACE_FIXTURES):
        try:
            fixture(umap_ops_device_v1, unit)
        except error:
            failed += 1
            print("FAIL neighbors_umap_surface_gpu fixture", unit, String(error))
    require(failed == 0, "neighbors_umap_surface_gpu failed " + String(failed) + " fixtures")
    print("PASS neighbors_umap_surface_gpu:", UMAP_SURFACE_FIXTURES, "UMAP fit/transform fixtures")
