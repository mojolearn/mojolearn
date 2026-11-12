# SPDX-License-Identifier: Apache-2.0
"""The device column of the UMAP operations table. The entries share the
metrics image's device context, so the cases submit in order; the kernels
keep their native parallelism (no concurrency claim)."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.neighbors_entry_table import NeighborsEntryKey, neighbors_vendor_word
from umap.ops_entry_device import umap_ops_device_v1, UMAP_OPS_DEVICE_API
from checks.native_tests.umap_ops_cases import UMAP_SHARED_CASES, require, run_shared_case, check_umap_table


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var vendor = neighbors_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "umap_ops_gpu: this build has no GPU vendor")
    var api = UMAP_OPS_DEVICE_API.get_or_create_ptr()
    check_umap_table(NeighborsEntryKey(mode, vendor), Int(api[].entries.unsafe_ptr()), umap_ops_device_v1)
    var failed = 0
    for case_id in range(UMAP_SHARED_CASES):  # driver-loop: over the ordered device cases
        try:
            run_shared_case(umap_ops_device_v1, case_id)
        except error:
            print("FAIL umap_ops_gpu case", case_id, String(error))
            failed += 1
    require(failed == 0, "umap_ops_gpu failed " + String(failed) + " cases")
    print("PASS umap_ops_gpu:", UMAP_SHARED_CASES, "native UMAP table cases")
