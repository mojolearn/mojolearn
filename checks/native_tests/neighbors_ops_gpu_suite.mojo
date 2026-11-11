# SPDX-License-Identifier: Apache-2.0
"""The device column of the neighbors operations table. The entries share
the base binding's device context and resident registry, so the cases
submit in order; the kernels keep their native parallelism (no concurrency
claim)."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.neighbors_entry_table import NeighborsEntryKey, neighbors_vendor_word
from neighbors.ops_entry_device import neighbors_ops_device_v1, NEIGHBORS_OPS_DEVICE_API
from checks.native_tests.neighbors_ops_cases import (
    NEIGHBORS_SHARED_CASES, NEIGHBORS_DEVICE_CASES, require, run_shared_case, run_device_case,
    check_neighbors_table,
)


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var vendor = neighbors_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "neighbors_ops_gpu: this build has no GPU vendor")
    var api = NEIGHBORS_OPS_DEVICE_API.get_or_create_ptr()
    check_neighbors_table(NeighborsEntryKey(mode, vendor), Int(api[].entries.unsafe_ptr()), neighbors_ops_device_v1)
    var failed = 0
    for case_id in range(NEIGHBORS_SHARED_CASES):  # driver-loop: over the ordered device cases
        try:
            run_shared_case(neighbors_ops_device_v1, case_id)
        except error:
            print("FAIL neighbors_ops_gpu shared case", case_id, String(error))
            failed += 1
    for case_id in range(NEIGHBORS_DEVICE_CASES):  # driver-loop: over the ordered device cases
        try:
            run_device_case(neighbors_ops_device_v1, case_id)
        except error:
            print("FAIL neighbors_ops_gpu device case", case_id, String(error))
            failed += 1
    require(failed == 0, "neighbors_ops_gpu failed " + String(failed) + " cases")
    print("PASS neighbors_ops_gpu:", NEIGHBORS_SHARED_CASES + NEIGHBORS_DEVICE_CASES, "native neighbors table cases")
