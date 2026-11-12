# SPDX-License-Identifier: Apache-2.0
"""The device column of the ANN operations table. The entries share
the x_ann device context and resident registry, so the cases
submit in order; the kernels keep their native parallelism (no concurrency
claim)."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.neighbors_entry_table import NeighborsEntryKey, neighbors_vendor_word
from x_ann.ops_entry_device import ann_ops_device_v1, ANN_OPS_DEVICE_API
from checks.native_tests.ann_ops_cases import (
    ANN_SHARED_CASES, ANN_DEVICE_CASES, require, run_shared_case, run_device_case,
    check_ann_table,
)


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var vendor = neighbors_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "ann_ops_gpu: this build has no GPU vendor")
    var api = ANN_OPS_DEVICE_API.get_or_create_ptr()
    check_ann_table(NeighborsEntryKey(mode, vendor), Int(api[].entries.unsafe_ptr()), ann_ops_device_v1)
    var failed = 0
    for case_id in range(ANN_SHARED_CASES):  # driver-loop: over the ordered device cases
        try:
            run_shared_case(ann_ops_device_v1, case_id)
        except error:
            print("FAIL ann_ops_gpu shared case", case_id, String(error))
            failed += 1
    for case_id in range(ANN_DEVICE_CASES):  # driver-loop: over the ordered device cases
        try:
            run_device_case(ann_ops_device_v1, case_id)
        except error:
            print("FAIL ann_ops_gpu device case", case_id, String(error))
            failed += 1
    require(failed == 0, "ann_ops_gpu failed " + String(failed) + " cases")
    print("PASS ann_ops_gpu:", ANN_SHARED_CASES + ANN_DEVICE_CASES, "native ANN table cases")
