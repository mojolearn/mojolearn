# SPDX-License-Identifier: Apache-2.0
"""The device column of the PCA owned-fit table. The entry shares the
estimators binding's device context, so the cases submit in order; the
kernels keep their native parallelism (no concurrency claim)."""
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from core.neighbors_entry_table import NeighborsEntryKey, neighbors_vendor_word
from decomposition.pca_ops_entry_device import pca_ops_device_v1, PCA_OPS_DEVICE_API
from checks.native_tests.pca_owned_ops_cases import PCA_OWNED_CASES, require, run_case, check_pca_table


def run_checks() raises:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var vendor = neighbors_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "pca_owned_ops_gpu: this build has no GPU vendor")
    var api = PCA_OPS_DEVICE_API.get_or_create_ptr()
    check_pca_table(NeighborsEntryKey(mode, vendor), Int(api[].entries.unsafe_ptr()), pca_ops_device_v1)
    var failed = 0
    for case_id in range(PCA_OWNED_CASES):  # driver-loop: over the ordered device cases
        try:
            run_case(pca_ops_device_v1, case_id)
        except error:
            print("FAIL pca_owned_ops_gpu case", case_id, String(error))
            failed += 1
    require(failed == 0, "pca_owned_ops_gpu failed " + String(failed) + " cases")
    print("PASS pca_owned_ops_gpu:", PCA_OWNED_CASES, "native PCA owned-fit table cases")
