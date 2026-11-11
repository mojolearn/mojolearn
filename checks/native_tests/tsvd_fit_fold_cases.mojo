# SPDX-License-Identifier: Apache-2.0
"""TSVD exact fit over the actual base-context resident fold provider."""
from std.memory import bitcast
from core.msel_device_x import msel_fold_store, msel_device_x_context
from core.pca_native_api import PCAFoldStore
from core.tsvd_native_api import TSVDNativeDriver, TSVD_FOLD
from core.tsvd_native_fold import tsvd_fold_native_v1
from checks.native_tests.tsvd_fit_cases import TSVDFitState, input_values, fit_oracle, require

comptime TSVD_FOLD_FIXTURES = 3


def fold_fixture(unit: Int) raises:
    var driver = TSVDNativeDriver(tsvd_fold_native_v1)
    require(driver.metadata(0) == 1 and driver.metadata(2) == 16, "TSVD resident fold metadata")
    var source = input_values("boundary-6x3", 18)
    var indices: List[Int64] = [5, 4, 3, 2, 1, 0]
    if unit == 1:
        source[0] = bitcast[DType.float32](UInt32(0x7FC00042))
    elif unit == 2:
        indices[0] = 6
    var ctx = msel_device_x_context()
    var store = msel_fold_store()
    var base_id = -1
    var index_id = -1
    var state = TSVDFitState(3)
    try:
        base_id = store[].put(ctx, Int(source.unsafe_ptr()), 18 * 4)
        index_id = store[].put(ctx, Int(indices.unsafe_ptr()), 6 * 8)
        # Upload waits. Host mutations must not change resident storage.
        source[0] = -999
        indices[0] = -7
        var request = state.request(0, 6, 3, TSVD_FOLD, PCAFoldStore(base_id, 6, index_id))
        var answer = driver.fit(request)
        require((answer.finite != 0) == (unit == 1) and (answer.in_range != 0) == (unit == 2),
                "TSVD resident fit distinguishes finite and index-range status")
        if unit == 0:
            fit_oracle(state, "boundary-6x3", 6, 3, True)
        state.check_guards()
    finally:
        if index_id >= 0:
            store[].free(ctx, index_id)
        if base_id >= 0:
            store[].free(ctx, base_id)
        _ = source^
        _ = indices^
        _ = state^
        _ = ctx^
