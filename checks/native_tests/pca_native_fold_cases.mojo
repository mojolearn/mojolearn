# SPDX-License-Identifier: Apache-2.0
"""Resident PCA gathers through the actual base-context native fold provider."""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.msel_device_x import msel_fold_store, msel_device_x_context
from core.pca_native_api import PCANativeDriver, PCAFoldStore
from core.pca_native_fold import pca_fold_native_v1
from checks.native_tests.pca_native_cases import (
    PCATestState, input_values, fit_oracle, projection_oracle, write_guarded, require,
)

comptime PCA_NATIVE_FOLD_FIXTURES = 3


def fold_fixture(unit: Int) raises:
    var driver = PCANativeDriver(pca_fold_native_v1)
    require(driver.metadata(0) == 1 and driver.metadata(2) == 21, "PCA fold mode/capabilities")
    var source = input_values("full-8x3", 24)
    var original = source.copy()
    var indices: List[Int64] = [7, 6, 5, 4, 3, 2, 1, 0]
    if unit == 1:
        source[0] = bitcast[DType.float32](UInt32(0x7FC00042))
    elif unit == 2:
        indices[0] = 8
    var ctx = msel_device_x_context()
    var store = msel_fold_store()
    var base_id = -1
    var index_id = -1
    try:
        base_id = store[].put(ctx, Int(source.unsafe_ptr()), 24 * 4)
        index_id = store[].put(ctx, Int(indices.unsafe_ptr()), 8 * 8)
        # Each put synchronizes its copy. Changing the original host storage
        # now must not replace resident data or the admitted index order.
        source[0] = 12345.0
        indices[0] = -7
        var fold = PCAFoldStore(base_id, 8, index_id)
        var state = PCATestState(8, 3, 3)
        var answer = driver.fit(state.fit_request(0, 8, 3, 3, True, 0, False, False, fold))
        if unit == 0:
            require(answer.keep == 3 and answer.finite == 0 and answer.in_range == 0,
                    "PCA resident full-fit status")
            fit_oracle(state, "full-8x3", 8, 3, 3, 3, True)
        else:
            require((answer.finite != 0) == (unit == 1) and (answer.in_range != 0) == (unit == 2),
                    "PCA resident fit finite/range status distinction")
            write_guarded(state.components, List[Float32](1, 0, 0, 0, 1, 0, 0, 0, 1))
            write_guarded(state.mean, List[Float32](0, 0, 0))
        var status = driver.query(state.query_request(0, Int(state.projected.unsafe_ptr()) + 4,
            8, 3, 3, 8, False, fold))
        require((status.finite != 0) == (unit == 1) and (status.in_range != 0) == (unit == 2),
                "PCA resident query finite/range status distinction")
        if unit == 0:
            var selected = List[Float32](unsafe_uninit_length=24)
            var sp = selected.unsafe_ptr()
            var op = original.unsafe_ptr()
            def row_word(index: Int) {imm sp, imm op}:
                sp.unsafe_store(index, op[(7 - index // 3) * 3 + index % 3])
            host_parallelize(row_word, 24)
            projection_oracle(selected, state, 8, 3, 3)
            _ = selected^
        state.check_guards()
        _ = state^
    finally:
        if index_id >= 0:
            store[].free(ctx, index_id)
        if base_id >= 0:
            store[].free(ctx, base_id)
        _ = source^
        _ = original^
        _ = indices^
        _ = ctx^
