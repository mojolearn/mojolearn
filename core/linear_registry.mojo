# SPDX-License-Identifier: Apache-2.0
"""Native linear state registry; public buffer pins stay at UNPACK/PACK."""
from std.memory import ArcPointer
from core.linear_model_state import LinearModelState, linear_state_predict_request
from core.linear_boundary_run import linear_predict_run
from core.native_descriptor_abi import NativeDescriptorCall
from core.state_registry import StateRegistry, StateHandle

# The state contains admitted host descriptors, including on a GPU execution
# route. It has no device-context identity to merge across vendor columns.
# A call must retain the admitted buffers while holding this native lease.
comptime LINEAR_STATES = StateRegistry[LinearModelState, "MojolearnLinearModelStatesV1"]


def linear_lease_predict(entry: NativeDescriptorCall, lease: ArcPointer[LinearModelState],
    x: Int, dst: Int, rows: Int, cols: Int, estimator: String, test_x: Bool = False) raises -> Int:
    """Predict through the leased state and the estimator's resolved
    prediction entry (core/linear_route.mojo: the device table's or the CPU
    table's, read once per estimator; this call reads no table slot). The
    entry's result word (1: the deferred finite test of X found a NaN or an
    infinity, see `linear_predict_run`)."""
    var request = linear_state_predict_request(lease[], x, dst, rows, cols, estimator)
    request.test_x = test_x
    var status = 0
    try:
        status = linear_predict_run(entry, request)
    finally:
        _ = lease
    return status
