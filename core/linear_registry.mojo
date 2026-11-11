# SPDX-License-Identifier: Apache-2.0
"""Native linear state registry; public buffer pins stay at UNPACK/PACK."""
from std.memory import ArcPointer
from core.linear_model_state import LinearModelState, linear_state_predict_request
from core.linear_boundary_run import linear_predict_run
from core.linear_boundary_types import LINEAR_PREDICT_ABI
from core.linear_entry_table import LinearEntryKey, LINEAR_TABLE_ESTIMATORS, linear_table_entry
from core.state_registry import StateRegistry, StateHandle

# The state contains admitted host descriptors, including on a GPU execution
# route. It has no device-context identity to merge across vendor columns.
# A call must retain the admitted buffers while holding this native lease.
comptime LINEAR_STATES = StateRegistry[LinearModelState, "MojolearnLinearModelStatesV1"]


def linear_lease_predict(key: LinearEntryKey, lease: ArcPointer[LinearModelState],
    x: Int, dst: Int, rows: Int, cols: Int, wide: Bool, estimator: String) raises:
    var request = linear_state_predict_request(lease[], x, dst, rows, cols, wide, estimator)
    var entry = linear_table_entry[LINEAR_TABLE_ESTIMATORS](key, 1, String(LINEAR_PREDICT_ABI), 1)
    try:
        linear_predict_run(entry, request)
    finally:
        _ = lease
