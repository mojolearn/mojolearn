# SPDX-License-Identifier: Apache-2.0
"""Linear prediction RUN: typed Mojo values and a compiled native endpoint."""
from core.linear_boundary_types import LinearPredictRequest, linear_predict_flags
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call


def linear_predict_run(entry: NativeDescriptorCall, request: LinearPredictRequest) raises -> Int:
    """The prediction entry's result word: 0, or 1 when `request.test_x`
    and X holds a NaN or an infinity (the caller refuses)."""
    return native_descriptor_call(entry, 0, [request.x, request.coef, request.dst],
        [request.rows, request.cols, linear_predict_flags(request)], [request.intercept])
