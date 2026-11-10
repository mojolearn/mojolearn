# SPDX-License-Identifier: Apache-2.0
"""Linear prediction RUN: typed Mojo values and a compiled native endpoint."""
from core.linear_boundary_types import LinearPredictRequest
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call


def linear_predict_run(entry: NativeDescriptorCall, request: LinearPredictRequest) raises:
    _ = native_descriptor_call(entry, 0, [request.x, request.coef, request.dst],
        [request.rows, request.cols, Int(request.wide)], [request.intercept])
