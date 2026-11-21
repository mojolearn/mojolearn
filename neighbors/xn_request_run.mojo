# SPDX-License-Identifier: Apache-2.0
"""One typed x_neighbors operation request on a native driver, and the
native scratch the neighbors RUN modules hand to operations.

`XnOpRequest` (neighbors/neighbors_graph_run.mojo) is the operation name and
its address, integer and float words; `xn_request_run` hands those words to
the image's native descriptor entry as spans. Nothing here imports Python:
the caller has released the GIL and keeps every addressed buffer alive.
"""
from core.native_call_output import NativeCallOutput
from core.native_descriptor_call import native_descriptor_call_spans
from core.neighbors_native_driver import XnDriver
from neighbors.neighbors_graph_run import XnOpRequest
from x_neighbors.descriptor_schema import xn_operation_id


def xn_request_run(driver: XnDriver, request: XnOpRequest) raises -> Int:
    """Run `request` and return the operation's result word. An empty span
    has no address."""
    var aa = 0
    var pp = 0
    var ff = 0
    if len(request.a) > 0:
        aa = Int(request.a.unsafe_ptr())
    if len(request.p) > 0:
        pp = Int(request.p.unsafe_ptr())
    if len(request.f) > 0:
        ff = Int(request.f.unsafe_ptr())
    var output = NativeCallOutput()
    try:
        return native_descriptor_call_spans(driver.entry, xn_operation_id(request.name), aa, len(request.a),
                                            pp, len(request.p), ff, len(request.f), output)
    finally:
        # Integer addresses carry no lifetime information.
        _ = request
        _ = output


def xn_scratch_f32(count: Int) -> List[Float32]:
    """Zeroed float32 scratch of `count` words (one word for an empty span,
    so an operation is never handed a null address)."""
    return List[Float32](length=max(count, 1), fill=Float32(0.0))


def xn_scratch_i32(count: Int) -> List[Int32]:
    """Zeroed int32 scratch of `count` words (one word for an empty span)."""
    return List[Int32](length=max(count, 1), fill=Int32(0))
