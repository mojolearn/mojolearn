# SPDX-License-Identifier: Apache-2.0
"""RUN half of the CNN DeviceTensor operations (`bindings/xcnn_tensor_native.mojo`
is the UNPACK/PACK boundary; lane doorsplit-cnn-xseq, 2026-11-11).

Typed values only: the shape-word and handle-word checks with the tensor's
refusal texts, reshape inference (`xcnn_run_infer_dims`), and the native
release of a resident allocation handle. The boundary reads the tensor's
slots and the caller's shape objects into words and releases the GIL around
`xcnn_tensor_run_free`, where the door released it before the split.
"""
from core.xcnn_entry_table import XcnnEntryKey, XC_RES_FREE
from x_cnn.door_run import xcnn_run_call, xcnn_run_infer_dims, xcnn_size


def xcnn_tensor_run_check_dim(size: Int, infer: Bool) raises:
    """One shape word: nonnegative, or -1 when the shape may infer one."""
    if size < (-1 if infer else 0):
        raise Error("ValueError: CNN tensor dimensions must be nonnegative")


def xcnn_tensor_run_check_handle(handle: Int) raises:
    """A tensor's explicit resident handle is a positive word."""
    if handle < 1:
        raise Error("TypeError: CNN tensor handle requires a positive native integer")


def xcnn_tensor_run_handle_live(handle: Int) -> Bool:
    """Whether a finalized tensor's handle word names an allocation to free."""
    return handle >= 1


def xcnn_tensor_run_reshape_dims(var dims: List[Int], shape: List[Int]) raises -> List[Int]:
    """`DeviceTensor.reshape(*dims)`: the requested words with at most one -1
    filled from the element count of `shape`."""
    return xcnn_run_infer_dims(dims^, xcnn_size(shape))


def xcnn_tensor_run_free(key: XcnnEntryKey, handle: Int) raises:
    """Release one resident allocation handle of the image `key`. The caller
    cleared every slot that named it and has released the GIL."""
    _ = xcnn_run_call(key, XC_RES_FREE, [handle], List[Int]())
