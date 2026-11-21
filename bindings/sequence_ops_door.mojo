# SPDX-License-Identifier: Apache-2.0
# The sequence layer and resident-tensor operations as calls of the
# whole-model native descriptor of the tensor's / layer's image
# (sequence/sequence_ops_dispatch.mojo, sequence/sequence_ops_device_dispatch.mojo;
# lane dw-autograd-seq, 2026-11-21). Each helper takes typed words, makes ONE
# descriptor call and returns typed values: no CPython call into the image, no
# Python list of words. Before, each was the x_sequence binding's Python entry
# of the same name called through `py_call_typed`, and the doors decided
# device against host, and the fused forms, by whether an entry NAME was
# present in the image's entry dictionary.
#
# THE ROUTE WORD. What an image can do is decided once per connected entry
# dictionary (`seq_ops_route`) and held in it as one word; no door looks an
# entry name up to decide a route:
#   SEQ_ROUTE_NATIVE  the dictionary carries the whole-model descriptor
#                     (`model_native_api_v1`: both x_sequence images). Its
#                     provider is connected at the same moment and held under
#                     `native_model_provider`, so a tensor owned by such an
#                     image never exists without its provider (the shutdown
#                     finalizer in bindings/sequence_device_native_pack.mojo
#                     depends on that).
#   SEQ_ROUTE_DEVICE  that image is a device image (its `native_key` vendor,
#                     read once at import, is not the CPU's): it serves all
#                     fourteen operations. Without this bit the image is the
#                     host image: LayerNorm and the MoE forward / backward on
#                     host arrays only.
#   0                 another family's tensor owner (the Embedding image,
#                     connected in `sequence_extra_bindings` with its own
#                     device context): it has no whole-model descriptor, so
#                     its resident tensors are still served by the tensor
#                     entries that image exports. That image is not this
#                     lane's; the five tensor helpers below are the only
#                     place that reads those entries.
#
# GIL. Every call here keeps the GIL (`release_gil=False`), as the Python
# entries did: the tensor and weight registries are not locked.
from std.python import PythonObject
from bindings.py_balanced import py_get, py_item, py_setitem
from bindings.py_typed_call import py_call_typed
from bindings.sequence_keys_unpack import sequence_keys
from providers.python.neural_descriptor_pyapi_door import neural_descriptor_execute, neural_descriptor_reuse
from autograd.model_descriptor import MODEL_ABI_NAME
from sequence.neural_model_dispatch import (
    MODEL_OPERATION_COUNT, SEQ_TENSOR_ALLOC, SEQ_TENSOR_FREE, SEQ_TENSOR_UPLOAD,
    SEQ_TENSOR_DOWNLOAD, SEQ_TENSOR_COPY,
)

comptime SEQ_ROUTE_NATIVE = 1
comptime SEQ_ROUTE_DEVICE = 2
comptime SEQ_ROUTE_ALL = 3


def seq_ops_route(entries: PythonObject) raises -> Int:
    """The route word of a connected entry dictionary. After the first
    resolution this is one dictionary read."""
    var nk = sequence_keys()
    var held = py_get(entries, nk[].k_sequence_ops_route)
    if held is not None:
        return Int(py=held)
    var route = 0
    var factory = py_get(entries, nk[].k_model_native_api_v1)
    if factory is not None:
        var provider = neural_descriptor_reuse(py_get(entries, nk[].k_native_model_provider), factory,
                                               String(MODEL_ABI_NAME), MODEL_OPERATION_COUNT)
        py_setitem(entries, nk[].k_native_model_provider, provider)
        route = SEQ_ROUTE_NATIVE
        # (tier, vendor) of the image, read once at import by the bridge;
        # vendor word 0 is the CPU (core/byte_lm_entry_table.mojo).
        var key = py_get(entries, nk[].k_native_key)
        if key is not None:
            var vendor = key[1]
            if Int(py=vendor) != 0:
                route = SEQ_ROUTE_ALL
    py_setitem(entries, nk[].k_sequence_ops_route, PythonObject(route))
    return route


def seq_ops_device(entries: PythonObject) raises -> Bool:
    """True when the image of `entries` serves the device forms (resident
    tensors, device LayerNorm, device MoE forward, resident MoE weights)."""
    return seq_ops_route(entries) == SEQ_ROUTE_ALL


def seq_ops_native(entries: PythonObject) raises -> Bool:
    """True when the image of `entries` carries the whole-model descriptor."""
    return (seq_ops_route(entries) & SEQ_ROUTE_NATIVE) != 0


def seq_ops_call(entries: PythonObject, operation: Int, addresses: List[Int], ints: List[Int],
                 floats: List[Float64]) raises -> Int:
    """One descriptor call on the provider `seq_ops_route` connected. The
    caller has established SEQ_ROUTE_NATIVE for `entries`."""
    var nk = sequence_keys()
    var provider = py_item(entries, nk[].k_native_model_provider)
    return neural_descriptor_execute(provider, operation, addresses, ints, floats, False)


def seq_tensor_new(entries: PythonObject, count: Int, zero: Bool) raises -> Int:
    """A new resident tensor of `count` >= 1 floats in the image of
    `entries`; its handle, which the caller owns."""
    if seq_ops_native(entries):
        var addresses = List[Int]()
        var ints: List[Int] = [count, 1 if zero else 0]
        var floats = List[Float64]()
        return seq_ops_call(entries, SEQ_TENSOR_ALLOC, addresses, ints, floats)
    var nk = sequence_keys()
    return Int(py=py_call_typed(py_item(entries, nk[].k_seq_tensor_alloc), count, 1 if zero else 0))


def seq_tensor_release(entries: PythonObject, handle: Int) raises:
    """Release a handle the caller owns."""
    if seq_ops_native(entries):
        var addresses = List[Int]()
        var ints: List[Int] = [handle]
        var floats = List[Float64]()
        _ = seq_ops_call(entries, SEQ_TENSOR_FREE, addresses, ints, floats)
        return
    var nk = sequence_keys()
    _ = py_call_typed(py_item(entries, nk[].k_seq_tensor_free), handle)


def seq_tensor_put(entries: PythonObject, handle: Int, address: Int, count: Int) raises:
    """The caller's `count` floats at `address` into the handle."""
    if seq_ops_native(entries):
        var addresses: List[Int] = [address]
        var ints: List[Int] = [handle, count]
        var floats = List[Float64]()
        _ = seq_ops_call(entries, SEQ_TENSOR_UPLOAD, addresses, ints, floats)
        return
    var nk = sequence_keys()
    _ = py_call_typed(py_item(entries, nk[].k_seq_tensor_upload), handle, address, count)


def seq_tensor_get(entries: PythonObject, handle: Int, address: Int, count: Int) raises:
    """The handle's first `count` floats into the caller's array at `address`."""
    if seq_ops_native(entries):
        var addresses: List[Int] = [address]
        var ints: List[Int] = [handle, count]
        var floats = List[Float64]()
        _ = seq_ops_call(entries, SEQ_TENSOR_DOWNLOAD, addresses, ints, floats)
        return
    var nk = sequence_keys()
    _ = py_call_typed(py_item(entries, nk[].k_seq_tensor_download), handle, address, count)


def seq_tensor_clone(entries: PythonObject, dst: Int, src: Int, count: Int) raises:
    """Device to device: `src`'s first `count` floats into `dst`."""
    if seq_ops_native(entries):
        var addresses = List[Int]()
        var ints: List[Int] = [dst, src, count]
        var floats = List[Float64]()
        _ = seq_ops_call(entries, SEQ_TENSOR_COPY, addresses, ints, floats)
        return
    var nk = sequence_keys()
    var entry = py_get(entries, nk[].k_seq_tensor_copy)
    if entry is None:
        raise Error("RuntimeError: native binding has no device-to-device tensor copy")
    _ = py_call_typed(entry, dst, src, count)
