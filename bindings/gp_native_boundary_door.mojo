# SPDX-License-Identifier: Apache-2.0
"""GP image pins and the native GP table entry around synchronous RUN
(core/gp_entry_table.mojo; lane gp-entry-table, 2026-11-12)."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_type, py_get, py_fields
from bindings.native_loaded_module import native_loaded_module
from bindings.gp_kernel_storage_boundary import GPPackedKernel
from bindings.classical_arrays import classical_address
from core.gp_entry_table import GPEntryKey
from gaussian_process.gp_native_boundary_run import GPNative, gp_native_run
from core.gp_native_api import (
    GPKernelView,
    GPRFitRequest, GPRQueryRequest, GPRCovarianceRequest, GPRSampleRequest,
    GPRGradientRequest, GPROptimizeRequest, GPCFitRequest, GPCFitAllRequest,
    GPCQueryRequest, GPCQueryAllRequest, GPCCombineRequest, GPThetaRequest,
    GPQueryResult,
)


def gp_entry_key(binding: PythonObject) raises -> GPEntryKey:
    """UNPACK: the GP table key an image recorded as its module attribute
    `gp_native_key` = `(mode, vendor, column)` when it published
    (core/gp_entry_table.mojo). One exact read of the module's `__dict__`."""
    var b = native_loaded_module("builtins")
    if py_type(binding) is not py_type(b):
        raise Error("TypeError: GP execution requires its connected native module")
    var key = py_get(py_fields(binding), PythonObject("gp_native_key"))
    if (key is None or py_type(key) is not b.tuple or len(key) != 3 or py_type(key[0]) is not b.int
            or py_type(key[1]) is not b.int or py_type(key[2]) is not b.int):
        raise Error("ImportError: mojolearn: the GP binding publishes no native gp_native_key table key; rebuild it")
    return GPEntryKey(Int(py=key[0]), Int(py=key[1]), Int(py=key[2]))


struct GPConnection(Movable):
    # `image` pins the connected module (and so the published entry's code)
    # for the public call; its address is the lease identity of
    # bindings/gp_resident_boundary.mojo. RUN reads only the typed `native`
    # connection (gaussian_process/gp_native_boundary_run.mojo).
    var image: PythonObject
    var native: GPNative

    def __init__(out self, binding: PythonObject) raises:
        self.native = GPNative(gp_entry_key(binding))
        self.image = binding


def gp_kernel_view(arrays: PythonObject) raises -> GPKernelView:
    var b = native_loaded_module("builtins")
    if py_type(arrays) is not b.tuple and py_type(arrays) is not b.list:
        return arrays.downcast_value_ptr[GPPackedKernel]()[].storage.view()
    # The packed storage remains pinned by the caller. This adapter also
    # serves the explicit cross-interpreter transport unpack boundary.
    return GPKernelView(Int(py=classical_address(arrays[0])), Int(py=classical_address(arrays[1])),
        Int(py=classical_address(arrays[2])), Int(py=classical_address(arrays[3])),
        Int(py=arrays[0].shape[0]), Int(py=arrays[4]))


def gp_seed_word(seed: PythonObject) raises -> UInt64:
    if py_type(seed) is not py_type(PythonObject(0)):
        raise Error("TypeError: GP seed must resolve to a native int")
    ref cpy = Python().cpython()
    var value = cpy.lib.get_function[UInt64]("PyLong_AsUnsignedLongLongMask")(seed._obj_ptr)
    _ = seed
    return value


def gp_run(connection: GPConnection, request: GPRFitRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRQueryRequest) raises -> GPQueryResult:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRCovarianceRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRSampleRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPRGradientRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPROptimizeRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCFitRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCFitAllRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCQueryRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCQueryAllRequest) raises -> GPQueryResult:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPCCombineRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection


def gp_run(connection: GPConnection, request: GPThetaRequest) raises -> Int:
    try:
        with GILReleased(Python()):
            return gp_native_run(connection.native, request)
    finally:
        _ = connection
