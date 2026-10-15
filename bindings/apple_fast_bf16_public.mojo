# SPDX-License-Identifier: Apache-2.0
"""Native public boundary for the opt-in Apple fast BF16 candidate.

The BF16 simdgroup kernel already exists in gemm.afn_apple_fast. This
boundary owns validation, host packing, output allocation, upload, dispatch,
download and synchronization. Python exports bind these functions directly.
No _buffer, Array method, NumPy operation or Python callback drives a call.
The CPython builtins used here (array.array and memoryview) only own/copy
bytes. Array slots are initialized natively; its Python constructor is not
called. A trace regression test checks that no mojolearn Python frame runs.

Packing intentionally copies operands through the C buffer API's native
memoryview.tobytes and array.frombytes operations, supporting strided and
read-only buffers without trusting an arbitrary __array_interface__ address.
Those copies are INSIDE the public operation and must stay in its measured
clock. A future zero-copy boundary is separate work, not assumed evidence.

The candidate remains default-off through AFN_GEMM_BF16_MMA. It refuses
instead of silently routing to the identical product when unavailable.
No numerical source or dispatch of the identical tier changes here.
"""

from std.collections import OwnedKwargsDict
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, u16_ptr
from core.neural_context import process_ctx
from gemm.afn_apple_fast import (
    AFN_GEMM_BF16_MMA,
    afn_gemm_bf16_bits_into,
    afn_gemm_bf16_into,
)
from gemm.checks.gemm_lowbit import bf16_narrow, bf16_widen
from gemm.contract import OP_NN, OP_NT, OP_TN
from max.gpu.host import DeviceContext


struct _Boundary(Defaultable, Movable):
    var builtins: PythonObject
    var array_type: PythonObject
    var array_class: PythonObject

    def __init__(out self):
        self.builtins = PythonObject(None)
        self.array_type = PythonObject(None)
        self.array_class = PythonObject(None)


comptime _BOUNDARY = _Global[
    StorageType=_Boundary, name="MojoAppleFastBf16Boundary",
    init_fn=_Boundary.__init__,
]


def initialize_fast_bf16_boundary() raises:
    # Import-time metadata only. Every callable subsequently used is a
    # CPython native builtin, never a Python Array or buffer helper method.
    comptime if AFN_GEMM_BF16_MMA:
        var state = _BOUNDARY.get_or_create_ptr()
        state[].builtins = Python.import_module("builtins")
        state[].array_type = Python.import_module("array").array
        state[].array_class = Python.import_module("mojolearn._array").Array


def fast_bf16_available_binding() raises -> PythonObject:
    return PythonObject(AFN_GEMM_BF16_MMA)


def _require_candidate() raises:
    comptime if not AFN_GEMM_BF16_MMA:
        raise Error("Apple fast BF16 requires a fast Apple binding built with MOJOLEARN_AFN_GEMM_BF16_MMA; no identical fallback is used")


def _shape_size(shape: PythonObject) raises -> Int:
    var total = 1
    for axis in range(len(shape)):
        var extent = Int(py=shape[axis])
        # Kernels use signed Int32 element counts and index products. This
        # is their representable address domain, not a benchmark-size cap.
        if extent <= 0 or extent > 2147483647 // total:
            raise Error("fast BF16: shape must be nonempty and fit signed 32-bit kernel indexing")
        total *= extent
    return total


struct _Input(Movable):
    var shape: PythonObject
    var store: PythonObject
    var address: Int
    var count: Int
    var bf16: Bool

    def __init__(out self, value: PythonObject, allow_f32: Bool, allow_bf16: Bool) raises:
        var state = _BOUNDARY.get_or_create_ptr()
        var builtins = state[].builtins
        var view = PythonObject(None)
        var shape = PythonObject(None)
        # Array.__buffer__ is a Python method on 3.12+. Read its native
        # backing memoryview directly on all supported Python versions.
        if builtins.type(value) is state[].array_class:
            view = value._mv
            shape = value.shape
            var dtype = String(value.dtype)
            if dtype != "<f4" and dtype != "<u2":
                raise Error("fast BF16: Array operands must be float32 or uint16 BF16 bits")
            # _mv is flat storage; recover a multidimensional view with
            # declared strides before packing Fortran-ordered Array data.
            if String(value.order) != "C":
                raise Error("fast BF16: Array operands must be C-contiguous; buffer-protocol operands may be strided")
        else:
            view = builtins.memoryview(value)
            shape = view.shape
        var format = String(view.format)
        self.bf16 = format == "H" or format == "@H" or format == "=H" or format == "<H"
        var f32 = format == "f" or format == "@f" or format == "=f" or format == "<f"
        if (not self.bf16 and not f32) or (self.bf16 and not allow_bf16) or (f32 and not allow_f32):
            raise Error("fast BF16: expected float32 or uint16 BF16 bits with native byte order")
        self.shape = shape
        self.count = _shape_size(shape)
        var itemsize = 2 if self.bf16 else 4
        if Int(py=view.nbytes) != self.count * itemsize or Int(py=view.itemsize) != itemsize:
            raise Error("fast BF16: buffer span and shape disagree")
        self.store = state[].array_type("H" if self.bf16 else "f")
        self.store.frombytes(view.tobytes("C"))
        self.address = Int(py=self.store.buffer_info()[0])


def _new_array(shape: PythonObject, bf16: Bool) raises -> PythonObject:
    var state = _BOUNDARY.get_or_create_ptr()
    var count = _shape_size(shape)
    var itemsize = 2 if bf16 else 4
    var store = state[].array_type("H" if bf16 else "f")
    store.frombytes(state[].builtins.bytes(count * itemsize))
    # object.__new__ and the slot setters are CPython C operations. Do not
    # call Array.__init__, _init_owned, _set_meta, reshape or from_buffer.
    var result = state[].builtins.object.__new__(state[].array_class)
    result._store = store
    result._base = PythonObject(None)
    result._pin = PythonObject(None)
    result._mv = state[].builtins.memoryview(store)
    result._addr = store.buffer_info()[0]
    result._readonly = PythonObject(False)
    result.shape = shape
    result.dtype = PythonObject("<u2" if bf16 else "<f4")
    result.ndim = PythonObject(len(shape))
    result.size = PythonObject(count)
    result.nbytes = PythonObject(count * itemsize)
    result.itemsize = PythonObject(itemsize)
    result.order = PythonObject("C")
    var strides = state[].builtins.list()
    var stride = itemsize
    for index in range(len(shape)):
        var axis = len(shape) - index - 1
        strides.insert(0, stride)
        stride *= Int(py=shape[axis])
    result.strides = state[].builtins.tuple(strides)
    return result


def fast_to_bf16_binding(value: PythonObject) raises -> PythonObject:
    _require_candidate()
    var source = _Input(value, True, False)
    var result = _new_array(source.shape, True)
    var dst_address = Int(py=result._addr)
    with GILReleased(Python()):
        var ctx = process_ctx["MojoAppleFastBf16PublicContext"]()
        var src = ctx.enqueue_create_buffer[DType.float32](source.count)
        var dst = ctx.enqueue_create_buffer[DType.uint16](source.count)
        ctx.enqueue_copy(dst_buf=src, src_ptr=f32_ptr(source.address))
        bf16_narrow(ctx, dst, src, source.count)
        ctx.enqueue_copy(dst_ptr=u16_ptr(dst_address), src_buf=dst)
        ctx.synchronize()
        _ = src^
        _ = dst^
        ctx.synchronize()
    return result


def fast_from_bf16_binding(value: PythonObject) raises -> PythonObject:
    _require_candidate()
    var source = _Input(value, False, True)
    var result = _new_array(source.shape, False)
    var dst_address = Int(py=result._addr)
    with GILReleased(Python()):
        var ctx = process_ctx["MojoAppleFastBf16PublicContext"]()
        var src = ctx.enqueue_create_buffer[DType.uint16](source.count)
        var dst = ctx.enqueue_create_buffer[DType.float32](source.count)
        ctx.enqueue_copy(dst_buf=src, src_ptr=u16_ptr(source.address))
        bf16_widen(ctx, dst, src, source.count)
        ctx.enqueue_copy(dst_ptr=f32_ptr(dst_address), src_buf=dst)
        ctx.synchronize()
        _ = src^
        _ = dst^
        ctx.synchronize()
    return result


def fast_matmul_bf16_binding(
    a: PythonObject, b: PythonObject, kwargs: OwnedKwargsDict[PythonObject]
) raises -> PythonObject:
    _require_candidate()
    var transpose_a = False
    var transpose_b = False
    var out = PythonObject(None)
    for entry in kwargs.items():
        var name = String(entry.key)
        if name == "transpose_a":
            transpose_a = Bool(entry.value)
        elif name == "transpose_b":
            transpose_b = Bool(entry.value)
        elif name == "out":
            out = entry.value
        else:
            raise Error("fast matmul_bf16: unknown keyword " + name)
    if transpose_a and transpose_b:
        raise Error("fast matmul_bf16 supports NN, NT and TN, not both transposes")
    var left = _Input(a, True, True)
    var right = _Input(b, False, True)
    if len(left.shape) != 2 or len(right.shape) != 2:
        raise Error("fast matmul_bf16 requires two-dimensional operands")
    var m = Int(py=left.shape[1 if transpose_a else 0])
    var k = Int(py=left.shape[0 if transpose_a else 1])
    var n = Int(py=right.shape[0 if transpose_b else 1])
    var kb = Int(py=right.shape[1 if transpose_b else 0])
    if k != kb:
        raise Error("fast matmul_bf16: contracted extents differ")
    var op = OP_TN if transpose_a else (OP_NT if transpose_b else OP_NN)
    var state = _BOUNDARY.get_or_create_ptr()
    var shape = state[].builtins.tuple([PythonObject(m), PythonObject(n)])
    var result = _new_array(shape, False)
    var out_view = PythonObject(None)
    if out is not PythonObject(None):
        if state[].builtins.type(out) is state[].array_class:
            if out.shape != shape or String(out.dtype) != "<f4" or String(out.order) != "C" or Bool(out._readonly):
                raise Error("fast matmul_bf16: out must be writable C-contiguous float32 with the output shape")
            out_view = out._mv
        else:
            out_view = state[].builtins.memoryview(out)
            var format = String(out_view.format)
            if out_view.shape != shape or not Bool(out_view.c_contiguous) or Bool(out_view.readonly) or (format != "f" and format != "@f" and format != "=f" and format != "<f"):
                raise Error("fast matmul_bf16: out must be writable C-contiguous float32 with the output shape")
    var dst_address = Int(py=result._addr)
    with GILReleased(Python()):
        var ctx = process_ctx["MojoAppleFastBf16PublicContext"]()
        var db = ctx.enqueue_create_buffer[DType.uint16](right.count)
        var dc = ctx.enqueue_create_buffer[DType.float32](m * n)
        ctx.enqueue_copy(dst_buf=db, src_ptr=u16_ptr(right.address))
        var served = False
        if left.bf16:
            var da = ctx.enqueue_create_buffer[DType.uint16](left.count)
            ctx.enqueue_copy(dst_buf=da, src_ptr=u16_ptr(left.address))
            served = afn_gemm_bf16_bits_into(ctx, dc, da, db, m, n, k, op)
            ctx.synchronize()
        else:
            var da = ctx.enqueue_create_buffer[DType.float32](left.count)
            ctx.enqueue_copy(dst_buf=da, src_ptr=f32_ptr(left.address))
            served = afn_gemm_bf16_into(ctx, dc, da, db, m, n, k, op)
            ctx.synchronize()
        if not served:
            raise Error("fast matmul_bf16: native Apple BF16 route refused the operation")
        ctx.enqueue_copy(dst_ptr=f32_ptr(dst_address), src_buf=dc)
        ctx.synchronize()
        _ = db^
        _ = dc^
        ctx.synchronize()
    if out is not PythonObject(None):
        var bytes_view = out_view.cast("B")
        bytes_view[:] = result._mv.cast("B")
        return out
    return result
