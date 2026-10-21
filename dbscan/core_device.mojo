# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""DBSCAN's core arrays on the device (lane cpu4-python).

`dbscan_core_arrays` keeps the fit's core rows, their training indices and
their labels, in ascending training index. The GPU estimators binding ran
`bindings/py2mojo_cluster_est.mojo`'s host walk (the mask check, the
compaction and the row copies over every row); that walk stays the host
column (`bindings/_mojolearn_estimators_host.mojo`). Here the mask goes up,
is checked and turned into flags by one kernel, `device_compact_equal_i32`
(an exclusive scan: the host loop's ascending order) lists the core rows,
and one gather kernel writes the three arrays: integer moves and float32
copies only, the same bytes as the host column on every vendor.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.python import Python, PythonObject
from max.gpu.host import DeviceBuffer, DeviceContext

from core.device_fold import device_compact_equal_i32

comptime _TPB = 256
comptime _U8 = MutPointer[UInt8, MutAnyOrigin]
comptime _I32 = MutPointer[Int32, MutAnyOrigin]
comptime _F32 = MutPointer[Float32, MutAnyOrigin]


@always_inline
def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + _TPB - 1) // _TPB if count > 0 else 1


def _core_flag_kernel(core: _U8, n_: Int32, flag: _I32, status: _I32):
    """flag[i] = core[i] (0 or 1); a byte above 1 sets status[0]."""
    var i = _tid()
    if i >= Int(n_):
        return
    var f = core.unsafe_load(i)
    if f > UInt8(1):
        status.unsafe_store(0, Int32(1))
    flag.unsafe_store(i, Int32(1) if f == UInt8(1) else Int32(0))


def _core_gather_kernel(
    rows: _I32, m_: Int32, d_: Int32, x: _F32, labels: _I32,
    idx: _I32, comp: _F32, lab: _I32,
):
    """One thread per (core row o, column j) of the (m, d) components:
    comp[o, j] = x[rows[o], j]; column 0 also writes idx[o] and lab[o]."""
    var t = _tid()
    var d = Int(d_)
    if t >= Int(m_) * d:
        return
    var o = t // d
    var j = t - o * d
    var r = Int(rows.unsafe_load(o))
    comp.unsafe_store(t, x.unsafe_load(r * d + j))
    if j == 0:
        idx.unsafe_store(o, Int32(r))
        lab.unsafe_store(o, labels.unsafe_load(r))


def _new_array(arr: PythonObject, code: StaticString, nbytes: Int) raises -> PythonObject:
    """An `array.array(code)` of `nbytes` zero bytes."""
    var zero = Python.import_module("builtins").bytes(nbytes)
    return arr.array(PythonObject(code), zero)


def _array_addr(a: PythonObject) raises -> Int:
    return Int(py=a.buffer_info()[0])


def dbscan_core_arrays_device(
    ctx: DeviceContext, x_addr: PythonObject, labels_addr: PythonObject,
    core_addr: PythonObject, params: PythonObject,
) raises -> PythonObject:
    """`dbscan_core_arrays` (params [n, d]) on the device: [idx int32 (m),
    components float32 (m, d), labels int32 (m)] as `array.array`s."""
    if len(params) != 2:
        raise Error("dbscan_core_arrays: params must be [n, d], got " + String(len(params)) + " values")
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    if n < 0 or d < 1 or (n > 0 and d > 2147483000 // n):
        raise Error("dbscan_core_arrays: n >= 0, d >= 1 and n * d within Int32")
    var arr = Python.import_module("array")
    if n == 0:
        var out0 = Python.list()
        out0.append(_new_array(arr, "i", 0))
        out0.append(_new_array(arr, "f", 0))
        out0.append(_new_array(arr, "i", 0))
        return out0
    var xa = Int(py=x_addr)
    var la = Int(py=labels_addr)
    var ca = Int(py=core_addr)
    if xa == 0 or la == 0 or ca == 0:
        raise Error("dbscan_core_arrays: null buffer address")
    var d_core = ctx.enqueue_create_buffer[DType.uint8](n)
    ctx.enqueue_copy(dst_buf=d_core, src_ptr=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ca))
    var flag = ctx.enqueue_create_buffer[DType.int32](n)
    var status = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(status, Int32(0))
    ctx.enqueue_function[_core_flag_kernel](
        d_core.unsafe_ptr(), Int32(n), flag.unsafe_ptr(), status.unsafe_ptr(),
        grid_dim=_blocks(n), block_dim=_TPB,
    )
    var hs = ctx.enqueue_create_host_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=status)
    ctx.synchronize()
    var bad = hs.unsafe_ptr().unsafe_load(0) != Int32(0)
    _ = hs^
    _ = status^
    _ = d_core^
    if bad:
        raise Error("mojolearn DBSCAN: the fit's core mask holds a value other than 0 or 1")
    var rows = ctx.enqueue_create_buffer[DType.int32](n)
    var m = device_compact_equal_i32(ctx, flag, n, Int32(1), rows)
    _ = flag^
    var idx = _new_array(arr, "i", 4 * m)
    var comp = _new_array(arr, "f", 4 * m * d)
    var lab = _new_array(arr, "i", 4 * m)
    if m > 0:
        var d_x = ctx.enqueue_create_buffer[DType.float32](n * d)
        ctx.enqueue_copy(dst_buf=d_x, src_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=xa))
        var d_l = ctx.enqueue_create_buffer[DType.int32](n)
        ctx.enqueue_copy(dst_buf=d_l, src_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=la))
        var o_idx = ctx.enqueue_create_buffer[DType.int32](m)
        var o_comp = ctx.enqueue_create_buffer[DType.float32](m * d)
        var o_lab = ctx.enqueue_create_buffer[DType.int32](m)
        ctx.enqueue_function[_core_gather_kernel](
            rows.unsafe_ptr(), Int32(m), Int32(d), d_x.unsafe_ptr(), d_l.unsafe_ptr(),
            o_idx.unsafe_ptr(), o_comp.unsafe_ptr(), o_lab.unsafe_ptr(),
            grid_dim=_blocks(m * d), block_dim=_TPB,
        )
        ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_array_addr(idx)), src_buf=o_idx)
        ctx.enqueue_copy(dst_ptr=MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=_array_addr(comp)), src_buf=o_comp)
        ctx.enqueue_copy(dst_ptr=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_array_addr(lab)), src_buf=o_lab)
        ctx.synchronize()
        _ = o_idx^
        _ = o_comp^
        _ = o_lab^
        _ = d_l^
        _ = d_x^
    _ = rows^
    var out = Python.list()
    out.append(idx)
    out.append(comp)
    out.append(lab)
    return out
