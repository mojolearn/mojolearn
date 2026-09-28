# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Device-resident matrices for the decomp lane's GPU binding (lane
decomp-apple, 2026-09-28). The host API (x_decomp/api.mojo) uploads every
input and downloads every output of every call, with two syncs and fresh
buffers each time; on Apple that was most of a fit (ew over 1M x 5: ~3 ms a
call, against ~0.3 ms on AMD). Here a matrix lives in a pooled device
buffer named by an integer id; the elementwise, product, fold and distance
entries launch on ids with NO sync, and only `dev_download` waits. The
kernels and their launch sequences are DevExec's own (x_decomp/device.mojo
launch_*), so every value is the same bits as the host API.

POOL: a freed buffer is kept and handed to the next request that fits
(capacity >= n and <= 2n), so a fit allocates each shape once. The one
DeviceContext runs its work in order, so a buffer reused by a later launch
is written only after every earlier launch that read it. Free capacity past
POOL_KEEP_BYTES is released after a sync."""
from std.ffi import _Global
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceBuffer, DeviceContext

from x_decomp.cells import F32Ptr
from x_decomp.device import (
    colsum_scratch,
    gemm_scratch,
    launch_colsum,
    launch_ew,
    launch_gemm,
    launch_rowsum,
    launch_sqdist,
    rowsum_scratch,
    xd_ctx,
)

comptime POOL_KEEP_BYTES = 1 << 31


struct _Pool(Defaultable, Movable):
    var bufs: List[DeviceBuffer[DType.float32]]
    var cap: List[Int]
    var live: List[Bool]
    var free_floats: Int

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.cap = List[Int]()
        self.live = List[Bool]()
        self.free_floats = 0


comptime X_DECOMP_POOL = _Global[StorageType=_Pool, name="MojoXDecompPool", init_fn=_Pool.__init__]


def pool_alloc(n: Int) raises -> Int:
    """An id whose buffer holds at least n floats (n >= 1)."""
    var want = max(n, 1)
    var p = X_DECOMP_POOL.get_or_create_ptr()
    var best = -1
    for i in range(len(p[].cap)):
        if not p[].live[i] and p[].cap[i] >= want and p[].cap[i] <= 2 * want:
            if best < 0 or p[].cap[i] < p[].cap[best]:
                best = i
    if best >= 0:
        p[].live[best] = True
        p[].free_floats -= p[].cap[best]
        return best
    var ctx = xd_ctx()
    var buf = ctx.enqueue_create_buffer[DType.float32](want)
    for i in range(len(p[].cap)):
        if not p[].live[i] and p[].cap[i] == 0:      # a released slot
            p[].bufs[i] = buf^
            p[].cap[i] = want
            p[].live[i] = True
            return i
    p[].bufs.append(buf^)
    p[].cap.append(want)
    p[].live.append(True)
    return len(p[].cap) - 1


def pool_free(id: Int) raises:
    var p = X_DECOMP_POOL.get_or_create_ptr()
    if id < 0 or id >= len(p[].cap) or not p[].live[id]:
        raise Error("x_decomp: freeing a device matrix that is not live")
    p[].live[id] = False
    p[].free_floats += p[].cap[id]
    if p[].free_floats * 4 > POOL_KEEP_BYTES:
        var ctx = xd_ctx()
        ctx.synchronize()
        for i in range(len(p[].cap)):
            if not p[].live[i] and p[].cap[i] > 0:
                p[].bufs[i] = ctx.enqueue_create_buffer[DType.float32](1)
                p[].free_floats -= p[].cap[i]
                p[].cap[i] = 0
        ctx.synchronize()


def _ptr(id: Int, n: Int) raises -> F32Ptr:
    var p = X_DECOMP_POOL.get_or_create_ptr()
    if id < 0 or id >= len(p[].cap) or not p[].live[id]:
        raise Error("x_decomp: a device matrix id that is not live")
    if p[].cap[id] < n:
        raise Error("x_decomp: a device matrix smaller than the call reads")
    return F32Ptr(unsafe_from_address=Int(p[].bufs[id].unsafe_ptr()))


def _id(o: PythonObject) raises -> Int:
    return Int(py=o)


def _n(p: PythonObject, i: Int) raises -> Int:
    var v = Int(py=p[i])
    if v < 0 or v > 2147483647:
        raise Error("x_decomp: dimension out of range")
    return v


# ---- Python entries (GPU binding only)
def dev_alloc_py(n: PythonObject) raises -> PythonObject:
    return PythonObject(pool_alloc(Int(py=n)))


def dev_free_py(id: PythonObject) raises -> PythonObject:
    pool_free(_id(id))
    return PythonObject(None)


def dev_upload_py(id: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Host floats into the device matrix; waits (the host store may change
    after this returns)."""
    var cnt = Int(py=n)
    var src = F32Ptr(unsafe_from_address=Int(py=addr))
    var p = X_DECOMP_POOL.get_or_create_ptr()
    var i = _id(id)
    _ = _ptr(i, cnt)
    with GILReleased(Python()):
        var ctx = xd_ctx()
        if cnt > 0:
            ctx.enqueue_copy(dst_buf=p[].bufs[i].create_sub_buffer[DType.float32](0, cnt), src_ptr=src)
        ctx.synchronize()
    return PythonObject(cnt)


def dev_download_py(id: PythonObject, addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The device matrix into host floats, after every launch before it."""
    var cnt = Int(py=n)
    var dst = F32Ptr(unsafe_from_address=Int(py=addr))
    var p = X_DECOMP_POOL.get_or_create_ptr()
    var i = _id(id)
    _ = _ptr(i, cnt)
    with GILReleased(Python()):
        var ctx = xd_ctx()
        if cnt > 0:
            ctx.enqueue_copy(dst_ptr=dst, src_buf=p[].bufs[i].create_sub_buffer[DType.float32](0, cnt))
        ctx.synchronize()
    return PythonObject(cnt)


def dev_ew_py(
    a: PythonObject, b: PythonObject, c: PythonObject, dst: PythonObject, p: PythonObject, s: PythonObject
) raises -> PythonObject:
    # p = [op, count, d, lb, bm, lc, cm], as x_decomp_ew
    var op = Int(py=p[0])
    var count = _n(p, 1)
    var d = _n(p, 2)
    var lb = _n(p, 3)
    var bm = Int(py=p[4])
    var lc = _n(p, 5)
    var cm = Int(py=p[6])
    if d <= 0:
        raise Error("x_decomp: ew needs a positive row width")
    var pa = _ptr(_id(a), count)
    var pb = _ptr(_id(b), lb)
    var pc = _ptr(_id(c), lc)
    var po = _ptr(_id(dst), count)
    launch_ew(xd_ctx(), op, pa, pb, bm, pc, cm, po, count, d, Float32(Float64(py=s)))
    return PythonObject(count)


def dev_gemm_py(a: PythonObject, b: PythonObject, c: PythonObject, p: PythonObject) raises -> PythonObject:
    var m = _n(p, 0)
    var k = _n(p, 1)
    var n = _n(p, 2)
    if m * n > 2147483647 or m * k > 2147483647 or k * n > 2147483647:
        raise Error("x_decomp: gemm exceeds the Int32 index bound")
    var ta = Int(py=p[3]) != 0
    var tb = Int(py=p[4]) != 0
    var ns = gemm_scratch(m, k, n)
    var sid = pool_alloc(ns)
    launch_gemm(xd_ctx(), _ptr(_id(a), m * k), _ptr(_id(b), k * n), _ptr(_id(c), m * n), _ptr(sid, ns),
                m, k, n, ta, tb)
    pool_free(sid)
    return PythonObject(m * n)


def dev_colsum_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var d = _n(p, 1)
    var ns = colsum_scratch(n, d)
    var sid = pool_alloc(ns)
    launch_colsum(xd_ctx(), _ptr(_id(a), n * d), _ptr(_id(dst), d), _ptr(sid, ns), n, d)
    pool_free(sid)
    return PythonObject(d)


def dev_rowsum_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    var n = _n(p, 0)
    var d = _n(p, 1)
    var ns = rowsum_scratch(n, d)
    var sid = pool_alloc(ns)
    launch_rowsum(xd_ctx(), _ptr(_id(a), n * d), _ptr(_id(dst), n), _ptr(sid, ns), n, d)
    pool_free(sid)
    return PythonObject(n)


def dev_sqdist_py(a: PythonObject, b: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [na, nb, d] or [na, nb, d, kind, pw], as x_decomp_sqdist
    var na = _n(p, 0)
    var nb = _n(p, 1)
    var d = _n(p, 2)
    if na * nb > 2147483647:
        raise Error("x_decomp: sqdist exceeds the Int32 index bound")
    var kind = _n(p, 3) if len(p) > 3 else 0
    var pw = Float32(Float64(py=p[4])) if len(p) > 4 else Float32(2)
    launch_sqdist(xd_ctx(), _ptr(_id(a), na * d), _ptr(_id(b), nb * d), _ptr(_id(dst), na * nb), na, nb, d, kind, pw)
    return PythonObject(na * nb)
