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
from core.device_zero import enqueue_fill

from x_decomp.cells import F32Ptr
from x_decomp.device import (
    _down,
    _up_into,
    absmax_scratch,
    colsum_scratch,
    launch_absmax,
    orth_on_device,
    orth_on_device_diag,
    gemm_scratch,
    launch_colsum,
    launch_ew,
    launch_gemm,
    launch_project,
    PROJECT_TILED,
    launch_rowsum,
    launch_sqdist,
    launch_trisolve,
    lda_rows_kernel,
    rand_kernel,
    rowsum_scratch,
    TPB,
    _blocks,
    xd_ctx,
)

comptime POOL_KEEP_BYTES = 1 << 30
comptime POOL_CLASSES = 40
#: at most this many free buffers wait in the pool: every live Metal buffer
#: costs each later dispatch (measured: after MinCovDet's tens of thousands
#: of small matrices, an unrelated lu(512) went 0.05 -> 0.2 s), so the pool
#: holds a fit's working set, not its history
comptime POOL_KEEP_COUNT = 128


def _class_of(n: Int) -> Int:
    """The power-of-two size class holding n floats (capacity 2^c >= n)."""
    var c = 0
    while (1 << c) < n:
        c += 1
    return c


struct _Pool(Defaultable, Movable):
    """Device buffers by id. A freed buffer waits in its power-of-two class
    for the next request of that class (O(1) either way); a released id is
    reused. Free capacity past POOL_KEEP_BYTES is released after a sync, so
    the number of live buffers stays near what a fit holds at once."""
    var bufs: List[DeviceBuffer[DType.float32]]
    var cls: List[Int]          # the size class, -1 for a released id
    var live: List[Bool]
    var free_by_class: List[List[Int]]
    var released: List[Int]
    var free_floats: Int
    var free_count: Int

    def __init__(out self):
        self.bufs = List[DeviceBuffer[DType.float32]]()
        self.cls = List[Int]()
        self.live = List[Bool]()
        self.free_by_class = List[List[Int]]()
        for _ in range(POOL_CLASSES):
            self.free_by_class.append(List[Int]())
        self.released = List[Int]()
        self.free_floats = 0
        self.free_count = 0


comptime X_DECOMP_POOL = _Global[StorageType=_Pool, name="MojoXDecompPool", init_fn=_Pool.__init__]


def pool_alloc(n: Int) raises -> Int:
    """An id whose buffer holds at least n floats (n >= 1)."""
    var c = _class_of(max(n, 1))
    if c >= POOL_CLASSES:
        raise Error("x_decomp: a device matrix past the pool's largest size class")
    var p = X_DECOMP_POOL.get_or_create_ptr()
    if len(p[].free_by_class[c]) > 0:
        var id = p[].free_by_class[c].pop()
        p[].live[id] = True
        p[].free_floats -= 1 << c
        p[].free_count -= 1
        return id
    var ctx = xd_ctx()
    var buf = ctx.enqueue_create_buffer[DType.float32](1 << c)
    if len(p[].released) > 0:
        var id = p[].released.pop()
        p[].bufs[id] = buf^
        p[].cls[id] = c
        p[].live[id] = True
        return id
    p[].bufs.append(buf^)
    p[].cls.append(c)
    p[].live.append(True)
    return len(p[].cls) - 1


def pool_free(id: Int) raises:
    var p = X_DECOMP_POOL.get_or_create_ptr()
    if id < 0 or id >= len(p[].cls) or not p[].live[id] or p[].cls[id] < 0:
        raise Error("x_decomp: freeing a device matrix that is not live")
    var c = p[].cls[id]
    p[].live[id] = False
    p[].free_by_class[c].append(id)
    p[].free_floats += 1 << c
    p[].free_count += 1
    if p[].free_floats * 4 > POOL_KEEP_BYTES or p[].free_count > POOL_KEEP_COUNT:
        # release every free buffer: its id holds a copy of one tiny buffer
        var ctx = xd_ctx()
        ctx.synchronize()
        var tiny = ctx.enqueue_create_buffer[DType.float32](1)
        for k in range(POOL_CLASSES):
            while len(p[].free_by_class[k]) > 0:
                var f = p[].free_by_class[k].pop()
                p[].bufs[f] = tiny
                p[].cls[f] = -1
                p[].released.append(f)
        p[].free_floats = 0
        p[].free_count = 0
        ctx.synchronize()


def _ptr(id: Int, n: Int) raises -> F32Ptr:
    var p = X_DECOMP_POOL.get_or_create_ptr()
    if id < 0 or id >= len(p[].cls) or not p[].live[id] or p[].cls[id] < 0:
        raise Error("x_decomp: a device matrix id that is not live")
    if (1 << p[].cls[id]) < n:
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
        _up_into(ctx, p[].bufs[i], src, cnt)
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
        _down(ctx, p[].bufs[i], dst, cnt)
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


def dev_project_py(a: PythonObject, b: PythonObject, c: PythonObject, flag: PythonObject, p: PythonObject) raises -> PythonObject:
    """c = a b^T (a m x k, b n x k; `dev_gemm` with tb=1's words) by the
    tiled projection kernel, enqueued (no sync), and flag[0] = 1 when an
    entry of a is not finite, else 0 (the random projection's transform;
    lane gap-nb-maxabs-grp). Returns -1, launching nothing, on a column
    whose shared memory does not hold the tile."""
    var m = _n(p, 0)
    var k = _n(p, 1)
    var n = _n(p, 2)
    if m * n > 2147483647 or m * k > 2147483647 or k * n > 2147483647:
        raise Error("x_decomp: projection exceeds the Int32 index bound")
    comptime if not PROJECT_TILED:
        return PythonObject(-1)
    var pf = _ptr(_id(flag), 1)
    var pa = _ptr(_id(a), m * k)
    var pb = _ptr(_id(b), k * n)
    var pc = _ptr(_id(c), m * n)
    var ctx = xd_ctx()
    var pool = X_DECOMP_POOL.get_or_create_ptr()
    enqueue_fill(ctx, pool[].bufs[_id(flag)], Float32(0))
    launch_project(ctx, pa, pb, pc, pf, m, k, n)
    return PythonObject(m * n)


def dev_rand_py(dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """p = [count, seed, stream, kind]: `rand_py`'s draws (x_decomp/cells.mojo
    `rand_cell`, the counter-based Philox stream) written into the device
    matrix, enqueued (no sync, no download): the random projections' matrix
    is made where transform reads it (lane gap-nb-maxabs-grp)."""
    var count = _n(p, 0)
    var seed = UInt32(Int(py=p[1]) & 0xFFFFFFFF)
    var stream = UInt32(Int(py=p[2]) & 0xFFFFFFFF)
    var kind = Int(py=p[3])
    if count == 0:
        return PythonObject(0)
    xd_ctx().enqueue_function[rand_kernel](
        _ptr(_id(dst), count), Int32(count), seed, stream, Int32(kind), grid_dim=_blocks(count), block_dim=TPB
    )
    return PythonObject(count)


def dev_trisolve_py(lu: PythonObject, idx: PythonObject, src: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """`trisolve_py` on device ids, enqueued (no sync)."""
    var n = _n(p, 0)
    var nrhs = _n(p, 1)
    var trans = _n(p, 2)
    if n >= 1 << 24:
        raise Error("x_decomp: trisolve row numbers exceed float32's exact integers")
    var sid = pool_alloc(max(n * nrhs, 1))
    launch_trisolve(xd_ctx(), _ptr(_id(lu), n * n), _ptr(_id(idx), n), _ptr(_id(src), n * nrhs),
                    _ptr(_id(dst), n * nrhs), _ptr(sid, n * nrhs), n, nrhs, trans)
    pool_free(sid)
    return PythonObject(n)


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


def dev_absmax_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    # p = [n, d, by_col], as x_decomp_absmax_sign
    var n = _n(p, 0)
    var d = _n(p, 1)
    var by_col = Int(py=p[2]) != 0
    var cnt = d if by_col else n
    var ns = absmax_scratch(n, d, by_col)
    var sid = pool_alloc(ns)
    launch_absmax(xd_ctx(), _ptr(_id(a), n * d), _ptr(_id(dst), cnt), _ptr(sid, ns), n, d, by_col)
    pool_free(sid)
    return PythonObject(cnt)


def dev_orth_py(a: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """dst = the orthonormalized columns of a (DevExec.orth's passes on the
    device copy; a is unchanged)."""
    var m = _n(p, 0)
    var l = _n(p, 1)
    var cells = m * l
    var ia = _id(a)
    var io = _id(dst)
    _ = _ptr(ia, cells)
    _ = _ptr(io, cells)
    var pool = X_DECOMP_POOL.get_or_create_ptr()
    var ctx = xd_ctx()
    if cells > 0:
        var sub = pool[].bufs[io].create_sub_buffer[DType.float32](0, cells)
        ctx.enqueue_copy(dst_buf=sub, src_buf=pool[].bufs[ia].create_sub_buffer[DType.float32](0, cells))
        with GILReleased(Python()):
            orth_on_device(ctx, sub, m, l)
    return PythonObject(cells)


def dev_orth_diag_py(a: PythonObject, dst: PythonObject, diag: PythonObject, p: PythonObject) raises -> PythonObject:
    """`dev_orth_py`, and the host floats at `diag` (l) = the product of the
    two passes' R diagonals (`orth_on_device_diag`, lane neural-pass17)."""
    var m = _n(p, 0)
    var l = _n(p, 1)
    var cells = m * l
    var ia = _id(a)
    var io = _id(dst)
    _ = _ptr(ia, cells)
    _ = _ptr(io, cells)
    var pd = F32Ptr(unsafe_from_address=Int(py=diag))
    var pool = X_DECOMP_POOL.get_or_create_ptr()
    var ctx = xd_ctx()
    var dd = ctx.enqueue_create_buffer[DType.float32](l if l > 0 else 1)
    enqueue_fill(ctx, dd, Float32(1.0))
    if cells > 0:
        var sub = pool[].bufs[io].create_sub_buffer[DType.float32](0, cells)
        ctx.enqueue_copy(dst_buf=sub, src_buf=pool[].bufs[ia].create_sub_buffer[DType.float32](0, cells))
        with GILReleased(Python()):
            orth_on_device_diag(ctx, sub, m, l, dd, True)
    if l > 0:
        ctx.enqueue_copy(dst_ptr=pd, src_buf=dd.create_sub_buffer[DType.float32](0, l))
    ctx.synchronize()
    _ = dd^
    return PythonObject(cells)


def dev_lda_rows_py(
    x: PythonObject, ew: PythonObject, d: PythonObject, e: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`x_decomp_lda_rows` on device matrices (lane/py-decomp-nbrs): d and e
    (n x k) updated in place on the device; DevExec.lda_rows' kernel and
    launch shape, with no upload, download or sync. p = [n, k, v, max_iter],
    f = [prior, tol], as x_decomp_lda_rows. Before, the host-only entry
    downloaded the resident X (n x v) and uploaded it again every E-step
    (12 GB per iteration at 1M x 1000)."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var v = _n(p, 2)
    var max_iter = _n(p, 3)
    if n * (v + k) > 2147483647:
        raise Error("x_decomp: lda_rows exceeds the Int32 index bound")
    var prior = Float32(Float64(py=f[0]))
    var tol = Float32(Float64(py=f[1]))
    var px = _ptr(_id(x), n * v)
    var pw = _ptr(_id(ew), k * v)
    var pd = _ptr(_id(d), n * k)
    var pe = _ptr(_id(e), n * k)
    if n == 0:
        return PythonObject(0)
    var sid = pool_alloc(n * (v + k))
    var iid = pool_alloc(n)
    xd_ctx().enqueue_function[lda_rows_kernel](
        px, pw, pd, pe, _ptr(sid, n * (v + k)), _ptr(iid, n),
        Int32(n), Int32(k), Int32(v), prior, Int32(max_iter), tol, grid_dim=_blocks(n), block_dim=TPB,
    )
    pool_free(sid)
    pool_free(iid)
    return PythonObject(n)
