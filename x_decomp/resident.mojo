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
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from core.device_pool import pool_give, pool_take
from core.device_scan import device_first_nonfinite

from x_decomp.cells import F32Ptr, OP_SCALE, OP_SELECT, ew_cell, rand_cell
from core.philox import philox4x32_10
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import HostBuffer
from core.device_scan import NONFINITE_NONE, SCAN_TPB, _scan_blocks, nonfinite_partial_kernel
from x_decomp.moves import MOVE_FILL0, MOVE_TAKE_ROWS
from x_decomp.moves_device import launch_move
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
    launch_knn_select,
    rand_kernel,
    als_scratch,
    launch_als_rows,
    launch_lda_bound,
    launch_lda_rows,
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


def dev_knn_select_py(dmat: PythonObject, dist: PythonObject, idx: PythonObject, p: PythonObject) raises -> PythonObject:
    """`knn_select_py` on device ids, enqueued (no sync)."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var k = _n(p, 2)
    var ex = _n(p, 3)
    if m >= 1 << 24 or n * m > 2147483647:
        raise Error("x_decomp: knn_select exceeds the index bounds")
    launch_knn_select(xd_ctx(), _ptr(_id(dmat), n * m), _ptr(_id(dist), n * k), _ptr(_id(idx), n * k), n, m, k, ex)
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
    launch_lda_rows(xd_ctx(), px, pw, pd, pe, _ptr(sid, n * (v + k)), _ptr(iid, n), n, k, v, prior, max_iter, tol)
    pool_free(sid)
    pool_free(iid)
    return PythonObject(n)


def dev_lda_bound_py(
    x: PythonObject, ddt: PythonObject, dcomp: PythonObject, dst: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """LatentDirichletAllocation._approx_bound's word-term cells on device
    matrices (lane gap-lda-als): dst (n x v) = x * logsumexp_t(ddt[:, t] +
    dcomp[t, :]) through the same `ew_cell` chain the composed path runs, with
    no n x v term matrix per topic. p = [n, k, v], f = [floor]."""
    var n = _n(p, 0)
    var k = _n(p, 1)
    var v = _n(p, 2)
    if n * v > 2147483647:
        raise Error("x_decomp: lda_bound exceeds the Int32 index bound")
    var floor = Float32(Float64(py=f[0]))
    launch_lda_bound(
        xd_ctx(), _ptr(_id(x), n * v), _ptr(_id(ddt), n * k), _ptr(_id(dcomp), k * v), _ptr(_id(dst), n * v),
        n, k, v, floor,
    )
    return PythonObject(n * v)


def dev_als_rows_py(
    c: PythonObject, y: PythonObject, yty: PythonObject, x: PythonObject, flags: PythonObject, p: PythonObject,
    reg: PythonObject,
) raises -> PythonObject:
    """`x_decomp_als_rows` on device matrices (lane gap-lda-als): every row's
    `als_row` into x (n x f) and flags (n), one block per row; element (u, i)
    of the confidences at c[u * su + i * si], so the item half-sweep reads the
    resident user x item matrix with su = 1, si = items (no transpose).
    p = [n, m, f, su, si, c_len]."""
    var n = _n(p, 0)
    var m = _n(p, 1)
    var f = _n(p, 2)
    var su = _n(p, 3)
    var si = _n(p, 4)
    var cl = _n(p, 5)
    if n == 0:
        return PythonObject(0)
    var ns = als_scratch(n, f)
    var sid = pool_alloc(ns if ns > 0 else 1)
    launch_als_rows(
        xd_ctx(), _ptr(_id(c), cl), _ptr(_id(y), m * f), _ptr(_id(yty), f * f), _ptr(_id(x), n * f),
        _ptr(sid, ns if ns > 0 else 1), _ptr(_id(flags), n), n, m, f, su, si, Float32(Float64(py=reg)),
    )
    pool_free(sid)
    return PythonObject(n)


# ---- lane/apple-fast-gap-cls2 (2026-10-03): GaussianRandomProjection.fit -----
# Board (M3 FAST): gaussian-rp istella 53.7 ms vs scikit-learn 24.3, taxi 3.3
# vs 1.6. The fit's matrix is a 10 x d device draw; the time is the host
# finiteness walk over X (python/mojolearn/_expansion_decomp.py
# `_M.shape_of_input` -> `_host_all_finite` -> bindings/host_helpers.mojo
# `all_finite_f32_binding`, one thread, a branch per word: 900k x 220 words
# on Istella). Three FAST + Apple switches (DEVSCAN default on, below), Python routes read
# back from `grp_cls2_py` (no env read):
#   MOJOLEARN_XD_FAST_CLS2_GRP_NOSCAN  bit 1: fit reads the shape only;
#       transform's device projection (`dev_project_py`) flags a non-finite X,
#       as SparseRandomProjection's MOJOLEARN_SPARSE_RP_DEVICE fit does.
#   MOJOLEARN_XD_FAST_CLS2_GRP_DEVSCAN bit 2: the refusal stays in fit, as a
#       device scan (`dev_first_nonfinite_py`: X up from the caller's buffer
#       into a pooled buffer, core/device_scan.mojo `device_first_nonfinite`).
#   MOJOLEARN_XD_FAST_CLS2_GRP_LAZY    bit 4: components_ is downloaded on
#       first read, not in fit (the fit then ends with no synchronize).
# The matrix (and so every output word) is main's under every switch.
# DEVSCAN is the FAST + Apple default since the M3 A/B (n=1, quality
# identical): gaussian-rp istella 54.3 -> 15.9 ms, taxi 4.7 -> 3.9 ms;
# -D MOJOLEARN_XD_FAST_CLS2_GRP_DEVSCAN_OFF turns it off. NOSCAN and LAZY
# stay opt-in (they move the NaN/inf refusal from fit to transform; pending
# Andrew's decision); NOSCAN, when defined, replaces the device scan.

comptime _CLS2_FAST_APPLE = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime GRP_CLS2_NOSCAN = _CLS2_FAST_APPLE and is_defined["MOJOLEARN_XD_FAST_CLS2_GRP_NOSCAN"]()
comptime GRP_CLS2_DEVSCAN = (
    _CLS2_FAST_APPLE
    and not is_defined["MOJOLEARN_XD_FAST_CLS2_GRP_DEVSCAN_OFF"]()
    and not GRP_CLS2_NOSCAN
)
comptime GRP_CLS2_LAZY = _CLS2_FAST_APPLE and is_defined["MOJOLEARN_XD_FAST_CLS2_GRP_LAZY"]()
#: lane apple-fast-gap-kapprox2 (2026-10-03), the FAST + Apple default since
#: the M3 A/Bs kap2-grp-fused-{taxi,istella}, kap2-srp-fused-{taxi,istella}
#: (gaussian-rp taxi 3.4 -> 2.1 ms, istella -4.6%; sparse-rp taxi -10.7%,
#: istella -8.5%; distortion identical; `-D MOJOLEARN_XD_FAST_GRP_FUSED_OFF`
#: reverts): the
#: random projections' whole fit in ONE binding call and ONE synchronize
#: (`grp_fit_fused_py`): X up into the pooled scan buffer, the nonfinite
#: partials into a pooled device buffer, the matrix drawn AND scaled by one
#: kernel (`rand_cell` then `ew_cell(OP_SCALE)`, the words of main's
#: `x_decomp_dev_rand` + `ew scale`), the partials and the matrix down in
#: the same queue. Main's DEVSCAN fit waits twice (scan, then the matrix
#: download), allocates the partials and a pinned host buffer per fit and
#: launches rand and scale separately. Refusal stays in fit; bit 8 of
#: `x_decomp_grp_cls2`. Needs DEVSCAN (not NOSCAN).
comptime GRP_FAST_FUSED = GRP_CLS2_DEVSCAN and not is_defined["MOJOLEARN_XD_FAST_GRP_FUSED_OFF"]()
#: lane apple-fast-w2-kfeat (2026-10-04), FAST + Apple, OPT-IN until its
#: quality pair and M3 A/B (`-D MOJOLEARN_XD_FAST_SRP_STRAT`; needs
#: GRP_FAST_FUSED): SparseRandomProjection's nonzero PATTERN is drawn column
#: by column by systematic sampling (`srp_strat_kernel`): column j takes a
#: uniform 24-bit offset o_j (Philox stream 4 at counter j) and component c
#: is nonzero iff floor(((c+1) D + o_j) / 2^24) > floor((c D + o_j) / 2^24),
#: D = the 2^-24 grid count main's `u <= density - 2^-25` keeps. Every entry
#: is still nonzero with probability D / 2^24 (main's), entries of different
#: columns stay independent, and the signs and the scale are main's words
#: (stream 3); only the within-column count changes, from Binomial(k, density)
#: to floor or ceil of k * density. WHY: sparse-rp istella's distortion
#: 1.883 vs sklearn's 0.474 is that count on one dominant raw column: with
#: k = 10 and density 1/sqrt(220), a column picked twice gives
#: |2 * 1.483 - 1| = 1.97 and once |1.483 - 1| = 0.48 (the projected squared
#: distance of a pair separated along that column is count * s^2 * dx^2,
#: s^2 = 1/(k density) = 1.483); main's seed-7 draw has 27 of 220 columns at
#: count >= 2 (sklearn's draw has its own luck). Stratifying removes that
#: variance term for every input and leaves the off-diagonal terms' law
#: alone. The 'auto' n_components and compute_inverse_components routes
#: (not fused) keep main's draw.
comptime GRP_FAST_SRP_STRAT = GRP_FAST_FUSED and is_defined["MOJOLEARN_XD_FAST_SRP_STRAT"]()
comptime GRP_CLS2_ANY = GRP_CLS2_NOSCAN or GRP_CLS2_DEVSCAN or GRP_CLS2_LAZY


def grp_cls2_py() raises -> PythonObject:
    """The random projections' FAST Apple fit switches compiled in: bit 1
    NOSCAN, bit 2 DEVSCAN, bit 4 LAZY, bit 8 FUSED, bit 16 SRP_STRAT (0 on
    every other build)."""
    var f = 0
    comptime if GRP_CLS2_NOSCAN:
        f |= 1
    comptime if GRP_CLS2_DEVSCAN:
        f |= 2
    comptime if GRP_CLS2_LAZY:
        f |= 4
    comptime if GRP_FAST_FUSED:
        f |= 8
    comptime if GRP_FAST_SRP_STRAT:
        f |= 16
    return PythonObject(f)


def dev_first_nonfinite_py(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """The first flat index of a NaN or infinity among the n floats at
    `addr`, or -1: one upload from the caller's buffer into a pooled device
    buffer (no fresh allocation per fit) and one device scan. Registered
    only under GRP_CLS2_DEVSCAN."""
    var cnt = Int(py=n)
    if cnt <= 0:
        return PythonObject(-1)
    var src = F32Ptr(unsafe_from_address=Int(py=addr))
    var r = -1
    with GILReleased(Python()):
        var ctx = xd_ctx()
        var buf = pool_take["MojoXDecompCls2Scan"](ctx, cnt)
        ctx.enqueue_copy(dst_buf=buf, src_ptr=src)
        r = device_first_nonfinite(ctx, buf, cnt)
        pool_give["MojoXDecompCls2Scan"](buf^)
    return PythonObject(r)


def dev_move_py(src: PythonObject, idx: PythonObject, dst: PythonObject, p: PythonObject) raises -> PythonObject:
    """`x_decomp_move` on device ids, enqueued (no sync); p = [op, count, a1,
    a2, a3, ist, ioff, nsrc, ndst, nidx] (lane apple-fast-py2mojo-decomp)."""
    var op = Int(py=p[0])
    var count = _n(p, 1)
    var a1 = _n(p, 2)
    var a2 = _n(p, 3)
    var a3 = _n(p, 4)
    var ist = _n(p, 5)
    var ioff = _n(p, 6)
    if op < MOVE_TAKE_ROWS or op > MOVE_FILL0:
        raise Error("x_decomp: unknown move op")
    if op != MOVE_FILL0 and a1 <= 0 and count > 0:
        raise Error("x_decomp: move needs a positive width")
    var ps = _ptr(_id(src), max(_n(p, 7), 1))
    var pi = _ptr(_id(idx), max(_n(p, 9), 1))
    var pd = _ptr(_id(dst), max(_n(p, 8), 1))
    launch_move(xd_ctx(), op, ps, pi, pd, count, a1, a2, a3, ist, ioff)
    return PythonObject(count)


# ---- GRP_FAST_FUSED (lane apple-fast-gap-kapprox2)
def grp_rand_scale_kernel(dst: F32Ptr, count: Int32, seed: UInt32, mode: Int32, s: Float32, thr: Float32):
    """The projection matrix in one pass. mode 0 (Gaussian): `rand_kernel`
    (stream 1, kind 1) then `ew_kernel(OP_SCALE)` with unit operands. mode 1
    (sparse, density < 1): the sign (stream 3, kind 2) scaled by s, kept
    where the uniform (stream 2, kind 0) is not above thr, else 0: main's
    `ew select(u, 0, scale(sign))`. mode 2: the scaled sign alone (density
    1). The same `rand_cell` / `ew_cell` words as main's three launches."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        var v: Float32
        if mode == 0:
            v = ew_cell(OP_SCALE, rand_cell(i, seed, UInt32(1), 1), Float32(1), Float32(1), s)
        else:
            var sg = ew_cell(OP_SCALE, rand_cell(i, seed, UInt32(3), 2), Float32(1), Float32(1), s)
            if mode == 1:
                v = ew_cell(OP_SELECT, rand_cell(i, seed, UInt32(2), 0), Float32(0), sg, thr)
            else:
                v = sg
        dst.unsafe_store(i, v)


def srp_strat_kernel(dst: F32Ptr, count: Int32, d: Int32, seed: UInt32, dgrid: UInt32, s: Float32):
    """GRP_FAST_SRP_STRAT: the sparse matrix (row-major, k x d, entry i =
    c d + j) with the column-stratified pattern (module comment above):
    nonzero iff the systematic-sampling boundary falls in [c D, (c+1) D)
    after column j's offset; the value is main's scaled sign
    (`ew_cell(OP_SCALE, rand_cell(i, seed, 3, 2))`), else 0. Integer
    arithmetic only (Metal has no float64); one Philox call per entry for
    the offset (recomputed per row: no scratch, no second launch)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(count):
        var dd = Int(d)
        var c = i // dd
        var j = i - c * dd
        var r = philox4x32_10(
            SIMD[DType.uint32, 4](UInt32(j & 0xFFFFFFFF), UInt32(4), UInt32(j >> 32), 0),
            SIMD[DType.uint32, 2](seed, UInt32(0x5EED)),
        )
        var off = UInt64(r[0] >> 8)
        var g = UInt64(dgrid)
        var lo = (UInt64(c) * g + off) >> 24
        var hi = (UInt64(c + 1) * g + off) >> 24
        var v = Float32(0)
        if hi != lo:
            v = ew_cell(OP_SCALE, rand_cell(i, seed, UInt32(3), 2), Float32(1), Float32(1), s)
        dst.unsafe_store(i, v)


struct _GrpStage(Defaultable, Movable):
    """The fused fit's pooled scan partials (device) and their pinned copy."""
    var part: Optional[DeviceBuffer[DType.int32]]
    var host: Optional[HostBuffer[DType.int32]]

    def __init__(out self):
        self.part = Optional[DeviceBuffer[DType.int32]]()
        self.host = Optional[HostBuffer[DType.int32]]()


comptime GRP_STAGE = _Global[StorageType=_GrpStage, name="MojoXDecompGrpFusedStage", init_fn=_GrpStage.__init__]


def grp_fit_fused_py(
    xaddr: PythonObject, n: PythonObject, dst: PythonObject, hout: PythonObject, p: PythonObject, s: PythonObject
) raises -> PythonObject:
    """p = [count, seed, mode], s = [scale, threshold]: the first flat
    index of a NaN or infinity among the n floats at xaddr (or -1), the
    count-entry matrix (`grp_rand_scale_kernel` mode) into the device
    matrix dst, and its words copied
    to the host floats at out; one synchronize. Registered only under
    GRP_FAST_FUSED."""
    var cnt = Int(py=n)
    var count = _n(p, 0)
    var seed = UInt32(Int(py=p[1]) & 0xFFFFFFFF)
    var mode = Int(py=p[2])
    var sd = 1
    var sgrid = 0
    comptime if GRP_FAST_SRP_STRAT:
        if mode == 3:
            sd = _n(p, 3)
            sgrid = _n(p, 4)
            if sd <= 0 or count % sd != 0 or sgrid <= 0 or sgrid > 16777216:
                raise Error("x_decomp_grp_fit_fused: stratified sparse mode needs n_features > 0 dividing count and 0 < D <= 2^24")
    var sc = Float32(Float64(py=s[0]))
    var thr = Float32(Float64(py=s[1]))
    var src = F32Ptr(unsafe_from_address=Int(py=xaddr))
    var host_out = F32Ptr(unsafe_from_address=Int(py=hout))
    var pd = _ptr(_id(dst), max(count, 1))
    var best = NONFINITE_NONE
    with GILReleased(Python()):
        var ctx = xd_ctx()
        var st = GRP_STAGE.get_or_create_ptr()
        if not st[].part:
            st[].part = ctx.enqueue_create_buffer[DType.int32](512)
            st[].host = ctx.enqueue_create_host_buffer[DType.int32](512)
        var blocks = _scan_blocks(cnt) if cnt > 0 else 0
        var buf = pool_take["MojoXDecompCls2Scan"](ctx, max(cnt, 1))
        if cnt > 0:
            ctx.enqueue_copy(dst_buf=buf.create_sub_buffer[DType.float32](0, cnt), src_ptr=src)
            ctx.enqueue_function[nonfinite_partial_kernel](
                st[].part.value().unsafe_ptr(), buf.unsafe_ptr(), Int32(cnt),
                grid_dim=(blocks, 1, 1), block_dim=(SCAN_TPB, 1, 1),
            )
            ctx.enqueue_copy(dst_ptr=st[].host.value().unsafe_ptr(),
                             src_buf=st[].part.value().create_sub_buffer[DType.int32](0, blocks))
        if count > 0:
            var strat = False
            comptime if GRP_FAST_SRP_STRAT:
                # p = [count, seed, 3, n_features, D]: the stratified sparse
                # pattern (srp_strat_kernel); modes 0-2 are main's kernel
                strat = mode == 3
                if strat:
                    ctx.enqueue_function[srp_strat_kernel](
                        pd, Int32(count), Int32(sd), seed, UInt32(sgrid), sc,
                        grid_dim=_blocks(count), block_dim=TPB,
                    )
            if not strat:
                ctx.enqueue_function[grp_rand_scale_kernel](
                    pd, Int32(count), seed, Int32(mode), sc, thr, grid_dim=_blocks(count), block_dim=TPB
                )
            ctx.enqueue_copy(dst_ptr=host_out, src_buf=_pool_buf_view(_id(dst), count))
        ctx.synchronize()
        var hp = st[].host.value().unsafe_ptr()
        for i in range(blocks):
            var v = hp[i]
            if v < best:
                best = v
        pool_give["MojoXDecompCls2Scan"](buf^)
    return PythonObject(-1 if best == NONFINITE_NONE else Int(best))


def _pool_buf_view(id: Int, count: Int) raises -> DeviceBuffer[DType.float32]:
    """The first `count` floats of pooled matrix id as a sub-buffer."""
    var p = X_DECOMP_POOL.get_or_create_ptr()
    return p[].bufs[id].create_sub_buffer[DType.float32](0, count)
