# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The native drivers' kit on the device, resident (lane hr-kit, 2026-10-02).

MinCovDet's fast_mcd (x_decomp/mcd.mojo) and LatentDirichletAllocation's
online pass (x_decomp/lda_online.mojo) make thousands of kit calls on
operands of a few hundred values. `Kit[DevExec, ...]` paid an upload, a
launch, a download and two syncs for each one. Here every matrix lives in a
pooled device buffer (x_decomp/resident.mojo) and every kit call is an
enqueued launch, with no sync:

- the launch sequences are DevExec's own (`launch_ew`, `launch_gemm`,
  `launch_colsum`, `launch_rowsum`, `launch_lu`, `rand_kernel`,
  `gamma_kernel`, `lda_rows_kernel`, `DevExec._eigh2_on`), so every value
  is the same bits as `Kit[DevExec, DevExec]`, which is the same bits as
  the host column's `Kit[HostExec, HostExec]` (every x-decomp lane's
  GPU == CPU claim);
- the host reads a value only where the search branches on it: a C-step's
  log determinant (the LU's diagonal and pivots, and the summed logs, in
  ONE sync), the distances it selects the next support from, the Philox
  draws a permutation sorts, and the eigenvalues the eigen solve already
  hands back. Online LDA's pass reads its (offset + n_batch_iter)^-decay
  weights for every mini-batch in ONE sync before the loop, then runs the
  whole pass with no sync until the final download;
- copies only (no arithmetic) in the three kernels below: a row gather, a
  diagonal gather and the pinvh mask.

Host floats a launch reads are held (`hold_f`, `hold_i`) until the next
sync, so an enqueued upload never reads freed memory."""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceBuffer, DeviceContext

from x_decomp.api import _f, _i, _n
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import (
    DevExec,
    TPB,
    _blocks,
    _down,
    colsum_scratch,
    gamma_kernel,
    gemm_scratch,
    jacobi2_eigh_on,
    launch_colsum,
    launch_ew,
    launch_gemm,
    launch_lu,
    launch_rowsum,
    lda_rows_kernel,
    pj_eigh_min,
    rand_kernel,
    rowsum_scratch,
    xd_ctx,
)
from x_decomp.kit import (
    Mat, OP_ABS, OP_ADD, OP_ADDS, OP_DIGAMMA, OP_DIV, OP_EXP, OP_LOGS, OP_MUL, OP_RECIP, OP_SCALE, OP_SUB,
    mat_from,
)
from x_decomp.mcd import Est, _F32_EPS, _FLT_MIN, _neg_inf, _order_by_det, _pos_inf, _write, argsort_values, smallest_sorted
from x_decomp.resident import X_DECOMP_POOL, _ptr, pool_alloc, pool_free

#: `_expansion_decomp._F64_EPS`, the `adds` of LDA's `norm_phi`
comptime _F64_EPS: Float64 = 2.220446049250313e-16


# ---- copies (no arithmetic)
def gather_rows_kernel(src: F32Ptr, idx: I32Ptr, dst: F32Ptr, m: Int32, c: Int32):
    """dst row a = src row idx[a] (`take_rows`)."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(m) * Int(c):
        var a = t // Int(c)
        var j = t % Int(c)
        dst.unsafe_store(t, src.unsafe_load(Int(idx.unsafe_load(a)) * Int(c) + j))


def diag_kernel(a: F32Ptr, dst: F32Ptr, n: Int32):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst.unsafe_store(i, a.unsafe_load(i * Int(n) + i))


def pinv_mask_kernel(keep: F32Ptr, w: F32Ptr, dst: F32Ptr, n: Int32, cut: Float32):
    """dst[j] = keep[j] where |w[j]| > cut, else 0 (`_pinvh`'s mask). The
    comparison is on the bit patterns of the two non-negative values, which
    order them as IEEE does (subnormals included, NaN never above), so no
    flush-to-zero can move it."""
    var j = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if j < Int(n):
        var ab = bitcast[DType.uint32](w.unsafe_load(j)) & UInt32(0x7FFFFFFF)
        var cb = bitcast[DType.uint32](cut) & UInt32(0x7FFFFFFF)
        var take = ab <= UInt32(0x7F800000) and cb <= UInt32(0x7F800000) and ab > cb
        dst.unsafe_store(j, keep.unsafe_load(j) if take else Float32(0))


# ---- the resident matrix
struct DMat(Movable):
    """A row-major float32 matrix in a pooled device buffer, or a view of
    rows of one (`own` False: the view never frees)."""
    var id: Int
    var off: Int
    var r: Int
    var c: Int
    var own: Bool

    def __init__(out self, r: Int, c: Int) raises:
        self.id = pool_alloc(max(r * c, 1))
        self.off = 0
        self.r = r
        self.c = c
        self.own = True

    def __init__(out self, *, rows_of: DMat, row0: Int, rows: Int):
        self.id = rows_of.id
        self.off = rows_of.off + row0 * rows_of.c
        self.r = rows
        self.c = rows_of.c
        self.own = False

    def __del__(deinit self):
        if self.own:
            try:
                pool_free(self.id)
            except:
                pass

    def n(self) -> Int:
        return self.r * self.c

    def p(self) raises -> F32Ptr:
        return _ptr(self.id, self.off + max(self.n(), 1)) + self.off


def _mode(xr: Int, xc: Int, ar: Int, ac: Int) raises -> Int:
    """`Kit.mode`: `_Kit.ew`'s `mode_of`, in its order."""
    if xr == ar and xc == ac:
        return 0
    if xr == 1 and xc == ac:
        return 1
    if xc == 1 and xr == ar:
        return 2
    if xr == 1 and xc == 1:
        return 3
    raise Error("x_decomp: cannot broadcast")


struct _Eig(Movable):
    var wh: Mat
    var wd: DMat
    var vd: DMat

    def __init__(out self, var wh: Mat, var wd: DMat, var vd: DMat):
        self.wh = wh^
        self.wd = wd^
        self.vd = vd^


struct DKit(Movable):
    """`Kit`'s calls on resident matrices, enqueued on x_decomp's one
    context."""
    var ctx: DeviceContext
    var one: DMat
    var hold_f: List[List[Float32]]
    var hold_i: List[List[Int32]]

    def __init__(out self) raises:
        self.ctx = xd_ctx()
        self.one = DMat(1, 1)
        self.hold_f = List[List[Float32]]()
        self.hold_i = List[List[Int32]]()
        # the unused broadcast operand: Kit's `one`, a 0
        var z = List[Float32](length=1, fill=Float32(0))
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        self.ctx.enqueue_copy(
            dst_buf=pool[].bufs[self.one.id].create_sub_buffer[DType.float32](0, 1),
            src_ptr=F32Ptr(unsafe_from_address=Int(z.unsafe_ptr())),
        )
        self.hold_f.append(z^)

    # ---- movement
    def _sub(self, D: DMat) raises -> DeviceBuffer[DType.float32]:
        _ = D.p()
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        return pool[].bufs[D.id].create_sub_buffer[DType.float32](D.off, max(D.n(), 1))

    def upload_into(mut self, D: DMat, var src: List[Float32]) raises:
        if D.n() > 0:
            var sp = F32Ptr(unsafe_from_address=Int(src.unsafe_ptr()))
            self.ctx.enqueue_copy(dst_buf=self._sub(D), src_ptr=sp)
        self.hold_f.append(src^)

    def upload(mut self, A: Mat) raises -> DMat:
        var out = DMat(A.r, A.c)
        self.upload_into(out, A.d.copy())
        return out^

    def get(self, D: DMat) raises -> Mat:
        """D's values into a host matrix, ENQUEUED: read it after `sync`."""
        if D.off != 0:
            raise Error("x_decomp: a row view is not downloaded")
        var out = Mat(D.r, D.c)
        if D.n() > 0:
            var pool = X_DECOMP_POOL.get_or_create_ptr()
            _ = D.p()
            _down(self.ctx, pool[].bufs[D.id], out.p(), D.n())
        return out^

    def sync(mut self) raises:
        self.ctx.synchronize()
        self.hold_f.clear()
        self.hold_i.clear()

    def copy(self, A: DMat) raises -> DMat:
        var out = DMat(A.r, A.c)
        if A.n() > 0:
            self.ctx.enqueue_copy(dst_buf=self._sub(out), src_buf=self._sub(A))
        return out^

    def gather_rows(mut self, X: DMat, sel: List[Int]) raises -> DMat:
        """`take_rows(X, sel)`: exact copies of X's rows in sel's order."""
        var m = len(sel)
        var out = DMat(m, X.c)
        if m * X.c == 0:
            return out^
        var idx = List[Int32](capacity=m)
        for a in range(m):
            if sel[a] < 0 or sel[a] >= X.r:
                raise Error("x_decomp: a row index out of range")
            idx.append(Int32(sel[a]))
        var iid = pool_alloc(m)
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        var sp = F32Ptr(unsafe_from_address=Int(idx.unsafe_ptr()))
        self.ctx.enqueue_copy(dst_buf=pool[].bufs[iid].create_sub_buffer[DType.float32](0, m), src_ptr=sp)
        self.ctx.enqueue_function[gather_rows_kernel](
            X.p(), I32Ptr(unsafe_from_address=Int(_ptr(iid, m))), out.p(), Int32(m), Int32(X.c),
            grid_dim=_blocks(m * X.c), block_dim=TPB,
        )
        pool_free(iid)
        self.hold_i.append(idx^)
        return out^

    # ---- the kit's calls (Kit's operands, modes and float32 scalars)
    def ew1(self, op: Int, A: DMat, s: Float64) raises -> DMat:
        var out = DMat(A.r, A.c)
        if A.n() == 0:
            return out^
        launch_ew(self.ctx, op, A.p(), self.one.p(), 3, self.one.p(), 3, out.p(), A.n(), A.c, Float32(s))
        return out^

    def ew2(self, op: Int, A: DMat, B: DMat) raises -> DMat:
        var bm = _mode(B.r, B.c, A.r, A.c)
        var out = DMat(A.r, A.c)
        if A.n() == 0:
            return out^
        launch_ew(self.ctx, op, A.p(), B.p(), bm, self.one.p(), 3, out.p(), A.n(), A.c, Float32(0))
        return out^

    def mm(self, A: DMat, B: DMat, ta: Bool, tb: Bool) raises -> DMat:
        var m = A.c if ta else A.r
        var k = A.r if ta else A.c
        var k2 = B.c if tb else B.r
        var n = B.r if tb else B.c
        if k != k2:
            raise Error("x_decomp: gemm inner dimensions differ")
        if m * n > 2147483647 or m * k > 2147483647 or k * n > 2147483647:
            raise Error("x_decomp: gemm exceeds the Int32 index bound")
        var out = DMat(m, n)
        if m * n == 0:
            return out^
        var ns = gemm_scratch(m, k, n)
        var sid = pool_alloc(max(ns, 1))
        launch_gemm(self.ctx, A.p(), B.p(), out.p(), _ptr(sid, max(ns, 1)), m, k, n, ta, tb)
        pool_free(sid)
        return out^

    def colsum(self, A: DMat) raises -> DMat:
        var out = DMat(1, A.c)
        var ns = colsum_scratch(A.r, A.c)
        var sid = pool_alloc(max(ns, 1))
        launch_colsum(self.ctx, A.p(), out.p(), _ptr(sid, max(ns, 1)), A.r, A.c)
        pool_free(sid)
        return out^

    def rowsum(self, A: DMat) raises -> DMat:
        var out = DMat(A.r, 1)
        var ns = rowsum_scratch(A.r, A.c)
        var sid = pool_alloc(max(ns, 1))
        launch_rowsum(self.ctx, A.p(), out.p(), _ptr(sid, max(ns, 1)), A.r, A.c)
        pool_free(sid)
        return out^

    def total(self, A: DMat) raises -> DMat:
        """`_Kit.total`: colsum(rowsum(A))."""
        return self.colsum(self.rowsum(A))

    def colmean(self, A: DMat) raises -> DMat:
        return self.ew1(OP_SCALE, self.colsum(A), 1.0 / Float64(A.r))

    def rand(self, r: Int, c: Int, seed: Int, stream: Int, kind: Int) raises -> DMat:
        var out = DMat(r, c)
        var cnt = r * c
        if cnt == 0:
            return out^
        self.ctx.enqueue_function[rand_kernel](
            out.p(), Int32(cnt), UInt32(seed & 0xFFFFFFFF), UInt32(stream & 0xFFFFFFFF), Int32(kind),
            grid_dim=_blocks(cnt), block_dim=TPB,
        )
        return out^

    def rand_gamma(self, r: Int, c: Int, seed: Int, stream: Int, shape: Float64) raises -> DMat:
        var out = DMat(r, c)
        var cnt = r * c
        if cnt == 0:
            return out^
        var a = Float32(shape)
        if not (a >= Float32(1)):
            raise Error("x_decomp: the gamma sampler takes shape >= 1")
        self.ctx.enqueue_function[gamma_kernel](
            out.p(), Int32(cnt), UInt32(seed & 0xFFFFFFFF), UInt32(stream & 0xFFFFFFFF), a,
            grid_dim=_blocks(cnt), block_dim=TPB,
        )
        return out^

    def lda_rows(self, X: DMat, EW: DMat, Dt: DMat, Et: DMat, prior: Float64, max_iter: Int, tol: Float64) raises:
        """`Kit.lda_rows`: Dt and Et (n x k) updated in place on the device."""
        var n = X.r
        var k = EW.r
        var v = X.c
        if n * (v + k) > 2147483647:
            raise Error("x_decomp: lda_rows exceeds the Int32 index bound")
        if n == 0:
            return
        var sid = pool_alloc(n * (v + k))
        var iid = pool_alloc(n)
        self.ctx.enqueue_function[lda_rows_kernel](
            X.p(), EW.p(), Dt.p(), Et.p(), _ptr(sid, n * (v + k)), _ptr(iid, n),
            Int32(n), Int32(k), Int32(v), Float32(prior), Int32(max_iter), Float32(tol),
            grid_dim=_blocks(n), block_dim=TPB,
        )
        pool_free(sid)
        pool_free(iid)

    def eigh(mut self, A: DMat) raises -> _Eig:
        """`Kit.eigh`: DevExec.eigh's solve on a device copy of A (one sync,
        inside the solve), then w and V back on the device."""
        var n = A.r
        var wh = Mat(1, n)
        var vh = Mat(n, n)
        if pj_eigh_min() > 0 or not jacobi2_eigh_on():
            # the opt-in A/B solvers take host memory: the same DevExec.eigh
            var ah = self.get(A)
            self.sync()
            DevExec.eigh(ah.p(), wh.p(), vh.p(), n)
        else:
            var da = self.ctx.enqueue_create_buffer[DType.float32](max(n * n, 1))
            if n > 0:
                self.ctx.enqueue_copy(dst_buf=da.create_sub_buffer[DType.float32](0, n * n), src_buf=self._sub(A))
            DevExec._eigh2_on(self.ctx, da, wh.p(), vh.p(), n)
            _ = da^
            self.hold_f.clear()
            self.hold_i.clear()
        var wd = self.upload(wh)
        var vd = self.upload(vh)
        return _Eig(wh^, wd^, vd^)

    def lu_logdet(mut self, A: DMat) raises -> Float64:
        """`_fast_logdet` over `_slogdet`: getrf on a copy (DevExec.lu's
        launches), the logs of |diag| summed as `total`, then ONE sync for
        the diagonal, the pivots and the sum the sign rule reads."""
        var n = A.r
        var lu = self.copy(A)
        var pid = pool_alloc(max(n, 1))
        var iid = pool_alloc(1)
        var sid = pool_alloc(2)
        var aid = pool_alloc(max(n, 1))
        launch_lu(
            self.ctx, lu.p(), I32Ptr(unsafe_from_address=Int(_ptr(pid, max(n, 1)))), _ptr(iid, 1), _ptr(sid, 2),
            _ptr(aid, max(n, 1)), n,
        )
        var diag = DMat(1, n)
        if n > 0:
            self.ctx.enqueue_function[diag_kernel](lu.p(), diag.p(), Int32(n), grid_dim=_blocks(n), block_dim=TPB)
        var t = self.total(self.ew1(OP_LOGS, self.ew1(OP_ABS, diag, 0.0), _FLT_MIN))
        var dh = self.get(diag)
        var th = self.get(t)
        var piv = List[Int32](length=max(n, 1), fill=Int32(0))
        if n > 0:
            var pool = X_DECOMP_POOL.get_or_create_ptr()
            self.ctx.enqueue_copy(
                dst_ptr=F32Ptr(unsafe_from_address=Int(piv.unsafe_ptr())),
                src_buf=pool[].bufs[pid].create_sub_buffer[DType.float32](0, n),
            )
        self.sync()
        pool_free(pid)
        pool_free(iid)
        pool_free(sid)
        pool_free(aid)
        _ = lu^
        for i in range(n):
            if dh.d[i] == Float32(0):
                return _neg_inf()             # sign 0 -> -inf
        var neg = 0
        for i in range(n):
            if dh.d[i] < Float32(0):
                neg += 1
        for i in range(n):
            if Int(piv[i]) != i:
                neg += 1
        if neg % 2 != 0:
            return _neg_inf()
        return Float64(th.d[0])


# ---- MinCovDet's fast_mcd on the resident kit (x_decomp/mcd.mojo's search)
struct DMcd:
    var k: DKit
    var seed: Int
    var draws: Int

    def __init__(out self, seed: Int) raises:
        self.k = DKit()
        self.seed = seed
        self.draws = 0

    def emp_cov(self, Xs: DMat) raises -> DMat:
        var Xc = self.k.ew2(OP_SUB, Xs, self.k.colmean(Xs))
        return self.k.ew1(OP_SCALE, self.k.mm(Xc, Xc, True, False), 1.0 / Float64(Xs.r))

    def mahal(self, X: DMat, loc: DMat, P: DMat) raises -> DMat:
        var Xc = self.k.ew2(OP_SUB, X, loc)
        return self.k.rowsum(self.k.ew2(OP_MUL, self.k.mm(Xc, P, False, False), Xc))

    def pinvh(mut self, A: DMat) raises -> DMat:
        """`_pinvh`: V diag(1/w) V^T over |w| > max|w| * n * float32 eps."""
        var n = A.r
        var e = self.k.eigh(A)
        var wmax: Float64 = 0.0
        if n > 0:
            wmax = abs(Float64(e.wh.d[0]))
            for j in range(1, n):
                var a = abs(Float64(e.wh.d[j]))
                if a > wmax:                  # Python max(): replace on >
                    wmax = a
        var cut = Float32((wmax * Float64(n)) * _F32_EPS)
        var keep = self.k.ew1(OP_RECIP, e.wd, 0.0)
        var inv = DMat(1, n)
        if n > 0:
            self.k.ctx.enqueue_function[pinv_mask_kernel](
                keep.p(), e.wd.p(), inv.p(), Int32(n), cut, grid_dim=_blocks(n), block_dim=TPB
            )
        return self.k.mm(self.k.ew2(OP_MUL, e.vd, inv), e.vd, False, True)

    def perm(mut self, n: Int) raises -> List[Int]:
        """`MinCovDet._perm`: a sort of the next Philox stream's draws."""
        self.draws += 1
        var r = self.k.rand(1, n, self.seed, 1000 + self.draws, 0)
        var h = self.k.get(r)
        self.k.sync()
        return argsort_values(h)

    def perm_head_sorted(mut self, n: Int, h: Int) raises -> List[Int]:
        """`sorted(self._perm(n)[:h])`."""
        self.draws += 1
        var r = self.k.rand(1, n, self.seed, 1000 + self.draws, 0)
        var v = self.k.get(r)
        self.k.sync()
        return smallest_sorted(v, h)

    def c_step(
        mut self, X: DMat, h: Int, n_iter: Int, has_init: Bool, loc0: Mat, cov0: Mat, want_dist: Bool
    ) raises -> Est:
        var iters = n_iter
        var dist_h = Mat(0, 0)
        var sel: List[Int]
        if not has_init:
            sel = self.perm_head_sorted(X.r, h)
        else:
            var c0 = self.k.upload(cov0)
            var l0 = self.k.upload(loc0)
            var P0 = self.pinvh(c0)
            var d0 = self.mahal(X, l0, P0)
            dist_h = self.k.get(d0)
            self.k.sync()
            sel = smallest_sorted(dist_h, h)
        var Xs = self.k.gather_rows(X, sel)
        var loc = self.k.colmean(Xs)
        var cov = self.emp_cov(Xs)
        var det = self.k.lu_logdet(cov)
        var P = DMat(0, 0)
        var has_p = False
        if det == _neg_inf():
            P = self.pinvh(cov)
            has_p = True
        var prev_det = _pos_inf()
        var have_prev = False
        var prev_loc = DMat(0, 0)
        var prev_cov = DMat(0, 0)
        var prev_sel = List[Int]()
        while det < prev_det and iters > 0 and det != _neg_inf():
            P = self.pinvh(cov)
            has_p = True
            var dd = self.mahal(X, loc, P)
            dist_h = self.k.get(dd)
            self.k.sync()
            prev_loc = loc^
            prev_cov = cov^
            prev_sel = sel^
            have_prev = True
            prev_det = det
            sel = smallest_sorted(dist_h, h)
            Xs = self.k.gather_rows(X, sel)
            loc = self.k.colmean(Xs)
            cov = self.emp_cov(Xs)
            det = self.k.lu_logdet(cov)
            iters -= 1
        if not has_p:
            # Python's final `_mahal(k, X, loc, None)` fails here (a +inf or
            # NaN log det at the first step); so does this.
            raise Error("x_decomp MinCovDet: the first C-step's log determinant is not finite")
        # sklearn's checks in its order, the last one that fires wins
        var use_prev = have_prev and det > prev_det
        if iters == 0:
            use_prev = False
        if use_prev:
            var plh = self.k.get(prev_loc)
            var pch = self.k.get(prev_cov)
            self.k.sync()
            return Est(plh^, pch^, prev_det, prev_sel^, dist_h^)
        var final = Mat(0, 0)
        if want_dist:
            var fd = self.mahal(X, loc, P)
            final = self.k.get(fd)
        var lh = self.k.get(loc)
        var ch = self.k.get(cov)
        self.k.sync()
        return Est(lh^, ch^, det, sel^, final^)

    def select_random(mut self, X: DMat, h: Int, trials: Int, keep: Int, n_iter: Int, want_dist: Bool) raises -> List[Est]:
        var est = List[Est]()
        var none = Mat(0, 0)
        for _ in range(trials):
            est.append(self.c_step(X, h, n_iter, False, none, none, want_dist))
        return _top(est, keep)

    def select_init(
        mut self, X: DMat, h: Int, inits: List[Est], keep: Int, n_iter: Int, want_dist: Bool
    ) raises -> List[Est]:
        var est = List[Est]()
        for t in range(len(inits)):
            est.append(self.c_step(X, h, n_iter, True, inits[t].loc, inits[t].cov, want_dist))
        return _top(est, keep)


def _top(est: List[Est], keep: Int) raises -> List[Est]:
    var order = _order_by_det(est, keep)
    var out = List[Est]()
    for a in range(len(order)):
        out.append(est[order[a]].copy())
    return out^


def fast_mcd_dev(
    X: Mat, p: List[Int], loc_out: F32Ptr, cov_out: F32Ptr, sup_out: I32Ptr, dist_out: F32Ptr,
) raises:
    """`fast_mcd` (x_decomp/mcd.mojo) with X resident: the same plan, the
    same C-steps, the same draws and orders."""
    var n = p[0]
    var d = p[1]
    var h = p[2]
    var run = DMcd(p[3])
    var Xd = run.k.upload(X)
    var best: Est
    var support = List[Int32](length=n, fill=Int32(0))
    var dist = List[Float32](length=n, fill=Float32(0))
    if n > 500:
        var n_sub = p[4]
        var n_ss = p[5]
        var h_sub = p[6]
        var n_trials = p[7]
        var n_m = p[8]
        var h_m = p[9]
        var n_best_m = p[10]
        var shuf = run.perm(n)
        var pool = List[Est]()
        for i in range(n_sub):
            var rows = List[Int](capacity=n_ss)
            for a in range(i * n_ss, (i + 1) * n_ss):
                rows.append(shuf[a])
            var cur = run.k.gather_rows(Xd, rows)
            var got = run.select_random(cur, h_sub, n_trials, 10, 2, False)
            for e in range(len(got)):
                pool.append(got[e].copy())
        var selection_all = run.perm(n)
        var selection = List[Int](capacity=n_m)
        for a in range(n_m):
            selection.append(selection_all[a])
        var Xm = run.k.gather_rows(Xd, selection)
        var merged = run.select_init(Xm, h_m, pool, n_best_m, 30, n < 1500)
        if n < 1500:
            ref m0 = merged[0]
            for a in range(len(selection)):
                dist[selection[a]] = m0.dist.d[a]
            for a in range(len(m0.sel)):
                support[selection[m0.sel[a]]] = Int32(1)
            _write(m0.loc, m0.cov, support, dist, d, loc_out, cov_out, sup_out, dist_out)
            run.k.sync()
            return
        var full = run.select_init(Xd, h, merged, 1, 30, True)
        best = full[0].copy()
    else:
        var first = run.select_random(Xd, h, 30, 10, 2, False)
        var full = run.select_init(Xd, h, first, 1, 30, True)
        best = full[0].copy()
    for a in range(len(best.sel)):
        support[best.sel[a]] = Int32(1)
    for i in range(n):
        dist[i] = best.dist.d[i]
    _write(best.loc, best.cov, support, dist, d, loc_out, cov_out, sup_out, dist_out)
    run.k.sync()
    _ = Xd^


# ---- LatentDirichletAllocation's online pass on the resident kit
def _dirichlet(k: DKit, A: DMat) raises -> DMat:
    """psi(A) - psi(rowsum(A))."""
    return k.ew2(OP_SUB, k.ew1(OP_DIGAMMA, A, 0.0), k.ew1(OP_DIGAMMA, k.rowsum(A), 0.0))


def lda_online_dev(
    X: Mat, mut comps: Mat, mut exp_dir: Mat, bs: Int, max_doc_iter: Int, seed: Int, mut draw: Int,
    mut n_batch_iter: Int, doc_prior: Float64, topic_prior: Float64, offset: Float64, decay: Float64,
    tol: Float64, total_samples: Float64,
) raises:
    """`lda_online_pass` (x_decomp/lda_online.mojo) with X, the components
    and their Dirichlet expectation resident: the same cells per mini-batch.
    The pass's weights (one per mini-batch, each `offset + n_batch_iter`
    rounded once to float32 then logs, scale and exp, elementwise cells)
    are computed in one vector and read in one sync before the loop."""
    var k = DKit()
    var n = X.r
    var nc = comps.r
    var nb = (n + bs - 1) // bs if n > 0 else 0
    var wts = Mat(nb, 1)
    if nb > 0:
        var base = Mat(nb, 1)
        for j in range(nb):
            base.d[j] = Float32(offset + Float64(n_batch_iter + j))
        var bd = k.upload(base)
        var wd = k.ew1(OP_EXP, k.ew1(OP_SCALE, k.ew1(OP_LOGS, bd, 1e-30), -decay), 0.0)
        wts = k.get(wd)
        k.sync()
    var Xd = k.upload(X)
    var C = k.upload(comps)
    var ED = k.upload(exp_dir)
    var a = 0
    var j = 0
    while a < n:
        var b = min(a + bs, n)
        var Xb = DMat(rows_of=Xd, row0=a, rows=b - a)
        # _e_step(k, Xb, cal_sstats=True, random_init=True)
        draw += 1
        var Dt = k.ew1(OP_SCALE, k.rand_gamma(Xb.r, nc, seed, 61 + draw, 100.0), 0.01)
        var Et = k.ew1(OP_EXP, _dirichlet(k, Dt), 0.0)
        k.lda_rows(Xb, ED, Dt, Et, doc_prior, max_doc_iter, tol)
        var norm_phi = k.ew1(OP_ADDS, k.mm(Et, ED, False, False), _F64_EPS)
        var R = k.ew2(OP_DIV, Xb, norm_phi)
        var ss = k.ew2(OP_MUL, k.mm(Et, R, True, False), ED)
        # the online update (batch_update=False)
        var weight = Float64(wts.d[j])
        var doc_ratio = total_samples / Float64(Xb.r)
        var upd = k.ew1(OP_ADDS, k.ew1(OP_SCALE, ss, doc_ratio), topic_prior)
        C = k.ew2(OP_ADD, k.ew1(OP_SCALE, C, 1.0 - weight), k.ew1(OP_SCALE, upd, weight))
        ED = k.ew1(OP_EXP, _dirichlet(k, C), 0.0)
        n_batch_iter += 1
        j += 1
        a = b
    comps = k.get(C)
    exp_dir = k.get(ED)
    k.sync()
    _ = Xd^


# ---- Python entries (GPU binding only; the arguments of api.mojo's)
def mcd_dev_py(
    x: PythonObject, loc: PythonObject, cov: PythonObject, sup: PythonObject, dist: PythonObject,
    p: PythonObject, dev: PythonObject,
) raises -> PythonObject:
    """`mcd_py` on the resident kit. `dev` is read and unused."""
    var q = List[Int]()
    for i in range(11):
        q.append(Int(py=p[i]))
    var n = q[0]
    var d = q[1]
    if n < 1 or d < 2 or n * d > 2147483647 or q[2] < 1 or q[2] > n:
        raise Error("x_decomp: mcd needs n >= 1, d >= 2 and 1 <= h <= n")
    if n > 500 and (q[4] < 1 or q[4] * q[5] > n or q[8] > n or q[8] < 1 or q[10] < 1):
        raise Error("x_decomp: mcd subset plan out of range")
    _ = Int(py=dev)
    var px = _f(x)
    var pl = _f(loc)
    var pc = _f(cov)
    var ps = _i(sup)
    var pd = _f(dist)
    with GILReleased(Python()):
        var X = mat_from(px, n, d)
        fast_mcd_dev(X, q, pl, pc, ps, pd)
    return PythonObject(n)


def lda_online_dev_py(
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject,
    dev: PythonObject,
) raises -> PythonObject:
    """`lda_online_py` on the resident kit. `dev` is read and unused."""
    var n = _n(p, 0)
    var v = _n(p, 1)
    var nc = _n(p, 2)
    var bs = _n(p, 3)
    var mdi = _n(p, 4)
    var seed = Int(py=p[5])
    var draw = Int(py=p[6])
    var nbi = Int(py=p[7])
    if bs < 1 or v < 1 or nc < 1 or n * v > 2147483647 or nc * v > 2147483647:
        raise Error("x_decomp: lda_online shape out of range")
    var fv = List[Float64]()
    for i in range(6):
        fv.append(Float64(py=f[i]))
    _ = Int(py=dev)
    var px = _f(x)
    var pc = _f(comps)
    var pe = _f(exp_dir)
    with GILReleased(Python()):
        var X = mat_from(px, n, v)
        var C = mat_from(pc, nc, v)
        var ED = mat_from(pe, nc, v)
        lda_online_dev(X, C, ED, bs, mdi, seed, draw, nbi, fv[0], fv[1], fv[2], fv[3], fv[4], fv[5])
        for i in range(nc * v):
            pc.unsafe_store(i, C.d[i])
            pe.unsafe_store(i, ED.d[i])
    return Python.tuple(draw, nbi)
