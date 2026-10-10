# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The native drivers' kit on the device, resident (lane hr-kit, 2026-10-02).

MinCovDet's fast_mcd (x_decomp/mcd.mojo) and LatentDirichletAllocation's
online pass (x_decomp/lda_online.mojo) make thousands of kit calls on
operands of a few hundred values. `Kit[DevExec]` would pay an upload, a
launch, a download and two syncs for each one. Here every matrix lives in a
pooled device buffer (x_decomp/resident.mojo) and every kit call is an
enqueued launch, with no sync:

- the launch sequences are DevExec's own (`launch_ew`, `launch_gemm`,
  `launch_colsum`, `launch_rowsum`, `launch_lu`, `rand_kernel`,
  `gamma_kernel`, `lda_rows_kernel`, `DevExec._eigh_par_on`), so every value
  is the same bits as `Kit[DevExec]`, which is the same bits as the CPU
  column's kit (every x-decomp lane's GPU == CPU claim);
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
from experiments.classical_identical_ideas.stats_controls import C57_CANDIDATE_STATE
from experiments.classical_identical_ideas.linear_controls import C23_MCD, C28_BUCKET_SOLVES
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from max.gpu.host import DeviceBuffer, DeviceContext

from x_decomp.api import _f, _i, _n
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.device import (
    DevExec,
    LU_SCAL_LEN,
    TPB,
    _blocks,
    _down,
    colsum_scratch,
    gamma_kernel,
    gemm_scratch,
    launch_colsum,
    launch_classical_code_rows,
    launch_ew,
    launch_gemm,
    launch_gemm_ordered,
    launch_lu,
    launch_rowsum,
    launch_sqdist,
    launch_trisolve,
    lasso_rows_kernel,
    lars_rows_kernel,
    omp_rows_kernel,
    launch_lda_bound,
    absmax_scratch,
    launch_absmax,
    cd_rows_kernel,
    lda_rows_kernel,
    rand_kernel,
    rowsum_scratch,
    xd_ctx,
)
from x_decomp.kit import (
    Mat, OP_GTS, OP_SELECT, mat_const, svd_order, OP_ABS, OP_ADD, OP_ADDS, OP_DIGAMMA, OP_DIV, OP_EXP, OP_LOGS, OP_MUL, OP_RECIP, OP_SCALE, OP_SUB,
    mat_from,
)
from x_decomp.classical_device import contrast_kernel
from core.blocked_moments import bm_centered_gram_panels
from core.blocked_moments_ops import C23_MCD_LEAF_ROWS
from std.atomic import Atomic
from std.builtin.sort import sort
from core.device_zero import enqueue_fill
from core.device_fold import device_exclusive_scan_total_from
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from x_decomp.mcd import Est, _key, _F32_EPS, _FLT_MIN, _neg_inf, _order_by_det, _pos_inf, _write, argsort_values, smallest_sorted
from x_decomp.mcd_fast import MCD_DEVICE_CSTEPS, fast_mcd_fast
from x_decomp.resident import X_DECOMP_POOL, _ptr, pool_alloc, pool_free
from x_decomp.moves import MOVE_FILL0, MOVE_TAKE_COLS, MOVE_TRANSPOSE
from x_decomp.cells import LARS_ROW_EXTRA
from x_decomp.dict_fast import DECOMP_FAST_DICT_DEV, dict_update_dev
from x_decomp.qr_bounded import QRB_CELLS
from x_decomp.moves_device import launch_move
from x_decomp.select_dev import enqueue_sel_reduce, order_small_kernel
from x_decomp.select_ops import SEL_ORDER_MAX

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



# ---- the C-step's selections on the device (lane cpu3-core) ---------------
# The search used to download every distance vector and every draw vector and
# select / sort on the host (`smallest_sorted`, `argsort_values`). Here the
# (value, index) order is a stable radix sort of the values' monotone images
# (ties keep index order, so the order is Python's tuple order), the h
# smallest are marked and compacted in index order by a scan, and the rows
# are gathered from the resident index vector: the same rows, the same
# order, so the same bits as the host column.
comptime U32Ptr = MutPointer[UInt32, MutAnyOrigin]


def mcd_sort_keys_kernel(v: F32Ptr, n: Int32, keys: U32Ptr, vals: U32Ptr, bad: I32Ptr):
    """keys[i] = `_key(v[i], i)`'s high word (v's monotone image, -0.0 as
    +0.0), vals[i] = i; a NaN sets bad[0] (`_key`'s refusal)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        # NaN and the zero test on the bit pattern (lane cpu4-misc review):
        # no fast-math fold of `x != x` and no flush of a subnormal to the
        # +0.0 key, so the key is `_key`'s on every vendor
        var ub = bitcast[DType.uint32](v.unsafe_load(i))
        var mag = ub & UInt32(0x7FFFFFFF)
        if mag > UInt32(0x7F800000):
            bad.unsafe_store(0, Int32(1))
        if mag == UInt32(0):
            ub = UInt32(0)
        var tw = ub ^ UInt32(0xFFFFFFFF) if (ub >> 31) == UInt32(1) else ub | UInt32(0x80000000)
        keys.unsafe_store(i, tw)
        vals.unsafe_store(i, UInt32(i))


def u32_to_idx_kernel(vals: U32Ptr, n: Int32, dst: I32Ptr):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst.unsafe_store(i, Int32(Int(vals.unsafe_load(i))))


def mark_head_kernel(order: U32Ptr, h: Int32, flag: I32Ptr):
    """flag[order[a]] = 1 for a < h (flag zeroed before)."""
    var a = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if a < Int(h):
        flag.unsafe_store(Int(order.unsafe_load(a)), Int32(1))


def emit_marked_kernel(flag: I32Ptr, scan: I32Ptr, n: Int32, dst: I32Ptr):
    """dst[scan[i]] = i for every marked i: the marked rows in index order."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        if flag.unsafe_load(i) != Int32(0):
            dst.unsafe_store(Int(scan.unsafe_load(i)), Int32(i))


def mark_rows_kernel(dst: I32Ptr, idx: I32Ptr, m: Int32):
    """dst[idx[a]] = 1 for a < m (the support vector)."""
    var a = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if a < Int(m):
        dst.unsafe_store(Int(idx.unsafe_load(a)), Int32(1))


def mark_rows_of_kernel(dst: I32Ptr, base: I32Ptr, idx: I32Ptr, m: Int32):
    """dst[base[idx[a]]] = 1 for a < m (a support inside a row subset)."""
    var a = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if a < Int(m):
        dst.unsafe_store(Int(base.unsafe_load(Int(idx.unsafe_load(a)))), Int32(1))


def scatter_rows_kernel(dst: F32Ptr, idx: I32Ptr, src: F32Ptr, m: Int32):
    """dst[idx[a]] = src[a] for a < m (distances back to their rows)."""
    var a = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if a < Int(m):
        dst.unsafe_store(Int(idx.unsafe_load(a)), src.unsafe_load(a))


def logdet_sign_kernel(diag: F32Ptr, piv: I32Ptr, n: Int32, dst: I32Ptr):
    """`_slogdet`'s sign rule on the device: dst[0] = 1 when a pivot is 0;
    dst[1] counts the negative pivots plus the row swaps (integer atomics,
    exact). dst zeroed before."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        var x = diag.unsafe_load(i)
        if x == Float32(0):
            dst.unsafe_store(0, Int32(1))
        var c = Int32(0)
        if x < Float32(0):
            c += 1
        if Int(piv.unsafe_load(i)) != i:
            c += 1
        if c != Int32(0):
            _ = Atomic.fetch_add(dst.unsafe_offset(1), c)


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
    # `smallest_dev`'s NaN refusal word, resident for the kit's life and
    # read at the next `sync` (lane idn-regress: was a sync a selection)
    var nan_word: DMat
    var nan_host: List[Int32]
    var nan_pending: Bool

    def __init__(out self) raises:
        self.ctx = xd_ctx()
        self.one = DMat(1, 1)
        self.hold_f = List[List[Float32]]()
        self.hold_i = List[List[Int32]]()
        self.nan_word = DMat(1, 1)
        self.nan_host = List[Int32](length=1, fill=Int32(0))
        self.nan_pending = False
        # the unused broadcast operand: Kit's `one`, a 0
        var z = List[Float32](length=1, fill=Float32(0))
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        self.ctx.enqueue_copy(
            dst_buf=pool[].bufs[self.one.id].create_sub_buffer[DType.float32](0, 1),
            src_ptr=F32Ptr(unsafe_from_address=Int(z.unsafe_ptr())),
        )
        self.hold_f.append(z^)
        var nv = pool[].bufs[self.nan_word.id].create_sub_buffer[DType.float32](0, 1)
        enqueue_fill(self.ctx, nv, Float32(0))
        _ = nv^

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
        var check = self.nan_pending
        if check:
            var pool = X_DECOMP_POOL.get_or_create_ptr()
            self.ctx.enqueue_copy(
                dst_ptr=F32Ptr(unsafe_from_address=Int(self.nan_host.unsafe_ptr())),
                src_buf=pool[].bufs[self.nan_word.id].create_sub_buffer[DType.float32](0, 1),
            )
        self.ctx.synchronize()
        self.hold_f.clear()
        self.hold_i.clear()
        if check:
            self.nan_pending = False
            if self.nan_host[0] != Int32(0):
                # `_key`'s refusal, as the host sort raised it; every fit
                # ends in a sync (`_write_dev`), so none goes unread
                raise Error("x_decomp MinCovDet: a NaN distance or draw has no order (refused)")

    def copy(self, A: DMat) raises -> DMat:
        var out = DMat(A.r, A.c)
        if A.n() > 0:
            self.ctx.enqueue_copy(dst_buf=self._sub(out), src_buf=self._sub(A))
        return out^

    def put(self, dst: DMat, src: DMat) raises:
        """src's words into dst (same shape; either may be a row view)."""
        if dst.r != src.r or dst.c != src.c:
            raise Error("x_decomp: put needs equal shapes")
        if src.n() > 0:
            self.ctx.enqueue_copy(dst_buf=self._sub(dst), src_buf=self._sub(src))

    # ---- selections on the device (lane cpu3-core)
    def _sorted_order(self, v: DMat, bad: DMat) raises -> DeviceBuffer[DType.uint32]:
        """v's indices in (value, index) order, resident (a stable radix
        sort of the monotone images); a NaN sets bad's word."""
        var n = v.n()
        var keys = self.ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        var vals = self.ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        var tk = self.ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        var tv = self.ctx.enqueue_create_buffer[DType.uint32](max(n, 1))
        var cnt = self.ctx.enqueue_create_buffer[DType.int32](max(frs_counts_len(n), 1))
        if n > 0:
            self.ctx.enqueue_function[mcd_sort_keys_kernel](
                v.p(), Int32(n), keys.unsafe_ptr(), vals.unsafe_ptr(),
                bad.p().bitcast[Int32](), grid_dim=_blocks(n), block_dim=TPB,
            )
            fast_radix_sort_pairs_u32(self.ctx, n, keys, vals, tk, tv, cnt)
        _ = keys^
        _ = tk^
        _ = tv^
        _ = cnt^
        return vals^

    def _bad_flag(mut self) raises -> DMat:
        var bad = DMat(1, 1)
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        var bv = pool[].bufs[bad.id].create_sub_buffer[DType.float32](0, 1)
        enqueue_fill(self.ctx, bv, Float32(0))
        _ = bv^
        return bad^

    def _refuse_nan(mut self, bad: DMat) raises:
        """One word home: `_key`'s NaN refusal, as the host sort raised it."""
        var h = List[Int32](length=1, fill=Int32(0))
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        self.ctx.enqueue_copy(
            dst_ptr=F32Ptr(unsafe_from_address=Int(h.unsafe_ptr())),
            src_buf=pool[].bufs[bad.id].create_sub_buffer[DType.float32](0, 1),
        )
        self.sync()
        if h[0] != Int32(0):
            raise Error("x_decomp MinCovDet: a NaN distance or draw has no order (refused)")

    def argsort_dev(mut self, v: DMat) raises -> DMat:
        """`argsort_values(v)` as a resident (n x 1) index vector."""
        var n = v.n()
        var bad = self._bad_flag()
        var order = self._sorted_order(v, bad)
        var out = DMat(n, 1)
        if n > 0:
            self.ctx.enqueue_function[u32_to_idx_kernel](
                order.unsafe_ptr(), Int32(n), out.p().bitcast[Int32](),
                grid_dim=_blocks(n), block_dim=TPB,
            )
        self._refuse_nan(bad)
        _ = order^
        return out^

    def smallest_dev(mut self, v: DMat, h: Int) raises -> DMat:
        """`smallest_sorted(v, h)` as a resident (h x 1) index vector: the h
        rows with the smallest (value, index) keys, in index order."""
        var n = v.n()
        if h <= 0:
            return DMat(0, 1)
        var take = min(h, n)
        var order = self._sorted_order(v, self.nan_word)
        var flag = self.ctx.enqueue_create_buffer[DType.int32](max(n, 1))
        var scan = self.ctx.enqueue_create_buffer[DType.int32](n + 1)
        enqueue_fill(self.ctx, flag, Int32(0))
        var out = DMat(take, 1)
        if n > 0:
            self.ctx.enqueue_function[mark_head_kernel](
                order.unsafe_ptr(), Int32(take), flag.unsafe_ptr(), grid_dim=_blocks(take), block_dim=TPB,
            )
            device_exclusive_scan_total_from(self.ctx, flag, scan, n)
            self.ctx.enqueue_function[emit_marked_kernel](
                flag.unsafe_ptr(), scan.unsafe_ptr(), Int32(n), out.p().bitcast[Int32](),
                grid_dim=_blocks(n), block_dim=TPB,
            )
        # the NaN word is read at the next sync (the C-step's log
        # determinant): the order is a permutation either way, so every
        # gather before it stays in range
        self.nan_pending = True
        _ = order^
        _ = flag^
        _ = scan^
        return out^

    def gather_dev(mut self, X: DMat, idx: DMat) raises -> DMat:
        """`take_rows(X, idx)` with the index vector resident (in range by
        construction: every index came from a device selection over X)."""
        var m = idx.n()
        var out = DMat(m, X.c)
        if m * X.c == 0:
            return out^
        self.ctx.enqueue_function[gather_rows_kernel](
            X.p(), idx.p().bitcast[Int32](), out.p(), Int32(m), Int32(X.c),
            grid_dim=_blocks(m * X.c), block_dim=TPB,
        )
        return out^

    # ---- the kit's calls (Kit's operands, modes and float32 scalars)
    def classical_centered_gram(self, X: DMat, means: DMat) raises -> DMat:
        var out = DMat(X.c, X.c)
        if X.c > 0:
            # C23_MCD (lane classical-decomp): the panel-256 reference cell's
            # value, row-parallel (core/blocked_moments.mojo)
            bm_centered_gram_panels(self.ctx, out.p(), X.p(), means.p(), X.r, X.c, C23_MCD_LEAF_ROWS)
        return out^

    def classical_contrast(self, Y: DMat, fun: Int, alpha: Float64, mut gp: DMat) raises -> DMat:
        var gx = DMat(Y.r, Y.c)
        gp = DMat(Y.r, Y.c)
        if Y.n() > 0:
            self.ctx.enqueue_function[contrast_kernel](Y.p(), gx.p(), gp.p(), Int32(Y.n()), Int32(fun), Float32(alpha),
                                                       grid_dim=_blocks(Y.n()), block_dim=TPB)
        return gx^

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

    def zeros(self, r: Int, c: Int) raises -> DMat:
        """`_M.zeros(r, c)` on the device."""
        var out = DMat(r, c)
        if r * c > 0:
            var sb = self._sub(out)
            enqueue_fill(self.ctx, sb, Float32(0))
            _ = sb^
        return out^

    def ew2s(self, op: Int, A: DMat, B: DMat, s: Float64) raises -> DMat:
        """`ew(op, A, B, s=s)` (lane py-runtime-b)."""
        var bm = _mode(B.r, B.c, A.r, A.c)
        var out = DMat(A.r, A.c)
        if A.n() == 0:
            return out^
        launch_ew(self.ctx, op, A.p(), B.p(), bm, self.one.p(), 3, out.p(), A.n(), A.c, Float32(s))
        return out^

    def ew3(self, op: Int, A: DMat, B: DMat, C: DMat, s: Float64) raises -> DMat:
        """`ew(op, A, B, C, s=s)` (lane py-runtime-b)."""
        var bm = _mode(B.r, B.c, A.r, A.c)
        var cm = _mode(C.r, C.c, A.r, A.c)
        var out = DMat(A.r, A.c)
        if A.n() == 0:
            return out^
        launch_ew(self.ctx, op, A.p(), B.p(), bm, C.p(), cm, out.p(), A.n(), A.c, Float32(s))
        return out^

    def reduce(self, A: DMat, op: Int) raises -> DMat:
        """`_Kit.reduce` on the device (x_decomp/select_dev.mojo)."""
        if A.n() == 0:
            raise Error("x_decomp: reduce of an empty matrix")
        var out = DMat(1, 1)
        enqueue_sel_reduce(A.p(), A.n(), op, out.p())
        return out^

    def place_row(self, D: DMat, V: DMat, row: Int) raises:
        """`_Kit.place_rows(D, V, row)`: V's words into D from row `row`."""
        if V.n() == 0:
            return
        var view = DMat(rows_of=D, row0=row, rows=V.n() // D.c)
        self.ctx.enqueue_copy(dst_buf=self._sub(view), src_buf=self._sub(V))

    def rows(self, D: DMat, a: Int, b: Int) raises -> DMat:
        """`_M.rows(a, b)` as an exact device copy."""
        var view = DMat(rows_of=D, row0=a, rows=b - a)
        return self.copy(view)

    def svd_host(mut self, A: DMat, mut s: Mat, mut v: Mat) raises:
        """`_Kit.svd`'s solve of a resident tall A (m >= n): DevExec's
        launches (`_svd_on`, the resident entry's sequence) on a device copy,
        s (1 x n) and v (n x n, vectors in columns) home, unordered."""
        var m = A.r
        var n = A.c
        if n <= 0 or m < n:
            raise Error("x_decomp: svd needs m >= n >= 1 (a tall matrix)")
        s = Mat(1, n)
        v = Mat(n, n)
        var da = self.ctx.enqueue_create_buffer[DType.float32](m * n)
        self.ctx.enqueue_copy(dst_buf=da, src_buf=self._sub(A))
        DevExec._svd_on(self.ctx, da, m, n, s.p(), v.p(), QRB_CELLS)
        _ = da^
        self.sync()

    def sqdist(self, A: DMat, B: DMat) raises -> DMat:
        """`_Kit.sqdist(A, B)` on the device (`launch_sqdist`)."""
        var out = DMat(A.r, B.r)
        if A.r * B.r > 0 and A.c > 0:
            launch_sqdist(self.ctx, A.p(), B.p(), out.p(), A.r, B.r, A.c, 0, Float32(2))
        return out^

    def trisolve(self, lu: DMat, idx: DMat, B: DMat, trans: Int) raises -> DMat:
        """`_Kit.trisolve` on device matrices (`launch_trisolve`)."""
        var n = lu.r
        var w = B.c
        var out = DMat(n, w)
        if n * w > 0:
            if n >= 1 << 24:
                raise Error("x_decomp: trisolve row numbers exceed float32's exact integers")
            var tmp = DMat(n * w, 1)
            launch_trisolve(self.ctx, lu.p(), idx.p(), B.p(), out.p(), tmp.p(), n, w, trans)
            self.ctx.synchronize()
        return out^

    def pad_zero_row(self, X: DMat) raises -> DMat:
        """`_Kit.pad_zero_row`: [X; 0] (copies only)."""
        var out = self.zeros(X.r + 1, X.c)
        if X.n() > 0:
            var view = DMat(rows_of=out, row0=0, rows=X.r)
            self.ctx.enqueue_copy(dst_buf=self._sub(view), src_buf=self._sub(X))
        return out^

    def const(mut self, v: Float64, r: Int, c: Int) raises -> DMat:
        return self.upload(mat_const(v, r, c))

    def cols(mut self, A: DMat, a: Int, b: Int) raises -> DMat:
        """`_M.cols(a, b)` on the device (the TAKE_COLS move)."""
        var w = b - a
        var out = DMat(A.r, w)
        if A.r * w == 0:
            return out^
        var ix = Mat(1, w)
        for j in range(w):  # small-loop(w: selected columns): the move's exact float column numbers
            ix.d[j] = Float32(a + j)
        var idx = self.upload(ix)
        launch_move(self.ctx, MOVE_TAKE_COLS, A.p(), idx.p(), out.p(), A.r * w, w, A.c, 0, 0, 0)
        self.sync()
        return out^

    def take_col(mut self, A: DMat, j: Int) raises -> DMat:
        return self.cols(A, j, j + 1)

    def neg_signs(mut self, V: DMat) raises -> DMat:
        var m1 = self.const(-1.0, 1, 1)
        var p1 = self.const(1.0, 1, 1)
        return self.ew3(OP_SELECT, self.ew1(OP_SCALE, V, -1.0), m1, p1, 0.0)

    def absmax_signs(mut self, A: DMat, by_col: Bool) raises -> DMat:
        var cnt = A.c if by_col else A.r
        var out = DMat(1, cnt)
        if cnt > 0 and A.n() > 0:
            var ns = absmax_scratch(A.r, A.c, by_col)
            var sc = DMat(max(ns, 1), 1)
            launch_absmax(self.ctx, A.p(), out.p(), sc.p(), A.r, A.c, by_col)
            self.ctx.synchronize()
        if not by_col:
            out = self.vec_t(out^)
        return self.neg_signs(out)

    def count_gt(mut self, A: DMat, s: Float64) raises -> Int:
        if A.n() == 0:
            return 0
        return Int(self.word(self.total(self.ew1(OP_GTS, A, s))))

    def svd(mut self, A: DMat, mut S: DMat, mut Vt: DMat) raises:
        """`_Kit.svd` of a resident tall A: DevExec's solve, the descending
        order of the values home (`svd_order`), S and Vt back up."""
        var s = Mat(0, 0)
        var v = Mat(0, 0)
        self.svd_host(A, s, v)
        var sh = Mat(0, 0)
        var vh = Mat(0, 0)
        svd_order(s, v, sh, vh)
        S = self.upload(sh)
        Vt = self.upload(vh)

    def lda_bound(self, X: DMat, ddt: DMat, dcomp: DMat, floor: Float64) raises -> DMat:
        """The `_approx_bound` term matrix (`lda_bound_kernel`)."""
        var P = DMat(X.r, X.c)
        if X.n() > 0:
            launch_lda_bound(self.ctx, X.p(), ddt.p(), dcomp.p(), P.p(), X.r, ddt.c, X.c, Float32(floor))
        return P^

    def diag(mut self, A: DMat) raises -> DMat:
        """`_Kit.diag` (the strided TAKE_COLS move)."""
        var n = A.r
        var out = DMat(1, n)
        if n > 0:
            var idx = self.const(0.0, 1, 1)
            launch_move(self.ctx, MOVE_TAKE_COLS, A.p(), idx.p(), out.p(), n, 1, n + 1, 0, 0, 0)
            self.sync()
        return out^

    def fill0(self, A: DMat, start: Int, stride: Int, count: Int) raises:
        """`_Kit.fill0` (the FILL0 move), in place."""
        if count > 0:
            launch_move(self.ctx, MOVE_FILL0, A.p(), self.one.p(), A.p(), count, start, stride, 0, 0, 0)

    def _code_rows(self, kind: Int, G: DMat, Q: DMat, W: DMat, a: Int, b: Int, alpha: Float64, tol: Float64) raises:
        """`dev_code_rows_py`'s launches (DevExec's): kind 0 lasso (W the
        warm start), 1 lars, 2 omp (W zeroed first)."""
        var n = Q.r
        var k = Q.c
        if n == 0 or k == 0:
            return
        if n * (k * k + LARS_ROW_EXTRA * k) > 2147483647:
            raise Error("x_decomp: code rows exceed the Int32 index bound")
        var per = n * k
        if kind == 1:
            per = n * (k * k + LARS_ROW_EXTRA * k)
        elif kind == 2:
            per = n * (k * k + 3 * k)
        var dn = self.ctx.enqueue_create_buffer[DType.float32](n)
        var ds = self.ctx.enqueue_create_buffer[DType.float32](per)
        if C28_BUCKET_SOLVES:
            if kind != 0:
                var wsub = self._sub(W)
                enqueue_fill(self.ctx, wsub, Float32(0.0))
            launch_classical_code_rows(self.ctx, G.p(), Q.p(), W.p(), rebind[F32Ptr](ds.unsafe_ptr()), rebind[F32Ptr](dn.unsafe_ptr()),
                                       n, k, kind, a, b, Float32(alpha), Float32(tol))
        elif kind == 0:
            self.ctx.enqueue_function[lasso_rows_kernel](
                G.p(), Q.p(), W.p(), ds.unsafe_ptr(), dn.unsafe_ptr(), Int32(n), Int32(k), Float32(alpha), Int32(a),
                Float32(tol), Int32(1 if b != 0 else 0), grid_dim=_blocks(n), block_dim=TPB,
            )
        else:
            var wsub = self._sub(W)
            enqueue_fill(self.ctx, wsub, Float32(0.0))
            if kind == 1:
                self.ctx.enqueue_function[lars_rows_kernel](
                    G.p(), Q.p(), W.p(), ds.unsafe_ptr(), dn.unsafe_ptr(), Int32(n), Int32(k), Int32(a), Int32(b),
                    grid_dim=_blocks(n), block_dim=TPB,
                )
            else:
                self.ctx.enqueue_function[omp_rows_kernel](
                    G.p(), Q.p(), W.p(), ds.unsafe_ptr(), dn.unsafe_ptr(), Int32(n), Int32(k), Int32(a),
                    grid_dim=_blocks(n), block_dim=TPB,
                )
        self.ctx.synchronize()
        _ = dn^
        _ = ds^

    def lasso_rows(self, G: DMat, Q: DMat, mut W: DMat, alpha: Float64, max_iter: Int, tol: Float64,
                   positive: Bool) raises:
        self._code_rows(0, G, Q, W, max_iter, 1 if positive else 0, alpha, tol)

    def lars_rows(self, G: DMat, Q: DMat, m: Int, nnz: Int) raises -> DMat:
        var W = DMat(Q.r, Q.c)
        self._code_rows(1, G, Q, W, m, nnz, 0.0, 0.0)
        return W^

    def omp_rows(self, G: DMat, Q: DMat, nnz: Int) raises -> DMat:
        var W = DMat(Q.r, Q.c)
        self._code_rows(2, G, Q, W, nnz, 0, 0.0, 0.0)
        return W^

    def dict_fused(self, D: DMat, A: DMat, B: DMat, mut Dn: DMat) raises -> Bool:
        """`_Kit._dict_dev`'s fused atom update (FAST builds), else False."""
        comptime if DECOMP_FAST_DICT_DEV:
            if D.n() == 0:
                return False
            Dn = DMat(D.r, D.c)
            _ = dict_update_dev(D.id, A.id, B.id, Dn.id, D.r, D.c)
            return True
        else:
            return False

    def word(mut self, A: DMat) raises -> Float64:
        """`A.s[0]` as Python reads it: one word home (a sync)."""
        var h = self.get(A)
        self.sync()
        return Float64(h.d[0])

    def t(self, A: DMat) raises -> DMat:
        """`_M.T`: a vector's same words, else the TRANSPOSE move on the device."""
        var out = DMat(A.c, A.r)
        if A.n() == 0:
            return out^
        if A.r == 1 or A.c == 1:
            self.ctx.enqueue_copy(dst_buf=self._sub(out), src_buf=self._sub(A))
            return out^
        launch_move(self.ctx, MOVE_TRANSPOSE, A.p(), A.p(), out.p(), A.n(), A.r, A.c, 0, 0, 0)
        return out^

    def vec_t(self, var A: DMat) -> DMat:
        """A vector's transpose: the same buffer, the dimensions swapped."""
        var r = A.r
        A.r = A.c
        A.c = r
        return A^

    def order_small(mut self, A: DMat) raises -> List[Int32]:
        """`_Kit.order_small` read as ints (the kernel's exact floats, home)."""
        var n = A.n()
        if n > SEL_ORDER_MAX:
            raise Error("x_decomp: order_small exceeds its bound")
        var out = DMat(n, 1)
        if n > 0:
            self.ctx.enqueue_function[order_small_kernel](A.p(), out.p(), Int32(n), grid_dim=_blocks(n), block_dim=TPB)
        var h = self.get(out)
        self.sync()
        var res = List[Int32](length=max(n, 1), fill=Int32(0))
        for i in range(n):  # small-loop(n: components): the coordinate order's int32 words for cd_rows
            res[i] = Int32(Int(h.d[i]))
        return res^

    def cd_rows(mut self, W: DMat, HHt: DMat, XHt: DMat, perm: List[Int32]) raises -> Float64:
        """`_Kit.cd_rows` on device matrices (`cd_rows_kernel`): W swept in
        place, the violation folded by `total` and read."""
        var n = W.r
        var kc = W.c
        var viol = DMat(n, 1)
        if n * kc > 0:
            var pd = DMat(kc, 1)
            var words = List[Float32](length=max(kc, 1), fill=Float32(0))
            for j in range(kc):  # small-loop(kc: components): the permutation's int32 bits as upload words
                words[j] = bitcast[DType.float32](perm[j])
            self.upload_into(pd, words^)
            self.ctx.enqueue_function[cd_rows_kernel](
                W.p(), HHt.p(), XHt.p(), pd.p().bitcast[Int32](), viol.p(), Int32(n), Int32(kc),
                grid_dim=_blocks(n), block_dim=TPB,
            )
            var tv = self.total(viol)
            var w = self.word(tv)
            _ = pd^
            return w
        return 0.0

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

    def mm_ordered(self, A: DMat, B: DMat, ta: Bool, tb: Bool) raises -> DMat:
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
        launch_gemm_ordered(self.ctx, A.p(), B.p(), out.p(), _ptr(sid, max(ns, 1)), m, k, n, ta, tb)
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
        # DevExec.eigh's round-robin solve on a device copy of A (one sync a
        # sweep inside the solve); cgr-decomp: the one-block jacobi2 route
        # this took is deleted, so the resident kit and DevExec agree
        var da = self.ctx.enqueue_create_buffer[DType.float32](max(n * n, 1))
        if n > 0:
            self.ctx.enqueue_copy(dst_buf=da.create_sub_buffer[DType.float32](0, n * n), src_buf=self._sub(A))
        _ = DevExec._eigh_par_on(self.ctx, da, wh.p(), vh.p(), n)
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
        var sid = pool_alloc(LU_SCAL_LEN)
        var aid = pool_alloc(max(n, 1))
        launch_lu(
            self.ctx, lu.p(), _ptr(pid, max(n, 1)).bitcast[Int32](), _ptr(iid, 1), _ptr(sid, LU_SCAL_LEN),
            _ptr(aid, max(n, 1)), n,
        )
        var diag = DMat(1, n)
        if n > 0:
            self.ctx.enqueue_function[diag_kernel](lu.p(), diag.p(), Int32(n), grid_dim=_blocks(n), block_dim=TPB)
        var t = self.total(self.ew1(OP_LOGS, self.ew1(OP_ABS, diag, 0.0), _FLT_MIN))
        var th = self.get(t)
        # the sign rule on the device (lane cpu3-core): two words come home
        var sg = DMat(1, 2)
        var pool = X_DECOMP_POOL.get_or_create_ptr()
        var sgv = pool[].bufs[sg.id].create_sub_buffer[DType.float32](0, 2)
        enqueue_fill(self.ctx, sgv, Float32(0))
        if n > 0:
            self.ctx.enqueue_function[logdet_sign_kernel](
                diag.p(), _ptr(pid, max(n, 1)).bitcast[Int32](), Int32(n),
                sg.p().bitcast[Int32](), grid_dim=_blocks(n), block_dim=TPB,
            )
        var sh = List[Int32](length=2, fill=Int32(0))
        self.ctx.enqueue_copy(
            dst_ptr=F32Ptr(unsafe_from_address=Int(sh.unsafe_ptr())),
            src_buf=pool[].bufs[sg.id].create_sub_buffer[DType.float32](0, 2),
        )
        self.sync()
        pool_free(pid)
        pool_free(iid)
        pool_free(sid)
        pool_free(aid)
        _ = lu^
        _ = sgv^
        _ = sg^
        if sh[0] != Int32(0):
            return _neg_inf()                 # sign 0 -> -inf
        if Int(sh[1]) % 2 != 0:
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

    def emp_cov_at(self, Xs: DMat, loc: DMat) raises -> DMat:
        # C57 retains this candidate's just-computed immutable subset mean.
        # No reuse across support changes; the centered products are unchanged.
        # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
        comptime if C23_MCD:
            return self.k.ew1(OP_SCALE, self.k.classical_centered_gram(Xs, loc), 1.0 / Float64(Xs.r))
        comptime if C57_CANDIDATE_STATE:
            var Xc = self.k.ew2(OP_SUB, Xs, loc)
            return self.k.ew1(OP_SCALE, self.k.mm(Xc, Xc, True, False), 1.0 / Float64(Xs.r))
        return self.emp_cov(Xs)

    def mahal(self, X: DMat, loc: DMat, P: DMat) raises -> DMat:
        var Xc = self.k.ew2(OP_SUB, X, loc)
        return self.k.rowsum(self.k.ew2(OP_MUL, self.k.mm(Xc, P, False, False), Xc))

    def pinvh(mut self, A: DMat) raises -> DMat:
        """`_pinvh`: V diag(1/w) V^T over |w| > max|w| * n * float32 eps."""
        var n = A.r
        var e = self.k.eigh(A)
        var wmax: Float64 = 0.0
        if n > 0:
            # w is ascending (the solve's own permutation), so max|w| is at
            # an end: no host walk over the n eigenvalues (lane cpu3-core)
            wmax = abs(Float64(e.wh.d[0]))
            var a_last = abs(Float64(e.wh.d[n - 1]))
            if a_last > wmax:                 # Python max(): replace on >
                wmax = a_last
        var cut = Float32((wmax * Float64(n)) * _F32_EPS)
        var keep = self.k.ew1(OP_RECIP, e.wd, 0.0)
        var inv = DMat(1, n)
        if n > 0:
            self.k.ctx.enqueue_function[pinv_mask_kernel](
                keep.p(), e.wd.p(), inv.p(), Int32(n), cut, grid_dim=_blocks(n), block_dim=TPB
            )
        return self.k.mm(self.k.ew2(OP_MUL, e.vd, inv), e.vd, False, True)

    def c_step(
        mut self, X: DMat, h: Int, n_iter: Int, has_init: Bool, loc0: DMat, cov0: DMat, want_dist: Bool
    ) raises -> DEst:
        """x_decomp/mcd.mojo's `c_step` with every selection, gather and
        distance resident (lane cpu3-core): the host reads only the log
        determinants the loop branches on (and the NaN refusal words)."""
        var iters = n_iter
        var dist = DMat(0, 1)
        var sel: DMat
        if not has_init:
            # `sorted(self._perm(n)[:h])`: the next Philox stream's draws
            self.draws += 1
            var r = self.k.rand(1, X.r, self.seed, 1000 + self.draws, 0)
            sel = self.k.smallest_dev(r, h)
        else:
            var P0 = self.pinvh(cov0)
            var d0 = self.mahal(X, loc0, P0)
            sel = self.k.smallest_dev(d0, h)
            dist = d0^
        var Xs = self.k.gather_dev(X, sel)
        var loc = self.k.colmean(Xs)
        var cov = self.emp_cov_at(Xs, loc)
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
        var prev_sel = DMat(0, 1)
        while det < prev_det and iters > 0 and det != _neg_inf():
            P = self.pinvh(cov)
            has_p = True
            var dd = self.mahal(X, loc, P)
            prev_loc = loc^
            prev_cov = cov^
            prev_sel = sel^
            have_prev = True
            prev_det = det
            sel = self.k.smallest_dev(dd, h)
            dist = dd^
            Xs = self.k.gather_dev(X, sel)
            loc = self.k.colmean(Xs)
            cov = self.emp_cov_at(Xs, loc)
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
            return DEst(prev_loc^, prev_cov^, prev_det, prev_sel^, dist^)
        var final = DMat(0, 1)
        if want_dist:
            final = self.mahal(X, loc, P)
        return DEst(loc^, cov^, det, sel^, final^)

    def select_random(
        mut self, X: DMat, h: Int, trials: Int, keep: Int, n_iter: Int, mut output: DCands
    ) raises:
        """`select_candidates` from random subsets: every trial's location
        and covariance packed into one trial arena, then the `keep` best (in
        `_order_by_det`'s order) appended to `output`. The later stages read
        only a kept candidate's location and covariance."""
        var tr = DCands(trials, X.c)
        var none = DMat(0, 0)
        for _t in range(trials):  # small-loop(trials: C-step candidates): one enqueued C-step a candidate, the plan's trial count
            var e = self.c_step(X, h, n_iter, False, none, none, False)
            tr.push(self.k, e.loc, e.cov, e.det)
        var top = _top_of(tr.det, keep)
        for a in range(len(top)):  # small-loop(top: kept candidates): two device copies a kept candidate
            var j = top[a]
            output.push(self.k, tr.loc(j), tr.cov(j), tr.det[j])

    def select_init(
        mut self, X: DMat, h: Int, inits: DCands, keep: Int, n_iter: Int, mut output: DCands
    ) raises:
        """`select_candidates` from initial estimates, the `keep` best
        locations and covariances appended to `output` (packed, as above)."""
        var tr = DCands(inits.count(), X.c)
        for t in range(inits.count()):  # small-loop(inits: C-step candidates): one enqueued C-step per kept candidate
            var e = self.c_step(X, h, n_iter, True, inits.loc(t), inits.cov(t), False)
            tr.push(self.k, e.loc, e.cov, e.det)
        var top = _top_of(tr.det, keep)
        for a in range(len(top)):  # small-loop(top: kept candidates): two device copies a kept candidate
            var j = top[a]
            output.push(self.k, tr.loc(j), tr.cov(j), tr.det[j])

    def select_init_best(
        mut self, X: DMat, h: Int, inits: DCands, n_iter: Int, want_dist: Bool
    ) raises -> DEst:
        """`select_candidates(...)[0]` from initial estimates: the best
        (det, j) candidate kept whole (support rows and distances), every
        other one released as soon as it loses. A NaN determinant is
        refused after every C-step ran, as `_order_by_det` refused it."""
        var best = Optional[DEst]()
        var best_key = UInt64(0)
        var nan_seen = False
        for t in range(inits.count()):  # small-loop(inits: C-step candidates): one enqueued C-step per kept candidate
            var e = self.c_step(X, h, n_iter, True, inits.loc(t), inits.cov(t), want_dist)
            var f = Float32(e.det)
            if f != f:
                nan_seen = True
                continue
            var key = _key(f, t)
            if not best or key < best_key:
                best_key = key
                best = Optional(e^)
        if nan_seen:
            raise Error("x_decomp MinCovDet: a NaN distance or draw has no order (refused)")
        if not best:
            raise Error("x_decomp MinCovDet: no candidate to select from")
        return best.take()


struct DCands(Movable):
    """Candidate locations (1 x d) and covariances (d x d) packed in two
    pooled buffers, with their log determinants (lane idn-regress). The
    search used to keep every candidate as its own four device buffers
    (location, covariance, support rows, distances): 333 subsets x 10 kept
    at 100k rows is 13k+ live Metal allocations, and every Metal launch pays
    ~0.25 us per live allocation (MTLResourceList residency), so each
    C-step launch cost milliseconds. Views of one allocation cost nothing.
    The values are copied word for word: no bit changes."""
    var L: DMat
    var C: DMat
    var det: List[Float64]
    var d: Int

    def __init__(out self, cap: Int, d: Int) raises:
        self.L = DMat(max(cap, 1), d)
        self.C = DMat(max(cap, 1) * d, d)
        self.det = List[Float64]()
        self.d = d

    def count(self) -> Int:
        return len(self.det)

    def loc(self, j: Int) raises -> DMat:
        return DMat(rows_of=self.L, row0=j, rows=1)

    def cov(self, j: Int) raises -> DMat:
        return DMat(rows_of=self.C, row0=j * self.d, rows=self.d)

    def push(mut self, k: DKit, loc: DMat, cov: DMat, det: Float64) raises:
        var j = len(self.det)
        if j >= self.L.r:
            raise Error("x_decomp MinCovDet: candidate arena full")
        k.put(self.loc(j), loc)
        k.put(self.cov(j), cov)
        self.det.append(det)


struct DEst(Movable):
    """`Est` with every matrix resident: location, covariance, the log
    determinant, the support rows (an index vector) and the distances."""
    var loc: DMat
    var cov: DMat
    var det: Float64
    var sel: DMat
    var dist: DMat

    def __init__(out self, var loc: DMat, var cov: DMat, det: Float64, var sel: DMat, var dist: DMat):
        self.loc = loc^
        self.cov = cov^
        self.det = det
        self.sel = sel^
        self.dist = dist^


def _top_of(dets: List[Float64], keep: Int) raises -> List[Int]:
    """`_order_by_det`'s order: `sorted(range(len(dets)), key=(det, j))[:keep]`."""
    var keys = List[UInt64](capacity=len(dets))
    for j in range(len(dets)):  # small-loop(dets: C-step candidates): one log determinant per candidate
        keys.append(_key(Float32(dets[j]), j))
    sort(keys)
    var out = List[Int]()
    for a in range(min(keep, len(keys))):  # small-loop(keep: kept candidates): the kept candidates' slots
        out.append(Int(keys[a] & UInt64(0xFFFFFFFF)))
    return out^


def _write_dev(
    mut k: DKit, loc: DMat, cov: DMat, sup: DMat, dist: DMat, n: Int, d: Int,
    loc_out: F32Ptr, cov_out: F32Ptr, sup_out: I32Ptr, dist_out: F32Ptr,
) raises:
    """`_write` from the resident answer: four copies, one wait."""
    if d > 0:
        k.ctx.enqueue_copy(dst_ptr=loc_out, src_buf=k._sub(loc))
        k.ctx.enqueue_copy(dst_ptr=cov_out, src_buf=k._sub(cov))
    if n > 0:
        k.ctx.enqueue_copy(dst_ptr=F32Ptr(unsafe_from_address=Int(sup_out)), src_buf=k._sub(sup))
        k.ctx.enqueue_copy(dst_ptr=dist_out, src_buf=k._sub(dist))
    k.sync()


def fast_mcd_dev(
    X: Mat, p: List[Int], loc_out: F32Ptr, cov_out: F32Ptr, sup_out: I32Ptr, dist_out: F32Ptr,
) raises:
    """`fast_mcd` (x_decomp/mcd.mojo) with X resident: the same plan, the
    same C-steps, the same draws and orders. Every permutation, selection,
    support and distance stays on the device (lane cpu3-core)."""
    # FAST on Apple, under MCD_BATCH_COMPAT (the old -D MOJOLEARN_MCD_DEVICE_CSTEPS
    # arm was DROPPED-quality, see x_decomp/mcd_fast.mojo): every candidate's
    # C-steps together on the device (lane/apple-fast-robust; M3 A/B min-cov-det
    # taxi 79,925 -> 215 ms, M2 robust-ee-taxi-x 64,578 -> 267.5 ms)
    comptime if MCD_DEVICE_CSTEPS:
        if fast_mcd_fast(X, p, loc_out, cov_out, sup_out, dist_out):
            return
    var n = p[0]
    var d = p[1]
    var h = p[2]
    var run = DMcd(p[3])
    var Xd = run.k.upload(X)
    var pool = X_DECOMP_POOL.get_or_create_ptr()
    var sup = DMat(n, 1)
    var dist = DMat(n, 1)
    var supv = pool[].bufs[sup.id].create_sub_buffer[DType.float32](0, max(n, 1))
    var distv = pool[].bufs[dist.id].create_sub_buffer[DType.float32](0, max(n, 1))
    enqueue_fill(run.k.ctx, supv, Float32(0))
    enqueue_fill(run.k.ctx, distv, Float32(0))
    _ = supv^
    _ = distv^
    var best: DEst
    if n > 500:
        var n_sub = p[4]
        var n_ss = p[5]
        var h_sub = p[6]
        var n_trials = p[7]
        var n_m = p[8]
        var h_m = p[9]
        var n_best_m = p[10]
        var shuf = run.k.argsort_dev(run.k.rand(1, n, run.seed, 1000 + run.draws + 1, 0))
        run.draws += 1
        var cands = DCands(n_sub * min(10, n_trials), d)
        for i in range(n_sub):  # small-loop(n_sub: row subsets): one gather and its C-steps a subset, the plan's subset count
            var rows = DMat(rows_of=shuf, row0=i * n_ss, rows=n_ss)
            var cur = run.k.gather_dev(Xd, rows)
            run.select_random(cur, h_sub, n_trials, 10, 2, cands)
        var selection_all = run.k.argsort_dev(run.k.rand(1, n, run.seed, 1000 + run.draws + 1, 0))
        run.draws += 1
        var selection = DMat(rows_of=selection_all, row0=0, rows=n_m)
        var Xm = run.k.gather_dev(Xd, selection)
        if n < 1500:
            # merged[0]: the best merged candidate, kept whole
            var m0 = run.select_init_best(Xm, h_m, cands, 30, True)
            var sp = sup.p().bitcast[Int32]()
            var selp = selection.p().bitcast[Int32]()
            if n_m > 0:
                run.k.ctx.enqueue_function[scatter_rows_kernel](
                    dist.p(), selp, m0.dist.p(), Int32(n_m), grid_dim=_blocks(n_m), block_dim=TPB,
                )
            var ms = m0.sel.n()
            if ms > 0:
                run.k.ctx.enqueue_function[mark_rows_of_kernel](
                    sp, selp, m0.sel.p().bitcast[Int32](), Int32(ms),
                    grid_dim=_blocks(ms), block_dim=TPB,
                )
            _write_dev(run.k, m0.loc, m0.cov, sup, dist, n, d, loc_out, cov_out, sup_out, dist_out)
            _ = Xd^
            return
        var merged = DCands(n_best_m, d)
        run.select_init(Xm, h_m, cands, n_best_m, 30, merged)
        _ = cands^
        best = run.select_init_best(Xd, h, merged, 30, True)
    else:
        var first = DCands(10, d)
        run.select_random(Xd, h, 30, 10, 2, first)
        best = run.select_init_best(Xd, h, first, 30, True)
    var bs = best.sel.n()
    if bs > 0:
        run.k.ctx.enqueue_function[mark_rows_kernel](
            sup.p().bitcast[Int32](), best.sel.p().bitcast[Int32](), Int32(bs),
            grid_dim=_blocks(bs), block_dim=TPB,
        )
    _write_dev(run.k, best.loc, best.cov, sup, best.dist, n, d, loc_out, cov_out, sup_out, dist_out)
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
        for j in range(nb):  # small-loop(nb: the pass's mini-batches, n / batch_size): one launch-group scalar per mini-batch, uploaded once
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
    p: PythonObject,
) raises -> PythonObject:
    """`mcd_py` on the resident kit."""
    var q = List[Int]()
    for i in range(11):
        q.append(Int(py=p[i]))
    var n = q[0]
    var d = q[1]
    if n < 1 or d < 2 or n * d > 2147483647 or q[2] < 1 or q[2] > n:
        raise Error("x_decomp: mcd needs n >= 1, d >= 2 and 1 <= h <= n")
    if n > 500 and (q[4] < 1 or q[4] * q[5] > n or q[8] > n or q[8] < 1 or q[10] < 1):
        raise Error("x_decomp: mcd subset plan out of range")
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
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`lda_online_py` on the resident kit."""
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
