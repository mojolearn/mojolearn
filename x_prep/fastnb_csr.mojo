# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""MultinomialNB / ComplementNB on a CSR count matrix (lane apple-fast-nb,
pass 2, 2026-10-02). FAST + Apple ONLY, ON by default (-D
MOJOLEARN_NB_TEXT_CSR_OFF reverts; M2 A/B nb-mnb-csr-text-x: multinomial-nb
text 221.7 -> 47.7 ms, accuracy .9831 logloss .5595 identical): the
IDENTICAL binding, the other vendors and a FAST build with the _OFF define do
not export `x_prep_nb_csr_fit` / `x_prep_nb_csr_jll`, and the Python layer
(`_expansion_prep._DiscreteNB`) keeps main's dense program.

Main's text fit stages the dense 78k x 4096 count matrix (1.3 GB) in the
host arena and uploads it, then folds it three times (`colb_part`,
`csb_part` x K). Here the CSR arrays (indptr, indices, data) go up ONCE from
their own buffers, no densification and no per-row Python staging:

* `nb_csr_count_kernel`: one block per CSR_ROWS rows, the threads striding
  the tile's nonzeros (coalesced indices and data), each nonzero's row found
  by a binary search over the tile's indptr slice in threadgroup memory,
  then a float32 atomic add into the (class, feature) table resident on the
  device; the class counts take one atomic add of 1 per row (exact below
  2^24 rows); a negative value raises a flag word (the caller refuses the
  input, as main's column-minimum check does).
* `nb_csr_jll_kernel` (predict): one thread per (row, class) accumulating
  data_j * feature_log_prob_[k, col_j] over the row's nonzeros ascending,
  plus the class log prior when the caller passes one (MultinomialNB; the
  single-class ComplementNB), `matmul_unit`'s statement.

The (class, feature) table and the class counts come back to the host
(K x d + K floats) and main's epilogue (`mnb_params` / `cnb_params`,
`class_log_prior`) runs on them unchanged, so feature_log_prob_ matches the
dense fit within float32 (the atomic fold order is new).
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_prep.common import FP, IP
from x_prep.prims import add, mul
from x_prep.device import x_prep_ctx

#: The switch: FAST and Apple, unless -D MOJOLEARN_NB_TEXT_CSR_OFF (default ON
#: since M2 A/B nb-mnb-csr-text-x, 221.7 -> 47.7 ms, same accuracy / logloss).
comptime NB_TEXT_CSR = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and not is_defined["MOJOLEARN_NB_TEXT_CSR_OFF"]()
)
#: Rows per block of the count kernel (its indptr slice in threadgroup memory).
comptime CSR_ROWS = 64
comptime CSR_TPB = 256


def nb_csr_count_kernel(
    indptr: IP, indices: IP, data: FP, y: IP, tab: FP, cnt: FP, flag: IP, n: Int32, d: Int32
):
    """Rows [b*CSR_ROWS, +CSR_ROWS) of block b: tab[y[i] * d + col] += val
    for every nonzero (atomic), cnt[y[i]] += 1 per row, flag[0] = 1 when a
    value is negative. A column outside [0, d) adds nothing."""
    var r0 = Int(block_idx.x) * CSR_ROWS
    var tid = Int(thread_idx.x)
    var nn = Int(n)
    var rows = min(nn, r0 + CSR_ROWS) - r0
    var sp = stack_allocation[CSR_ROWS + 1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var sy = stack_allocation[CSR_ROWS, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    if rows <= 0:
        return
    if tid <= rows:
        sp[tid] = indptr.unsafe_load(r0 + tid)
    if tid < rows:
        var k = y.unsafe_load(r0 + tid)
        sy[tid] = k
        _ = Atomic.fetch_add(cnt.unsafe_offset(Int(k)), Float32(1))
    barrier()
    var lo = Int(sp[0])
    var hi = Int(sp[rows])
    var dd = Int(d)
    var neg = False
    var j = lo + tid
    while j < hi:
        # the row of nonzero j: the last r in [0, rows) with sp[r] <= j
        var a = 0
        var b = rows
        while b - a > 1:
            var m = (a + b) // 2
            if Int(sp[m]) <= j:
                a = m
            else:
                b = m
        var c = Int(indices.unsafe_load(j))
        var v = data.unsafe_load(j)
        if v < Float32(0):
            neg = True
        if c >= 0 and c < dd:
            _ = Atomic.fetch_add(tab.unsafe_offset(Int(sy[a]) * dd + c), v)
        j += CSR_TPB
    if neg:
        _ = Atomic.max(flag.unsafe_offset(0), Int32(1))


def nb_csr_jll_kernel(
    indptr: IP, indices: IP, data: FP, flp: FP, clp: FP, dst: FP, n: Int32, d: Int32, K: Int32, has_clp: Int32
):
    """dst[i*K + k] = (clp[k] +) sum_j data[j] * flp[k*d + indices[j]] over
    row i's nonzeros ascending, each product rounded before its add."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var kk = Int(K)
    if t >= Int(n) * kk:
        return
    var i = t // kk
    var k = t - i * kk
    var dd = Int(d)
    var base = k * dd
    var acc = Float32(0)
    for j in range(Int(indptr.unsafe_load(i)), Int(indptr.unsafe_load(i + 1))):
        var c = Int(indices.unsafe_load(j))
        if c >= 0 and c < dd:
            acc = add(acc, mul(data.unsafe_load(j), flp.unsafe_load(base + c)))
    if has_clp != 0:
        acc = add(acc, clp.unsafe_load(k))
    dst.unsafe_store(t, acc)


def _up_i32(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.int32]:
    var buf = ctx.enqueue_create_buffer[DType.int32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=IP(unsafe_from_address=addr))
    return buf^


def _up_f32(ctx: DeviceContext, addr: Int, n: Int) raises -> DeviceBuffer[DType.float32]:
    var buf = ctx.enqueue_create_buffer[DType.float32](n if n > 0 else 1)
    if n > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=FP(unsafe_from_address=addr))
    return buf^


def _nb_csr_fit(
    pa: Int, ia: Int, da: Int, ya: Int, n: Int, d: Int, K: Int, nnz: Int, fa: Int, ca: Int, ga: Int
) raises:
    var ctx = x_prep_ctx()
    var dp = _up_i32(ctx, pa, n + 1)
    var di = _up_i32(ctx, ia, nnz)
    var dv = _up_f32(ctx, da, nnz)
    var dy = _up_i32(ctx, ya, n)
    var dt = ctx.enqueue_create_buffer[DType.float32](K * d)
    var dc = ctx.enqueue_create_buffer[DType.float32](K)
    var dg = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(dt, Float32(0))
    ctx.enqueue_memset(dc, Float32(0))
    ctx.enqueue_memset(dg, Int32(0))
    ctx.enqueue_function[nb_csr_count_kernel](
        dp.unsafe_ptr(), di.unsafe_ptr(), dv.unsafe_ptr(), dy.unsafe_ptr(), dt.unsafe_ptr(), dc.unsafe_ptr(),
        dg.unsafe_ptr(), Int32(n), Int32(d),
        grid_dim=(n + CSR_ROWS - 1) // CSR_ROWS, block_dim=CSR_TPB,
    )
    ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=fa), src_buf=dt)
    ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=ca), src_buf=dc)
    ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=ga), src_buf=dg)
    ctx.synchronize()
    _ = dp^
    _ = di^
    _ = dv^
    _ = dy^
    _ = dt^
    _ = dc^
    _ = dg^


def nb_csr_fit_py(
    indptr: PythonObject, indices: PythonObject, data: PythonObject, y: PythonObject, sizes: PythonObject,
    out_fc: PythonObject, out_cnt: PythonObject, out_flag: PythonObject,
) raises -> PythonObject:
    """`x_prep_nb_csr_fit`: host addresses of indptr (n + 1 int32), indices
    (nnz int32), data (nnz float32) and the class codes (n int32); sizes =
    [n, d, K, nnz]; out_fc (K * d float32), out_cnt (K float32) and out_flag
    (1 int32: 1 when a value is negative) are written. Returns 1."""
    var n = Int(py=sizes[0])
    var d = Int(py=sizes[1])
    var K = Int(py=sizes[2])
    var nnz = Int(py=sizes[3])
    var pa = Int(py=indptr)
    var ia = Int(py=indices)
    var da = Int(py=data)
    var ya = Int(py=y)
    var fa = Int(py=out_fc)
    var ca = Int(py=out_cnt)
    var ga = Int(py=out_flag)
    if n < 1 or d < 1 or K < 1 or nnz < 0 or pa == 0 or ya == 0 or fa == 0 or ca == 0 or ga == 0:
        raise Error("x_prep: invalid CSR fit buffers")
    if nnz > 0 and (ia == 0 or da == 0):
        raise Error("x_prep: invalid CSR fit buffers")
    if K * d > 2147483647 or nnz > 2147483647:
        raise Error("x_prep: CSR fit exceeds the Int32 index bound")
    with GILReleased(Python()):
        _nb_csr_fit(pa, ia, da, ya, n, d, K, nnz, fa, ca, ga)
    return PythonObject(1)


def _nb_csr_jll(
    pa: Int, ia: Int, da: Int, wa: Int, ba: Int, n: Int, d: Int, K: Int, nnz: Int, oa: Int
) raises:
    var ctx = x_prep_ctx()
    var dp = _up_i32(ctx, pa, n + 1)
    var di = _up_i32(ctx, ia, nnz)
    var dv = _up_f32(ctx, da, nnz)
    var dw = _up_f32(ctx, wa, K * d)
    var db = _up_f32(ctx, ba, K if ba != 0 else 0)
    var do_ = ctx.enqueue_create_buffer[DType.float32](n * K)
    var cells = n * K
    ctx.enqueue_function[nb_csr_jll_kernel](
        dp.unsafe_ptr(), di.unsafe_ptr(), dv.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(), do_.unsafe_ptr(),
        Int32(n), Int32(d), Int32(K), Int32(1 if ba != 0 else 0),
        grid_dim=(cells + CSR_TPB - 1) // CSR_TPB, block_dim=CSR_TPB,
    )
    ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=oa), src_buf=do_)
    ctx.synchronize()
    _ = dp^
    _ = di^
    _ = dv^
    _ = dw^
    _ = db^
    _ = do_^


def nb_csr_jll_py(
    indptr: PythonObject, indices: PythonObject, data: PythonObject, flp: PythonObject, clp: PythonObject,
    sizes: PythonObject, dst: PythonObject,
) raises -> PythonObject:
    """`x_prep_nb_csr_jll`: the joint log likelihood (n x K float32 at dst)
    of a CSR matrix; flp = feature_log_prob_ (K x d), clp = class_log_prior_
    (K) or address 0 for none; sizes = [n, d, K, nnz]. Returns 1."""
    var n = Int(py=sizes[0])
    var d = Int(py=sizes[1])
    var K = Int(py=sizes[2])
    var nnz = Int(py=sizes[3])
    var pa = Int(py=indptr)
    var ia = Int(py=indices)
    var da = Int(py=data)
    var wa = Int(py=flp)
    var ba = Int(py=clp)
    var oa = Int(py=dst)
    if n < 1 or d < 1 or K < 1 or nnz < 0 or pa == 0 or wa == 0 or oa == 0:
        raise Error("x_prep: invalid CSR scoring buffers")
    if nnz > 0 and (ia == 0 or da == 0):
        raise Error("x_prep: invalid CSR scoring buffers")
    if n * K > 2147483647 or K * d > 2147483647 or nnz > 2147483647:
        raise Error("x_prep: CSR scoring exceeds the Int32 index bound")
    with GILReleased(Python()):
        _nb_csr_jll(pa, ia, da, wa, ba, n, d, K, nnz, oa)
    return PythonObject(1)
