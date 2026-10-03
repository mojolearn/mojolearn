# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OneClassSVM's SMO on the GPU over the grid (cpu-gpu-cleanup
c-xneighbors): libsvm's one-class solve, `ocsvm_smo_item` of
x_neighbors/items.mojo, with every per-sample scan and update spread over
ceil(n / OCSVM_TPB) blocks. The host column (x_neighbors/ocsvm_host.mojo)
runs the item.

One SMO iteration is three launches on the lane's stream:
  * sel_i: commits the previous step's alpha_i, alpha_j (each by the thread
    that owns the sample), then each block's maximum of (-g_t, t) over
    alpha_t < C_t into a partial;
  * sel_j: every block folds the sel_i partials to (gmax, gi), then its
    minimum of (obj_j, -j) (carrying g_j) and maximum of (g_j, j) over
    alpha_j > 0 into partials;
  * step: every block folds all partials (the same values in every block),
    takes the stopping test and the two-variable step (`ocsvm_pair`, pure,
    from the carried gradients), and updates its own gradient entries;
    thread 0 of block 0 records the step for the next sel_i.
The scans are maxima / minima of (value, index) under a total order with
the LAST index on an equal value and NaN never taken: libsvm's `>=` / `<=`
scans ascending. Any reduction order returns the item's index and value,
so the device's alpha, iteration count and gradients are the item's bits.
rho is the blocked fold `ocsvm_rho_part_item` / `ocsvm_rho_fin_item`, the
item's `ocsvm_rho`.

State (int32 `st`): it, stop, go, the pending (i, j); float32 `sf` the
pending (alpha_i, alpha_j). Each launch reads only the slots the launch
before it wrote. The host reads `st` once per OCSVM_CHUNK iterations, the
convergence check."""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from std.sys import llvm_intrinsic
from std.sys.info import is_apple_gpu
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from x_neighbors.items import (
    FP, IP, _add, _sub, xn_fold_blocks, ocsvm_g0, ocsvm_obj, ocsvm_pair, ocsvm_g_step,
    ocsvm_rho_part_item, ocsvm_rho_fin_item,
)
from x_neighbors.device_ops import xn_ctx, _grid, _tid, _buf, _buf_i, _down, BLOCK, kernel_kernel
from std.python import PythonObject
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

comptime OCSVM_TPB = 256
#: SMO iterations enqueued between two reads of the stop flag
comptime OCSVM_CHUNK = 64

# lane/apple-fast-gap-cls2 (2026-10-03), FAST + Apple, every switch default
# ON (below). Board (M3 FAST): ocsvm taxi 372 ms vs scikit-learn 181 (10,000 rows).
#   MOJOLEARN_XN_FAST_CLS2_OCSVM_RES: the 10,000 x 10,000 Gram never leaves
#     the device. Main forms it with `xn_kernel` (x_neighbors/device_ops.mojo
#     op_kernel), DOWNLOADS its 400 MB into a fresh host array
#     (python/mojolearn/_expansion_neighbors.py OneClassSVM.fit `_kernel`),
#     and `op_ocsvm` UPLOADS it back (`_buf(ctx, q, n * n, True)`).
#     `op_ocsvm_x` takes X, forms the same Gram with the same kernel into the
#     device buffer the solve reads: the same Q words, the same solve.
#   MOJOLEARN_XN_FAST_CLS2_OCSVM_2L: two launches per SMO iteration, not
#     three: the step kernel also writes the NEXT iteration's sel_i partials
#     (its own samples' gradients and alphas are final there; ping-pong
#     partial buffers, so no block overwrites a partial another block of the
#     same launch folds), and sel_j commits the pending pair. The same
#     maxima / minima under the same total order: the same alpha bits.
#   MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256: 256 iterations between two reads
#     of the stop flag, not 64 (fewer synchronizes; the iterations past the
#     stop are no-op launches, as they are now).
comptime _OC_FA = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
# All three are the FAST + Apple default since the M3 A/B (n=1, quality
# identical): ocsvm taxi 375.7 -> 65.0 ms (RES+2L+CHUNK256; RES alone -77%).
# -D MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF / _2L_OFF / _CHUNK256_OFF turn them
# off; the old -D names stay harmless.
comptime OCSVM_CLS2_RES = _OC_FA and not is_defined["MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF"]()
comptime OCSVM_CLS2_2L = _OC_FA and not is_defined["MOJOLEARN_XN_FAST_CLS2_OCSVM_2L_OFF"]()
comptime OCSVM_CLS2_CHUNK256 = _OC_FA and not is_defined["MOJOLEARN_XN_FAST_CLS2_OCSVM_CHUNK256_OFF"]()
comptime OCSVM_CHUNK_RUN = 256 if OCSVM_CLS2_CHUNK256 else OCSVM_CHUNK

comptime S_IT = 0
comptime S_STOP = 1
comptime S_GO = 2
comptime S_PI = 3
comptime S_PJ = 4
comptime S_LEN = 8


@always_inline
def _xn_barrier():
    comptime if is_apple_gpu():
        # orders device memory too (x_linear/team.mojo `team_barrier`)
        llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(3), Int32(1))
    else:
        barrier()


@always_inline
def _takes_max(v1: Float32, i1: Int32, v2: Float32, i2: Int32) -> Bool:
    """(v2, i2) beats (v1, i1) in the lexicographic maximum; -1 is empty."""
    if i2 < 0:
        return False
    if i1 < 0:
        return True
    return v2 > v1 or (v2 == v1 and i2 > i1)


@always_inline
def _takes_min(v1: Float32, i1: Int32, v2: Float32, i2: Int32) -> Bool:
    """(v2, -i2) beats (v1, -i1) in the lexicographic minimum; -1 is empty."""
    if i2 < 0:
        return False
    if i1 < 0:
        return True
    return v2 < v1 or (v2 == v1 and i2 > i1)


@always_inline
def _ninf() -> Float32:
    return bitcast[DType.float32](UInt32(0xFF800000))


@always_inline
def _pinf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


def ocsvm_g0_kernel(q: FP, alpha: FP, g: FP, st: IP, sf: FP, n_: Int64):
    """The initial gradient, one thread per sample; the state reset."""
    var n = Int(n_)
    var t = Int(block_idx.x) * OCSVM_TPB + Int(thread_idx.x)
    if t < n:
        g.unsafe_store(t, ocsvm_g0(q, alpha, n, t))
    if t == 0:
        st.unsafe_store(S_IT, Int32(0))
        st.unsafe_store(S_STOP, Int32(0))
        st.unsafe_store(S_GO, Int32(0))
        st.unsafe_store(S_PI, Int32(-1))
        st.unsafe_store(S_PJ, Int32(-1))
        sf.unsafe_store(0, Float32(0))
        sf.unsafe_store(1, Float32(0))


def ocsvm_commit_kernel(alpha: FP, st: IP, sf: FP, n_: Int64):
    """The pending step's alpha_i, alpha_j into alpha (idempotent)."""
    var t = Int(block_idx.x) * OCSVM_TPB + Int(thread_idx.x)
    if t < Int(n_):
        if t == Int(st.unsafe_load(S_PI)):
            alpha.unsafe_store(t, sf.unsafe_load(0))
        elif t == Int(st.unsafe_load(S_PJ)):
            alpha.unsafe_store(t, sf.unsafe_load(1))


def ocsvm_sel_i_kernel(
    cv: FP, alpha: FP, g: FP, st: IP, sf: FP, p1v: FP, p1i: IP, n_: Int64, max_iter_: Int64,
):
    var n = Int(n_)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var t = b * OCSVM_TPB + tid
    if t < n:
        if t == Int(st.unsafe_load(S_PI)):
            alpha.unsafe_store(t, sf.unsafe_load(0))
        elif t == Int(st.unsafe_load(S_PJ)):
            alpha.unsafe_store(t, sf.unsafe_load(1))
    var go = st.unsafe_load(S_STOP) == Int32(0) and Int(st.unsafe_load(S_IT)) < Int(max_iter_)
    if b == 0 and tid == 0:
        st.unsafe_store(S_GO, Int32(1) if go else Int32(0))
    if not go:
        return
    var rv = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var bv = _ninf()
    var bi = Int32(-1)
    if t < n and alpha.unsafe_load(t) < cv.unsafe_load(t):
        var ng = -g.unsafe_load(t)
        if ng >= _ninf():
            bv = ng
            bi = Int32(t)
    rv[tid] = bv
    ri[tid] = bi
    _xn_barrier()
    var w = OCSVM_TPB // 2
    while w > 0:
        if tid < w:
            if _takes_max(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                rv[tid] = rv[tid + w]
                ri[tid] = ri[tid + w]
        _xn_barrier()
        w //= 2
    if tid == 0:
        p1v.unsafe_store(b, rv[0])
        p1i.unsafe_store(b, ri[0])


def ocsvm_sel_j_kernel(
    q: FP, alpha: FP, g: FP, st: IP, p1v: FP, p1i: IP,
    p2v: FP, p2i: IP, p2g: FP, p3v: FP, p3i: IP, n_: Int64, nb_: Int64,
):
    if st.unsafe_load(S_GO) == Int32(0):
        return
    var n = Int(n_)
    var nb = Int(nb_)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var t = b * OCSVM_TPB + tid
    var rv = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rg = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var rv2 = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri2 = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    # ---- (gmax, gi): the sel_i partials folded --------------------------
    var bv = _ninf()
    var bi = Int32(-1)
    var k = tid
    while k < nb:
        if _takes_max(bv, bi, p1v.unsafe_load(k), p1i.unsafe_load(k)):
            bv = p1v.unsafe_load(k)
            bi = p1i.unsafe_load(k)
        k += OCSVM_TPB
    rv[tid] = bv
    ri[tid] = bi
    _xn_barrier()
    var w = OCSVM_TPB // 2
    while w > 0:
        if tid < w:
            if _takes_max(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                rv[tid] = rv[tid + w]
                ri[tid] = ri[tid + w]
        _xn_barrier()
        w //= 2
    var gmax = rv[0]
    var gi = Int(ri[0])
    _xn_barrier()
    # ---- this block's (obj_j, -j) minimum and (g_j, j) maximum ----------
    var ov = _pinf()
    var oi = Int32(-1)
    var og = Float32(0)
    var mv = _ninf()
    var mi = Int32(-1)
    if gi >= 0 and t < n and alpha.unsafe_load(t) > Float32(0):
        var gjv = g.unsafe_load(t)
        if gjv >= _ninf():
            mv = gjv
            mi = Int32(t)
        if _add(gmax, gjv) > Float32(0):
            var obj = ocsvm_obj(gmax, gjv, q.unsafe_load(gi * n + gi), q.unsafe_load(t * n + t), q.unsafe_load(gi * n + t))
            if obj <= _pinf():
                ov = obj
                oi = Int32(t)
                og = gjv
    rv[tid] = ov
    ri[tid] = oi
    rg[tid] = og
    rv2[tid] = mv
    ri2[tid] = mi
    _xn_barrier()
    w = OCSVM_TPB // 2
    while w > 0:
        if tid < w:
            if _takes_min(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                rv[tid] = rv[tid + w]
                ri[tid] = ri[tid + w]
                rg[tid] = rg[tid + w]
            if _takes_max(rv2[tid], ri2[tid], rv2[tid + w], ri2[tid + w]):
                rv2[tid] = rv2[tid + w]
                ri2[tid] = ri2[tid + w]
        _xn_barrier()
        w //= 2
    if tid == 0:
        p2v.unsafe_store(b, rv[0])
        p2i.unsafe_store(b, ri[0])
        p2g.unsafe_store(b, rg[0])
        p3v.unsafe_store(b, rv2[0])
        p3i.unsafe_store(b, ri2[0])


def ocsvm_step_kernel(
    q: FP, cv: FP, alpha: FP, g: FP, st: IP, sf: FP, p1v: FP, p1i: IP,
    p2v: FP, p2i: IP, p2g: FP, p3v: FP, p3i: IP, n_: Int64, nb_: Int64, eps: Float32,
):
    if st.unsafe_load(S_GO) == Int32(0):
        return
    var n = Int(n_)
    var nb = Int(nb_)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var t = b * OCSVM_TPB + tid
    var rv = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rg = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var rv2 = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri2 = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rv3 = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri3 = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    # ---- every partial folded: (gmax, gi), (obj, gj, g_j), (gmax2, j) ----
    var av = _ninf()
    var ai_ = Int32(-1)
    var ov = _pinf()
    var oi = Int32(-1)
    var og = Float32(0)
    var mv = _ninf()
    var mi = Int32(-1)
    var k = tid
    while k < nb:
        if _takes_max(av, ai_, p1v.unsafe_load(k), p1i.unsafe_load(k)):
            av = p1v.unsafe_load(k)
            ai_ = p1i.unsafe_load(k)
        if _takes_min(ov, oi, p2v.unsafe_load(k), p2i.unsafe_load(k)):
            ov = p2v.unsafe_load(k)
            oi = p2i.unsafe_load(k)
            og = p2g.unsafe_load(k)
        if _takes_max(mv, mi, p3v.unsafe_load(k), p3i.unsafe_load(k)):
            mv = p3v.unsafe_load(k)
            mi = p3i.unsafe_load(k)
        k += OCSVM_TPB
    rv[tid] = av
    ri[tid] = ai_
    rv2[tid] = ov
    ri2[tid] = oi
    rg[tid] = og
    rv3[tid] = mv
    ri3[tid] = mi
    _xn_barrier()
    var w = OCSVM_TPB // 2
    while w > 0:
        if tid < w:
            if _takes_max(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                rv[tid] = rv[tid + w]
                ri[tid] = ri[tid + w]
            if _takes_min(rv2[tid], ri2[tid], rv2[tid + w], ri2[tid + w]):
                rv2[tid] = rv2[tid + w]
                ri2[tid] = ri2[tid + w]
                rg[tid] = rg[tid + w]
            if _takes_max(rv3[tid], ri3[tid], rv3[tid + w], ri3[tid + w]):
                rv3[tid] = rv3[tid + w]
                ri3[tid] = ri3[tid + w]
        _xn_barrier()
        w //= 2
    var gmax = rv[0]
    var gi = Int(ri[0])
    var gj = Int(ri2[0])
    var g_j = rg[0]
    var gmax2 = _ninf()
    if Int(ri3[0]) >= 0:
        gmax2 = rv3[0]
    # ---- the stopping test (the item's), the same in every block ---------
    if gi < 0 or gj < 0 or _add(gmax, gmax2) < eps:
        if b == 0 and tid == 0:
            st.unsafe_store(S_STOP, Int32(1))
        return
    # ---- the step from the carried gradients; alpha is not written here --
    var old_ai = alpha.unsafe_load(gi)
    var old_aj = alpha.unsafe_load(gj)
    var a = ocsvm_pair(q, cv, n, gi, gj, old_ai, old_aj, -gmax, g_j)
    var dai = _sub(a[0], old_ai)
    var daj = _sub(a[1], old_aj)
    if t < n:
        ocsvm_g_step(q, g, n, gi, gj, dai, daj, t)
    if b == 0 and tid == 0:
        st.unsafe_store(S_IT, st.unsafe_load(S_IT) + Int32(1))
        st.unsafe_store(S_PI, Int32(gi))
        st.unsafe_store(S_PJ, Int32(gj))
        sf.unsafe_store(0, a[0])
        sf.unsafe_store(1, a[1])



def ocsvm_sel_j2_kernel(
    q: FP, alpha: FP, g: FP, st: IP, sf: FP, p1v: FP, p1i: IP,
    p2v: FP, p2i: IP, p2g: FP, p3v: FP, p3i: IP, n_: Int64, nb_: Int64, max_iter_: Int64,
):
    """OCSVM_CLS2_2L: `ocsvm_sel_j_kernel` that also does sel_i's other two
    jobs: commits the pending (alpha_i, alpha_j) of its own samples and
    decides `go` (block 0 records it for the step)."""
    var n = Int(n_)
    var nb = Int(nb_)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var t = b * OCSVM_TPB + tid
    if t < n:
        if t == Int(st.unsafe_load(S_PI)):
            alpha.unsafe_store(t, sf.unsafe_load(0))
        elif t == Int(st.unsafe_load(S_PJ)):
            alpha.unsafe_store(t, sf.unsafe_load(1))
    var go = st.unsafe_load(S_STOP) == Int32(0) and Int(st.unsafe_load(S_IT)) < Int(max_iter_)
    if b == 0 and tid == 0:
        st.unsafe_store(S_GO, Int32(1) if go else Int32(0))
    if not go:
        return
    var rv = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rg = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var rv2 = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri2 = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var bv = _ninf()
    var bi = Int32(-1)
    var k = tid
    while k < nb:
        if _takes_max(bv, bi, p1v.unsafe_load(k), p1i.unsafe_load(k)):
            bv = p1v.unsafe_load(k)
            bi = p1i.unsafe_load(k)
        k += OCSVM_TPB
    rv[tid] = bv
    ri[tid] = bi
    _xn_barrier()
    var w = OCSVM_TPB // 2
    while w > 0:
        if tid < w:
            if _takes_max(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                rv[tid] = rv[tid + w]
                ri[tid] = ri[tid + w]
        _xn_barrier()
        w //= 2
    var gmax = rv[0]
    var gi = Int(ri[0])
    _xn_barrier()
    var ov = _pinf()
    var oi = Int32(-1)
    var og = Float32(0)
    var mv = _ninf()
    var mi = Int32(-1)
    if gi >= 0 and t < n and alpha.unsafe_load(t) > Float32(0):
        var gjv = g.unsafe_load(t)
        if gjv >= _ninf():
            mv = gjv
            mi = Int32(t)
        if _add(gmax, gjv) > Float32(0):
            var obj = ocsvm_obj(gmax, gjv, q.unsafe_load(gi * n + gi), q.unsafe_load(t * n + t), q.unsafe_load(gi * n + t))
            if obj <= _pinf():
                ov = obj
                oi = Int32(t)
                og = gjv
    rv[tid] = ov
    ri[tid] = oi
    rg[tid] = og
    rv2[tid] = mv
    ri2[tid] = mi
    _xn_barrier()
    w = OCSVM_TPB // 2
    while w > 0:
        if tid < w:
            if _takes_min(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                rv[tid] = rv[tid + w]
                ri[tid] = ri[tid + w]
                rg[tid] = rg[tid + w]
            if _takes_max(rv2[tid], ri2[tid], rv2[tid + w], ri2[tid + w]):
                rv2[tid] = rv2[tid + w]
                ri2[tid] = ri2[tid + w]
        _xn_barrier()
        w //= 2
    if tid == 0:
        p2v.unsafe_store(b, rv[0])
        p2i.unsafe_store(b, ri[0])
        p2g.unsafe_store(b, rg[0])
        p3v.unsafe_store(b, rv2[0])
        p3i.unsafe_store(b, ri2[0])


def ocsvm_step2_kernel(
    q: FP, cv: FP, alpha: FP, g: FP, st: IP, sf: FP, p1v: FP, p1i: IP,
    p2v: FP, p2i: IP, p2g: FP, p3v: FP, p3i: IP, n_: Int64, nb_: Int64, eps: Float32,
    n1v: FP, n1i: IP,
):
    """OCSVM_CLS2_2L: `ocsvm_step_kernel`, then this block's sel_i partial of
    the NEXT iteration into (n1v, n1i) (the other ping-pong buffer, never
    the one this launch folds): its samples' gradients are updated by their
    own threads here and their alphas are alpha with the step's pair."""
    if st.unsafe_load(S_GO) == Int32(0):
        return
    var n = Int(n_)
    var nb = Int(nb_)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var t = b * OCSVM_TPB + tid
    var rv = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rg = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var rv2 = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri2 = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var rv3 = stack_allocation[OCSVM_TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ri3 = stack_allocation[OCSVM_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var av = _ninf()
    var ai_ = Int32(-1)
    var ov = _pinf()
    var oi = Int32(-1)
    var og = Float32(0)
    var mv = _ninf()
    var mi = Int32(-1)
    var k = tid
    while k < nb:
        if _takes_max(av, ai_, p1v.unsafe_load(k), p1i.unsafe_load(k)):
            av = p1v.unsafe_load(k)
            ai_ = p1i.unsafe_load(k)
        if _takes_min(ov, oi, p2v.unsafe_load(k), p2i.unsafe_load(k)):
            ov = p2v.unsafe_load(k)
            oi = p2i.unsafe_load(k)
            og = p2g.unsafe_load(k)
        if _takes_max(mv, mi, p3v.unsafe_load(k), p3i.unsafe_load(k)):
            mv = p3v.unsafe_load(k)
            mi = p3i.unsafe_load(k)
        k += OCSVM_TPB
    rv[tid] = av
    ri[tid] = ai_
    rv2[tid] = ov
    ri2[tid] = oi
    rg[tid] = og
    rv3[tid] = mv
    ri3[tid] = mi
    _xn_barrier()
    var w = OCSVM_TPB // 2
    while w > 0:
        if tid < w:
            if _takes_max(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                rv[tid] = rv[tid + w]
                ri[tid] = ri[tid + w]
            if _takes_min(rv2[tid], ri2[tid], rv2[tid + w], ri2[tid + w]):
                rv2[tid] = rv2[tid + w]
                ri2[tid] = ri2[tid + w]
                rg[tid] = rg[tid + w]
            if _takes_max(rv3[tid], ri3[tid], rv3[tid + w], ri3[tid + w]):
                rv3[tid] = rv3[tid + w]
                ri3[tid] = ri3[tid + w]
        _xn_barrier()
        w //= 2
    var gmax = rv[0]
    var gi = Int(ri[0])
    var gj = Int(ri2[0])
    var g_j = rg[0]
    var gmax2 = _ninf()
    if Int(ri3[0]) >= 0:
        gmax2 = rv3[0]
    if gi < 0 or gj < 0 or _add(gmax, gmax2) < eps:
        if b == 0 and tid == 0:
            st.unsafe_store(S_STOP, Int32(1))
        return
    var old_ai = alpha.unsafe_load(gi)
    var old_aj = alpha.unsafe_load(gj)
    var a = ocsvm_pair(q, cv, n, gi, gj, old_ai, old_aj, -gmax, g_j)
    var dai = _sub(a[0], old_ai)
    var daj = _sub(a[1], old_aj)
    if t < n:
        ocsvm_g_step(q, g, n, gi, gj, dai, daj, t)
    if b == 0 and tid == 0:
        st.unsafe_store(S_IT, st.unsafe_load(S_IT) + Int32(1))
        st.unsafe_store(S_PI, Int32(gi))
        st.unsafe_store(S_PJ, Int32(gj))
        sf.unsafe_store(0, a[0])
        sf.unsafe_store(1, a[1])
    # ---- the next iteration's sel_i partial (`ocsvm_sel_i_kernel`'s scan) --
    var bv = _ninf()
    var bi = Int32(-1)
    if t < n:
        var at: Float32
        if t == gi:
            at = a[0]
        elif t == gj:
            at = a[1]
        else:
            at = alpha.unsafe_load(t)
        if at < cv.unsafe_load(t):
            var ng = -g.unsafe_load(t)
            if ng >= _ninf():
                bv = ng
                bi = Int32(t)
    _xn_barrier()
    rv[tid] = bv
    ri[tid] = bi
    _xn_barrier()
    w = OCSVM_TPB // 2
    while w > 0:
        if tid < w:
            if _takes_max(rv[tid], ri[tid], rv[tid + w], ri[tid + w]):
                rv[tid] = rv[tid + w]
                ri[tid] = ri[tid + w]
        _xn_barrier()
        w //= 2
    if tid == 0:
        n1v.unsafe_store(b, rv[0])
        n1i.unsafe_store(b, ri[0])


def ocsvm_rho_part_kernel(g: FP, alpha: FP, cv: FP, pf: FP, pc: IP, n_: Int64):
    var t = _tid()
    if t < xn_fold_blocks(Int(n_)):
        ocsvm_rho_part_item(t, g, alpha, cv, pf, pc, Int(n_))


def ocsvm_rho_fin_kernel(pf: FP, pc: IP, info: FP, n_: Int64):
    var t = _tid()
    if t < 1:
        ocsvm_rho_fin_item(t, pf, pc, info, Int(n_))


def op_ocsvm(q: Int, cv: Int, alpha: Int, info: Int, iters: Int, n: Int, eps: Float32, max_iter: Int) raises:
    var ctx = xn_ctx()
    var d_q = _buf(ctx, q, n * n, True)
    _ocsvm_solve(ctx, d_q, cv, alpha, info, iters, n, eps, max_iter)
    _ = d_q^
    _ = ctx^


def op_ocsvm_x(
    x: Int, cv: Int, alpha: Int, info: Int, iters: Int, n: Int, d: Int, kind: Int, gamma: Float32,
    coef0: Float32, degree: Int, eps: Float32, max_iter: Int,
) raises:
    """OCSVM_CLS2_RES: `op_ocsvm` over the Gram of the n x d rows at `x`
    formed on the device by the kernel the Python side would have run
    (`kernel_kernel`): the same words, never downloaded."""
    comptime if not OCSVM_CLS2_RES:
        raise Error("x_neighbors: op_ocsvm_x is a FAST Apple switch (off under -D MOJOLEARN_XN_FAST_CLS2_OCSVM_RES_OFF)")
    else:
        var ctx = xn_ctx()
        var d_x = _buf(ctx, x, n * d, True)
        var d_q = _buf(ctx, 0, n * n, False)
        # x and y are the same rows: one pointer passed twice (not two
        # mutable borrows of one buffer)
        var px = d_x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        ctx.enqueue_function[kernel_kernel](
            px, px, d_q.unsafe_ptr(), Int64(n), Int64(n), Int64(d), Int64(kind),
            gamma, coef0, Int64(degree),
            grid_dim=_grid(n * n), block_dim=(BLOCK if n * n > 1 else 1),
        )
        _ocsvm_solve(ctx, d_q, cv, alpha, info, iters, n, eps, max_iter)
        _ = d_q^
        _ = d_x^
        _ = ctx^


def ocsvm_resident_binding(a_: PythonObject, i_: PythonObject, f_: PythonObject) raises -> PythonObject:
    """xn_ocsvm_x: addresses (x, cv, alpha, info, iters), ints (n, d, kind,
    degree, max_iter), floats (gamma, coef0, eps)."""
    var a = List[Int]()
    for k in range(5):
        var v = Int(py=a_[k])
        if v == 0:
            raise Error("x_neighbors: null buffer address")
        a.append(v)
    var iv = List[Int]()
    for k in range(5):
        var v = Int(py=i_[k])
        if v < 0:
            raise Error("x_neighbors: a negative size was passed")
        iv.append(v)
    var gamma = Float32(Float64(py=f_[0]))
    var coef0 = Float32(Float64(py=f_[1]))
    var eps = Float32(Float64(py=f_[2]))
    op_ocsvm_x(a[0], a[1], a[2], a[3], a[4], iv[0], iv[1], iv[2], gamma, coef0, iv[3], eps, iv[4])
    return PythonObject(None)


def _ocsvm_solve(
    ctx: DeviceContext, mut d_q: DeviceBuffer[DType.float32], cv: Int, alpha: Int, info: Int, iters: Int,
    n: Int, eps: Float32, max_iter: Int,
) raises:
    """`op_ocsvm`'s solve over the device Gram `d_q`."""
    var nb = (n + OCSVM_TPB - 1) // OCSVM_TPB if n > 0 else 1
    var nf = xn_fold_blocks(n)
    var d_cv = _buf(ctx, cv, n, True)
    var d_alpha = _buf(ctx, alpha, n, True)
    var d_g = _buf(ctx, 0, n, False)
    var d_info = _buf(ctx, info, 1, False)
    var d_st = _buf_i(ctx, 0, S_LEN, False)
    var d_sf = _buf(ctx, 0, 2, False)
    var d_p1v = _buf(ctx, 0, nb, False)
    var d_p1i = _buf_i(ctx, 0, nb, False)
    var d_p2v = _buf(ctx, 0, nb, False)
    var d_p2i = _buf_i(ctx, 0, nb, False)
    var d_p2g = _buf(ctx, 0, nb, False)
    var d_p3v = _buf(ctx, 0, nb, False)
    var d_p3i = _buf_i(ctx, 0, nb, False)
    var d_pf = _buf(ctx, 0, 3 * nf, False)
    var d_pc = _buf_i(ctx, 0, nf, False)
    ctx.enqueue_function[ocsvm_g0_kernel](
        d_q.unsafe_ptr(), d_alpha.unsafe_ptr(), d_g.unsafe_ptr(), d_st.unsafe_ptr(), d_sf.unsafe_ptr(), Int64(n),
        grid_dim=nb, block_dim=OCSVM_TPB,
    )
    var h_st = List[Int32](length=S_LEN, fill=Int32(0))
    comptime if OCSVM_CLS2_2L:
        # the ping-pong partner of (p1v, p1i)
        var d_q1v = _buf(ctx, 0, nb, False)
        var d_q1i = _buf_i(ctx, 0, nb, False)
        ctx.enqueue_function[ocsvm_sel_i_kernel](
            d_cv.unsafe_ptr(), d_alpha.unsafe_ptr(), d_g.unsafe_ptr(), d_st.unsafe_ptr(), d_sf.unsafe_ptr(),
            d_p1v.unsafe_ptr(), d_p1i.unsafe_ptr(), Int64(n), Int64(max_iter),
            grid_dim=nb, block_dim=OCSVM_TPB,
        )
        var pa_v = d_p1v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pa_i = d_p1i.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pb_v = d_q1v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var pb_i = d_q1i.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var par = 0
        while True:
            for _c in range(OCSVM_CHUNK_RUN):
                var cv_ = pa_v if par == 0 else pb_v
                var ci_ = pa_i if par == 0 else pb_i
                var nv_ = pb_v if par == 0 else pa_v
                var ni_ = pb_i if par == 0 else pa_i
                ctx.enqueue_function[ocsvm_sel_j2_kernel](
                    d_q.unsafe_ptr(), d_alpha.unsafe_ptr(), d_g.unsafe_ptr(), d_st.unsafe_ptr(), d_sf.unsafe_ptr(),
                    cv_, ci_, d_p2v.unsafe_ptr(), d_p2i.unsafe_ptr(), d_p2g.unsafe_ptr(),
                    d_p3v.unsafe_ptr(), d_p3i.unsafe_ptr(), Int64(n), Int64(nb), Int64(max_iter),
                    grid_dim=nb, block_dim=OCSVM_TPB,
                )
                ctx.enqueue_function[ocsvm_step2_kernel](
                    d_q.unsafe_ptr(), d_cv.unsafe_ptr(), d_alpha.unsafe_ptr(), d_g.unsafe_ptr(), d_st.unsafe_ptr(),
                    d_sf.unsafe_ptr(), cv_, ci_, d_p2v.unsafe_ptr(), d_p2i.unsafe_ptr(),
                    d_p2g.unsafe_ptr(), d_p3v.unsafe_ptr(), d_p3i.unsafe_ptr(), Int64(n), Int64(nb), eps,
                    nv_, ni_, grid_dim=nb, block_dim=OCSVM_TPB,
                )
                par = 1 - par
            ctx.enqueue_copy(dst_ptr=h_st.unsafe_ptr(), src_buf=d_st)
            ctx.synchronize()
            if h_st[S_STOP] != Int32(0) or Int(h_st[S_IT]) >= max_iter:
                break
        _ = d_q1v^
        _ = d_q1i^
    while not OCSVM_CLS2_2L:
        for _c in range(OCSVM_CHUNK_RUN):
            ctx.enqueue_function[ocsvm_sel_i_kernel](
                d_cv.unsafe_ptr(), d_alpha.unsafe_ptr(), d_g.unsafe_ptr(), d_st.unsafe_ptr(), d_sf.unsafe_ptr(),
                d_p1v.unsafe_ptr(), d_p1i.unsafe_ptr(), Int64(n), Int64(max_iter),
                grid_dim=nb, block_dim=OCSVM_TPB,
            )
            ctx.enqueue_function[ocsvm_sel_j_kernel](
                d_q.unsafe_ptr(), d_alpha.unsafe_ptr(), d_g.unsafe_ptr(), d_st.unsafe_ptr(),
                d_p1v.unsafe_ptr(), d_p1i.unsafe_ptr(), d_p2v.unsafe_ptr(), d_p2i.unsafe_ptr(), d_p2g.unsafe_ptr(),
                d_p3v.unsafe_ptr(), d_p3i.unsafe_ptr(), Int64(n), Int64(nb),
                grid_dim=nb, block_dim=OCSVM_TPB,
            )
            ctx.enqueue_function[ocsvm_step_kernel](
                d_q.unsafe_ptr(), d_cv.unsafe_ptr(), d_alpha.unsafe_ptr(), d_g.unsafe_ptr(), d_st.unsafe_ptr(),
                d_sf.unsafe_ptr(), d_p1v.unsafe_ptr(), d_p1i.unsafe_ptr(), d_p2v.unsafe_ptr(), d_p2i.unsafe_ptr(),
                d_p2g.unsafe_ptr(), d_p3v.unsafe_ptr(), d_p3i.unsafe_ptr(), Int64(n), Int64(nb), eps,
                grid_dim=nb, block_dim=OCSVM_TPB,
            )
        # the convergence check: the stop flag and the iteration count
        ctx.enqueue_copy(dst_ptr=h_st.unsafe_ptr(), src_buf=d_st)
        ctx.synchronize()
        if h_st[S_STOP] != Int32(0) or Int(h_st[S_IT]) >= max_iter:
            break
    ctx.enqueue_function[ocsvm_commit_kernel](
        d_alpha.unsafe_ptr(), d_st.unsafe_ptr(), d_sf.unsafe_ptr(), Int64(n),
        grid_dim=nb, block_dim=OCSVM_TPB,
    )
    ctx.enqueue_function[ocsvm_rho_part_kernel](
        d_g.unsafe_ptr(), d_alpha.unsafe_ptr(), d_cv.unsafe_ptr(), d_pf.unsafe_ptr(), d_pc.unsafe_ptr(), Int64(n),
        grid_dim=_grid(nf), block_dim=(BLOCK if nf > 1 else 1),
    )
    ctx.enqueue_function[ocsvm_rho_fin_kernel](
        d_pf.unsafe_ptr(), d_pc.unsafe_ptr(), d_info.unsafe_ptr(), Int64(n),
        grid_dim=_grid(1), block_dim=1,
    )
    _down(ctx, d_alpha, alpha, n)
    _down(ctx, d_info, info, 1)
    ctx.synchronize()
    IP(unsafe_from_address=iters).unsafe_store(0, h_st[S_IT])
    _ = d_cv^
    _ = d_alpha^
    _ = d_g^
    _ = d_info^
    _ = d_st^
    _ = d_sf^
    _ = d_p1v^
    _ = d_p1i^
    _ = d_p2v^
    _ = d_p2i^
    _ = d_p2g^
    _ = d_p3v^
    _ = d_p3i^
    _ = d_pf^
    _ = d_pc^
    _ = h_st^
