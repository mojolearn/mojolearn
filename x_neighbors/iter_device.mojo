# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Resident GPU drivers of the lane's iterations (lane neighbors-apple2).

`op_lp_iterate`: LabelPropagation / LabelSpreading's fit loop
(`_LabelPropagationBase.fit` in python/mojolearn/_expansion_neighbors.py) on
the device with the graph uploaded ONCE. The Python loop called three ops per
iteration, and each op uploaded its inputs: the n x n graph crossed to the
device on every iteration. Here the same kernels (`absdiff_sum_kernel`,
`matmul_kernel`, `lp_clamp_kernel` / `ls_clamp_kernel`, the generated
one-item-per-thread drivers of the same items) run in the same order on the
same values; only the stopping sum crosses back, one float per iteration,
compared in double exactly as Python compared it. The CPU column runs the
same loop over the items (`x_neighbors/iter_host.mojo`).
"""
from std.memory import bitcast, memcpy
from std.atomic import Atomic
from core.host_lanes import host_row_tasks
from std.time import perf_counter_ns
from std.os import getenv
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_mul_add
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator

from std.sys.compile import is_defined
from x_neighbors.cc_sparse import cc_iterate_sparse, cc_iterate_csr
from x_neighbors.nan_cells import nan_cells_host
from x_neighbors.pr_sparse import PrGraph, pr_graph_from_dense, pagerank_dangling_sum, pagerank_step_sparse_item
from x_neighbors.items import FP, IP, absdiff_sum_item, _sub, _add, knn_sq_item, knn_impute_finish
from checks.numerics import identical_mul, identical_div, identical_sqrt
from std.memory import bitcast as _bc
from x_neighbors.device_ops import (
    xn_ctx, _buf, _buf_i, _down, _down_i, _grid, _tid, BLOCK,
    absdiff_sum_kernel, matmul_kernel, lp_clamp_kernel, ls_clamp_kernel,
    pagerank_step_kernel, cc_step_kernel,
    pcs_sketch_kernel, pcs_conv_kernel, pcs_copy0_kernel, op_knn_sq, op_knn_impute_cells,
    kernel_kernel, rowsum_kernel, scale_div_kernel, kpca_center_kernel, unary_kernel, svgp_var_kernel,
)
from x_neighbors.items import matmul_tn_acc_item, K_RBF, U_IDENTITY
from x_neighbors.lp_spmm import lp_spmm_kernel
from core.device_zero import enqueue_fill


def op_lp_iterate(
    g: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, c: Int, max_iter: Int, variant: Int, tol_hi: Int, tol_lo: Int,
    alpha: Float32,
) raises:
    """variant 0 propagation (lp_clamp), 1 spreading (ls_clamp). `ld` in:
    the initial label distributions, out: the last. info (int32 x 2): the
    fit's n_iter_ and converged."""
    # Imports must be at function scope, even when their use is opt-in.
    from x_neighbors.lp_batched import op_lp_iterate_batched

    var tol = bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo))
    comptime if is_defined["MOJOLEARN_XN_LP_BATCH"]() and not is_defined["MOJOLEARN_XN_LP_DEVICE_FOLD"]():
        if n * c > 0 and max_iter > 0:
            op_lp_iterate_batched(g, ld, ystatic, unlabeled, info, n, c, max_iter, variant, tol, alpha)
            return
    var nc = n * c
    var ctx = xn_ctx()
    var d_g = _buf(ctx, g, n * n, True)
    var d_a = _buf(ctx, ld, nc, True)
    var d_b = _buf(ctx, 0, nc, False)
    ctx.enqueue_memset(d_b, Float32(0))
    var d_nxt = _buf(ctx, 0, nc, False)
    var d_ys = _buf(ctx, ystatic, nc, True)
    var d_unl = _buf_i(ctx, unlabeled, n, True)
    var d_s = _buf(ctx, 0, 1, False)
    var hs = List[Float32](length=1, fill=Float32(0))
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var prev: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur_is_a = True
    var n_iter = 0
    var converged = False
    # The stopping sum's ONE-ITEM fold over n * c values ran on ONE GPU
    # thread every iteration (1.5 ms at n = 5,000 with many classes). By
    # default the host folds it: the current distributions come back each
    # iteration (they are what the step just wrote), the previous ones are
    # the host copy from the iteration before, and `absdiff_sum_item` runs
    # its same statements on them. `-D MOJOLEARN_XN_LP_DEVICE_FOLD` keeps
    # the device fold.
    var hc = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    var hp = List[Float32](length=nc if nc > 0 else 1, fill=Float32(0))
    # G's nonzero entries (a knn graph holds ~k per row), row-major CSR,
    # built once on the host from the caller's array. Sparse only when it
    # pays (under an eighth nonzero) and only in the host-fold path, which
    # has each iteration's x on the host for the finiteness test.
    # `-D MOJOLEARN_XN_LP_DENSE` keeps the dense product.
    var pg = FP(unsafe_from_address=g)
    var h_ip = List[Int32](length=n + 1, fill=Int32(0))
    var h_cols = List[Int32]()
    var h_vals = List[Float32]()
    var use_sparse = False
    comptime if not is_defined["MOJOLEARN_XN_LP_DENSE"]() and not is_defined["MOJOLEARN_XN_LP_DEVICE_FOLD"]():
        var nnz = 0
        for q in range(n * n):
            if pg.unsafe_load(q) != Float32(0):
                nnz += 1
        use_sparse = nnz > 0 and nnz * 8 < n * n
        if use_sparse:
            for i in range(n):
                for jj in range(n):
                    var v = pg.unsafe_load(i * n + jj)
                    if v != Float32(0):
                        h_cols.append(Int32(jj))
                        h_vals.append(v)
                h_ip[i + 1] = Int32(len(h_cols))
    var nnzb = len(h_cols) if len(h_cols) > 0 else 1
    var d_ip = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var d_cols = ctx.enqueue_create_buffer[DType.int32](nnzb)
    var d_vals = ctx.enqueue_create_buffer[DType.float32](nnzb)
    if use_sparse:
        ctx.enqueue_copy(dst_buf=d_ip, src_ptr=h_ip.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_cols, src_ptr=h_cols.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=d_vals, src_ptr=h_vals.unsafe_ptr())
    for it in range(max_iter):
        n_iter = it
        comptime if is_defined["MOJOLEARN_XN_LP_DEVICE_FOLD"]():
            ctx.enqueue_function[absdiff_sum_kernel](
                cur, prev, d_s.unsafe_ptr(), Int64(nc), grid_dim=1, block_dim=1,
            )
            ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=d_s)
            ctx.synchronize()
        else:
            if cur_is_a:
                ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=d_a)
            else:
                ctx.enqueue_copy(dst_ptr=hc.unsafe_ptr(), src_buf=d_b)
            ctx.synchronize()
            absdiff_sum_item(
                0, FP(unsafe_from_address=Int(hc.unsafe_ptr())),
                FP(unsafe_from_address=Int(hp.unsafe_ptr())),
                FP(unsafe_from_address=Int(hs.unsafe_ptr())), nc,
            )
            for q in range(nc):
                hp[q] = hc[q]
        if Float64(hs[0]) < tol:
            converged = True
            break
        var sparse_now = use_sparse
        if sparse_now:
            for q in range(nc):
                var bits = bitcast[DType.uint32](hc[q]) & UInt32(0x7F800000)
                if bits == UInt32(0x7F800000):
                    sparse_now = False
                    break
        if sparse_now:
            ctx.enqueue_function[lp_spmm_kernel](
                d_ip.unsafe_ptr(), d_cols.unsafe_ptr(), d_vals.unsafe_ptr(), cur, d_nxt.unsafe_ptr(),
                Int64(n), Int64(c), grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
        else:
            ctx.enqueue_function[matmul_kernel](
                d_g.unsafe_ptr(), cur, d_nxt.unsafe_ptr(), Int64(n), Int64(n), Int64(c),
                grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
        # prev = ld; ld = clamp(nxt): the clamp writes over the buffer the
        # old prev held, then the two names swap.
        if variant == 0:
            ctx.enqueue_function[lp_clamp_kernel](
                d_nxt.unsafe_ptr(), d_ys.unsafe_ptr(), d_unl.unsafe_ptr(), prev, Int64(n), Int64(c),
                grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
            )
        else:
            ctx.enqueue_function[ls_clamp_kernel](
                d_nxt.unsafe_ptr(), d_ys.unsafe_ptr(), prev, Int64(nc), alpha,
                grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
        var t = cur
        cur = prev
        prev = t
        cur_is_a = not cur_is_a
    if not converged:
        n_iter += 1
    if cur_is_a:
        _down(ctx, d_a, ld, nc)
    else:
        _down(ctx, d_b, ld, nc)
    ctx.synchronize()
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = hs^
    _ = hc^
    _ = hp^
    _ = h_ip^
    _ = h_cols^
    _ = h_vals^
    _ = d_ip^
    _ = d_cols^
    _ = d_vals^
    _ = d_g^
    _ = d_a^
    _ = d_b^
    _ = d_nxt^
    _ = d_ys^
    _ = d_unl^
    _ = d_s^
    _ = ctx^


def op_pr_iterate(
    q: Int, x: Int, p: Int, dw: Int, dangling: Int, info: Int,
    n: Int, max_iter: Int, thr_hi: Int, thr_lo: Int, alpha: Float32,
) raises:
    """PageRank.fit's power iteration (`pagerank_step` then `absdiff_sum`
    per iteration, x <- the step) with Q resident. `thr` is Python's
    n * tol as float64 bits. `x` in: the start, out: the last iterate.
    info (int32 x 2): iterations run, converged."""
    var thr = bitcast[DType.float64]((UInt64(thr_hi) << UInt64(32)) | UInt64(thr_lo))
    var ctx = xn_ctx()
    var d_q = _buf(ctx, q, n * n, True)
    var d_a = _buf(ctx, x, n, True)
    var d_b = _buf(ctx, 0, n, False)
    var d_p = _buf(ctx, p, n, True)
    var d_dw = _buf(ctx, dw, n, True)
    var d_dg = _buf_i(ctx, dangling, n, True)
    var d_s = _buf(ctx, 0, 1, False)
    var hs = List[Float32](length=1, fill=Float32(0))
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur_is_a = True
    var n_iter = 0
    var converged = False
    for it in range(max_iter):
        ctx.enqueue_function[pagerank_step_kernel](
            d_q.unsafe_ptr(), cur, d_p.unsafe_ptr(), d_dw.unsafe_ptr(), d_dg.unsafe_ptr(), nxt,
            Int64(n), alpha, grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
        )
        ctx.enqueue_function[absdiff_sum_kernel](
            nxt, cur, d_s.unsafe_ptr(), Int64(n), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=d_s)
        ctx.synchronize()
        var t = cur
        cur = nxt
        nxt = t
        cur_is_a = not cur_is_a
        n_iter = it + 1
        if Float64(hs[0]) < thr:
            converged = True
            break
    if cur_is_a:
        _down(ctx, d_a, x, n)
    else:
        _down(ctx, d_b, x, n)
    ctx.synchronize()
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = hs^
    _ = d_q^
    _ = d_a^
    _ = d_b^
    _ = d_p^
    _ = d_dw^
    _ = d_dg^
    _ = d_s^
    _ = ctx^



def pr_dangling_sum_kernel(x: FP, dangling: IP, res: FP, n_: Int64):
    """ONE thread: the item's dangling-mass chain, once per iteration."""
    if _tid() == 0:
        res.unsafe_store(0, pagerank_dangling_sum(x, dangling, Int(n_)))


def pagerank_step_sparse_kernel(
    indptr: IP, rows: IP, vals: FP, x: FP, p: FP, dw: FP, dsum: FP, res: FP, n_: Int64, alpha_: Float32,
):
    var t = _tid()
    if t < Int(n_):
        pagerank_step_sparse_item(t, indptr, rows, vals, x, p, dw, dsum.unsafe_load(0), res, alpha_)


def op_pr_iterate_sparse(
    a: Int, x: Int, p: Int, dw: Int, info: Int,
    n: Int, max_iter: Int, thr_hi: Int, thr_lo: Int, binary: Int, alpha: Float32,
) raises:
    """`op_pr_iterate` over the nonzero cells of the dense adjacency `a`
    (x_neighbors/pr_sparse.mojo): the scan on the host, the iteration on
    the device over the column lists. `x` in: the start, out: the last
    iterate. info (int32 x 2): iterations run, converged."""
    var thr = bitcast[DType.float64]((UInt64(thr_hi) << UInt64(32)) | UInt64(thr_lo))
    var timing = String(getenv("MOJOLEARN_PR_TIMING")) == "1"
    var t_start = perf_counter_ns()
    var g = pr_graph_from_dense(FP(unsafe_from_address=a), n, binary != 0)
    if timing:
        print("pr_iterate_sparse: scan", (perf_counter_ns() - t_start) // 1000000, "ms, nnz", g.nnz,
              "tasks", host_row_tasks(n, 2 * n))
        t_start = perf_counter_ns()
    var ctx = xn_ctx()
    var d_ip = _buf_i(ctx, Int(g.indptr.unsafe_ptr()), n + 1, True)
    var d_rows = _buf_i(ctx, Int(g.rows.unsafe_ptr()), g.nnz, True)
    var d_vals = _buf(ctx, Int(g.vals.unsafe_ptr()), g.nnz, True)
    var d_dg = _buf_i(ctx, Int(g.dangling.unsafe_ptr()), n, True)
    var d_a = _buf(ctx, x, n, True)
    var d_b = _buf(ctx, 0, n, False)
    var d_p = _buf(ctx, p, n, True)
    var d_dw = _buf(ctx, dw, n, True)
    var d_s = _buf(ctx, 0, 1, False)
    var d_ds = _buf(ctx, 0, 1, False)
    var hs = List[Float32](length=1, fill=Float32(0))
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur_is_a = True
    var n_iter = 0
    var converged = False
    for it in range(max_iter):
        ctx.enqueue_function[pr_dangling_sum_kernel](
            cur, d_dg.unsafe_ptr(), d_ds.unsafe_ptr(), Int64(n), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_function[pagerank_step_sparse_kernel](
            d_ip.unsafe_ptr(), d_rows.unsafe_ptr(), d_vals.unsafe_ptr(), cur, d_p.unsafe_ptr(), d_dw.unsafe_ptr(),
            d_ds.unsafe_ptr(), nxt, Int64(n), alpha, grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
        )
        ctx.enqueue_function[absdiff_sum_kernel](
            nxt, cur, d_s.unsafe_ptr(), Int64(n), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=d_s)
        ctx.synchronize()
        var t = cur
        cur = nxt
        nxt = t
        cur_is_a = not cur_is_a
        n_iter = it + 1
        if Float64(hs[0]) < thr:
            converged = True
            break
    if cur_is_a:
        _down(ctx, d_a, x, n)
    else:
        _down(ctx, d_b, x, n)
    ctx.synchronize()
    if timing:
        print("pr_iterate_sparse: device", (perf_counter_ns() - t_start) // 1000000, "ms,", n_iter, "iterations")
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = hs^
    _ = d_ip^
    _ = d_rows^
    _ = d_vals^
    _ = d_dg^
    _ = d_a^
    _ = d_b^
    _ = d_p^
    _ = d_dw^
    _ = d_s^
    _ = d_ds^
    _ = g^
    _ = ctx^

def op_nan_cells(x: Int, cells: Int, colmiss: Int, info: Int, n: Int, d: Int) raises:
    """lane/neural-pass71: the NaN cells of x (n x d): flat indices
    ascending into `cells`, the NaN count per column, the total in info[0].
    The host pass on every column (x_neighbors/nan_cells.mojo)."""
    nan_cells_host(FP(unsafe_from_address=x), IP(unsafe_from_address=cells),
                   IP(unsafe_from_address=colmiss), IP(unsafe_from_address=info), n, d)


def op_cc_iterate_csr(indptr: Int, indices: Int, lab: Int, info: Int, n: Int, nnz: Int) raises:
    """lane/neural-pass69: `op_cc_iterate` from a CSR adjacency (indptr n + 1,
    indices nnz): the host walk of x_neighbors/cc_sparse.mojo on every
    column, no dense matrix."""
    comptime if is_defined["MOJOLEARN_XN_CC_HOST"]():
        cc_iterate_csr(IP(unsafe_from_address=indptr), IP(unsafe_from_address=indices),
                       IP(unsafe_from_address=lab), IP(unsafe_from_address=info), n, nnz)
        return
    _cc_csr_device(indptr, indices, lab, info, n, nnz)


# lane/neural-pass95 (2026-10-01): weak connected components of a CSR graph
# on the device. The output is the min-label fixed point the host rounds
# reach (every node labelled by the lowest node of its component, which
# Python numbers in order of appearance), and that fixed point does not
# depend on how it is reached, so the device reaches it the fast way:
# hooking (each edge lowers the larger of its two labels' entries to the
# smaller, an atomic min; labels only fall, each to a node of the same
# component no larger than itself) and pointer jumping (every node to the
# root of its label chain), rounds until an edge changes nothing. The
# labels are integers: the same words on every column. The step count in
# info is the device's round count (Python reads only the labels).
# `-D MOJOLEARN_XN_CC_HOST=1` restores the host rounds (x_neighbors/cc_sparse.mojo).
def cc_hook_kernel(indptr: IP, indices: IP, lab: IP, n: Int32, changed: IP):
    var u = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if u < Int(n):
        for e in range(Int(indptr.unsafe_load(u)), Int(indptr.unsafe_load(u + 1))):
            var v = Int(indices.unsafe_load(e))
            var a = lab.unsafe_load(u)
            var b = lab.unsafe_load(v)
            if a != b:
                var lo = a if a < b else b
                var hi = b if a < b else a
                # a racing read only costs a spare round: the flag is set
                # whenever this edge could still lower an entry
                if lab.unsafe_load(Int(hi)) > lo:
                    Atomic[DType.int32].min(lab + Int(hi), lo)
                    changed.unsafe_store(0, Int32(1))


def cc_jump_kernel(lab: IP, n: Int32):
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v < Int(n):
        var p = Int(lab.unsafe_load(v))
        while Int(lab.unsafe_load(p)) != p:
            p = Int(lab.unsafe_load(p))
        lab.unsafe_store(v, Int32(p))


def _cc_csr_device(indptr: Int, indices: Int, lab: Int, info: Int, n: Int, nnz: Int) raises:
    # Every host side of a copy is ONE pinned host buffer (lane/neural-pass95,
    # 2026-10-02): the peer's MI325X paid ~100 ms on the first fit after any
    # fork / posix_spawn in the process (ps, a notebook's subprocess) when
    # the CSR, the labels and the per-round flag went through pageable
    # memory, which fork's copy-on-write unmaps from the GPU's view; pinned
    # host memory is excluded from fork (MADV_DONTFORK). Copies only.
    var ctx = xn_ctx()
    var o_ix = n + 1
    var o_l = o_ix + max(nnz, 1)
    var o_c = o_l + max(n, 1)
    var hb = ctx.enqueue_create_host_buffer[DType.int32](o_c + 1)
    ctx.synchronize()
    var hp = hb.unsafe_ptr()
    memcpy(dest=hp, src=IP(unsafe_from_address=indptr), count=n + 1)
    if nnz > 0:
        memcpy(dest=hp + o_ix, src=IP(unsafe_from_address=indices), count=nnz)
    if n > 0:
        memcpy(dest=hp + o_l, src=IP(unsafe_from_address=lab), count=n)
    var d_ip = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var d_ix = ctx.enqueue_create_buffer[DType.int32](max(nnz, 1))
    var d_l = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_c = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_copy(dst_buf=d_ip, src_ptr=hp)
    if nnz > 0:
        ctx.enqueue_copy(dst_buf=d_ix, src_ptr=hp + o_ix)
    if n > 0:
        ctx.enqueue_copy(dst_buf=d_l, src_ptr=hp + o_l)
    var blocks = (n + 255) // 256
    var rounds = 0
    while n > 0:
        rounds += 1
        d_c.enqueue_fill(Int32(0))
        ctx.enqueue_function[cc_hook_kernel](d_ip.unsafe_ptr(), d_ix.unsafe_ptr(), d_l.unsafe_ptr(), Int32(n),
                                             d_c.unsafe_ptr(), grid_dim=blocks, block_dim=256)
        ctx.enqueue_function[cc_jump_kernel](d_l.unsafe_ptr(), Int32(n), grid_dim=blocks, block_dim=256)
        ctx.enqueue_copy(dst_ptr=hp + o_c, src_buf=d_c)
        ctx.synchronize()
        if hp[o_c] == 0:
            break
    if n > 0:
        ctx.enqueue_copy(dst_ptr=hp + o_l, src_buf=d_l)
        ctx.synchronize()
        memcpy(dest=IP(unsafe_from_address=lab), src=hp + o_l, count=n)
    IP(unsafe_from_address=info).unsafe_store(0, Int32(rounds))
    _ = hb^
    _ = d_ip^
    _ = d_ix^
    _ = d_l^
    _ = d_c^


def op_cc_iterate(a: Int, lab: Int, info: Int, n: Int) raises:
    """connected_components' min-label iteration (`cc_step` until the labels
    stop changing) with A resident; the labels come back each step (n
    integers) for the host's equality test, as Python compared them.
    `lab` in: 0..n-1, out: the fixed point. info (int32 x 1): steps.
    Lane neural-pass22: unless MOJOLEARN_XN_CC_GPU is defined, the rounds run
    on the HOST as the sparse walk of x_neighbors/cc_sparse.mojo (the same
    labels and round count; the dense rounds here read row t and column t
    of the matrix for every node in every round, 800 million cells a round
    at the board's 20,000 nodes); the resident loop below stays the device
    body and the reference."""
    comptime if not is_defined["MOJOLEARN_XN_CC_GPU"]():
        cc_iterate_sparse(FP(unsafe_from_address=a), IP(unsafe_from_address=lab), IP(unsafe_from_address=info), n)
        return
    var ctx = xn_ctx()
    var d_a = _buf(ctx, a, n * n, True)
    var d_l0 = _buf_i(ctx, lab, n, True)
    var d_l1 = _buf_i(ctx, 0, n, False)
    var cur_host = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    var nxt_host = List[Int32](length=n if n > 0 else 1, fill=Int32(0))
    var pl = IP(unsafe_from_address=lab)
    for i in range(n):
        cur_host[i] = pl.unsafe_load(i)
    var cur: IP = d_l0.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt: IP = d_l1.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt_is_1 = True
    var steps = 0
    while True:
        ctx.enqueue_function[cc_step_kernel](
            d_a.unsafe_ptr(), cur, nxt, Int64(n),
            grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
        )
        if nxt_is_1:
            ctx.enqueue_copy(dst_ptr=nxt_host.unsafe_ptr(), src_buf=d_l1)
        else:
            ctx.enqueue_copy(dst_ptr=nxt_host.unsafe_ptr(), src_buf=d_l0)
        ctx.synchronize()
        steps += 1
        var same = True
        for i in range(n):
            if nxt_host[i] != cur_host[i]:
                same = False
                break
        if same:
            break
        for i in range(n):
            cur_host[i] = nxt_host[i]
        var t = cur
        cur = nxt
        nxt = t
        nxt_is_1 = not nxt_is_1
    for i in range(n):
        pl.unsafe_store(i, cur_host[i])
    IP(unsafe_from_address=info).unsafe_store(0, Int32(steps))
    _ = cur_host^
    _ = nxt_host^
    _ = d_a^
    _ = d_l0^
    _ = d_l1^
    _ = ctx^


comptime PCS_ROW_TPB = 256
comptime PCS_ROW_MAX_NC = 2048


def pcs_conv_row_kernel(acc: FP, sk: FP, res: FP, n_: Int64, nc_: Int64, degree_: Int64, p_: Int64):
    """`pcs_conv_item` for one ROW per block: the row's running product and
    its degree-p sketch staged in threadgroup memory (read-only after one
    barrier), then each thread folds its components' convolutions with the
    item's statements, a ascending. nc <= PCS_ROW_MAX_NC."""
    var nc = Int(nc_)
    var degree = Int(degree_)
    var p = Int(p_)
    var r = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var ar = stack_allocation[PCS_ROW_MAX_NC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sr = stack_allocation[PCS_ROW_MAX_NC, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var q = tid
    while q < nc:
        ar[q] = acc.unsafe_load(r * nc + q)
        sr[q] = sk.unsafe_load((r * degree + p) * nc + q)
        q += PCS_ROW_TPB
    barrier()
    # Four consecutive components per thread per pass: ar[a] is read once
    # for four independent chains; each chain is the item's, a ascending.
    var h0 = tid * 4
    while h0 < nc:
        var s0 = Float32(0)
        var s1 = Float32(0)
        var s2 = Float32(0)
        var s3 = Float32(0)
        for a in range(nc):
            var av = ar[a]
            var b = h0 - a
            if b < 0:
                b += nc
            s0 = ftz(identical_mul_add(av, sr[b], s0))
            b += 1
            if b == nc:
                b = 0
            s1 = ftz(identical_mul_add(av, sr[b], s1))
            b += 1
            if b == nc:
                b = 0
            s2 = ftz(identical_mul_add(av, sr[b], s2))
            b += 1
            if b == nc:
                b = 0
            s3 = ftz(identical_mul_add(av, sr[b], s3))
        res.unsafe_store(r * nc + h0, s0)
        if h0 + 1 < nc:
            res.unsafe_store(r * nc + h0 + 1, s1)
        if h0 + 2 < nc:
            res.unsafe_store(r * nc + h0 + 2, s2)
        if h0 + 3 < nc:
            res.unsafe_store(r * nc + h0 + 3, s3)
        h0 += PCS_ROW_TPB * 4


#: lane neighbors-apple3 (2026-09-28), FAST on Apple, OPT-IN until its A/B
#: and quality check pass (`-D MOJOLEARN_XN_PCS_SPARSE`): the convolution
#: over the running product's NONZERO components only. A count sketch of a
#: row of d features has at most d nonzero components (each feature lands in
#: one), so at degree 2 a row's 500 outputs fold about 8 terms each, not
#: 500. The skipped terms are products with a zero; FAST's words can differ
#: from the full fold's only in the sign of a zero.
comptime PCS_SPARSE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_XN_PCS_SPARSE"]()
)

def op_pcs_resident(
    x: Int, hidx: Int, hbit: Int, res: Int,
    n: Int, d_in: Int, nf: Int, nc: Int, degree: Int, gamma: Float32, coef0: Float32,
) raises:
    """PolynomialCountSketch.transform: `pcs_item` (one thread per ROW, the
    convolution O(nc^2) per row) as three kernels over the same statements:
    one sketch per (row, degree), then per degree p >= 1 one thread per
    output cell folding the convolution in the same ascending order, the
    running product ping-ponging on the device."""
    # The loop keeps its compile-time guard; only import placement changes.
    from x_neighbors.pcs_sparse import PCS_SPARSE_MAX_NC, pcs_conv_row_sparse_kernel

    var ctx = xn_ctx()
    var d_x = _buf(ctx, x, n * d_in, True)
    var d_hi = _buf_i(ctx, hidx, degree * nf, True)
    var d_hb = _buf_i(ctx, hbit, degree * nf, True)
    var d_sk = _buf(ctx, 0, n * degree * nc, False)
    var d_a = _buf(ctx, 0, n * nc, False)
    var d_b = _buf(ctx, 0, n * nc, False)
    var nd = n * degree
    var cells = n * nc
    ctx.enqueue_function[pcs_sketch_kernel](
        d_x.unsafe_ptr(), d_hi.unsafe_ptr(), d_hb.unsafe_ptr(), d_sk.unsafe_ptr(),
        Int64(n), Int64(d_in), Int64(nf), Int64(nc), Int64(degree), gamma, coef0,
        grid_dim=_grid(nd), block_dim=(BLOCK if nd > 1 else 1),
    )
    ctx.enqueue_function[pcs_copy0_kernel](
        d_sk.unsafe_ptr(), d_a.unsafe_ptr(), Int64(n), Int64(nc), Int64(degree),
        grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
    )
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var nxt: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur_is_a = True
    for p in range(1, degree):
        var row_kernel = nc <= PCS_ROW_MAX_NC
        comptime if is_defined["MOJOLEARN_XN_PCS_CELL"]():
            row_kernel = False
        var sparse_kernel = False
        comptime if PCS_SPARSE:
            sparse_kernel = row_kernel and nc <= PCS_SPARSE_MAX_NC
            if sparse_kernel:
                ctx.enqueue_function[pcs_conv_row_sparse_kernel](
                    cur, d_sk.unsafe_ptr(), nxt, Int64(n), Int64(nc), Int64(degree), Int64(p),
                    grid_dim=n, block_dim=PCS_ROW_TPB,
                )
        if sparse_kernel:
            pass
        elif row_kernel:
            ctx.enqueue_function[pcs_conv_row_kernel](
                cur, d_sk.unsafe_ptr(), nxt, Int64(n), Int64(nc), Int64(degree), Int64(p),
                grid_dim=n, block_dim=PCS_ROW_TPB,
            )
        else:
            ctx.enqueue_function[pcs_conv_kernel](
                cur, d_sk.unsafe_ptr(), nxt, Int64(n), Int64(nc), Int64(degree), Int64(p),
                grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
            )
        var t = cur
        cur = nxt
        nxt = t
        cur_is_a = not cur_is_a
    if cur_is_a:
        _down(ctx, d_a, res, cells)
    else:
        _down(ctx, d_b, res, cells)
    ctx.synchronize()
    _ = d_x^
    _ = d_hi^
    _ = d_hb^
    _ = d_sk^
    _ = d_a^
    _ = d_b^
    _ = ctx^


comptime KNN_TILE_TPB = 128
comptime KNN_TILE_ROWS = 64
comptime KNN_TILE_MAX_D = 64


def knn_sq_tiled_kernel(
    x: FP, y: FP, dist: FP, idx: IP, n_: Int64, m_: Int64, d_: Int64, k_: Int64, ex_: Int64,
):
    """`knn_sq_item` for x row t = this thread, with the y rows staged
    KNN_TILE_ROWS at a time in threadgroup memory (read-only between two
    barriers) instead of read by every thread from device memory. Candidate
    columns are offered in the same ascending order to the same strict-<
    insertion, each value by the item's statements, so the lists are the
    item's. d <= KNN_TILE_MAX_D."""
    var n = Int(n_)
    var m = Int(m_)
    var d = Int(d_)
    var k = Int(k_)
    var ex = Int(ex_)
    var tid = Int(thread_idx.x)
    var t = Int(block_idx.x) * KNN_TILE_TPB + tid
    var ys = stack_allocation[KNN_TILE_ROWS * KNN_TILE_MAX_D, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var inf = _bc[DType.float32](UInt32(0x7F800000))
    var live = t < n
    if live:
        for s in range(k):
            dist.unsafe_store(t * k + s, inf)
            idx.unsafe_store(t * k + s, Int32(-1))
    var worst = inf
    var j0 = 0
    while j0 < m:
        var rows = min(KNN_TILE_ROWS, m - j0)
        var q = tid
        while q < rows * d:
            ys[q] = y.unsafe_load(j0 * d + q)
            q += KNN_TILE_TPB
        barrier()
        if live:
            for jj in range(rows):
                var j = j0 + jj
                if ex != 0 and j == t:
                    continue
                var acc = Float32(0)
                for f in range(d):
                    var df = _sub(x.unsafe_load(t * d + f), ys[jj * d + f])
                    acc = ftz(identical_mul_add(df, df, acc))
                var v = acc
                if not (v < worst):
                    continue
                var s = k - 1
                while s > 0 and v < dist.unsafe_load(t * k + s - 1):
                    dist.unsafe_store(t * k + s, dist.unsafe_load(t * k + s - 1))
                    idx.unsafe_store(t * k + s, idx.unsafe_load(t * k + s - 1))
                    s -= 1
                dist.unsafe_store(t * k + s, v)
                idx.unsafe_store(t * k + s, Int32(j))
                worst = dist.unsafe_load(t * k + k - 1)
        barrier()
        j0 += rows


def op_knn_sq_tiled(
    x: Int, y: Int, dist: Int, idx: Int, n: Int, m: Int, d: Int, k: Int, exclude_self: Int,
) raises:
    """The fused k-NN (`knn_sq`) with y staged per block; d above
    KNN_TILE_MAX_D takes the one-thread-per-row item kernel."""
    if d > KNN_TILE_MAX_D:
        op_knn_sq(x, y, dist, idx, n, m, d, k, exclude_self)
        return
    var ctx = xn_ctx()
    var d_x = _buf(ctx, x, n * d, True)
    var d_y = _buf(ctx, y, m * d, True)
    var d_dist = _buf(ctx, 0, n * k, False)
    var d_idx = _buf_i(ctx, 0, n * k, False)
    ctx.enqueue_function[knn_sq_tiled_kernel](
        d_x.unsafe_ptr(), d_y.unsafe_ptr(), d_dist.unsafe_ptr(), d_idx.unsafe_ptr(),
        Int64(n), Int64(m), Int64(d), Int64(k), Int64(exclude_self),
        grid_dim=(n + KNN_TILE_TPB - 1) // KNN_TILE_TPB, block_dim=KNN_TILE_TPB,
    )
    _down(ctx, d_dist, dist, n * k)
    _down_i(ctx, d_idx, idx, n * k)
    ctx.synchronize()
    _ = d_x^
    _ = d_y^
    _ = d_dist^
    _ = d_idx^
    _ = ctx^


comptime IMP_TPB = 128
comptime IMP_ROWS = 64
comptime IMP_MAX_D = 64


def knn_impute_tiled_kernel(
    cells: IP, x: FP, fx: FP, best_d: FP, best_i: IP, res: FP,
    n_: Int64, m_: Int64, d_: Int64, k_: Int64, weights_: Int64, nc_: Int64,
):
    """`knn_impute_cell_item` for the missing cell of this thread, with the
    fit rows staged IMP_ROWS at a time in threadgroup memory. The donor scan
    is `knn_impute_item`'s statements with fx read from the stage, donors in
    the same ascending order; the tail is `knn_impute_finish`. d <= IMP_MAX_D."""
    var m = Int(m_)
    var d = Int(d_)
    var k = Int(k_)
    var nc = Int(nc_)
    var tid = Int(thread_idx.x)
    var q0 = Int(block_idx.x) * IMP_TPB + tid
    var fs = stack_allocation[IMP_ROWS * IMP_MAX_D, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var live = q0 < nc
    var t = 0
    var r = 0
    var c = 0
    if live:
        t = Int(cells.unsafe_load(q0))
        r = t // d
        c = t - r * d
    var inf = _bc[DType.float32](UInt32(0x7F800000))
    var bd = best_d + t * k
    var bi = best_i + t * k
    if live:
        for s in range(k):
            bd.unsafe_store(s, inf)
            bi.unsafe_store(s, Int32(-1))
    var n_donors = 0
    var j0 = 0
    while j0 < m:
        var rows = min(IMP_ROWS, m - j0)
        var q = tid
        while q < rows * d:
            fs[q] = fx.unsafe_load(j0 * d + q)
            q += IMP_TPB
        barrier()
        if live:
            for jj in range(rows):
                var j = j0 + jj
                var dv = fs[jj * d + c]
                if dv != dv:
                    continue
                n_donors += 1
                var acc = Float32(0)
                var present = 0
                for f in range(d):
                    var a = x.unsafe_load(r * d + f)
                    var b = fs[jj * d + f]
                    if a != a or b != b:
                        continue
                    present += 1
                    var df = _sub(a, b)
                    acc = ftz(identical_mul_add(df, df, acc))
                if present == 0:
                    continue
                var sq = ftz(identical_mul(ftz(identical_div(acc, Float32(present))), Float32(d)))
                var dist = ftz(identical_sqrt(sq))
                if not (dist < bd.unsafe_load(k - 1)):
                    continue
                var s = k - 1
                while s > 0 and dist < bd.unsafe_load(s - 1):
                    bd.unsafe_store(s, bd.unsafe_load(s - 1))
                    bi.unsafe_store(s, bi.unsafe_load(s - 1))
                    s -= 1
                bd.unsafe_store(s, dist)
                bi.unsafe_store(s, Int32(j))
        barrier()
        j0 += rows
    if live:
        knn_impute_finish(t, fx, bd, bi, res, m, d, k, Int(weights_), n_donors)


def op_knn_impute_tiled(
    cells: Int, x: Int, fx: Int, res: Int,
    n: Int, m: Int, d: Int, k: Int, weights: Int, nc: Int,
) raises:
    """`knn_impute_cells` with the fit rows staged per block; d above
    IMP_MAX_D takes the per-cell item kernel."""
    if d > IMP_MAX_D:
        op_knn_impute_cells(cells, x, fx, res, n, m, d, k, weights, nc)
        return
    var ctx = xn_ctx()
    var d_cells = _buf_i(ctx, cells, nc, True)
    var d_x = _buf(ctx, x, n * d, True)
    var d_fx = _buf(ctx, fx, m * d, True)
    var d_bd = _buf(ctx, 0, n * d * k, False)
    var d_bi = _buf_i(ctx, 0, n * d * k, False)
    var d_res = _buf(ctx, res, n * d, True)
    var split = k <= IMPS_KMAX
    comptime if is_defined["MOJOLEARN_XN_IMPUTE_NO_SPLIT"]():
        split = False
    if split:
        ctx.enqueue_function[knn_impute_split_kernel](
            d_cells.unsafe_ptr(), d_x.unsafe_ptr(), d_fx.unsafe_ptr(), d_bd.unsafe_ptr(), d_bi.unsafe_ptr(),
            d_res.unsafe_ptr(), Int64(n), Int64(m), Int64(d), Int64(k), Int64(weights), Int64(nc),
            grid_dim=(nc + IMPS_CELLS - 1) // IMPS_CELLS, block_dim=IMPS_TPB,
        )
    else:
        ctx.enqueue_function[knn_impute_tiled_kernel](
            d_cells.unsafe_ptr(), d_x.unsafe_ptr(), d_fx.unsafe_ptr(), d_bd.unsafe_ptr(), d_bi.unsafe_ptr(),
            d_res.unsafe_ptr(), Int64(n), Int64(m), Int64(d), Int64(k), Int64(weights), Int64(nc),
            grid_dim=(nc + IMP_TPB - 1) // IMP_TPB, block_dim=IMP_TPB,
        )
    _down(ctx, d_res, res, n * d)
    ctx.synchronize()
    _ = d_cells^
    _ = d_x^
    _ = d_fx^
    _ = d_bd^
    _ = d_bi^
    _ = d_res^
    _ = ctx^


comptime IMPS_TPB = 128
comptime IMPS_SPLIT = 8
comptime IMPS_CELLS = IMPS_TPB // IMPS_SPLIT
comptime IMPS_ROWS = 32
comptime IMPS_KMAX = 16


def knn_impute_split_kernel(
    cells: IP, x: FP, fx: FP, best_d: FP, best_i: IP, res: FP,
    n_: Int64, m_: Int64, d_: Int64, k_: Int64, weights_: Int64, nc_: Int64,
):
    """`knn_impute_tiled_kernel` with each missing cell's donor scan split
    over IMPS_SPLIT threads (donor j to split j % IMPS_SPLIT, ascending
    within a split), then merged. The item's insertion (strict <, donors
    ascending) keeps exactly the k smallest donors by (distance, donor
    index), ties to the lower index; each split keeps that for its donors,
    and the merge takes the k smallest (distance, index) pairs of the
    splits' lists, which is the same list. Distances by the item's
    statements; n_donors is the sum of the splits' counts; the tail is
    `knn_impute_finish`. k <= IMPS_KMAX, d <= IMP_MAX_D."""
    var m = Int(m_)
    var d = Int(d_)
    var k = Int(k_)
    var nc = Int(nc_)
    var tid = Int(thread_idx.x)
    var ci = tid // IMPS_SPLIT
    var sp = tid - ci * IMPS_SPLIT
    var q0 = Int(block_idx.x) * IMPS_CELLS + ci
    var fs = stack_allocation[IMPS_ROWS * IMP_MAX_D, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var pd = stack_allocation[IMPS_TPB * IMPS_KMAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var pi = stack_allocation[IMPS_TPB * IMPS_KMAX, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var pn = stack_allocation[IMPS_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var live = q0 < nc
    var t = 0
    var r = 0
    var c = 0
    if live:
        t = Int(cells.unsafe_load(q0))
        r = t // d
        c = t - r * d
    var inf = _bc[DType.float32](UInt32(0x7F800000))
    var lb = tid * IMPS_KMAX
    for s in range(k):
        pd[lb + s] = inf
        pi[lb + s] = Int32(-1)
    var n_donors = 0
    var j0 = 0
    while j0 < m:
        var rows = min(IMPS_ROWS, m - j0)
        var q = tid
        while q < rows * d:
            fs[q] = fx.unsafe_load(j0 * d + q)
            q += IMPS_TPB
        barrier()
        if live:
            var jj = sp
            while jj < rows:
                var j = j0 + jj
                var dv = fs[jj * d + c]
                if dv == dv:
                    n_donors += 1
                    var acc = Float32(0)
                    var present = 0
                    for f in range(d):
                        var a = x.unsafe_load(r * d + f)
                        var b = fs[jj * d + f]
                        if a != a or b != b:
                            continue
                        present += 1
                        var df = _sub(a, b)
                        acc = ftz(identical_mul_add(df, df, acc))
                    if present != 0:
                        var sq = ftz(identical_mul(ftz(identical_div(acc, Float32(present))), Float32(d)))
                        var dist = ftz(identical_sqrt(sq))
                        if dist < pd[lb + k - 1]:
                            var s = k - 1
                            while s > 0 and dist < pd[lb + s - 1]:
                                pd[lb + s] = pd[lb + s - 1]
                                pi[lb + s] = pi[lb + s - 1]
                                s -= 1
                            pd[lb + s] = dist
                            pi[lb + s] = Int32(j)
                jj += IMPS_SPLIT
        barrier()
        j0 += rows
    pn[tid] = Int32(n_donors)
    barrier()
    if live and sp == 0:
        var bd = best_d + t * k
        var bi = best_i + t * k
        var total = 0
        var head = InlineArray[Int, IMPS_SPLIT](fill=0)
        for u in range(IMPS_SPLIT):
            total += Int(pn[tid + u])
        for s in range(k):
            var best_u = -1
            var best_d_v = inf
            var best_j = Int32(-1)
            for u in range(IMPS_SPLIT):
                var h = head[u]
                if h < k:
                    var ix = pi[(tid + u) * IMPS_KMAX + h]
                    if ix >= 0:
                        var dv2 = pd[(tid + u) * IMPS_KMAX + h]
                        if best_u < 0 or dv2 < best_d_v or (dv2 == best_d_v and ix < best_j):
                            best_u = u
                            best_d_v = dv2
                            best_j = ix
            if best_u < 0:
                bd.unsafe_store(s, inf)
                bi.unsafe_store(s, Int32(-1))
            else:
                bd.unsafe_store(s, best_d_v)
                bi.unsafe_store(s, best_j)
                head[best_u] = head[best_u] + 1
        knn_impute_finish(t, fx, bd, bi, res, m, d, k, Int(weights_), total)


# ============================================================================
# FUSED KERNEL CHAINS (lane/py-dn-kern, 2026-09-28). KernelPCA.transform,
# OneClassSVM.score_samples and SVGP's fit statistics and prediction used to
# be chains of `xn_*` calls, each uploading its inputs and downloading its
# output, so the nq x n_fit kernel matrix crossed the bus about four times.
# These drivers launch the SAME item kernels in the SAME order over row tiles
# that stay on the device, and download only the final output. Every item
# computes its cells from its own row, so a row tile changes no statement.
# The one carried fold (SVGP's Kuf Kfu and Kuf y) continues each cell's
# float32 accumulator from tile to tile: `matmul_tn_acc_item`.
# -D MOJOLEARN_XN_FUSED_SABOTAGE adds 1e-3 to the first output cell of each
# fused device driver (the new device path's negative control).
# ============================================================================

#: cells of the per-tile kernel matrix (a tile is at most this many floats)
comptime XN_FUSED_CELLS = 1 << 24
#: `kind` of `kpca_transform` when q IS the precomputed kernel (d == nf)
comptime XN_PRECOMPUTED_KIND = 100


@always_inline
def _p(mut b: DeviceBuffer[DType.float32]) -> FP:
    return b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _tile_rows(n: Int, width: Int) -> Int:
    var t = XN_FUSED_CELLS // max(width, 1)
    return max(1, min(n, t))


def matmul_tn_acc_kernel(a: FP, b: FP, res: FP, rows_: Int64, n_: Int64, m_: Int64):
    var t = _tid()
    if t < Int(n_) * Int(m_):
        matmul_tn_acc_item(t, a, b, res, Int(rows_), Int(n_), Int(m_))


def _fused_sabotage(ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], count: Int) raises:
    comptime if is_defined["MOJOLEARN_XN_FUSED_SABOTAGE"]():
        if count > 0:
            var h = ctx.enqueue_create_host_buffer[DType.float32](1)
            var sub = buf.create_sub_buffer[DType.float32](0, 1)
            ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=sub)
            ctx.synchronize()
            var hp = h.unsafe_ptr()
            hp[0] = hp[0] + Float32(1e-3)
            ctx.enqueue_copy(dst_buf=sub, src_ptr=h.unsafe_ptr())
            ctx.synchronize()
            _ = sub^
            _ = h^


def _launch_kernel(
    ctx: DeviceContext, q: FP, y: FP, res: FP, rows: Int, m: Int, d: Int, kind: Int, degree: Int,
    gamma: Float32, coef0: Float32,
) raises:
    var cells = rows * m
    ctx.enqueue_function[kernel_kernel](
        q, y, res, Int64(rows), Int64(m), Int64(d), Int64(kind), gamma, coef0, Int64(degree),
        grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
    )


def _launch_matmul(ctx: DeviceContext, a: FP, b: FP, res: FP, n: Int, k: Int, m: Int) raises:
    var cells = n * m
    ctx.enqueue_function[matmul_kernel](
        a, b, res, Int64(n), Int64(k), Int64(m),
        grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
    )


def op_kpca_transform(
    q: Int, fitx: Int, fit_cols: Int, fit_all: Int, alphas: Int, res: Int,
    nq: Int, nf: Int, d: Int, c: Int, kind: Int, degree: Int, gamma: Float32, coef0: Float32, s: Float32,
) raises:
    """KernelPCA.transform: K = kernel(q, fitx) (kind
    XN_PRECOMPUTED_KIND: q IS the precomputed K, d == nf), pred = rowsum(K) / s, Kc = kpca_center(K,
    fit_cols, pred, fit_all), res = Kc alphas; per row tile on the device."""
    var ctx = xn_ctx()
    var pre = kind == XN_PRECOMPUTED_KIND
    var d_q = _buf(ctx, q, nq * d, True)
    var d_fx = _buf(ctx, fitx, 0 if pre else nf * d, not pre)
    var d_cols = _buf(ctx, fit_cols, nf, True)
    var d_all = _buf(ctx, fit_all, 1, True)
    var d_al = _buf(ctx, alphas, nf * c, True)
    var d_res = _buf(ctx, 0, nq * c, False)
    var tr = _tile_rows(nq, nf)
    var d_k = _buf(ctx, 0, 0 if pre else tr * nf, False)
    var d_kc = _buf(ctx, 0, tr * nf, False)
    var d_rs = _buf(ctx, 0, tr, False)
    var d_pr = _buf(ctx, 0, tr, False)
    var qp: FP = _p(d_q)
    var kp: FP = _p(d_k)
    var rp: FP = _p(d_res)
    var r0 = 0
    while r0 < nq:
        var rows = min(tr, nq - r0)
        var K = qp + r0 * d
        if not pre:
            _launch_kernel(ctx, qp + r0 * d, _p(d_fx), kp, rows, nf, d, kind, degree, gamma, coef0)
            K = kp
        ctx.enqueue_function[rowsum_kernel](
            K, _p(d_rs), Int64(rows), Int64(nf),
            grid_dim=_grid(rows), block_dim=(BLOCK if rows > 1 else 1),
        )
        ctx.enqueue_function[scale_div_kernel](
            _p(d_rs), _p(d_pr), Int64(rows), s,
            grid_dim=_grid(rows), block_dim=(BLOCK if rows > 1 else 1),
        )
        var cells = rows * nf
        ctx.enqueue_function[kpca_center_kernel](
            K, _p(d_cols), _p(d_pr), _p(d_all), _p(d_kc), Int64(rows), Int64(nf),
            grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
        )
        _launch_matmul(ctx, _p(d_kc), _p(d_al), rp + r0 * c, rows, nf, c)
        r0 += rows
    _fused_sabotage(ctx, d_res, nq * c)
    _down(ctx, d_res, res, nq * c)
    ctx.synchronize()
    _ = d_q^
    _ = d_fx^
    _ = d_cols^
    _ = d_all^
    _ = d_al^
    _ = d_res^
    _ = d_k^
    _ = d_kc^
    _ = d_rs^
    _ = d_pr^
    _ = ctx^


def op_kernel_matmul(
    q: Int, y: Int, w: Int, res: Int,
    n: Int, m: Int, d: Int, c: Int, kind: Int, degree: Int, gamma: Float32, coef0: Float32,
) raises:
    """res = kernel(q, y) w (n x c): OneClassSVM.score_samples' two ops per
    row tile on the device."""
    var ctx = xn_ctx()
    var d_q = _buf(ctx, q, n * d, True)
    var d_y = _buf(ctx, y, m * d, True)
    var d_w = _buf(ctx, w, m * c, True)
    var d_res = _buf(ctx, 0, n * c, False)
    var tr = _tile_rows(n, m)
    var d_k = _buf(ctx, 0, tr * m, False)
    var qp: FP = _p(d_q)
    var rp: FP = _p(d_res)
    var r0 = 0
    while r0 < n:
        var rows = min(tr, n - r0)
        _launch_kernel(ctx, qp + r0 * d, _p(d_y), _p(d_k), rows, m, d, kind, degree, gamma, coef0)
        _launch_matmul(ctx, _p(d_k), _p(d_w), rp + r0 * c, rows, m, c)
        r0 += rows
    _fused_sabotage(ctx, d_res, n * c)
    _down(ctx, d_res, res, n * c)
    ctx.synchronize()
    _ = d_q^
    _ = d_y^
    _ = d_w^
    _ = d_res^
    _ = d_k^
    _ = ctx^


def _launch_scaled_rbf(
    ctx: DeviceContext, q: FP, z: FP, kbuf: FP, dst: FP, rows: Int, m: Int, d: Int, gamma: Float32, variance: Float32,
) raises:
    """SVGP's `_k`: the rbf kernel (coef0 0, degree 0), then
    `unary(K, identity, variance, 0)`."""
    _launch_kernel(ctx, q, z, kbuf, rows, m, d, K_RBF, 0, gamma, Float32(0))
    var cells = rows * m
    ctx.enqueue_function[unary_kernel](
        kbuf, dst, Int64(cells), Int64(U_IDENTITY), variance, Float32(0),
        grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
    )


def op_svgp_stats(
    x: Int, z: Int, y: Int, bmat: Int, bvec: Int, n: Int, m: Int, d: Int, gamma: Float32, variance: Float32,
) raises:
    """SVGP.fit's B = Kuf Kfu (m x m) and b = Kuf y (m): Kfu = variance *
    rbf(x, z) per row tile on the device, both products carried over the
    tiles (`matmul_tn_acc_item`); Kuf is never formed (it is Kfu^T bit for
    bit: the rbf item squares x - z, and IEEE subtraction is antisymmetric)."""
    var ctx = xn_ctx()
    var d_x = _buf(ctx, x, n * d, True)
    var d_z = _buf(ctx, z, m * d, True)
    var d_y = _buf(ctx, y, n, True)
    var d_b = _buf(ctx, 0, m * m, False)
    var d_bv = _buf(ctx, 0, m, False)
    enqueue_fill(ctx, d_b, Float32(0))
    enqueue_fill(ctx, d_bv, Float32(0))
    var tr = _tile_rows(n, m)
    var d_k = _buf(ctx, 0, tr * m, False)
    var d_ks = _buf(ctx, 0, tr * m, False)
    var xp: FP = _p(d_x)
    var yp: FP = _p(d_y)
    var r0 = 0
    while r0 < n:
        var rows = min(tr, n - r0)
        _launch_scaled_rbf(ctx, xp + r0 * d, _p(d_z), _p(d_k), _p(d_ks), rows, m, d, gamma, variance)
        ctx.enqueue_function[matmul_tn_acc_kernel](
            _p(d_ks), _p(d_ks), _p(d_b), Int64(rows), Int64(m), Int64(m),
            grid_dim=_grid(m * m), block_dim=(BLOCK if m * m > 1 else 1),
        )
        ctx.enqueue_function[matmul_tn_acc_kernel](
            _p(d_ks), yp + r0, _p(d_bv), Int64(rows), Int64(m), Int64(1),
            grid_dim=_grid(m), block_dim=(BLOCK if m > 1 else 1),
        )
        r0 += rows
    _fused_sabotage(ctx, d_b, m * m)
    _down(ctx, d_b, bmat, m * m)
    _down(ctx, d_bv, bvec, m)
    ctx.synchronize()
    _ = d_x^
    _ = d_z^
    _ = d_y^
    _ = d_b^
    _ = d_bv^
    _ = d_k^
    _ = d_ks^
    _ = ctx^


def op_svgp_predict(
    q: Int, z: Int, alpha: Int, cmat: Int, mean: Int, var_: Int,
    n: Int, m: Int, d: Int, gamma: Float32, variance: Float32, kdiag: Float32,
) raises:
    """SVGP.predict_f: Ksu = variance * rbf(q, z), mean = Ksu alpha,
    var = svgp_var(Ksu, C), per row tile on the device."""
    var ctx = xn_ctx()
    var d_q = _buf(ctx, q, n * d, True)
    var d_z = _buf(ctx, z, m * d, True)
    var d_al = _buf(ctx, alpha, m, True)
    var d_c = _buf(ctx, cmat, m * m, True)
    var d_mean = _buf(ctx, 0, n, False)
    var d_var = _buf(ctx, 0, n, False)
    var tr = _tile_rows(n, m)
    var d_k = _buf(ctx, 0, tr * m, False)
    var d_ks = _buf(ctx, 0, tr * m, False)
    var qp: FP = _p(d_q)
    var mp: FP = _p(d_mean)
    var vp: FP = _p(d_var)
    var r0 = 0
    while r0 < n:
        var rows = min(tr, n - r0)
        _launch_scaled_rbf(ctx, qp + r0 * d, _p(d_z), _p(d_k), _p(d_ks), rows, m, d, gamma, variance)
        _launch_matmul(ctx, _p(d_ks), _p(d_al), mp + r0, rows, m, 1)
        ctx.enqueue_function[svgp_var_kernel](
            _p(d_ks), _p(d_c), vp + r0, Int64(rows), Int64(m), kdiag,
            grid_dim=_grid(rows), block_dim=(BLOCK if rows > 1 else 1),
        )
        r0 += rows
    _fused_sabotage(ctx, d_mean, n)
    _down(ctx, d_mean, mean, n)
    _down(ctx, d_var, var_, n)
    ctx.synchronize()
    _ = d_q^
    _ = d_z^
    _ = d_al^
    _ = d_c^
    _ = d_mean^
    _ = d_var^
    _ = d_k^
    _ = d_ks^
    _ = ctx^


from x_neighbors.lp_knn import op_lp_knn_graph, lp_knn_product_item, lp_knn_finite


def lp_knn_product_kernel(cols: IP, vals: FP, x: FP, res: FP, n_: Int64, m_: Int64, k_: Int64, c_: Int64, finite_: Int64):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n_) * Int(c_):
        lp_knn_product_item(t, cols, vals, x, res, Int(n_), Int(m_), Int(k_), Int(c_), finite_ != Int64(0))


def op_lp_knn_product(cols: Int, vals: Int, x: Int, res: Int, n: Int, m: Int, k: Int, c: Int) raises:
    var finite = lp_knn_finite(FP(unsafe_from_address=x), m * c)
    var ctx = xn_ctx()
    var dc = _buf_i(ctx, cols, n * k, True)
    var dv = _buf(ctx, vals, n * k, True)
    var dx = _buf(ctx, x, m * c, True)
    var dr = _buf(ctx, 0, n * c, False)
    if n * c > 0:
        ctx.enqueue_function[lp_knn_product_kernel](dc.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), _p(dv), _p(dx), _p(dr),
            Int64(n), Int64(m), Int64(k), Int64(c), Int64(1 if finite else 0),
            grid_dim=_grid(n * c), block_dim=(BLOCK if n * c > 1 else 1))
    _down(ctx, dr, res, n * c)
    ctx.synchronize()


# lane/neural-pass95 follow-up (2026-10-02): NearestCentroid's statistics
# with the rows staged. nc_stats_item ran one thread per feature through
# every class's mean chain, the std chain and the centroid chain, each a
# million rows read at a stride of d words (MI325X taxi 1M x 11: 351 ms on
# 11 threads; main's nc_shrink 177). Here a block takes 16 features and
# stages NCS_TR rows of them (and the labels) in threadgroup memory; every
# (class, feature) mean chain and the feature's centroid chain is its own
# thread (`nc_means_kernel`), then the std chains (`nc_std_kernel`, which
# needs the means). Each chain keeps nc_stats_item's statements and rows
# ascending: the same words.
comptime NCS_TR = 128
comptime NCS_TC = 16
comptime NCS_NT = 256
comptime NCS_RU = 16


@always_inline
def _ncs_stage(x: FP, lab: IP, n: Int, d: Int, c0: Int, r0: Int, xs: UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED], ls: UnsafePointer[Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]):
    var tid = Int(thread_idx.x)
    var cnt = min(NCS_TR, n - r0)
    for u in range(tid, NCS_TR * NCS_TC, NCS_NT):
        var r = u // NCS_TC
        var c = c0 + u % NCS_TC
        var v = Float32(0)
        if r < cnt and c < d:
            v = x.unsafe_load((r0 + r) * d + c)
        xs[u] = v
    for u in range(tid, NCS_TR, NCS_NT):
        ls[u] = lab.unsafe_load(r0 + u) if u < cnt else Int32(-1)


def nc_means_kernel(x: FP, lab: IP, cent: FP, dsc: FP, n_: Int64, d_: Int64, nc_: Int64):
    """Block b: features [16b, 16b + 16). Chain q = g * 16 + c (g < classes)
    is class g's mean of feature 16b + c; chain classes * 16 + c is the
    feature's centroid chain."""
    var n = Int(n_)
    var d = Int(d_)
    var ncl = Int(nc_)
    var tid = Int(thread_idx.x)
    var c0 = Int(block_idx.x) * NCS_TC
    var xs = stack_allocation[NCS_TR * NCS_TC, Float32, address_space=AddressSpace.SHARED]()
    var ls = stack_allocation[NCS_TR, Int32, address_space=AddressSpace.SHARED]()
    var chains = (ncl + 1) * NCS_TC
    # each thread runs chains tid, tid + NT, ... (at most a few)
    comptime MAXC = 8
    var acc = SIMD[DType.float32, MAXC](0)
    var cnt = SIMD[DType.int32, MAXC](0)
    var r0 = 0
    while r0 < n:
        barrier()
        _ncs_stage(x, lab, n, d, c0, r0, xs, ls)
        barrier()
        var rows = min(NCS_TR, n - r0)
        comptime for k in range(MAXC):
            var q = tid + k * NCS_NT
            if q < chains:
                var g = q // NCS_TC
                var c = q % NCS_TC
                if c0 + c < d:
                    var a = acc[k]
                    var m = cnt[k]
                    var r = 0
                    while r + NCS_RU <= rows:
                        var bv = SIMD[DType.float32, NCS_RU]()
                        var bl = SIMD[DType.int32, NCS_RU]()
                        comptime for u in range(NCS_RU):
                            bv[u] = xs[(r + u) * NCS_TC + c]
                            bl[u] = ls[r + u]
                        comptime for u in range(NCS_RU):
                            if g >= ncl or Int(bl[u]) == g:
                                a = _add(a, bv[u])
                                if g < ncl:
                                    m += 1
                        r += NCS_RU
                    while r < rows:
                        if g >= ncl or Int(ls[r]) == g:
                            a = _add(a, xs[r * NCS_TC + c])
                            if g < ncl:
                                m += 1
                        r += 1
                    acc[k] = a
                    cnt[k] = m
        r0 += NCS_TR
    comptime for k in range(MAXC):
        var q = tid + k * NCS_NT
        if q < chains:
            var g = q // NCS_TC
            var c = q % NCS_TC
            var f = c0 + c
            if f < d:
                if g < ncl:
                    if cnt[k] == 0:
                        cent.unsafe_store(g * d + f, Float32(0))
                    else:
                        cent.unsafe_store(g * d + f, ftz(identical_div(acc[k], Float32(Int(cnt[k])))))
                else:
                    dsc.unsafe_store(f, ftz(identical_div(acc[k], Float32(n))))


def nc_std_kernel(x: FP, lab: IP, cent: FP, std: FP, n_: Int64, d_: Int64, nc_: Int64):
    """Block b: features [16b, 16b + 16), thread c < 16 the std chain."""
    var n = Int(n_)
    var d = Int(d_)
    var ncl = Int(nc_)
    var tid = Int(thread_idx.x)
    var c0 = Int(block_idx.x) * NCS_TC
    var xs = stack_allocation[NCS_TR * NCS_TC, Float32, address_space=AddressSpace.SHARED]()
    var ls = stack_allocation[NCS_TR, Int32, address_space=AddressSpace.SHARED]()
    var f = c0 + tid
    var live = tid < NCS_TC and f < d
    var ss = Float32(0)
    var r0 = 0
    while r0 < n:
        barrier()
        _ncs_stage(x, lab, n, d, c0, r0, xs, ls)
        barrier()
        if live:
            var rows = min(NCS_TR, n - r0)
            var r = 0
            while r + NCS_RU <= rows:
                var bv = SIMD[DType.float32, NCS_RU]()
                var bc = SIMD[DType.float32, NCS_RU]()
                comptime for u in range(NCS_RU):
                    bv[u] = xs[(r + u) * NCS_TC + tid]
                    bc[u] = cent.unsafe_load(Int(ls[r + u]) * d + f)
                comptime for u in range(NCS_RU):
                    var df = _sub(bv[u], bc[u])
                    ss = ftz(identical_mul_add(df, df, ss))
                r += NCS_RU
            while r < rows:
                var df = _sub(xs[r * NCS_TC + tid], cent.unsafe_load(Int(ls[r]) * d + f))
                ss = ftz(identical_mul_add(df, df, ss))
                r += 1
        r0 += NCS_TR
    if live:
        if n - ncl <= 0:
            std.unsafe_store(f, Float32(0))
        else:
            std.unsafe_store(f, ftz(identical_sqrt(ftz(identical_div(ss, Float32(n - ncl))))))


#: FAST on Apple (lane/apple-fast-classical, 2026-10-02): the staged chains
#: above split over row chunks as well as feature tiles. `nc_means_kernel`
#: and `nc_std_kernel` run one block per 16 features (14 blocks at Istella's
#: 220), each walking every row; here a block is (16 features, NCC_ROWS rows)
#: and a second launch sums the chunk partials. FAST's words move (the sums
#: are chunked); `MOJOLEARN_XN_NC_CHUNKS=0` is the A/B arm.
comptime XN_NC_CHUNKED = GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
comptime NCC_ROWS = 16384


def nc_means_part_kernel(x: FP, lab: IP, psum: FP, pcnt: FP, n_: Int64, d_: Int64, nc_: Int64, tiles_: Int64):
    """Block b: features [16 t, 16 t + 16) (t = b % tiles) over chunk
    b // tiles; chain q = g * 16 + c as `nc_means_kernel`'s, its sum (and
    count) into the chunk's partials."""
    var n = Int(n_)
    var d = Int(d_)
    var ncl = Int(nc_)
    var tiles = Int(tiles_)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var c0 = (b % tiles) * NCS_TC
    var ch = b // tiles
    var lo = ch * NCC_ROWS
    var hi = min(n, lo + NCC_ROWS)
    var xs = stack_allocation[NCS_TR * NCS_TC, Float32, address_space=AddressSpace.SHARED]()
    var ls = stack_allocation[NCS_TR, Int32, address_space=AddressSpace.SHARED]()
    var chains = (ncl + 1) * NCS_TC
    comptime MAXC = 8
    var acc = SIMD[DType.float32, MAXC](0)
    var cnt = SIMD[DType.int32, MAXC](0)
    var r0 = lo
    while r0 < hi:
        barrier()
        _ncs_stage(x, lab, n, d, c0, r0, xs, ls)
        barrier()
        var rows = min(NCS_TR, hi - r0)
        comptime for k in range(MAXC):
            var q = tid + k * NCS_NT
            if q < chains:
                var g = q // NCS_TC
                var c = q % NCS_TC
                if c0 + c < d:
                    var a = acc[k]
                    var m = cnt[k]
                    for r in range(rows):
                        if g >= ncl or Int(ls[r]) == g:
                            a += xs[r * NCS_TC + c]
                            m += 1
                    acc[k] = a
                    cnt[k] = m
        r0 += NCS_TR
    comptime for k in range(MAXC):
        var q = tid + k * NCS_NT
        if q < chains:
            var g = q // NCS_TC
            var c = q % NCS_TC
            var f = c0 + c
            if f < d:
                psum.unsafe_store((ch * (ncl + 1) + g) * d + f, acc[k])
                pcnt.unsafe_store((ch * (ncl + 1) + g) * d + f, Float32(Int(cnt[k])))


def nc_means_red_kernel(psum: FP, pcnt: FP, cent: FP, dsc: FP, n_: Int64, d_: Int64, nc_: Int64, nch_: Int64):
    """Thread (g, f): the chunk partials summed; class g's mean of feature f
    (g == classes: the centroid of every row)."""
    var d = Int(d_)
    var ncl = Int(nc_)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < (ncl + 1) * d:
        var g = t // d
        var f = t - g * d
        var a = Float32(0)
        var m = Float32(0)
        for ch in range(Int(nch_)):
            a += psum.unsafe_load((ch * (ncl + 1) + g) * d + f)
            m += pcnt.unsafe_load((ch * (ncl + 1) + g) * d + f)
        if g < ncl:
            cent.unsafe_store(g * d + f, a / m if m > 0 else Float32(0))
        else:
            dsc.unsafe_store(f, a / Float32(Int(n_)))


def nc_std_part_kernel(x: FP, lab: IP, cent: FP, pss: FP, n_: Int64, d_: Int64, tiles_: Int64):
    """Block (16 features, a chunk): thread c < 16 the chunk's squared
    deviations of feature 16 t + c from its class mean."""
    var n = Int(n_)
    var d = Int(d_)
    var tiles = Int(tiles_)
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var c0 = (b % tiles) * NCS_TC
    var ch = b // tiles
    var lo = ch * NCC_ROWS
    var hi = min(n, lo + NCC_ROWS)
    var xs = stack_allocation[NCS_TR * NCS_TC, Float32, address_space=AddressSpace.SHARED]()
    var ls = stack_allocation[NCS_TR, Int32, address_space=AddressSpace.SHARED]()
    var f = c0 + tid
    var live = tid < NCS_TC and f < d
    var ss = Float32(0)
    var r0 = lo
    while r0 < hi:
        barrier()
        _ncs_stage(x, lab, n, d, c0, r0, xs, ls)
        barrier()
        if live:
            var rows = min(NCS_TR, hi - r0)
            for r in range(rows):
                var df = xs[r * NCS_TC + tid] - cent.unsafe_load(Int(ls[r]) * d + f)
                ss += df * df
        r0 += NCS_TR
    if live:
        pss.unsafe_store(ch * d + f, ss)


def nc_std_red_kernel(pss: FP, std: FP, n_: Int64, d_: Int64, nc_: Int64, nch_: Int64):
    var d = Int(d_)
    var f = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if f < d:
        var ss = Float32(0)
        for ch in range(Int(nch_)):
            ss += pss.unsafe_load(ch * d + f)
        var dof = Int(n_) - Int(nc_)
        std.unsafe_store(f, sqrt(ss / Float32(dof)) if dof > 0 else Float32(0))


def _nc_chunked_on() -> Bool:
    return String(getenv("MOJOLEARN_XN_NC_CHUNKS")) != "0"


def op_nc_stats(x: Int, lab: Int, nk: Int, cent: Int, std: Int, dsc: Int, n: Int, d: Int, n_classes: Int) raises:
    if (n_classes + 1) * NCS_TC > 8 * NCS_NT:
        raise Error("nc_stats: more classes than the staged kernel's chains (" + String(n_classes) + ")")
    var ctx = xn_ctx()
    var d_x = _buf(ctx, x, n * d, True)
    var d_lab = _buf_i(ctx, lab, n, True)
    var d_cent = _buf(ctx, cent, n_classes * d, False)
    var d_std = _buf(ctx, std, d, False)
    var d_dsc = _buf(ctx, dsc, d, False)
    var tiles = (d + NCS_TC - 1) // NCS_TC
    comptime if XN_NC_CHUNKED:
        if _nc_chunked_on() and n > NCC_ROWS:
            var nch = (n + NCC_ROWS - 1) // NCC_ROWS
            var cells = (n_classes + 1) * d
            var d_ps = ctx.enqueue_create_buffer[DType.float32](nch * cells)
            var d_pc = ctx.enqueue_create_buffer[DType.float32](nch * cells)
            var d_pss = ctx.enqueue_create_buffer[DType.float32](nch * d)
            ctx.enqueue_function[nc_means_part_kernel](
                d_x.unsafe_ptr(), d_lab.unsafe_ptr(), d_ps.unsafe_ptr(), d_pc.unsafe_ptr(),
                Int64(n), Int64(d), Int64(n_classes), Int64(tiles), grid_dim=tiles * nch, block_dim=NCS_NT,
            )
            ctx.enqueue_function[nc_means_red_kernel](
                d_ps.unsafe_ptr(), d_pc.unsafe_ptr(), d_cent.unsafe_ptr(), d_dsc.unsafe_ptr(),
                Int64(n), Int64(d), Int64(n_classes), Int64(nch), grid_dim=(cells + 255) // 256, block_dim=256,
            )
            ctx.enqueue_function[nc_std_part_kernel](
                d_x.unsafe_ptr(), d_lab.unsafe_ptr(), d_cent.unsafe_ptr(), d_pss.unsafe_ptr(),
                Int64(n), Int64(d), Int64(tiles), grid_dim=tiles * nch, block_dim=NCS_NT,
            )
            ctx.enqueue_function[nc_std_red_kernel](
                d_pss.unsafe_ptr(), d_std.unsafe_ptr(), Int64(n), Int64(d), Int64(n_classes), Int64(nch),
                grid_dim=(d + 255) // 256, block_dim=256,
            )
            _down(ctx, d_cent, cent, n_classes * d)
            _down(ctx, d_std, std, d)
            _down(ctx, d_dsc, dsc, d)
            ctx.synchronize()
            _ = d_ps^
            _ = d_pc^
            _ = d_pss^
            _ = d_x^
            _ = d_lab^
            _ = d_cent^
            _ = d_std^
            _ = d_dsc^
            return
    ctx.enqueue_function[nc_means_kernel](
        d_x.unsafe_ptr(), d_lab.unsafe_ptr(), d_cent.unsafe_ptr(), d_dsc.unsafe_ptr(),
        Int64(n), Int64(d), Int64(n_classes), grid_dim=max(tiles, 1), block_dim=NCS_NT,
    )
    ctx.enqueue_function[nc_std_kernel](
        d_x.unsafe_ptr(), d_lab.unsafe_ptr(), d_cent.unsafe_ptr(), d_std.unsafe_ptr(),
        Int64(n), Int64(d), Int64(n_classes), grid_dim=max(tiles, 1), block_dim=NCS_NT,
    )
    _down(ctx, d_cent, cent, n_classes * d)
    _down(ctx, d_std, std, d)
    _down(ctx, d_dsc, dsc, d)
    ctx.synchronize()
    _ = d_x^
    _ = d_lab^
    _ = d_cent^
    _ = d_std^
    _ = d_dsc^
