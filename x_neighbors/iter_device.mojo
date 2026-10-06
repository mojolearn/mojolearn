from experiments.classical_identical_ideas.graph_controls import C43_RESIDENT_NORMALIZATION
from x_neighbors.classical_graph import classical_graph_degree, classical_graph_product
from experiments.classical_identical_ideas.stats_controls import C54_PREDICT_TILES
from experiments.classical_identical_ideas.graph_controls import C33_FROZEN_CHUNKS, C33_CHUNK
from x_neighbors.items import matmul_item
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Resident GPU drivers of the lane's iterations (lane neighbors-apple2).

`op_lp_iterate`: LabelPropagation / LabelSpreading's fit loop
(`_LabelPropagationBase.fit` in python/mojolearn/_expansion_neighbors.py) on
the device with the graph uploaded ONCE. The Python loop called three ops per
iteration, and each op uploaded its inputs: the n x n graph crossed to the
device on every iteration. Here the same kernels (`absdiff_sum_k0` / `_k1`,
`matmul_kernel`, `lp_clamp_kernel` / `ls_clamp_kernel`, the generated
one-item-per-thread drivers of the same items) run in the same order on the
same values; only the stopping sum crosses back, one float per iteration,
compared in double exactly as Python compared it. The CPU column runs the
same loop over the items (`x_neighbors/iter_host.mojo`).
"""
from checks.kernel_matrix import lib_smem_page_fits_for, TARGET_COLUMN
from x_neighbors.svgp_ff import (
    matmul_tn_acc_ff_item, svgp_ff_init_item, svgp_ff_chol_item, svgp_ff_chol_s_item, svgp_ff_column_item,
    svgp_ff_x_item, svgp_ff_qmu_item, svgp_ff_qsqrt_item, svgp_ff_part_item, svgp_ff_fin_item, svgp_ff_nbn,
    svgp_ff_nbm, svgp_ff_ws_size, matmul_tn_sym_ff_tile_item, svgp_sym_nb,
    svgp_ff_col_solve_item, svgp_ff_col_fin_item, svgp_ff_mat,
)
from x_neighbors.svgp_ff import (
    SVGP_TREE, SVGP_TREE_ON, SV8, svgp_ff_tree_comb, svgp_ff_tree_slot, svgp_ff_bound, svgp_ff_failed,
)
from x_linear.ff import FF
from x_linear.ff import ff_of, ff_add, ff_sub, ff_mul, ff_div, ff_sqrt, ff_ld, ff_st
from std.memory import bitcast, memcpy
from std.atomic import Atomic
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
from std.python import PythonObject
from x_neighbors.nan_cells_device import (
    nan_cells_device, nan_group_sum_kernel, nan_top_scan_kernel, nan_down_scan_kernel,
    NC_TPB, NC_PER, NC_CHUNK, NC_SMEM_FITS,
)
from x_neighbors.graph_dev import pr_iterate_gpu
from x_neighbors.items import FP, IP, absdiff_sum_item, xn_fold_blocks, _sub, _add, knn_sq_item, knn_impute_finish
from x_neighbors.items import XN_TREE, XN_TREE_ON, xn_tree_slot
from checks.numerics import identical_mul, identical_div, identical_sqrt
from std.memory import bitcast as _bc
from x_neighbors.device_ops import (
    xn_ctx, _buf, _buf_i, _down, _down_i, _grid, _tid, BLOCK,
    absdiff_sum_k0, absdiff_sum_k1, matmul_kernel, lp_clamp_kernel, ls_clamp_kernel,
    pagerank_step_kernel, cc_step_kernel,
    pcs_sketch_kernel, pcs_conv_kernel, pcs_copy0_kernel, op_knn_sq, op_knn_impute_cells,
    kernel_kernel, rowsum_kernel, scale_div_kernel, kpca_center_kernel, unary_kernel, svgp_var_kernel,
)
from x_neighbors.items import K_RBF, U_IDENTITY
from x_neighbors.lp_spmm import lp_ell_fill_kernel, lp_nonfinite_kernel, lp_prod_kernel, lp_rowcount_kernel
from core.device_zero import enqueue_fill
from core.pinned_reduce import pinned_block_sum
from checks.numerics import NUMERIC_IDENTICAL
from checks.numerics import identical_exp
from x_neighbors.items import lp_clamp_item, ls_clamp_item
from x_neighbors.items import absdiff_part_item, absdiff_fin_item


def _absdiff_launch(ctx: DeviceContext, a: FP, b: FP, s: FP, part: FP, count: Int) raises:
    """The `absdiff_sum` op's two stages on resident buffers: one thread per
    XN_FOLD_BLOCK block into `part`, then the partials folded into s[0]
    (the bits of `absdiff_sum_item`, the host column's fold)."""
    var nb = xn_fold_blocks(count)
    ctx.enqueue_function[absdiff_sum_k0](
        a, b, s, part, Int64(count), grid_dim=_grid(nb), block_dim=(BLOCK if nb > 1 else 1),
    )
    comptime if XN_TREE_ON:
        # lane apple-fast-purity: the partials folded by one block of XN_TREE
        # threads (`xn_tree_fold`'s steps), not by one thread
        ctx.enqueue_function[xn_tree_fold_kernel](part, s, Int64(nb), grid_dim=1, block_dim=XN_TREE)
    else:
        ctx.enqueue_function[absdiff_sum_k1](a, b, s, part, Int64(count), grid_dim=1, block_dim=1)


comptime _XN_TREE_SMEM_FITS = lib_smem_page_fits_for[TARGET_COLUMN, XN_TREE * 4]()


def xn_tree_fold_kernel(part: FP, res: FP, nb_: Int64):
    """ONE block of XN_TREE threads: `xn_tree_fold` (items.mojo) with thread
    s on slot s, then the halving steps in threadgroup memory; res[0]."""
    comptime assert _XN_TREE_SMEM_FITS, "xn_tree_fold_kernel: a 1 KB threadgroup page must fit"
    var s = Int(thread_idx.x)
    var sh = stack_allocation[XN_TREE, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    sh[s] = xn_tree_slot(part, s, Int(nb_))
    barrier()
    var h = XN_TREE // 2
    while h > 0:
        if s < h:
            sh[s] = _add(sh[s], sh[s + h])
        barrier()
        h //= 2
    if s == 0:
        res.unsafe_store(0, sh[0])


def _c33_stop(sum: FP, state: IP, threshold: Float32, iteration: Int32):
    # First convergence wins, including which ping-pong buffer owns it.
    if state[0] == 0 and sum[0] < threshold:
        state[0] = 1
        state[1] = iteration
        state[2] = iteration % 2


def _c33_product(state: IP, graph: FP, cur: FP, nxt: FP, n: Int32, c: Int32, degree: FP, raw_graph: Bool, variant: Int32):
    var t = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if state[0] == 0 and t < Int(n)*Int(c):
        if raw_graph:
            classical_graph_product(t,graph,degree,cur,nxt,Int(n),Int(c),Int(variant))
        else:
            matmul_item(t,graph,cur,nxt,Int(n),Int(n),Int(c))


def _c33_clamp(state: IP, nxt: FP, ys: FP, unl: IP, out: FP, n: Int32, c: Int32, variant: Int32, alpha: Float32):
    var t = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if state[0] != 0:
        return
    if variant == 0:
        if t < Int(n):
            lp_clamp_item(t,nxt,ys,unl,out,Int(n),Int(c))
    elif t < Int(n)*Int(c):
        ls_clamp_item(t,nxt,ys,out,Int(n)*Int(c),alpha)


def _c43_graph_degree(a: FP,degree: FP,n: Int32,variant: Int32):
    var t=_tid()
    if t < Int(n):
        classical_graph_degree(t,a,degree,Int(n),Int(variant))


def _c43_graph_product(a: FP,degree: FP,x: FP,dst: FP,n: Int32,c: Int32,variant: Int32):
    var t=_tid()
    if t < Int(n)*Int(c):
        classical_graph_product(t,a,degree,x,dst,Int(n),Int(c),Int(variant))


def op_lp_iterate(
    g: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, c: Int, max_iter: Int, variant: Int, tol_hi: Int, tol_lo: Int,
    alpha: Float32,
) raises:
    """variant 0 propagation, 1 spreading; C43 adds 2 for raw affinity.
    `ld` in:
    the initial label distributions, out: the last. info (int32 x 2): the
    fit's n_iter_ and converged.

    Everything on the device (lane/cgr-kernel): the stopping sum is the
    blocked `absdiff_sum` fold (`absdiff_sum_item`'s bits) with only the
    scalar read back; G's nonzeros are found on the device, once, and the
    product runs over them while the distributions are finite
    (`lp_prod_kernel`, the dense product's bits). Before, the host scanned
    G for its CSR and downloaded the distributions every iteration for the
    stopping sum and the finiteness test (and `-D MOJOLEARN_XN_LP_BATCH`,
    `_DENSE` and `_DEVICE_FOLD` chose other routes; they are gone)."""
    var tol = bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo))
    var raw_graph = C43_RESIDENT_NORMALIZATION and variant >= 2
    var clamp_variant = variant % 2
    var nc = n * c
    var ctx = xn_ctx()
    var d_g = _buf(ctx, g, n * n, True)
    var d_degree = ctx.enqueue_create_buffer[DType.float32](max(n,1) if raw_graph else 1)
    if raw_graph and n > 0:
        ctx.enqueue_function[_c43_graph_degree](_p(d_g),_p(d_degree),Int32(n),Int32(clamp_variant),grid_dim=_grid(n),block_dim=BLOCK)
    var d_a = _buf(ctx, ld, nc, True)
    var d_b = _buf(ctx, 0, nc, False)
    ctx.enqueue_memset(d_b, Float32(0))
    var d_nxt = _buf(ctx, 0, nc, False)
    var d_ys = _buf(ctx, ystatic, nc, True)
    var d_unl = _buf_i(ctx, unlabeled, n, True)
    var d_s = _buf(ctx, 0, 1, False)
    var d_part = ctx.enqueue_create_buffer[DType.float32](max(xn_fold_blocks(nc), 1))
    var hs = List[Float32](length=1, fill=Float32(0))
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var prev: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur_is_a = True
    var n_iter = 0
    var converged = False
    # G's nonzeros per row (sparse only when it pays: under an eighth
    # nonzero), counted and laid out on the device; two integers come back
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_stats = ctx.enqueue_create_buffer[DType.int32](2)
    var d_flag = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(d_stats, Int32(0))
    if n > 0:
        ctx.enqueue_function[lp_rowcount_kernel](
            d_g.unsafe_ptr(), Int64(n), d_cnt.unsafe_ptr(), d_stats.unsafe_ptr(),
            grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
        )
    var hst = List[Int32](length=2, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=d_stats)
    ctx.synchronize()
    var nnz = Int(hst[0])
    var maxk = Int(hst[1])
    var use_sparse = not raw_graph and nnz > 0 and nnz * 8 < n * n
    var ell = n * maxk if use_sparse else 1
    var d_cols = ctx.enqueue_create_buffer[DType.int32](max(ell, 1))
    var d_vals = ctx.enqueue_create_buffer[DType.float32](max(ell, 1))
    if use_sparse:
        ctx.enqueue_function[lp_ell_fill_kernel](
            d_g.unsafe_ptr(), Int64(n), Int64(maxk), d_cols.unsafe_ptr(), d_vals.unsafe_ptr(),
            grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
        )
    comptime if C33_FROZEN_CHUNKS:
        # Upward Float32 threshold gives exactly Float64(sum) < tol.
        var threshold = Float32(tol)
        if Float64(threshold) < tol:
            threshold = bitcast[DType.float32](bitcast[DType.uint32](threshold) + UInt32(1))
        var state = ctx.enqueue_create_buffer[DType.int32](3)
        ctx.enqueue_memset(state,Int32(0))
        var status = List[Int32](length=3,fill=Int32(0))
        for chunk in range(0,max_iter,C33_CHUNK):
            for it in range(chunk,min(chunk+C33_CHUNK,max_iter)):
                n_iter = it
                _absdiff_launch(ctx,cur,prev,_p(d_s),_p(d_part),nc)
                ctx.enqueue_function[_c33_stop](_p(d_s),state.unsafe_ptr(),threshold,Int32(it),grid_dim=1,block_dim=1)
                ctx.enqueue_function[_c33_product](state.unsafe_ptr(),_p(d_g),cur,_p(d_nxt),Int32(n),Int32(c),_p(d_degree),raw_graph,Int32(clamp_variant),grid_dim=_grid(nc),block_dim=BLOCK)
                ctx.enqueue_function[_c33_clamp](state.unsafe_ptr(),_p(d_nxt),_p(d_ys),d_unl.unsafe_ptr(),prev,Int32(n),Int32(c),Int32(clamp_variant),alpha,grid_dim=_grid(nc),block_dim=BLOCK)
                var tmp = cur
                cur = prev
                prev = tmp
                cur_is_a = not cur_is_a
            ctx.enqueue_copy(dst_ptr=status.unsafe_ptr(),src_buf=state)
            ctx.synchronize()
            if status[0] != 0:
                converged = True
                n_iter = Int(status[1])
                cur_is_a = status[2] == 0
                break
        _ = state^
    else:
        for it in range(max_iter):
            n_iter = it
            _absdiff_launch(ctx, cur, prev, _p(d_s), _p(d_part), nc)
            ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=d_s)
            ctx.synchronize()
            if Float64(hs[0]) < tol:
                converged = True
                break
            if raw_graph:
                ctx.enqueue_function[_c43_graph_product](_p(d_g),_p(d_degree),cur,_p(d_nxt),Int32(n),Int32(c),Int32(clamp_variant),grid_dim=_grid(nc),block_dim=BLOCK)
            elif use_sparse:
                ctx.enqueue_memset(d_flag, Int32(0))
                ctx.enqueue_function[lp_nonfinite_kernel](
                    cur, Int64(nc), d_flag.unsafe_ptr(), grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
                )
                ctx.enqueue_function[lp_prod_kernel](
                    d_flag.unsafe_ptr(), d_cnt.unsafe_ptr(), d_cols.unsafe_ptr(), d_vals.unsafe_ptr(), Int64(maxk),
                    d_g.unsafe_ptr(), cur, d_nxt.unsafe_ptr(), Int64(n), Int64(c),
                    grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
                )
            else:
                ctx.enqueue_function[matmul_kernel](
                    d_g.unsafe_ptr(), cur, d_nxt.unsafe_ptr(), Int64(n), Int64(n), Int64(c),
                    grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
                )
            # prev = ld; ld = clamp(nxt): the clamp writes over the buffer the
            # old prev held, then the two names swap.
            if clamp_variant == 0:
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
    _ = hst^
    _ = d_cnt^
    _ = d_stats^
    _ = d_flag^
    _ = d_cols^
    _ = d_vals^
    _ = d_g^
    _ = d_degree^
    _ = d_a^
    _ = d_b^
    _ = d_nxt^
    _ = d_ys^
    _ = d_unl^
    _ = d_s^
    _ = d_part^
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
    var d_part = ctx.enqueue_create_buffer[DType.float32](max(xn_fold_blocks(n), 1))
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
        _absdiff_launch(ctx, nxt, cur, _p(d_s), _p(d_part), n)
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
    _ = d_part^
    _ = ctx^



def op_pr_iterate_sparse(
    a: Int, x: Int, p: Int, dw: Int, info: Int,
    n: Int, max_iter: Int, thr_hi: Int, thr_lo: Int, binary: Int, alpha: Float32,
) raises:
    """`op_pr_iterate` over the column lists of the dense adjacency `a`, all
    on the device (lane hr-graph, x_neighbors/graph_par.mojo `pr_drive`):
    the row sums, column lists and offsets built from `a`, the step one
    thread per node over its column ascending, the dangling mass and
    |x' - x| blocked folds. `x` in: the start, out: the last iterate. info
    (int32 x 2): iterations run, converged."""
    pr_iterate_gpu(a, x, p, dw, info, n, max_iter,
                   bitcast[DType.float64]((UInt64(thr_hi) << UInt64(32)) | UInt64(thr_lo)), binary, alpha)

def op_nan_cells(x: Int, cells: Int, colmiss: Int, info: Int, n: Int, d: Int, colmiss_only: Int = 0) raises:
    """lane/neural-pass71: the NaN cells of x (n x d): flat indices
    ascending into `cells`, the NaN count per column, the total in info[0].
    lane hr-small-passes (2026-10-02): a device NaN mask and a deterministic
    prefix-sum compaction (x_neighbors/nan_cells_device.mojo), the same
    integers in the same order as the CPU column's pass
    (x_neighbors/nan_cells.mojo)."""
    nan_cells_device(x, cells, colmiss, info, n, d, colmiss_only)


def op_cc_iterate_csr(indptr: Int, indices: Int, lab: Int, info: Int, n: Int, nnz: Int) raises:
    """lane/neural-pass69: `op_cc_iterate` from a CSR adjacency (indptr n + 1,
    indices nnz), no dense matrix: hooking and pointer jumping on the device
    (`_cc_csr_device`). FAST on Apple under `-D MOJOLEARN_CC_FAST`:
    `_cc_csr_fast` (batched rounds, the relabel on the device)."""
    comptime if CC_FAST:
        _cc_csr_fast(indptr, indices, lab, info, n, nnz)
    else:
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
# The opt-in host rounds (`-D MOJOLEARN_XN_CC_HOST`) were removed (hr-optin-flags).
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


# C33: preserve the first fixed-point round while later scheduled work is inert.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
def _c33_cc_hook(ip: IP, ix: IP, lab: IP, n: Int32, changed: IP, state: IP):
    if state.unsafe_load(0) == 0:
        cc_hook_kernel(ip, ix, lab, n, changed)


def _c33_cc_jump(lab: IP, n: Int32, state: IP):
    if state.unsafe_load(0) == 0:
        cc_jump_kernel(lab, n)


def _c33_cc_finish(changed: IP, state: IP, round_: Int32):
    if thread_idx.x == 0 and state.unsafe_load(0) == 0 and changed.unsafe_load(0) == 0:
        state.unsafe_store(0, Int32(1))
        state.unsafe_store(1, round_)


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
    comptime if C33_FROZEN_CHUNKS:
        var state = ctx.enqueue_create_buffer[DType.int32](2)
        state.enqueue_fill(Int32(0))
        var hstate = List[Int32](length=2,fill=Int32(0))
        while n > 0:
            for offset in range(C33_CHUNK):
                d_c.enqueue_fill(Int32(0))
                ctx.enqueue_function[_c33_cc_hook](d_ip.unsafe_ptr(),d_ix.unsafe_ptr(),d_l.unsafe_ptr(),Int32(n),d_c.unsafe_ptr(),state.unsafe_ptr(),grid_dim=blocks,block_dim=256)
                ctx.enqueue_function[_c33_cc_jump](d_l.unsafe_ptr(),Int32(n),state.unsafe_ptr(),grid_dim=blocks,block_dim=256)
                ctx.enqueue_function[_c33_cc_finish](d_c.unsafe_ptr(),state.unsafe_ptr(),Int32(rounds+offset+1),grid_dim=1,block_dim=1)
            rounds += C33_CHUNK
            ctx.enqueue_copy(dst_ptr=hstate.unsafe_ptr(),src_buf=state)
            ctx.synchronize()
            if hstate[0] != 0:
                rounds=Int(hstate[1])
                break
        _ = state^
        _ = hstate^
    else:
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


# lane/apple-fast-graph (2026-10-02), FAST on Apple only, the default since
# 2026-10-04 (rollback `-D MOJOLEARN_CC_FAST_OFF`): the same hooking and pointer jumping as
# `_cc_csr_device`, with the host waits taken out of the loop. The board's
# race (20,000 nodes, 54,528 edges) spends its ~90 ms on overhead, not on
# the kernels: one host wait and one memset per round, a pinned staging
# buffer with its own wait, three uploads, and Python's relabel over the
# labels. Here the CSR and the start labels go straight to the device (no
# staging), CC_FAST_BATCH rounds run between two reads of the change word,
# the change word is never cleared (every writer of a round stores the ROUND
# NUMBER, so after a batch the word equals the batch's last round iff that
# round still lowered a label; a round that changes nothing is the fixed
# point, so the spare rounds of a batch change nothing), and the relabel is a
# device flag-and-scan: a root (lab[v] == v) is numbered by the count of
# roots below it, which is the order of first appearance of the min-labels
# over the nodes (the component's minimum node is the first node that
# carries its label), the integers `_cc_relabel` builds; every node takes
# its root's number. One download of the labels and the count. Same labels
# as the host rounds + Python relabel; the round count in info[0] includes
# the batch's spare rounds (Python reads only the labels and the count).
#: FAST Apple, default ON since 2026-10-04 (rollback
#: `-D MOJOLEARN_CC_FAST_OFF`; the old -D name is harmless). Source
#: lane/apple-fast-graph@1fa36a7ec (ported 2026-10-04, lane
#: apple-fast-rec-misc). connected_components on a CSR graph: batched
#: hook + jump rounds (one change-word read per CC_FAST_BATCH rounds), the
#: first-appearance relabel and the component count on the device.
#: Known: never ran. M3 graph-cc-fast-taxi failed to parse at
#: `cc_relabel_kernel(..., out: IP, ...)` ("expected argument name": `out`
#: is an argument convention); renamed `dst`. Python half rebased onto
#: main's `_p2m_iota` / `_p2m_relabel` (both already device ops). Board:
#: connected-components taxi 0.68 (already a win); prior gap 88 ms vs
#: networkx 7.3 ms was launch + wait overhead, not kernel time.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, tag
#: rab3-ccfast): connected-components taxi 8.17 -> 3.60 ms (-56.0%);
#: n_components 588 both arms, output digest identical. KEEP.
comptime CC_FAST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_CC_FAST_OFF"]()
)
#: hook + jump rounds between two reads of the change word
comptime CC_FAST_BATCH = 4


def cc_hook_round_kernel(indptr: IP, indices: IP, lab: IP, n: Int32, changed: IP, rid: Int32):
    """`cc_hook_kernel` whose change word is the round number instead of 1
    (never cleared; see the lane note above)."""
    var u = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if u < Int(n):
        for e in range(Int(indptr.unsafe_load(u)), Int(indptr.unsafe_load(u + 1))):
            var v = Int(indices.unsafe_load(e))
            var a = lab.unsafe_load(u)
            var b = lab.unsafe_load(v)
            if a != b:
                var lo = a if a < b else b
                var hi = b if a < b else a
                if lab.unsafe_load(Int(hi)) > lo:
                    _ = Atomic[DType.int32].min(lab + Int(hi), lo)
                    changed.unsafe_store(0, rid)


def cc_root_count_kernel(lab: IP, bcount: IP, n_: Int64):
    """Per block of NC_CHUNK nodes, the number of roots (lab[v] == v): the
    first stage of the compaction scan, the shape of `nan_count_kernel`."""
    var n = Int(n_)
    var t = Int(thread_idx.x)
    var base = Int(block_idx.x) * NC_CHUNK
    var sh = stack_allocation[NC_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var c = Int32(0)
    for k in range(NC_PER):
        var j = base + k * NC_TPB + t
        if j < n and Int(lab.unsafe_load(j)) == j:
            c += 1
    sh[t] = c
    barrier()
    var s = NC_TPB // 2
    while s > 0:
        if t < s:
            sh[t] = sh[t] + sh[t + s]
        barrier()
        s //= 2
    if t == 0:
        bcount.unsafe_store(Int(block_idx.x), sh[0])


def cc_root_rank_kernel(lab: IP, boff: IP, newid: IP, n_: Int64):
    """Root r gets newid[r] = the number of roots below r (the block's
    offset plus the block's exclusive scan): the shape of
    `nan_scatter_kernel`, writing the rank at the root instead of the root
    at the rank."""
    var n = Int(n_)
    var t = Int(thread_idx.x)
    var base = Int(block_idx.x) * NC_CHUNK
    var sh = stack_allocation[NC_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var running = boff.unsafe_load(Int(block_idx.x))
    for k in range(NC_PER):
        var j = base + k * NC_TPB + t
        var v = Int32(0)
        if j < n and Int(lab.unsafe_load(j)) == j:
            v = 1
        sh[t] = v
        barrier()
        var off = 1
        while off < NC_TPB:
            var add = Int32(0)
            if t >= off:
                add = sh[t - off]
            barrier()
            sh[t] = sh[t] + add
            barrier()
            off *= 2
        var incl = sh[t]
        var tot = sh[NC_TPB - 1]
        barrier()
        if v != 0:
            newid.unsafe_store(j, running + incl - v)
        running += tot


def cc_relabel_kernel(lab: IP, newid: IP, dst: IP, n: Int32):
    """dst[v] = newid[lab[v]]: every node takes its root's rank. (`dst`, not
    `out`: `out` is an argument convention and fails to parse as a name.)"""
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v < Int(n):
        dst.unsafe_store(v, newid.unsafe_load(Int(lab.unsafe_load(v))))


def _cc_csr_fast(indptr: Int, indices: Int, lab: Int, info: Int, n: Int, nnz: Int) raises:
    """`_cc_csr_device`'s contract plus the relabel: out, `lab` holds the
    compacted labels (component k = the k-th component in order of first
    appearance over the nodes, scipy's numbering) and info[1] the number of
    components + 1 (0 would mean "not relabelled here"). Start labels are
    the identity (what `connected_components` passes)."""
    comptime assert NC_SMEM_FITS, "cc_csr_fast: a 1 KB threadgroup page must fit"
    var ctx = xn_ctx()
    # the CSR and the start labels straight to the device (the copy engine)
    var d_ip = _buf_i(ctx, indptr, n + 1, True)
    var d_ix = _buf_i(ctx, indices, nnz, True)
    var d_l = _buf_i(ctx, lab, n, True)
    var d_out = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_new = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var nb = max((n + NC_CHUNK - 1) // NC_CHUNK, 1)
    var ng = (nb + NC_CHUNK - 1) // NC_CHUNK
    var d_bc = ctx.enqueue_create_buffer[DType.int32](nb)
    var d_bo = ctx.enqueue_create_buffer[DType.int32](nb)
    var d_gs = ctx.enqueue_create_buffer[DType.int32](ng)
    var d_go = ctx.enqueue_create_buffer[DType.int32](ng)
    var d_c = ctx.enqueue_create_buffer[DType.int32](1)
    var d_cnt = ctx.enqueue_create_buffer[DType.int32](1)
    # the change word starts below every round number (a kernel, not a memset)
    enqueue_fill(ctx, d_c, Int32(0))
    var blocks = (n + 255) // 256
    var hflag = List[Int32](length=1, fill=Int32(0))
    var rounds = 0
    while n > 0:
        for _ in range(CC_FAST_BATCH):
            rounds += 1
            ctx.enqueue_function[cc_hook_round_kernel](
                d_ip.unsafe_ptr(), d_ix.unsafe_ptr(), d_l.unsafe_ptr(), Int32(n), d_c.unsafe_ptr(), Int32(rounds),
                grid_dim=blocks, block_dim=256,
            )
            ctx.enqueue_function[cc_jump_kernel](d_l.unsafe_ptr(), Int32(n), grid_dim=blocks, block_dim=256)
        # one read per batch: did the batch's last round still lower a label?
        ctx.enqueue_copy(dst_ptr=hflag.unsafe_ptr(), src_buf=d_c)
        ctx.synchronize()
        if Int(hflag[0]) != rounds:
            break
    # the relabel: root flags per block -> group sums -> top scan (the count
    # into d_cnt) -> block offsets -> root ranks -> every node its root's rank
    var lp = d_l.unsafe_ptr()
    ctx.enqueue_function[cc_root_count_kernel](lp, d_bc.unsafe_ptr(), Int64(n), grid_dim=nb, block_dim=NC_TPB)
    ctx.enqueue_function[nan_group_sum_kernel](d_bc.unsafe_ptr(), d_gs.unsafe_ptr(), Int64(nb), grid_dim=ng, block_dim=NC_TPB)
    ctx.enqueue_function[nan_top_scan_kernel](d_gs.unsafe_ptr(), d_go.unsafe_ptr(), d_cnt.unsafe_ptr(), Int64(ng),
                                              grid_dim=1, block_dim=NC_TPB)
    ctx.enqueue_function[nan_down_scan_kernel](d_bc.unsafe_ptr(), d_go.unsafe_ptr(), d_bo.unsafe_ptr(), Int64(nb),
                                               grid_dim=ng, block_dim=NC_TPB)
    ctx.enqueue_function[cc_root_rank_kernel](lp, d_bo.unsafe_ptr(), d_new.unsafe_ptr(), Int64(n), grid_dim=nb, block_dim=NC_TPB)
    if n > 0:
        ctx.enqueue_function[cc_relabel_kernel](lp, d_new.unsafe_ptr(), d_out.unsafe_ptr(), Int32(n),
                                                grid_dim=blocks, block_dim=256)
    var hcnt = List[Int32](length=1, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=hcnt.unsafe_ptr(), src_buf=d_cnt)
    _down_i(ctx, d_out, lab, n)
    ctx.synchronize()
    var ip = IP(unsafe_from_address=info)
    ip.unsafe_store(0, Int32(rounds))
    ip.unsafe_store(1, hcnt[0] + Int32(1))
    _ = hflag^
    _ = hcnt^
    _ = d_ip^
    _ = d_ix^
    _ = d_l^
    _ = d_out^
    _ = d_new^
    _ = d_bc^
    _ = d_bo^
    _ = d_gs^
    _ = d_go^
    _ = d_c^
    _ = d_cnt^


def cc_hook_dense_kernel(a: FP, lab: IP, n: Int32, changed: IP):
    """`cc_hook_kernel` on the dense matrix (lane hr2-graph-embed): thread v
    walks COLUMN v (a warp reads one row's consecutive cells: coalesced), and
    every nonzero a[t, v] hooks t and v. Hooking is symmetric in its two
    ends, so the columns cover both directions of weak connectivity."""
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nn = Int(n)
    if v < nn:
        for t in range(nn):
            if a.unsafe_load(t * nn + v) != Float32(0):
                var x = lab.unsafe_load(t)
                var y = lab.unsafe_load(v)
                if x != y:
                    var lo = x if x < y else y
                    var hi = y if x < y else x
                    if lab.unsafe_load(Int(hi)) > lo:
                        _ = Atomic[DType.int32].min(lab + Int(hi), lo)
                        changed.unsafe_store(0, Int32(1))


def _c33_cc_dense(a: FP, lab: IP, n: Int32, changed: IP, state: IP):
    if state.unsafe_load(0) == 0:
        cc_hook_dense_kernel(a, lab, n, changed)


def cc_label_init_kernel(lab: IP, n: Int32):
    var v = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if v < Int(n):
        lab.unsafe_store(v, Int32(v))


def op_cc_iterate(a: Int, lab: Int, info: Int, n: Int) raises:
    """connected_components on a dense adjacency, on the device (lane
    hr2-graph-embed, 2026-10-02; before, the default was the host's sparse
    walk, x_neighbors/cc_sparse.mojo): A uploaded once, then hooking
    (`cc_hook_dense_kernel`) and pointer jumping (`cc_jump_kernel`) rounds
    until no edge lowers a label, from the identity labels. The result is
    the unique min-label fixed point (every node labelled by the lowest node
    of its weak component), the host column's rounds word for word; info
    holds the device's round count (Python reads only the labels)."""
    var ctx = xn_ctx()
    var d_a = _buf(ctx, a, n * n, True)
    var d_l = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var d_c = ctx.enqueue_create_buffer[DType.int32](1)
    var hb = ctx.enqueue_create_host_buffer[DType.int32](max(n, 1))
    var blocks = (n + 255) // 256
    var rounds = 0
    if n > 0:
        ctx.enqueue_function[cc_label_init_kernel](d_l.unsafe_ptr(), Int32(n), grid_dim=blocks, block_dim=256)
    comptime if C33_FROZEN_CHUNKS:
        var state=ctx.enqueue_create_buffer[DType.int32](2)
        state.enqueue_fill(Int32(0))
        var hstate=List[Int32](length=2,fill=Int32(0))
        while n > 0:
            for offset in range(C33_CHUNK):
                d_c.enqueue_fill(Int32(0))
                ctx.enqueue_function[_c33_cc_dense](d_a.unsafe_ptr(),d_l.unsafe_ptr(),Int32(n),d_c.unsafe_ptr(),state.unsafe_ptr(),grid_dim=blocks,block_dim=256)
                ctx.enqueue_function[_c33_cc_jump](d_l.unsafe_ptr(),Int32(n),state.unsafe_ptr(),grid_dim=blocks,block_dim=256)
                ctx.enqueue_function[_c33_cc_finish](d_c.unsafe_ptr(),state.unsafe_ptr(),Int32(rounds+offset+1),grid_dim=1,block_dim=1)
            rounds += C33_CHUNK
            ctx.enqueue_copy(dst_ptr=hstate.unsafe_ptr(),src_buf=state)
            ctx.synchronize()
            if hstate[0] != 0:
                rounds=Int(hstate[1])
                break
        _ = state^
        _ = hstate^
    else:
        while n > 0:
            rounds += 1
            d_c.enqueue_fill(Int32(0))
            ctx.enqueue_function[cc_hook_dense_kernel](d_a.unsafe_ptr(), d_l.unsafe_ptr(), Int32(n), d_c.unsafe_ptr(),
                                                       grid_dim=blocks, block_dim=256)
            ctx.enqueue_function[cc_jump_kernel](d_l.unsafe_ptr(), Int32(n), grid_dim=blocks, block_dim=256)
            ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=d_c)
            ctx.synchronize()
            if hb.unsafe_ptr()[0] == 0:
                break
    if n > 0:
        ctx.enqueue_copy(dst_ptr=hb.unsafe_ptr(), src_buf=d_l)
        ctx.synchronize()
        memcpy(dest=IP(unsafe_from_address=lab), src=hb.unsafe_ptr(), count=n)
    IP(unsafe_from_address=info).unsafe_store(0, Int32(rounds))
    _ = hb^
    _ = d_a^
    _ = d_l^
    _ = d_c^
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



def op_pcs_resident(
    x: Int, hidx: Int, hbit: Int, res: Int,
    n: Int, d_in: Int, nf: Int, nc: Int, degree: Int, gamma: Float32, coef0: Float32,
) raises:
    """PolynomialCountSketch.transform: `pcs_item` (one thread per ROW, the
    convolution O(nc^2) per row) as three kernels over the same statements:
    one sketch per (row, degree), then per degree p >= 1 one thread per
    output cell folding the convolution in the same ascending order, the
    running product ping-ponging on the device."""
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
        if row_kernel:
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

#: lane/fam2-neighbors (2026-10-04), IDENTICAL on every vendor, the fused
#: k-NN behind LocalOutlierFactor, LabelPropagation / LabelSpreading and
#: KNNImputer's neighbor search (`knn_sq_tiled_kernel`, `knn_sq_tiled2_kernel`).
#: Neither switch changes a bit, so the host column (`knn_sq_item`) is untouched.
#:
#: XN_KNN_LEAN (default ON; -D MOJOLEARN_IDN_XN_KNN_LEAN_OFF): one ftz per
#: pair-feature, not four. `knn_sq_item`'s step is
#: acc = ftz(fma(df, df, acc)), df = ftz(ftz(x) - ftz(y)). The inputs are
#: flushed once when they are staged (ftz is idempotent). The flush of df is
#: dropped: a subnormal df has df * df < 2^-252, so fma(df, df, acc) is acc
#: itself (acc is +0 or a normal number after its own ftz, and a term that
#: far under half an ulp of either never moves the rounding), the value the
#: flushed df = +-0 gives; a device that reads a subnormal operand as zero
#: computes the same.
#:
#: XN_KNN_PRUNE (default ON; -D MOJOLEARN_IDN_XN_KNN_PRUNE_OFF): a pair's
#: chain stops once its partial sum is not below its row's current k-th
#: distance. The chain never decreases (every term is >= 0), so the full
#: value would fail the insertion's strict `<` too: the same lists. In the
#: 2-D kernel a thread skips a feature chunk once all 16 of its pairs are
#: out (the rows' k-th distances are published in threadgroup memory per y
#: tile); in the row kernel the check runs every 8 features.
#: Both off under MOJOLEARN_IDN_ALL_OFF.
comptime _XN_KNN_IDN = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
comptime XN_KNN_LEAN = _XN_KNN_IDN and not is_defined["MOJOLEARN_IDN_XN_KNN_LEAN_OFF"]()
comptime XN_KNN_PRUNE = _XN_KNN_IDN and not is_defined["MOJOLEARN_IDN_XN_KNN_PRUNE_OFF"]()


@always_inline
def _knn_in(v: Float32) -> Float32:
    """A staged input: flushed once under XN_KNN_LEAN."""
    comptime if XN_KNN_LEAN:
        return ftz(v)
    return v


@always_inline
def _knn_step(xv: Float32, yv: Float32, acc: Float32) -> Float32:
    """One feature of `knn_sq_item`'s chain; xv and yv are `_knn_in` values."""
    comptime if XN_KNN_LEAN:
        var dl = xv - yv
        return ftz(identical_mul_add(dl, dl, acc))
    var df = _sub(xv, yv)
    return ftz(identical_mul_add(df, df, acc))


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
            ys[q] = _knn_in(y.unsafe_load(j0 * d + q))
            q += KNN_TILE_TPB
        barrier()
        if live:
            for jj in range(rows):
                var j = j0 + jj
                if ex != 0 and j == t:
                    continue
                var acc = Float32(0)
                for f in range(d):
                    acc = _knn_step(_knn_in(x.unsafe_load(t * d + f)), ys[jj * d + f], acc)
                    comptime if XN_KNN_PRUNE:
                        if (f & 7) == 7 and not (acc < worst):
                            break
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


# lane/neural-pass101 (2026-10-01): the fused k-NN for any d, tiled in two
# dimensions. A block owns KNN2_TX rows of x; per KNN2_TY rows of y it builds
# the KNN2_TX x KNN2_TY distance tile, every pair's chain over the features
# ascending (KNN2_FC features of both rows staged in threadgroup memory at a
# time, each thread KNN2_PT pairs in registers), with knn_sq_item's
# statements, then the x row's thread offers the tile's columns in ascending
# order to the item's strict-< insertion. The same values in the same order,
# so the same lists. The one-thread-per-row kernel this replaces for d above
# KNN_TILE_MAX_D read every y row from device memory in every thread
# (LabelPropagation on istella, 220 features: 23 s at 20,000 rows on the M4).
# `-D MOJOLEARN_XN_KNN_ROWWISE=1` restores it.
comptime KNN2_TX = 64
comptime KNN2_TY = 64
comptime KNN2_FC = 16
comptime KNN2_TPB = 256
#: a thread's register block: KNN2_R x rows by KNN2_R y rows (16 x 16 threads)
comptime KNN2_R = 4
comptime KNN2_XS = KNN2_FC + 1  # padded rows (bank spread)
comptime KNN2_SMEM_BYTES = 4 * (KNN2_TX * KNN2_XS + KNN2_TY * KNN2_XS + KNN2_TX * (KNN2_TY + 1) + KNN2_TX)


def knn_sq_tiled2_kernel(
    x: FP, y: FP, dist: FP, idx: IP, n_: Int64, m_: Int64, d_: Int64, k_: Int64, ex_: Int64,
):
    var n = Int(n_)
    var m = Int(m_)
    var d = Int(d_)
    var k = Int(k_)
    var ex = Int(ex_)
    var tid = Int(thread_idx.x)
    var x0 = Int(block_idx.x) * KNN2_TX
    var xs = stack_allocation[KNN2_TX * KNN2_XS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var ys = stack_allocation[KNN2_TY * KNN2_XS, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dt = stack_allocation[KNN2_TX * (KNN2_TY + 1), Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    # XN_KNN_PRUNE: the block's rows' current k-th distances
    var ws = stack_allocation[KNN2_TX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var inf = _bc[DType.float32](UInt32(0x7F800000))
    var t = x0 + tid
    var owner = tid < KNN2_TX and t < n
    var worst = inf
    if owner:
        for s in range(k):
            dist.unsafe_store(t * k + s, inf)
            idx.unsafe_store(t * k + s, Int32(-1))
    var ta = tid // 16
    var tb = tid - ta * 16
    comptime if XN_KNN_PRUNE:
        if tid < KNN2_TX:
            ws[tid] = worst
        barrier()
    var j0 = 0
    while j0 < m:
        var rows = min(KNN2_TY, m - j0)
        var acc = InlineArray[Float32, KNN2_R * KNN2_R](fill=Float32(0))
        # XN_KNN_PRUNE: this thread's four rows' k-th distances for this y
        # tile (written before the barrier that ended the last tile), and
        # whether all 16 of its pairs are out
        var wv = InlineArray[Float32, KNN2_R](fill=inf)
        var dead = False
        comptime if XN_KNN_PRUNE:
            comptime for i in range(KNN2_R):
                wv[i] = ws[ta + 16 * i]
        var f0 = 0
        while f0 < d:
            var fc = min(KNN2_FC, d - f0)
            var q = tid
            while q < KNN2_TX * KNN2_FC:
                var r = q // KNN2_FC
                var f = q - r * KNN2_FC
                xs[r * KNN2_XS + f] = _knn_in(x.unsafe_load((x0 + r) * d + f0 + f)) if (x0 + r < n and f < fc) else Float32(0)
                ys[r * KNN2_XS + f] = _knn_in(y.unsafe_load((j0 + r) * d + f0 + f)) if (r < rows and f < fc) else Float32(0)
                q += KNN2_TPB
            barrier()
            if not dead:
                for f in range(fc):
                    var xv = InlineArray[Float32, KNN2_R](fill=Float32(0))
                    var yv = InlineArray[Float32, KNN2_R](fill=Float32(0))
                    comptime for i in range(KNN2_R):
                        xv[i] = xs[(ta + 16 * i) * KNN2_XS + f]
                        yv[i] = ys[(tb + 16 * i) * KNN2_XS + f]
                    comptime for i in range(KNN2_R):
                        comptime for jj in range(KNN2_R):
                            acc[i * KNN2_R + jj] = _knn_step(xv[i], yv[jj], acc[i * KNN2_R + jj])
                comptime if XN_KNN_PRUNE:
                    var alive = False
                    comptime for i in range(KNN2_R):
                        comptime for jj in range(KNN2_R):
                            if acc[i * KNN2_R + jj] < wv[i]:
                                alive = True
                    dead = not alive
            barrier()
            f0 += KNN2_FC
        comptime for i in range(KNN2_R):
            comptime for jj in range(KNN2_R):
                dt[(ta + 16 * i) * (KNN2_TY + 1) + tb + 16 * jj] = acc[i * KNN2_R + jj]
        barrier()
        if owner:
            for bb in range(rows):
                var j = j0 + bb
                if ex != 0 and j == t:
                    continue
                var v = dt[tid * (KNN2_TY + 1) + bb]
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
            comptime if XN_KNN_PRUNE:
                ws[tid] = worst
        barrier()
        j0 += rows


def op_knn_sq_tiled(
    x: Int, y: Int, dist: Int, idx: Int, n: Int, m: Int, d: Int, k: Int, exclude_self: Int,
) raises:
    """The fused k-NN (`knn_sq`) with y staged per block; d above
    KNN_TILE_MAX_D takes the one-thread-per-row item kernel."""
    if d > KNN_TILE_MAX_D:
        # the 2-D tiled kernel's threadgroup pages (25,344 bytes) under every
        # column's limit, or the row kernel (every shared page has a fits gate)
        comptime if is_defined["MOJOLEARN_XN_KNN_ROWWISE"]() or not lib_smem_page_fits_for[TARGET_COLUMN, KNN2_SMEM_BYTES]():
            op_knn_sq(x, y, dist, idx, n, m, d, k, exclude_self)
            return
        # direct device copies in and out (no host-thread staging)
        var ctx2 = xn_ctx()
        var d_x2 = ctx2.enqueue_create_buffer[DType.float32](max(n * d, 1))
        var d_y2 = ctx2.enqueue_create_buffer[DType.float32](max(m * d, 1))
        if n * d > 0:
            ctx2.enqueue_copy(dst_buf=d_x2, src_ptr=FP(unsafe_from_address=x))
        if m * d > 0:
            ctx2.enqueue_copy(dst_buf=d_y2, src_ptr=FP(unsafe_from_address=y))
        var d_dist2 = ctx2.enqueue_create_buffer[DType.float32](max(n * k, 1))
        var d_idx2 = _buf_i(ctx2, 0, n * k, False)
        ctx2.enqueue_function[knn_sq_tiled2_kernel](
            d_x2.unsafe_ptr(), d_y2.unsafe_ptr(), d_dist2.unsafe_ptr(), d_idx2.unsafe_ptr(),
            Int64(n), Int64(m), Int64(d), Int64(k), Int64(exclude_self),
            grid_dim=(n + KNN2_TX - 1) // KNN2_TX, block_dim=KNN2_TPB,
        )
        if n * k > 0:
            ctx2.enqueue_copy(dst_ptr=FP(unsafe_from_address=dist), src_buf=d_dist2)
        _down_i(ctx2, d_idx2, idx, n * k)
        ctx2.synchronize()
        _ = d_x2^
        _ = d_y2^
        _ = d_dist2^
        _ = d_idx2^
        _ = ctx2^
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
    comptime if XN_IMPUTE_TIE_MEAN:
        if weights == 0:
            ctx.enqueue_function[knn_impute_tie_mean_kernel](
                d_cells.unsafe_ptr(), d_x.unsafe_ptr(), d_fx.unsafe_ptr(), d_bd.unsafe_ptr(), d_bi.unsafe_ptr(),
                d_res.unsafe_ptr(), Int64(m), Int64(d), Int64(k), Int64(nc),
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


# lane/apple-fast-gap-manprep (2026-10-03): KNNImputer quality on taxi was
# masked_rmse 6.15 vs scikit-learn 5.26. Taxi's columns are mostly discrete
# (passengers, hour, weekday, day, fees) and the fit rows are in time order,
# so many donors tie at the k-th distance and the lower-index rule always
# took the EARLIEST trips (early January for late-February queries), a
# biased draw. The tie mean imputes the expectation of a uniform tie-break:
# the donors strictly closer than the k-th distance D_k, plus the remaining
# k - c_lt slots filled by the mean of ALL donors at exactly D_k. Order-free,
# deterministic, one GPU thread per missing cell, the same distance
# statements as the scan. Uniform weights only (distance weights keep the
# scan's result). FAST + Apple default since the M3 A/B gmp-imp-taxi
# (masked_rmse 6.152 -> 5.109, scikit-learn 5.26; with COLMISS_ONLY);
# -D MOJOLEARN_XN_FAST_IMPUTE_TIE_MEAN_OFF restores the lower-index rule.
# IDENTICAL keeps the lower-index rule (a cross-vendor bit change is the
# orchestrator's call).
comptime XN_IMPUTE_TIE_MEAN = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                               and not is_defined["MOJOLEARN_XN_FAST_IMPUTE_TIE_MEAN_OFF"]())


def knn_impute_tie_mean_kernel(
    cells: IP, x: FP, fx: FP, best_d: FP, best_i: IP, res: FP,
    m_: Int64, d_: Int64, k_: Int64, nc_: Int64,
):
    """Rewrites the imputed value of each missing cell whose scan found k
    donors as the tie mean (above). Fewer than k donors: every donor is
    already used, the scan's value stays."""
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
    var dk = Float32(0)
    if live:
        t = Int(cells.unsafe_load(q0))
        r = t // d
        c = t - r * d
        if Int(best_i.unsafe_load(t * k + k - 1)) < 0:
            live = False
        else:
            dk = best_d.unsafe_load(t * k + k - 1)
    var sum_lt = Float32(0)
    var sum_eq = Float32(0)
    var c_lt = 0
    var c_eq = 0
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
                var dv = fs[jj * d + c]
                if dv != dv:
                    continue
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
                if dist < dk:
                    sum_lt = _add(sum_lt, dv)
                    c_lt += 1
                elif dist == dk:
                    sum_eq = _add(sum_eq, dv)
                    c_eq += 1
        barrier()
        j0 += rows
    if live and c_eq > 0 and c_lt < k:
        var fill = ftz(identical_mul(ftz(identical_div(sum_eq, Float32(c_eq))), Float32(k - c_lt)))
        res.unsafe_store(t, ftz(identical_div(_add(sum_lt, fill), Float32(k))))


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
# float-float accumulator from tile to tile: `matmul_tn_acc_ff_item`.
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


#: SVGP_FAST_RBFTILE, DEFAULT in FAST + Apple (lane/apple-fast-w2-svgp):
#: SVGP's scaled rbf (Kuu, the Kfu tiles, Ksu in predict) as one tiled
#: launch, 64 rows x 64 inducing points per 256-thread block, 16 features
#: staged in threadgroup memory per step, 4 x 4 cells per thread, variance
#: scale fused. Each cell keeps `kernel_item`'s chain (_sub, mul_add,
#: features ascending) and `unary_item`'s identity step. M3, one run per
#: arm: istella 336.3 -> 299.1 ms, taxi 285.9 -> 283.9 ms; w2-svgp-rbftile-q
#: SVGP-FAST-PAIR PASS (r2/rmse/elbo gates; A r2 taxi -0.19498, istella
#: -0.10602). MOJOLEARN_SVGP_FAST_RBFTILE_OFF restores the per-cell kernel.
comptime SVGP_FAST_RBFTILE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SVGP_FAST_RBFTILE_OFF"]()
)
#: X6 (IDENTICAL, every vendor; lane ml-cluster-nbrs 2026-10-04): the same
#: tiled launch. Each cell is `kernel_item`'s K_RBF chain (`_sub`, the pinned
#: `identical_mul_add`, features ascending from +0 over exactly d) and
#: `unary_item`'s identity step, so the words are the per-cell kernel's.
#: `-D MOJOLEARN_IDN_SVGP_RBFTILE_OFF` (or MOJOLEARN_IDN_ALL_OFF) restores it.
comptime IDN_SVGP_RBFTILE = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_SVGP_RBFTILE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime SVGP_RBFTILE = SVGP_FAST_RBFTILE or IDN_SVGP_RBFTILE
comptime SVGP_RBF_T = 64
comptime SVGP_RBF_TP = SVGP_RBF_T + 1
comptime SVGP_RBF_FK = 16
comptime SVGP_RBF_THREADS = 256
comptime _SVGP_RBF_SMEM_FITS = lib_smem_page_fits_for[TARGET_COLUMN, 2 * SVGP_RBF_FK * SVGP_RBF_TP * 4]()


def svgp_rbf_tile_kernel(q: FP, z: FP, dst: FP, rows_: Int64, m_: Int64, d_: Int64, gamma: Float32, variance: Float32):
    comptime assert _SVGP_RBF_SMEM_FITS, "svgp_rbf_tile_kernel: an 8.3 KB threadgroup page must fit"
    comptime T = SVGP_RBF_T
    comptime TP = SVGP_RBF_TP
    comptime FK = SVGP_RBF_FK
    comptime NT = SVGP_RBF_THREADS
    var rows = Int(rows_)
    var m = Int(m_)
    var d = Int(d_)
    var ncb = (m + T - 1) // T
    var b = Int(block_idx.x)
    var rb = b // ncb
    var cb = b - rb * ncb
    var r0 = rb * T
    var c0 = cb * T
    var tid = Int(thread_idx.x)
    var ty = tid // 16
    var tx = tid - ty * 16
    var xs = stack_allocation[FK * TP, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var zs = stack_allocation[FK * TP, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var acc = SIMD[DType.float32, 16](0)
    var f0 = 0
    while f0 < d:
        var fk = min(FK, d - f0)
        # stage: element e = row * FK + feature (a row's features are adjacent in memory)
        var e = tid
        while e < FK * T:
            var rr = e // FK
            var ff = e - rr * FK
            var xv = Float32(0)
            var zv = Float32(0)
            if ff < fk:
                if r0 + rr < rows:
                    xv = q.unsafe_load((r0 + rr) * d + f0 + ff)
                if c0 + rr < m:
                    zv = z.unsafe_load((c0 + rr) * d + f0 + ff)
            xs[ff * TP + rr] = xv
            zs[ff * TP + rr] = zv
            e += NT
        barrier()
        for ff in range(fk):
            comptime for u in range(4):
                var xv = xs[ff * TP + ty + 16 * u]
                comptime for v in range(4):
                    var df = _sub(xv, zs[ff * TP + tx + 16 * v])
                    acc[u * 4 + v] = ftz(identical_mul_add(df, df, acc[u * 4 + v]))
        barrier()
        f0 += FK
    comptime for u in range(4):
        comptime for v in range(4):
            var i = r0 + ty + 16 * u
            var j = c0 + tx + 16 * v
            if i < rows and j < m:
                var k = ftz(identical_exp(ftz(identical_mul(-gamma, acc[u * 4 + v]))))
                dst.unsafe_store(i * m + j, ftz(identical_mul_add(variance, ftz(k), Float32(0))))


# C54 predicts four independent inducing-point outputs per query thread.
# The feature fold, exponential, and variance scale match the incumbent;
# query feature loads are reused. Scratch is four scalar accumulators.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
def _c54_svgp_predict_kernel(q: FP,z: FP,dst: FP,rows: Int32,m: Int32,d: Int32,gamma: Float32,variance: Float32):
    var tile = _tid()
    var per_row = (Int(m)+3)//4
    var row = tile//per_row
    var first = (tile%per_row)*4
    if row >= Int(rows):
        return
    var sums = InlineArray[Float32,4](fill=Float32(0))
    for feature in range(Int(d)):
        var qv=q.unsafe_load(row*Int(d)+feature)
        comptime for lane in range(4):
            if first+lane < Int(m):
                var delta=_sub(qv,z.unsafe_load((first+lane)*Int(d)+feature))
                sums[lane]=ftz(identical_mul_add(delta,delta,sums[lane]))
    comptime for lane in range(4):
        if first+lane < Int(m):
            var kval=ftz(identical_exp(ftz(identical_mul(-gamma,sums[lane]))))
            dst.unsafe_store(row*Int(m)+first+lane,ftz(identical_mul_add(variance,ftz(kval),Float32(0))))


def _launch_scaled_rbf(
    ctx: DeviceContext, q: FP, z: FP, kbuf: FP, dst: FP, rows: Int, m: Int, d: Int, gamma: Float32, variance: Float32,
    prediction: Bool = False,
) raises:
    """SVGP's `_k`: the rbf kernel (coef0 0, degree 0), then
    `unary(K, identity, variance, 0)`."""
    comptime if C54_PREDICT_TILES:
        if prediction:
            if rows*m > 0:
                ctx.enqueue_function[_c54_svgp_predict_kernel](q,z,dst,Int32(rows),Int32(m),Int32(d),gamma,variance,grid_dim=_grid(rows*((m+3)//4)),block_dim=BLOCK)
            return
    comptime if SVGP_RBFTILE:
        if rows * m > 0:
            var nblk = ((rows + SVGP_RBF_T - 1) // SVGP_RBF_T) * ((m + SVGP_RBF_T - 1) // SVGP_RBF_T)
            ctx.enqueue_function[svgp_rbf_tile_kernel](
                q, z, dst, Int64(rows), Int64(m), Int64(d), gamma, variance,
                grid_dim=nblk, block_dim=SVGP_RBF_THREADS,
            )
        return
    _launch_kernel(ctx, q, z, kbuf, rows, m, d, K_RBF, 0, gamma, Float32(0))
    var cells = rows * m
    ctx.enqueue_function[unary_kernel](
        kbuf, dst, Int64(cells), Int64(U_IDENTITY), variance, Float32(0),
        grid_dim=_grid(cells), block_dim=(BLOCK if cells > 1 else 1),
    )


def matmul_tn_acc_ff_kernel(a: FP, b: FP, rh: FP, rl: FP, rows_: Int64, n_: Int64, m_: Int64):
    var t = _tid()
    if t < Int(n_) * Int(m_):
        matmul_tn_acc_ff_item(t, a, b, rh, rl, Int(rows_), Int(n_), Int(m_))


#: lane apple-fast-gap-kapprox2: the FAST + Apple default since the M3 A/B
#: kap2-svgp-symtile-taxi (svgp taxi 611 -> 463 ms, r2/rmse identical); -D
#: MOJOLEARN_SVGP_FAST_SYMTILE_OFF reverts.
#: lane/idn-gates (2026-10-04): SYMTILE and COLSPLIT are also the IDENTICAL
#: default on every vendor (two_prod commutes bit for bit and every entry
#: keeps its p-ascending fold; the four solves are the column item's own);
#: -D MOJOLEARN_IDN_GATES_OFF (or either _OFF) restores the old IDENTICAL form.
comptime _SVGP_GATE_ON = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()) or (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_GATES_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime SVGP_FAST_SYMTILE = _SVGP_GATE_ON and not is_defined["MOJOLEARN_SVGP_FAST_SYMTILE_OFF"]()


#: lane apple-fast-gap-kapprox2: the FAST + Apple default since the M3 A/B
#: kap2-svgp-colsplit-taxi (svgp taxi 613 -> 438 ms, r2/rmse identical); -D
#: MOJOLEARN_SVGP_FAST_COLSPLIT_OFF reverts.
comptime SVGP_FAST_COLSPLIT = _SVGP_GATE_ON and not is_defined["MOJOLEARN_SVGP_FAST_COLSPLIT_OFF"]()


def svgp_ff_col_solve_kernel(kuu: FP, bh: FP, bl: FP, w: FP, xb: FP, m_: Int64, n_: Int64, jitter: Float32):
    var t = _tid()
    if t < 4 * Int(m_):
        svgp_ff_col_solve_item(t, kuu, bh, bl, w, xb, Int(m_), Int(n_), jitter)


def svgp_ff_col_fin_kernel(kuu: FP, cmat: FP, w: FP, m_: Int64, n_: Int64, jitter: Float32):
    var t = _tid()
    if t < Int(m_) * Int(m_):
        svgp_ff_col_fin_item(t, kuu, cmat, w, Int(m_), Int(n_), jitter)


def matmul_tn_sym_ff_kernel(a: FP, rh: FP, rl: FP, rows_: Int64, m_: Int64):
    var t = _tid()
    var nb = svgp_sym_nb(Int(m_))
    if t < nb * nb:
        matmul_tn_sym_ff_tile_item(t, a, rh, rl, Int(rows_), Int(m_))


def _up_direct(ctx: DeviceContext, addr: Int, count: Int) raises -> DeviceBuffer[DType.float32]:
    """A device buffer with `count` floats copied straight from the host
    pointer by the device copy engine (no host-thread staging)."""
    var buf = ctx.enqueue_create_buffer[DType.float32](max(count, 1))
    if count > 0:
        ctx.enqueue_copy(dst_buf=buf, src_ptr=FP(unsafe_from_address=addr))
    return buf^


def svgp_ff_init_kernel(kuu: FP, bh: FP, bl: FP, w: FP, m_: Int64, n_: Int64, noise: Float32, jitter: Float32):
    var t = _tid()
    if t < Int(m_) * Int(m_):
        svgp_ff_init_item(t, kuu, bh, bl, w, Int(m_), Int(n_), noise, jitter)


def svgp_ff_chol_kernel(w: FP, m_: Int64, n_: Int64, j_: Int64):
    var t = _tid()
    if t < 2 * (Int(m_) - Int(j_)):
        svgp_ff_chol_item(t, w, Int(m_), Int(n_), Int(j_))


def svgp_ff_chol_s_kernel(w: FP, m_: Int64, n_: Int64, j_: Int64):
    var t = _tid()
    if t < Int(m_) - Int(j_):
        svgp_ff_chol_s_item(t, w, Int(m_), Int(n_), Int(j_))


def svgp_ff_column_kernel(kuu: FP, bh: FP, bl: FP, cmat: FP, w: FP, m_: Int64, n_: Int64, jitter: Float32):
    var t = _tid()
    if t < Int(m_):
        svgp_ff_column_item(t, kuu, bh, bl, cmat, w, Int(m_), Int(n_), jitter)


def svgp_ff_x_kernel(bvh: FP, bvl: FP, alpha: FP, w: FP, m_: Int64, n_: Int64, noise: Float32):
    var t = _tid()
    if t < Int(m_):
        svgp_ff_x_item(t, bvh, bvl, alpha, w, Int(m_), Int(n_), noise)


def svgp_ff_qmu_kernel(kuu: FP, qmu: FP, w: FP, m_: Int64, n_: Int64, jitter: Float32):
    var t = _tid()
    if t < Int(m_):
        svgp_ff_qmu_item(t, kuu, qmu, w, Int(m_), Int(n_), jitter)


def svgp_ff_qsqrt_kernel(qsqrt: FP, w: FP, m_: Int64, n_: Int64):
    var t = _tid()
    if t < Int(m_) * Int(m_):
        svgp_ff_qsqrt_item(t, qsqrt, w, Int(m_), Int(n_))


def svgp_ff_part_kernel(y: FP, bvh: FP, bvl: FP, w: FP, m_: Int64, n_: Int64):
    var t = _tid()
    if t < svgp_ff_nbn(Int(n_)) + svgp_ff_nbm(Int(m_)):
        svgp_ff_part_item(t, y, bvh, bvl, w, Int(m_), Int(n_))


def svgp_ff_fin_kernel(w: FP, info: FP, nbn_: Int64, nbm_: Int64, nf: Float32, noise: Float32, kdiag: Float32):
    if _tid() == 0:
        svgp_ff_fin_item(w, info, Int(nbn_), Int(nbm_), nf, noise, kdiag)


comptime _SVGP_TREE_SMEM_FITS = lib_smem_page_fits_for[TARGET_COLUMN, SVGP_TREE * 8 * 4]()


def svgp_ff_fin_tree_kernel(w: FP, info: FP, nbn_: Int64, nbm_: Int64, nf: Float32, noise: Float32, kdiag: Float32):
    """ONE block of SVGP_TREE threads (lane apple-fast-purity): `svgp_ff_fin_item`'s
    tree fold with thread s on slot s, the halving steps in threadgroup
    memory, then thread 0 writes the bound."""
    comptime assert _SVGP_TREE_SMEM_FITS, "svgp_ff_fin_tree_kernel: an 8 KB threadgroup page must fit"
    var s = Int(thread_idx.x)
    var sh = stack_allocation[SVGP_TREE * 8, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var v = svgp_ff_tree_slot(w, s, Int(nbn_), Int(nbm_))
    comptime for L in range(8):
        sh[s * 8 + L] = v[L]
    barrier()
    var h = SVGP_TREE // 2
    while h > 0:
        if s < h:
            var a = SV8(0)
            var b = SV8(0)
            comptime for L in range(8):
                a[L] = sh[s * 8 + L]
                b[L] = sh[(s + h) * 8 + L]
            var c = svgp_ff_tree_comb(a, b)
            comptime for L in range(8):
                sh[s * 8 + L] = c[L]
        barrier()
        h //= 2
    if s == 0:
        if svgp_ff_failed(w, info):
            return
        svgp_ff_bound(w, info, FF(sh[0], sh[1]), FF(sh[2], sh[3]), FF(sh[4], sh[5]), sh[6], sh[7], nf, noise, kdiag)


# ---- lane apple-fast-w2-svgp (2026-10-04): FAST + Apple candidates for
# SVGP's fit (and predict). BLKCHOL is default; BSPLIT stays opt-in.

#: SVGP_FAST_BLKCHOL, DEFAULT in FAST + Apple (lane/apple-fast-w2-svgp).
#: M3, one run per arm (measured without RBFTILE; combined retime owed):
#: taxi 287.0 -> 212.5 ms, istella 336.6 -> 265.7 ms; w2-svgp-blkchol-q
#: SVGP-FAST-PAIR PASS. MOJOLEARN_SVGP_FAST_BLKCHOL_OFF restores the
#: one-column-per-launch factor.
#: The three m x m float-float Cholesky factors
#: (L_u, L_s, then Q) one panel of SVGP_CHOL_PB columns per launch instead of
#: one column per launch (m = 512: 1,024 dependent launches -> 64). Each block
#: forms the panel's diagonal block G = A - L L^T over the columns before the
#: panel (k ascending), factors it in threadgroup memory one column per
#: barrier, then each thread finishes its own row's panel columns. Every
#: entry is the same chain as `_chol_col`: the pivot A[c,c] - L[c,k]^2 and
#: the row A[i,c] - L[i,k] L[c,k], k ascending from 0, then sqrt / divide.
#: The words only move with FAST's free contraction.
comptime SVGP_FAST_BLKCHOL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SVGP_FAST_BLKCHOL_OFF"]()
)
#: X5 (IDENTICAL, every vendor; lane ml-cluster-nbrs 2026-10-04): the same
#: panel factor. Its chains are `_chol_col`'s through the shared ff helpers
#: (x_linear/ff.mojo: two_prod and ff_mul on the pinned `fmad`), k ascending,
#: so no contraction choice can move a word: the factor of the one-column
#: launches bit for bit. `-D MOJOLEARN_IDN_SVGP_BLKCHOL_OFF` (or
#: MOJOLEARN_IDN_ALL_OFF) restores one column per launch.
comptime IDN_SVGP_BLKCHOL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_SVGP_BLKCHOL_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime SVGP_BLKCHOL = SVGP_FAST_BLKCHOL or IDN_SVGP_BLKCHOL
#: panel width (columns per launch)
comptime SVGP_CHOL_PB = 16
comptime _SVGP_CHOL_SMEM_FITS = lib_smem_page_fits_for[TARGET_COLUMN, 4 * SVGP_CHOL_PB * SVGP_CHOL_PB * 4]()


def svgp_ff_chol_panel_kernel(w: FP, m_: Int64, n_: Int64, j0_: Int64, nfac_: Int64):
    """Columns j0 .. j0 + tb - 1 of nfac factors (2: L_u then L_s, as
    `svgp_ff_chol_item`; 1: Q, as `svgp_ff_chol_s_item`). Block b: factor
    b // bpf, rows j0 + (b % bpf) * BLOCK + thread."""
    comptime assert _SVGP_CHOL_SMEM_FITS, "svgp_ff_chol_panel_kernel: a 4 KB threadgroup page must fit"
    comptime PB = SVGP_CHOL_PB
    var m = Int(m_)
    var n = Int(n_)
    var j0 = Int(j0_)
    var span = m - j0
    var tb = min(PB, span)
    var bpf = (span + BLOCK - 1) // BLOCK
    var b = Int(block_idx.x)
    var fac = b // bpf
    var rb = b - fac * bpf
    var tid = Int(thread_idx.x)
    var ah: FP
    var al: FP
    var lh: FP
    var ll: FP
    var ok: FP
    if Int(nfac_) == 1:
        ah = svgp_ff_mat(w, m, n, 8)
        al = svgp_ff_mat(w, m, n, 9)
        lh = svgp_ff_mat(w, m, n, 10)
        ll = svgp_ff_mat(w, m, n, 11)
        ok = w + 2
    elif fac == 0:
        ah = svgp_ff_mat(w, m, n, 0)
        al = svgp_ff_mat(w, m, n, 1)
        lh = svgp_ff_mat(w, m, n, 4)
        ll = svgp_ff_mat(w, m, n, 5)
        ok = w
    else:
        ah = svgp_ff_mat(w, m, n, 2)
        al = svgp_ff_mat(w, m, n, 3)
        lh = svgp_ff_mat(w, m, n, 6)
        ll = svgp_ff_mat(w, m, n, 7)
        ok = w + 1
    # G (the panel block continued over k < j0) and its factor D, PB x PB, row major
    var gh = stack_allocation[PB * PB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var gl = stack_allocation[PB * PB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dh = stack_allocation[PB * PB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dl = stack_allocation[PB * PB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var tri = tb * (tb + 1) // 2
    var e = tid
    while e < tri:
        var r = 0
        var c = e
        while c > r:
            c -= r + 1
            r += 1
        var gi = j0 + r
        var gj = j0 + c
        var tv = ff_ld(ah, al, gi * m + gj)
        for k in range(j0):
            tv = ff_sub(tv, ff_mul(ff_ld(lh, ll, gi * m + k), ff_ld(lh, ll, gj * m + k)))
        gh[r * PB + c] = tv.hi
        gl[r * PB + c] = tv.lo
        e += BLOCK
    barrier()
    for c in range(tb):
        if tid >= c and tid < tb:
            # the pivot, every thread in the same order as `_chol_col`
            var s = FF(gh[c * PB + c], gl[c * PB + c])
            for k in range(c):
                var l = FF(dh[c * PB + k], dl[c * PB + k])
                s = ff_sub(s, ff_mul(l, l))
            var rt = ff_sqrt(s)
            if tid == c:
                if not (s.hi > 0) and rb == 0:
                    ok.unsafe_store(0, Float32(0))
                dh[c * PB + c] = rt.hi
                dl[c * PB + c] = rt.lo
            else:
                var tv = FF(gh[tid * PB + c], gl[tid * PB + c])
                for k in range(c):
                    tv = ff_sub(tv, ff_mul(FF(dh[tid * PB + k], dl[tid * PB + k]), FF(dh[c * PB + k], dl[c * PB + k])))
                var q = ff_div(tv, rt)
                dh[tid * PB + c] = q.hi
                dl[tid * PB + c] = q.lo
        barrier()
    var rr = rb * BLOCK + tid
    if fac >= Int(nfac_) or rr >= span:
        return
    var i = j0 + rr
    if rr < tb:
        for c in range(rr + 1):
            ff_st(lh, ll, i * m + j0 + c, FF(dh[rr * PB + c], dl[rr * PB + c]))
        return
    # row i below the panel: all panel columns' chains over k < j0 together
    # (k outer, each column's own order unchanged), then the panel columns
    var th = SIMD[DType.float32, PB](0)
    var tl = SIMD[DType.float32, PB](0)
    comptime for c in range(PB):
        if c < tb:
            var a = ff_ld(ah, al, i * m + j0 + c)
            th[c] = a.hi
            tl[c] = a.lo
    for k in range(j0):
        var lik = ff_ld(lh, ll, i * m + k)
        comptime for c in range(PB):
            if c < tb:
                var tv = ff_sub(FF(th[c], tl[c]), ff_mul(lik, ff_ld(lh, ll, (j0 + c) * m + k)))
                th[c] = tv.hi
                tl[c] = tv.lo
    comptime for c in range(PB):
        if c < tb:
            var tv = FF(th[c], tl[c])
            comptime for k in range(c):
                tv = ff_sub(tv, ff_mul(FF(th[k], tl[k]), FF(dh[c * PB + k], dl[c * PB + k])))
            var q = ff_div(tv, FF(dh[c * PB + c], dl[c * PB + c]))
            th[c] = q.hi
            tl[c] = q.lo
            ff_st(lh, ll, i * m + j0 + c, q)


def _svgp_chol_panels(ctx: DeviceContext, wp: FP, m: Int, n: Int, nfac: Int) raises:
    var j0 = 0
    while j0 < m:
        var bpf = (m - j0 + BLOCK - 1) // BLOCK
        ctx.enqueue_function[svgp_ff_chol_panel_kernel](
            wp, Int64(m), Int64(n), Int64(j0), Int64(nfac), grid_dim=nfac * bpf, block_dim=BLOCK,
        )
        j0 += SVGP_CHOL_PB


#: MOJOLEARN_SVGP_FAST_BSPLIT (with SYMTILE on): B = Kuf Kfu and b = Kuf y
#: over SVGP_BSPLIT_R (B) and SVGP_BSPLIT_RV (b) row slices of every Kfu
#: tile, each slice its own float-float partial carried across the tiles,
#: then the partials summed slice-ascending. SYMTILE's launch has 8,256
#: busy threads (m = 512) each 32,768 rows deep, and b's has m threads with
#: one dependent float-float chain each: too few to fill the M3 Ultra. Bits
#: change (float-float re-association, ~1e-14 relative in B and b).
#: Source lane/apple-fast-w2-svgp@146898d2c.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-05,
#: rab19-svgpbsplit): svgp istella 224.6 -> 167.1 ms, taxi 211.3 -> 149.1 ms,
#: r2 equal. KEEP: the FAST + Apple default (with SYMTILE); rollback
#: -D MOJOLEARN_SVGP_FAST_BSPLIT_OFF. See docs/apple-fast/EXPERIMENTS.md.
comptime SVGP_FAST_BSPLIT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and SVGP_FAST_SYMTILE
    and not is_defined["MOJOLEARN_SVGP_FAST_BSPLIT_OFF"]()
)
comptime SVGP_BSPLIT_R = 4
comptime SVGP_BSPLIT_RV = 32


def matmul_tn_sym_ff_split_kernel(a: FP, ph: FP, pl: FP, rows_: Int64, m_: Int64):
    var t = _tid()
    var m = Int(m_)
    var nb = svgp_sym_nb(m)
    var nbb = nb * nb
    if t < SVGP_BSPLIT_R * nbb:
        var s = t // nbb
        var rows = Int(rows_)
        var lo = rows * s // SVGP_BSPLIT_R
        var hi = rows * (s + 1) // SVGP_BSPLIT_R
        matmul_tn_sym_ff_tile_item(t - s * nbb, a + lo * m, ph + s * m * m, pl + s * m * m, hi - lo, m)


def matmul_tn_vec_ff_split_kernel(a: FP, y: FP, ph: FP, pl: FP, rows_: Int64, m_: Int64):
    var t = _tid()
    var m = Int(m_)
    if t < SVGP_BSPLIT_RV * m:
        var s = t // m
        var rows = Int(rows_)
        var lo = rows * s // SVGP_BSPLIT_RV
        var hi = rows * (s + 1) // SVGP_BSPLIT_RV
        matmul_tn_acc_ff_item(t - s * m, a + lo * m, y + lo, ph + s * m, pl + s * m, hi - lo, m, 1)


def svgp_ff_split_sum_kernel(ph: FP, pl: FP, rh: FP, rl: FP, count_: Int64, parts_: Int64):
    """(rh, rl)[t] = the parts' cell t summed part-ascending, float-float."""
    var t = _tid()
    var count = Int(count_)
    if t < count:
        var acc = ff_ld(ph, pl, t)
        for s in range(1, Int(parts_)):
            acc = ff_add(acc, ff_ld(ph, pl, s * count + t))
        ff_st(rh, rl, t, acc)


def op_svgp_fit_ff(
    x: Int, z: Int, y: Int, alpha: Int, cmat: Int, qmu: Int, qsqrt: Int, info: Int, n: Int, m: Int, d: Int,
    gamma: Float32, variance: Float32, noise: Float32, jitter: Float32, kdiag: Float32,
) raises:
    """SVGP.fit in float-float on the device, one resident chain
    (lane/cgr-kernel): Kuu = variance * rbf(z, z), then B = Kuf Kfu and
    b = Kuf y accumulated in float-float over the Kfu row tiles (rows
    ascending), then the float-float solve (x_neighbors/svgp_ff.mojo). Kuu,
    B and b never leave the device; only alpha, C, q_mu, q_sqrt and
    [elbo, ok] come back. Before, B and b (hi and lo) were downloaded after
    the statistics and uploaded again for the solve."""
    var ctx = xn_ctx()
    var d_x = _up_direct(ctx, x, n * d)
    var d_z = _up_direct(ctx, z, m * d)
    var d_y = _up_direct(ctx, y, n)
    var mm = m * m
    var d_kuu = ctx.enqueue_create_buffer[DType.float32](max(mm, 1))
    var d_bh = ctx.enqueue_create_buffer[DType.float32](max(mm, 1))
    var d_bl = ctx.enqueue_create_buffer[DType.float32](max(mm, 1))
    var d_bvh = ctx.enqueue_create_buffer[DType.float32](max(m, 1))
    var d_bvl = ctx.enqueue_create_buffer[DType.float32](max(m, 1))
    enqueue_fill(ctx, d_bh, Float32(0))
    enqueue_fill(ctx, d_bl, Float32(0))
    enqueue_fill(ctx, d_bvh, Float32(0))
    enqueue_fill(ctx, d_bvl, Float32(0))
    var tr = max(_tile_rows(n, m), m)
    var d_k = ctx.enqueue_create_buffer[DType.float32](max(tr * m, 1))
    var d_ks = ctx.enqueue_create_buffer[DType.float32](max(tr * m, 1))
    var xp: FP = _p(d_x)
    var yp: FP = _p(d_y)
    if mm > 0:
        _launch_scaled_rbf(ctx, _p(d_z), _p(d_z), _p(d_k), _p(d_kuu), m, m, d, gamma, variance)
    var tile = _tile_rows(n, m)
    # MOJOLEARN_SVGP_FAST_BSPLIT: the row-slice partials of B and b
    var d_pbh = ctx.enqueue_create_buffer[DType.float32](max(SVGP_BSPLIT_R * mm, 1) if SVGP_FAST_BSPLIT else 1)
    var d_pbl = ctx.enqueue_create_buffer[DType.float32](max(SVGP_BSPLIT_R * mm, 1) if SVGP_FAST_BSPLIT else 1)
    var d_pvh = ctx.enqueue_create_buffer[DType.float32](max(SVGP_BSPLIT_RV * m, 1) if SVGP_FAST_BSPLIT else 1)
    var d_pvl = ctx.enqueue_create_buffer[DType.float32](max(SVGP_BSPLIT_RV * m, 1) if SVGP_FAST_BSPLIT else 1)
    comptime if SVGP_FAST_BSPLIT:
        enqueue_fill(ctx, d_pbh, Float32(0))
        enqueue_fill(ctx, d_pbl, Float32(0))
        enqueue_fill(ctx, d_pvh, Float32(0))
        enqueue_fill(ctx, d_pvl, Float32(0))
    var r0 = 0
    while r0 < n:
        var rows = min(tile, n - r0)
        _launch_scaled_rbf(ctx, xp + r0 * d, _p(d_z), _p(d_k), _p(d_ks), rows, m, d, gamma, variance)
        comptime if SVGP_FAST_BSPLIT:
            var nbs2 = svgp_sym_nb(m)
            ctx.enqueue_function[matmul_tn_sym_ff_split_kernel](
                _p(d_ks), _p(d_pbh), _p(d_pbl), Int64(rows), Int64(m),
                grid_dim=_grid(SVGP_BSPLIT_R * nbs2 * nbs2), block_dim=BLOCK,
            )
            ctx.enqueue_function[matmul_tn_vec_ff_split_kernel](
                _p(d_ks), yp + r0, _p(d_pvh), _p(d_pvl), Int64(rows), Int64(m),
                grid_dim=_grid(SVGP_BSPLIT_RV * m), block_dim=BLOCK,
            )
        else:
            comptime if SVGP_FAST_SYMTILE:
                var nbs = svgp_sym_nb(m)
                ctx.enqueue_function[matmul_tn_sym_ff_kernel](
                    _p(d_ks), _p(d_bh), _p(d_bl), Int64(rows), Int64(m),
                    grid_dim=_grid(nbs * nbs), block_dim=BLOCK,
                )
            else:
                ctx.enqueue_function[matmul_tn_acc_ff_kernel](
                    _p(d_ks), _p(d_ks), _p(d_bh), _p(d_bl), Int64(rows), Int64(m), Int64(m),
                    grid_dim=_grid(mm), block_dim=(BLOCK if mm > 1 else 1),
                )
            ctx.enqueue_function[matmul_tn_acc_ff_kernel](
                _p(d_ks), yp + r0, _p(d_bvh), _p(d_bvl), Int64(rows), Int64(m), Int64(1),
                grid_dim=_grid(m), block_dim=(BLOCK if m > 1 else 1),
            )
        r0 += rows
    comptime if SVGP_FAST_BSPLIT:
        if mm > 0:
            ctx.enqueue_function[svgp_ff_split_sum_kernel](
                _p(d_pbh), _p(d_pbl), _p(d_bh), _p(d_bl), Int64(mm), Int64(SVGP_BSPLIT_R),
                grid_dim=_grid(mm), block_dim=BLOCK,
            )
            ctx.enqueue_function[svgp_ff_split_sum_kernel](
                _p(d_pvh), _p(d_pvl), _p(d_bvh), _p(d_bvl), Int64(m), Int64(SVGP_BSPLIT_RV),
                grid_dim=_grid(m), block_dim=BLOCK,
            )
    var d_alpha = _buf(ctx, 0, m, False)
    var d_c = _buf(ctx, 0, mm, False)
    var d_qmu = _buf(ctx, 0, m, False)
    var d_qs = _buf(ctx, 0, mm, False)
    var d_info = _buf(ctx, 0, 2, False)
    var d_w = ctx.enqueue_create_buffer[DType.float32](max(svgp_ff_ws_size(m, n), 1))
    enqueue_fill(ctx, d_w, Float32(0))
    var wp = _p(d_w)
    var d_xb = ctx.enqueue_create_buffer[DType.float32](max(2 * mm, 1) if SVGP_FAST_COLSPLIT else 1)
    if mm > 0:
        ctx.enqueue_function[svgp_ff_init_kernel](
            _p(d_kuu), _p(d_bh), _p(d_bl), wp, Int64(m), Int64(n), noise, jitter,
            grid_dim=_grid(mm), block_dim=BLOCK,
        )
    comptime if SVGP_BLKCHOL:
        _svgp_chol_panels(ctx, wp, m, n, 2)
    else:
        for j in range(m):
            ctx.enqueue_function[svgp_ff_chol_kernel](wp, Int64(m), Int64(n), Int64(j),
                                                      grid_dim=_grid(2 * (m - j)), block_dim=BLOCK)
    if m > 0:
        comptime if SVGP_FAST_COLSPLIT:
            ctx.enqueue_function[svgp_ff_col_solve_kernel](
                _p(d_kuu), _p(d_bh), _p(d_bl), wp, _p(d_xb), Int64(m), Int64(n), jitter,
                grid_dim=_grid(4 * m), block_dim=BLOCK,
            )
            ctx.enqueue_function[svgp_ff_col_fin_kernel](
                _p(d_kuu), _p(d_c), wp, Int64(m), Int64(n), jitter, grid_dim=_grid(mm), block_dim=BLOCK,
            )
        else:
            ctx.enqueue_function[svgp_ff_column_kernel](
                _p(d_kuu), _p(d_bh), _p(d_bl), _p(d_c), wp, Int64(m), Int64(n), jitter,
                grid_dim=_grid(m), block_dim=BLOCK,
            )
        ctx.enqueue_function[svgp_ff_x_kernel](
            _p(d_bvh), _p(d_bvl), _p(d_alpha), wp, Int64(m), Int64(n), noise, grid_dim=_grid(m), block_dim=BLOCK,
        )
        ctx.enqueue_function[svgp_ff_qmu_kernel](
            _p(d_kuu), _p(d_qmu), wp, Int64(m), Int64(n), jitter, grid_dim=_grid(m), block_dim=BLOCK,
        )
    comptime if SVGP_BLKCHOL:
        _svgp_chol_panels(ctx, wp, m, n, 1)
    else:
        for j in range(m):
            ctx.enqueue_function[svgp_ff_chol_s_kernel](wp, Int64(m), Int64(n), Int64(j),
                                                        grid_dim=_grid(m - j), block_dim=BLOCK)
    if mm > 0:
        ctx.enqueue_function[svgp_ff_qsqrt_kernel](_p(d_qs), wp, Int64(m), Int64(n), grid_dim=_grid(mm), block_dim=BLOCK)
    var nbn = svgp_ff_nbn(n)
    var nbm = svgp_ff_nbm(m)
    if nbn + nbm > 0:
        ctx.enqueue_function[svgp_ff_part_kernel](
            yp, _p(d_bvh), _p(d_bvl), wp, Int64(m), Int64(n), grid_dim=_grid(nbn + nbm), block_dim=BLOCK,
        )
    comptime if SVGP_TREE_ON:
        # lane apple-fast-purity: one block of SVGP_TREE threads folds the
        # nbn + nbm block partials (svgp_ff_fin_item's tree order)
        ctx.enqueue_function[svgp_ff_fin_tree_kernel](
            wp, _p(d_info), Int64(nbn), Int64(nbm), Float32(n), noise, kdiag, grid_dim=1, block_dim=SVGP_TREE,
        )
    else:
        ctx.enqueue_function[svgp_ff_fin_kernel](
            wp, _p(d_info), Int64(nbn), Int64(nbm), Float32(n), noise, kdiag, grid_dim=1, block_dim=1,
        )
    _down(ctx, d_alpha, alpha, m)
    _down(ctx, d_c, cmat, mm)
    _down(ctx, d_qmu, qmu, m)
    _down(ctx, d_qs, qsqrt, mm)
    _down(ctx, d_info, info, 2)
    ctx.synchronize()
    _ = d_x^
    _ = d_z^
    _ = d_y^
    _ = d_kuu^
    _ = d_bh^
    _ = d_bl^
    _ = d_bvh^
    _ = d_bvl^
    _ = d_k^
    _ = d_ks^
    _ = d_alpha^
    _ = d_c^
    _ = d_qmu^
    _ = d_qs^
    _ = d_info^
    _ = d_w^
    _ = d_xb^
    _ = d_pbh^
    _ = d_pbl^
    _ = d_pvh^
    _ = d_pvl^
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
        _launch_scaled_rbf(ctx, qp + r0 * d, _p(d_z), _p(d_k), _p(d_ks), rows, m, d, gamma, variance, True)
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
    comptime if LP_IDN_RESIDENT:
        # lane/fam2-neighbors: the finiteness scan of x on the device (a
        # host walk over m x c before, `lp_knn_finite`); the same product
        var ctx_i = xn_ctx()
        var dc_i = _buf_i(ctx_i, cols, n * k, True)
        var dv_i = _buf(ctx_i, vals, n * k, True)
        var dx_i = _buf(ctx_i, x, m * c, True)
        var dr_i = _buf(ctx_i, 0, n * c, False)
        var d_fl = ctx_i.enqueue_create_buffer[DType.int32](4)
        ctx_i.enqueue_memset(d_fl, Int32(0))
        if m * c > 0:
            ctx_i.enqueue_function[lpki_nonfinite_kernel](
                dx_i.unsafe_ptr(), d_fl.unsafe_ptr(), Int64(m * c),
                grid_dim=_grid(m * c), block_dim=(BLOCK if m * c > 1 else 1),
            )
        if n * c > 0:
            ctx_i.enqueue_function[lpki_product_kernel](
                dc_i.unsafe_ptr(), dv_i.unsafe_ptr(), dx_i.unsafe_ptr(), dr_i.unsafe_ptr(), d_fl.unsafe_ptr(),
                Int64(n), Int64(m), Int64(k), Int64(c),
                grid_dim=_grid(n * c), block_dim=(BLOCK if n * c > 1 else 1),
            )
        _down(ctx_i, dr_i, res, n * c)
        ctx_i.synchronize()
        _ = dc_i^
        _ = dv_i^
        _ = dx_i^
        _ = dr_i^
        _ = d_fl^
        _ = ctx_i^
        return
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
#: are chunked). No runtime switch: the A/B arm is main's build.
#: lane/fam-neighbors (2026-10-04): also the IDENTICAL default on every
#: vendor (XN_NC_IDN_CHUNKED, x_neighbors/items.mojo), with the pinned
#: arithmetic in the `comptime if XN_NC_IDN_CHUNKED` arms below; the host
#: column folds the same chunks.
from x_neighbors.items import XN_NC_IDN_CHUNKED, XN_NC_CHUNK_ROWS
comptime XN_NC_CHUNKED = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()) or XN_NC_IDN_CHUNKED
comptime NCC_ROWS = XN_NC_CHUNK_ROWS


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
                    comptime if XN_NC_IDN_CHUNKED:
                        for r in range(rows):
                            if g >= ncl or Int(ls[r]) == g:
                                a = _add(a, xs[r * NCS_TC + c])
                                m += 1
                    else:
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
        comptime if XN_NC_IDN_CHUNKED:
            for ch in range(Int(nch_)):
                a = _add(a, psum.unsafe_load((ch * (ncl + 1) + g) * d + f))
                m = _add(m, pcnt.unsafe_load((ch * (ncl + 1) + g) * d + f))
            if g < ncl:
                if m > 0:
                    cent.unsafe_store(g * d + f, ftz(identical_div(a, m)))
                else:
                    cent.unsafe_store(g * d + f, Float32(0))
            else:
                dsc.unsafe_store(f, ftz(identical_div(a, Float32(Int(n_)))))
        else:
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
            comptime if XN_NC_IDN_CHUNKED:
                for r in range(rows):
                    var df = _sub(xs[r * NCS_TC + tid], cent.unsafe_load(Int(ls[r]) * d + f))
                    ss = ftz(identical_mul_add(df, df, ss))
            else:
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
        var dof = Int(n_) - Int(nc_)
        comptime if XN_NC_IDN_CHUNKED:
            for ch in range(Int(nch_)):
                ss = _add(ss, pss.unsafe_load(ch * d + f))
            if dof > 0:
                std.unsafe_store(f, ftz(identical_sqrt(ftz(identical_div(ss, Float32(dof))))))
            else:
                std.unsafe_store(f, Float32(0))
        else:
            for ch in range(Int(nch_)):
                ss += pss.unsafe_load(ch * d + f)
            std.unsafe_store(f, sqrt(ss / Float32(dof)) if dof > 0 else Float32(0))


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
    # FAST on Apple: rows in NCC_ROWS chunks across the grid, then a fold per
    # cell; otherwise one block per feature tile walking every row.
    var chunked = False
    comptime if XN_NC_CHUNKED:
        chunked = n > NCC_ROWS
    var nch = (n + NCC_ROWS - 1) // NCC_ROWS if chunked else 1
    var cells = (n_classes + 1) * d
    var d_ps = ctx.enqueue_create_buffer[DType.float32](nch * cells if chunked else 1)
    var d_pc = ctx.enqueue_create_buffer[DType.float32](nch * cells if chunked else 1)
    var d_pss = ctx.enqueue_create_buffer[DType.float32](nch * d if chunked else 1)
    if chunked:
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
    else:
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
    _ = d_ps^
    _ = d_pc^
    _ = d_pss^
    _ = d_x^
    _ = d_lab^
    _ = d_cent^
    _ = d_std^
    _ = d_dsc^


# ---------------------------------------------------------------- lp_iterate_knn
#: LP_FAST_RESIDENT (lane/apple-fast-neighbors2, 2026-10-02; FAST + Apple
#: default, `-D MOJOLEARN_LP_FAST_RESIDENT_OFF` off): LabelPropagation /
#: LabelSpreading's fit loop over the compact kNN graph as ONE resident op.
#: Cause: with kernel='knn' `_LabelPropagationBase.fit`
#: (python/mojolearn/_expansion_neighbors.py) runs the loop in Python, three
#: binding calls per iteration: `absdiff_sum` (a one-item serial fold over
#: n x C on the host), `lp_knn_product` (uploads cols, vals and the
#: distributions, downloads the product) and `lp_clamp` / `ls_clamp` (upload,
#: download) -- about 10 MB across the boundary and two drains per iteration
#: at the board's 200,000 rows, for up to 1,000 iterations. Here the graph
#: goes up once, the stopping sum is block partials + a one-block fold on the
#: device (`pinned_block_sum`, FAST's block sum), the stopping test sets a
#: device flag the later kernels of the batch honour, and LPK_BATCH iterations
#: are enqueued per drain. The product is the finite-x item (a kNN graph's
#: values and the distributions are finite); the fold order differs from the
#: item's ascending chain (FAST promises quality, not bits); tol is compared
#: in float32 on the device. Expected: the per-iteration boundary crossings
#: gone (LabelPropagation taxi runs its 1,000 steps).
comptime LPK_TPB = 256
comptime LPK_BATCH = C33_CHUNK if C33_FROZEN_CHUNKS else 16
comptime LPK_FAST_BUILD = GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL
#: LP_FAST_RESIDENT: the default on FAST + Apple since the M3 A/B
#: (n2-lp-res-taxi: label-propagation taxi 6,793 -> 2,086 ms, -69%, accuracy
#: .7016 same). `-D MOJOLEARN_LP_FAST_RESIDENT_OFF` keeps the Python loop;
#: the old `MOJOLEARN_LP_FAST_RESIDENT` env name is harmless (no longer
#: read). The Python layer asks `x_neighbors_lp_fast_resident()`.
comptime LP_FAST_RESIDENT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_LP_FAST_RESIDENT_OFF"]()
)


#: lane/fam2-neighbors (2026-10-04), IDENTICAL on every vendor, default ON:
#: the same loop resident in IDENTICAL. The Python loop paid three binding
#: calls per iteration (absdiff_sum, lp_knn_product, lp_clamp / ls_clamp:
#: the distributions up and down three times, a host finiteness walk over
#: them, `lp_knn_finite`, and a scalar readback every iteration). Here the
#: graph goes up once and every launch is the op's own item:
#: `absdiff_part_item` + `absdiff_fin_item` (the `absdiff_sum` op's bits),
#: the stopping test against the float32 ceiling of tol (the same decision
#: as Python's float64 compare of a float32 sum), the finiteness scan on the
#: device, `lp_knn_product_item`, `lp_clamp_item` / `ls_clamp_item`. The
#: stop flag gates the later launches of the batch; LPK_BATCH iterations per
#: drain. No bit changes: the host column keeps the Python loop over the
#: same items. -D MOJOLEARN_IDN_LP_RESIDENT_OFF (or MOJOLEARN_IDN_ALL_OFF)
#: restores the Python loop.
comptime LP_IDN_RESIDENT = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_LP_RESIDENT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


def lp_fast_resident_binding() raises -> PythonObject:
    """1 when this binary takes the resident kNN-graph loop by default
    (LP_FAST_RESIDENT on FAST + Apple, LP_IDN_RESIDENT on IDENTICAL)."""
    comptime if LP_FAST_RESIDENT or LP_IDN_RESIDENT:
        return PythonObject(1)
    return PythonObject(0)


def lpki_absdiff_part_kernel(a: FP, b: FP, part: FP, flag: IP, count_: Int64):
    """`absdiff_part_item` per fold block; nothing once the flag is set."""
    if flag.unsafe_load(0) != 0:
        return
    var count = Int(count_)
    var t = _tid()
    if t < xn_fold_blocks(count):
        absdiff_part_item(t, a, b, part, part, count)


def lpki_check_kernel(a: FP, b: FP, sres: FP, part: FP, flag: IP, count_: Int64, tol_: Float32):
    """ONE item: `absdiff_fin_item`, then the stopping test. Below tol the
    stop flag (flag[0]) is set (the loop stops before this step), else the
    step is counted (flag[1]) and the finiteness flag (flag[2]) is cleared
    for this step's scan."""
    if flag.unsafe_load(0) != 0:
        return
    var t = _tid()
    if t < 1:
        absdiff_fin_item(t, a, b, sres, part, Int(count_))
        if sres.unsafe_load(0) < tol_:
            flag.unsafe_store(0, Int32(1))
        else:
            flag.unsafe_store(1, flag.unsafe_load(1) + Int32(1))
            flag.unsafe_store(2, Int32(0))


def lpki_nonfinite_kernel(x: FP, flag: IP, count_: Int64):
    """flag[2] = 1 when any x is Inf or NaN (`lp_knn_finite`'s test, every
    element its own thread); nothing once the stop flag is set."""
    if flag.unsafe_load(0) != 0:
        return
    var t = _tid()
    if t < Int(count_):
        var bits = bitcast[DType.uint32](x.unsafe_load(t)) & UInt32(0x7F800000)
        if bits == UInt32(0x7F800000):
            flag.unsafe_store(2, Int32(1))


def lpki_product_kernel(cols: IP, vals: FP, x: FP, res: FP, flag: IP, n_: Int64, m_: Int64, k_: Int64, c_: Int64):
    """`lp_knn_product_item` with the device's finiteness flag; nothing once
    the stop flag is set."""
    if flag.unsafe_load(0) != 0:
        return
    var t = _tid()
    if t < Int(n_) * Int(c_):
        lp_knn_product_item(t, cols, vals, x, res, Int(n_), Int(m_), Int(k_), Int(c_), flag.unsafe_load(2) == 0)


def _lpki_tol_ceil(tol: Float64) -> Float32:
    """The least float32 >= tol, so `s < result` decides `Float64(s) < tol`
    for every float32 s (the sum is >= 0: a tol <= 0 never stops)."""
    if not (tol > Float64(0)):
        return Float32(0)
    var t32 = Float32(tol)
    if Float64(t32) < tol:
        t32 = bitcast[DType.float32](bitcast[DType.uint32](t32) + UInt32(1))
    return t32


def _op_lp_iterate_knn_idn(
    cols: Int, vals: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, k: Int, c: Int, max_iter: Int, variant: Int, tol: Float64, alpha: Float32,
) raises:
    """LP_IDN_RESIDENT: `_LabelPropagationBase.fit`'s Python loop over the
    compact kNN graph, resident, every launch the op's own item."""
    var nc = n * c
    var nfb = xn_fold_blocks(nc)
    var ctx = xn_ctx()
    var d_cols = _buf_i(ctx, cols, n * k, True)
    var d_vals = _buf(ctx, vals, n * k, True)
    var d_a = _buf(ctx, ld, nc, True)
    var d_b = _buf(ctx, 0, nc, False)
    ctx.enqueue_memset(d_b, Float32(0))
    var d_nxt = _buf(ctx, 0, nc, False)
    var d_ys = _buf(ctx, ystatic, nc, True)
    var d_unl = _buf_i(ctx, unlabeled, n, True)
    var d_s = _buf(ctx, 0, 1, False)
    var d_part = _buf(ctx, 0, nfb, False)
    var h_fl = List[Int32](length=4, fill=Int32(0))
    var d_fl = _buf_i(ctx, Int(h_fl.unsafe_ptr()), 4, True)
    var flp: IP = d_fl.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var prev: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    var tol32 = _lpki_tol_ceil(tol)
    var done = 0
    var converged = False
    while done < max_iter and not converged and nc > 0:
        var steps = min(LPK_BATCH, max_iter - done)
        for _ in range(steps):
            ctx.enqueue_function[lpki_absdiff_part_kernel](
                cur, prev, d_part.unsafe_ptr(), flp, Int64(nc),
                grid_dim=_grid(nfb), block_dim=(BLOCK if nfb > 1 else 1),
            )
            ctx.enqueue_function[lpki_check_kernel](
                cur, prev, d_s.unsafe_ptr(), d_part.unsafe_ptr(), flp, Int64(nc), tol32,
                grid_dim=1, block_dim=1,
            )
            ctx.enqueue_function[lpki_nonfinite_kernel](
                cur, flp, Int64(nc), grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
            ctx.enqueue_function[lpki_product_kernel](
                d_cols.unsafe_ptr(), d_vals.unsafe_ptr(), cur, d_nxt.unsafe_ptr(), flp,
                Int64(n), Int64(n), Int64(k), Int64(c), grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
            # prev = ld; ld = clamp(nxt): the clamp writes over the buffer
            # the old prev held, then the two names swap (op_lp_iterate)
            ctx.enqueue_function[lpk_clamp_kernel](
                d_nxt.unsafe_ptr(), d_ys.unsafe_ptr(), d_unl.unsafe_ptr(), prev, flp,
                Int64(n), Int64(c), Int64(variant), alpha,
                grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
            )
            var t = cur
            cur = prev
            prev = t
        ctx.enqueue_copy(dst_ptr=h_fl.unsafe_ptr(), src_buf=d_fl)
        ctx.synchronize()
        converged = h_fl[0] != 0
        done += steps
    # steps taken: the loop's n_iter_ whether it converged (it) or ran out
    var n_iter = Int(h_fl[1])
    if nc == 0:
        # the Python loop over nothing: the empty sum 0 against tol
        converged = Float64(0) < tol
        n_iter = 0 if (converged or max_iter <= 0) else max_iter
    elif n_iter % 2 == 0:
        _down(ctx, d_a, ld, nc)
    else:
        _down(ctx, d_b, ld, nc)
    ctx.synchronize()
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(n_iter))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    _ = h_fl^
    _ = d_fl^
    _ = d_s^
    _ = d_part^
    _ = d_cols^
    _ = d_vals^
    _ = d_a^
    _ = d_b^
    _ = d_nxt^
    _ = d_ys^
    _ = d_unl^
    _ = ctx^


def lpk_absdiff_partial_kernel(a: FP, b: FP, part: FP, flag: IP, count_: Int64):
    """Block partials of sum |a - b|; nothing once the flag is set."""
    if flag.unsafe_load(0) != 0:
        return
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * LPK_TPB + tid
    var v = Float32(0)
    if i < Int(count_):
        v = abs(a.unsafe_load(i) - b.unsafe_load(i))
    var s = pinned_block_sum[LPK_TPB](v)
    if tid == 0:
        part.unsafe_store(Int(block_idx.x), s)


def lpk_check_kernel(part: FP, nparts_: Int64, flag: IP, iters: IP, tol_: Float32):
    """ONE block of LPK_TPB threads: the partials folded; below tol the flag
    is set (the loop stops before this step), else the step is counted."""
    if flag.unsafe_load(0) != 0:
        return
    var tid = Int(thread_idx.x)
    var acc = Float32(0)
    var i = tid
    while i < Int(nparts_):
        acc += part.unsafe_load(i)
        i += LPK_TPB
    var s = pinned_block_sum[LPK_TPB](acc)
    if tid == 0:
        if s < tol_:
            flag.unsafe_store(0, 1)
        else:
            iters.unsafe_store(0, iters.unsafe_load(0) + 1)


def lpk_product_kernel(cols: IP, vals: FP, x: FP, res: FP, flag: IP, n_: Int64, k_: Int64, c_: Int64):
    """`lp_knn_product_item` (finite x); nothing once the flag is set."""
    if flag.unsafe_load(0) != 0:
        return
    var t = _tid()
    if t < Int(n_) * Int(c_):
        lp_knn_product_item(t, cols, vals, x, res, Int(n_), Int(n_), Int(k_), Int(c_), True)


def lpk_clamp_kernel(
    ld: FP, ystatic: FP, unlabeled: IP, res: FP, flag: IP, n_: Int64, c_: Int64, variant_: Int64, alpha_: Float32,
):
    """variant 0 `lp_clamp_item` per row, 1 `ls_clamp_item` per cell;
    nothing once the flag is set."""
    if flag.unsafe_load(0) != 0:
        return
    var t = _tid()
    var n = Int(n_)
    var c = Int(c_)
    if variant_ == 0:
        if t < n:
            lp_clamp_item(t, ld, ystatic, unlabeled, res, n, c)
    else:
        if t < n * c:
            ls_clamp_item(t, ld, ystatic, res, n * c, alpha_)


def op_lp_iterate_knn(
    cols: Int, vals: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, k: Int, c: Int, max_iter: Int, variant: Int, tol_hi: Int, tol_lo: Int, alpha: Float32,
) raises:
    """The Python loop of `_LabelPropagationBase.fit` over the compact kNN
    graph, resident (see LPK_TPB above). `ld` in: the initial distributions,
    out: the last. info (int32 x 2): n_iter_, converged. tol is Python's
    float64 bits."""
    comptime if not LPK_FAST_BUILD:
        comptime if LP_IDN_RESIDENT:
            _op_lp_iterate_knn_idn(
                cols, vals, ld, ystatic, unlabeled, info, n, k, c, max_iter, variant,
                bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo)), alpha,
            )
        else:
            raise Error("lp_iterate_knn: the FAST tier only (LP_FAST_RESIDENT), or IDENTICAL without MOJOLEARN_IDN_LP_RESIDENT_OFF")
    else:
        var tol = bitcast[DType.float64]((UInt64(tol_hi) << UInt64(32)) | UInt64(tol_lo))
        var nc = n * c
        var ctx = xn_ctx()
        var d_cols = _buf_i(ctx, cols, n * k, True)
        var d_vals = ctx.enqueue_create_buffer[DType.float32](n * k)
        ctx.enqueue_copy(dst_buf=d_vals, src_ptr=FP(unsafe_from_address=vals))
        var d_a = ctx.enqueue_create_buffer[DType.float32](nc)
        ctx.enqueue_copy(dst_buf=d_a, src_ptr=FP(unsafe_from_address=ld))
        var d_b = ctx.enqueue_create_buffer[DType.float32](nc)
        ctx.enqueue_memset(d_b, Float32(0))
        var d_nxt = ctx.enqueue_create_buffer[DType.float32](nc)
        var d_ys = ctx.enqueue_create_buffer[DType.float32](nc)
        ctx.enqueue_copy(dst_buf=d_ys, src_ptr=FP(unsafe_from_address=ystatic))
        var d_unl = _buf_i(ctx, unlabeled, n, True)
        var nparts = max(1, (nc + LPK_TPB - 1) // LPK_TPB)
        var d_part = ctx.enqueue_create_buffer[DType.float32](nparts)
        var h_fl = List[Int32](length=2, fill=Int32(0))
        var d_fl = _buf_i(ctx, Int(h_fl.unsafe_ptr()), 2, True)
        var flp: IP = d_fl.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var itp: IP = flp + 1
        var cur: FP = d_a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var prev: FP = d_b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        var tol32 = Float32(tol)
        var done = 0
        var converged = False
        while done < max_iter and not converged:
            var steps = min(LPK_BATCH, max_iter - done)
            for _ in range(steps):
                ctx.enqueue_function[lpk_absdiff_partial_kernel](
                    cur, prev, d_part.unsafe_ptr(), flp, Int64(nc),
                    grid_dim=nparts, block_dim=LPK_TPB,
                )
                ctx.enqueue_function[lpk_check_kernel](
                    d_part.unsafe_ptr(), Int64(nparts), flp, itp, tol32,
                    grid_dim=1, block_dim=LPK_TPB,
                )
                ctx.enqueue_function[lpk_product_kernel](
                    d_cols.unsafe_ptr(), d_vals.unsafe_ptr(), cur, d_nxt.unsafe_ptr(), flp,
                    Int64(n), Int64(k), Int64(c), grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
                )
                # prev = ld; ld = clamp(nxt): the clamp writes over the buffer
                # the old prev held, then the two names swap (op_lp_iterate)
                ctx.enqueue_function[lpk_clamp_kernel](
                    d_nxt.unsafe_ptr(), d_ys.unsafe_ptr(), d_unl.unsafe_ptr(), prev, flp,
                    Int64(n), Int64(c), Int64(variant), alpha,
                    grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
                )
                var t = cur
                cur = prev
                prev = t
            ctx.enqueue_copy(dst_ptr=h_fl.unsafe_ptr(), src_buf=d_fl)
            ctx.synchronize()
            converged = h_fl[0] != 0
            done += steps
        # steps taken: the loop's n_iter_ whether it converged (it) or ran out
        var n_iter = Int(h_fl[1])
        if n_iter % 2 == 0:
            ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=ld), src_buf=d_a)
        else:
            ctx.enqueue_copy(dst_ptr=FP(unsafe_from_address=ld), src_buf=d_b)
        ctx.synchronize()
        var inf = IP(unsafe_from_address=info)
        inf.unsafe_store(0, Int32(n_iter))
        inf.unsafe_store(1, Int32(1 if converged else 0))
        _ = h_fl^
        _ = d_fl^
        _ = d_part^
        _ = d_cols^
        _ = d_vals^
        _ = d_a^
        _ = d_b^
        _ = d_nxt^
        _ = d_ys^
        _ = d_unl^
        _ = ctx^
