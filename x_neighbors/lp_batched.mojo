# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LabelPropagation / LabelSpreading's fit loop in batches behind one drain
(lane neighbors-apple3, 2026-09-28). Its own module, imported only where a
build selects it (x_neighbors/iter_device.mojo, `-D MOJOLEARN_XN_LP_BATCH`)."""
from std.memory import bitcast
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from x_neighbors.device_ops import (
    xn_ctx, _buf, _buf_i, _grid, BLOCK,
    matmul_kernel, lp_clamp_kernel, ls_clamp_kernel,
)
from x_neighbors.items import FP, IP, absdiff_sum_item
from x_neighbors.lp_spmm import lp_spmm_kernel

#: lane neighbors-apple3 (2026-09-28), OPT-IN until its A/B passes
#: (`-D MOJOLEARN_XN_LP_BATCH`): the iteration in batches of LP_BATCH steps
#: behind ONE drain. `op_lp_iterate` drains once per step to read the
#: distributions for the stopping sum (about 0.2 ms a step on Apple;
#: LabelPropagation on taxi runs its 1,000 steps).
comptime LP_BATCH = 16


def _lp_all_finite(p: FP, count: Int) -> Bool:
    for q in range(count):
        var bits = bitcast[DType.uint32](p.unsafe_load(q)) & UInt32(0x7F800000)
        if bits == UInt32(0x7F800000):
            return False
    return True


def op_lp_iterate_batched(
    g: Int, ld: Int, ystatic: Int, unlabeled: Int, info: Int,
    n: Int, c: Int, max_iter: Int, variant: Int, tol: Float64,
    alpha: Float32,
) raises:
    """`op_lp_iterate`: the same kernels in the same order on the same
    values, and the same stopping decisions. State S_0 is the caller's
    `ld`; a step makes S_(i+1) from S_i. A batch enqueues up to LP_BATCH
    steps, each writing its own slot of one device buffer, and reads the
    slots back in one copy. The host then walks them in order and makes,
    for every state that a later step of the batch was computed from, the
    two decisions `op_lp_iterate` makes before that step: the stopping sum
    against the state before it (stop: the answer is that state, the later
    slots are dropped) and the finiteness test of the sparse product (not
    finite: the later slots are dropped and the next step runs alone with
    the dense kernel)."""
    var nc = n * c
    var ctx = xn_ctx()
    var d_g = _buf(ctx, g, n * n, True)
    var d_ring = _buf(ctx, 0, (LP_BATCH + 1) * nc, False)
    var d_nxt = _buf(ctx, 0, nc, False)
    var d_ys = _buf(ctx, ystatic, nc, True)
    var d_unl = _buf_i(ctx, unlabeled, n, True)
    var views = List[DeviceBuffer[DType.float32]]()
    for sl in range(LP_BATCH + 1):
        views.append(d_ring.create_sub_buffer[DType.float32](sl * nc, nc))
    var d_tail = d_ring.create_sub_buffer[DType.float32](nc, LP_BATCH * nc)
    var hs = List[Float32](length=1, fill=Float32(0))
    var hc = List[Float32](length=nc, fill=Float32(0))
    var hp = List[Float32](length=nc, fill=Float32(0))
    var hr = List[Float32](length=LP_BATCH * nc, fill=Float32(0))
    var p_hc = FP(unsafe_from_address=Int(hc.unsafe_ptr()))
    var p_hp = FP(unsafe_from_address=Int(hp.unsafe_ptr()))
    var p_hs = FP(unsafe_from_address=Int(hs.unsafe_ptr()))
    var p_hr = FP(unsafe_from_address=Int(hr.unsafe_ptr()))
    var p_ld = FP(unsafe_from_address=ld)
    for q in range(nc):
        p_hc.unsafe_store(q, p_ld.unsafe_load(q))
    # G's nonzero entries, as `op_lp_iterate` builds them
    var pg = FP(unsafe_from_address=g)
    var h_ip = List[Int32](length=n + 1, fill=Int32(0))
    var h_cols = List[Int32]()
    var h_vals = List[Float32]()
    var use_sparse = False
    comptime if not is_defined["MOJOLEARN_XN_LP_DENSE"]():
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
    var it = 0
    var converged = False
    while it < max_iter:
        # the decisions before the step from S_it (hc), S_(it-1) in hp
        absdiff_sum_item(0, p_hc, p_hp, p_hs, nc)
        if Float64(hs[0]) < tol:
            converged = True
            break
        var sparse_now = use_sparse and _lp_all_finite(p_hc, nc)
        var steps = min(LP_BATCH, max_iter - it)
        if use_sparse and not sparse_now:
            steps = 1
        ctx.enqueue_copy(dst_buf=views[0], src_ptr=p_hc)
        for j in range(steps):
            if sparse_now:
                ctx.enqueue_function[lp_spmm_kernel](
                    d_ip.unsafe_ptr(), d_cols.unsafe_ptr(), d_vals.unsafe_ptr(), views[j].unsafe_ptr(),
                    d_nxt.unsafe_ptr(), Int64(n), Int64(c),
                    grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
                )
            else:
                ctx.enqueue_function[matmul_kernel](
                    d_g.unsafe_ptr(), views[j].unsafe_ptr(), d_nxt.unsafe_ptr(), Int64(n), Int64(n), Int64(c),
                    grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
                )
            if variant == 0:
                ctx.enqueue_function[lp_clamp_kernel](
                    d_nxt.unsafe_ptr(), d_ys.unsafe_ptr(), d_unl.unsafe_ptr(), views[j + 1].unsafe_ptr(),
                    Int64(n), Int64(c),
                    grid_dim=_grid(n), block_dim=(BLOCK if n > 1 else 1),
                )
            else:
                ctx.enqueue_function[ls_clamp_kernel](
                    d_nxt.unsafe_ptr(), d_ys.unsafe_ptr(), views[j + 1].unsafe_ptr(), Int64(nc), alpha,
                    grid_dim=_grid(nc), block_dim=(BLOCK if nc > 1 else 1),
                )
        ctx.enqueue_copy(dst_ptr=p_hr, src_buf=d_tail)
        ctx.synchronize()
        var w = 0
        while w < steps:
            # S_(it+1) is slot w + 1, the host ring's row w
            for q in range(nc):
                p_hp.unsafe_store(q, p_hc.unsafe_load(q))
                p_hc.unsafe_store(q, p_hr.unsafe_load(w * nc + q))
            it += 1
            w += 1
            if w < steps:
                # a later slot was computed from this state: its decisions
                absdiff_sum_item(0, p_hc, p_hp, p_hs, nc)
                if Float64(hs[0]) < tol:
                    converged = True
                    break
                if use_sparse and not _lp_all_finite(p_hc, nc):
                    break
        if converged:
            break
    for q in range(nc):
        p_ld.unsafe_store(q, p_hc.unsafe_load(q))
    var inf = IP(unsafe_from_address=info)
    inf.unsafe_store(0, Int32(it))
    inf.unsafe_store(1, Int32(1 if converged else 0))
    ctx.synchronize()
    _ = views^
    _ = d_tail^
    _ = hs^
    _ = hc^
    _ = hp^
    _ = hr^
    _ = h_ip^
    _ = h_cols^
    _ = h_vals^
    _ = d_ip^
    _ = d_cols^
    _ = d_vals^
    _ = d_g^
    _ = d_ring^
    _ = d_nxt^
    _ = d_ys^
    _ = d_unl^
    _ = ctx^
