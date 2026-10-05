# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/apple-fast-w4-decomp (2026-10-04): FAST + Apple candidates.

LLE_FAST_DEV_LU (`-D MOJOLEARN_LLE_FAST_DEV_LU`): LocallyLinearEmbedding's
shift-invert factor F0 (n x n, 10,000 x 10,000 = 400 MB at the board's n)
is factored where it already lives. Main (`_lle_smallest` ->
`_Kit.lu` + `_Kit.lu_aux`) moved it through the host five times: F0.copy()
downloads it (and copies it into a fresh array), `x_decomp_lu` uploads it,
factors it and downloads the factor, `x_decomp_lu_aux` uploads the factor
and (clamp) downloads it again, and the first `trisolve` uploads it once
more. On Apple a download into host memory runs at ~3 GB/s (~130 ms per
400 MB) and fresh host pages cost about as much again. Here
`dev_lu_aux_py` copies F0 into a pooled device matrix and runs
`launch_lu` then DevExec.lu_aux's five launches on it, enqueued, the
same kernels in the same order on the same words: the factor, the pivots'
row orders, the diagonal and the four stats are bit-identical to main's.
Only the 4 stats come back to the host.

RSVD_FAST_DIRECT_IN (FAST + Apple default; `-D MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF`
rolls back): randomized_svd's
input goes up from the caller's own buffer into a pooled device matrix
(`_M._on_device` + `x_decomp_dev_upload`, the random projections' and
KernelPCA's resident pattern) instead of `_M.from_input`'s copy into a fresh
`array.array` (880 MB of fresh host pages at the board's 1,000,000 x 220)
that the first product then uploads. The host finiteness refusal
(`_host_all_finite`) is unchanged. Same words reach the device: no bit moves.

Both are read by Python through `x_decomp_w4_flags()` (bit 1 LLE_FAST_DEV_LU,
bit 2 RSVD_FAST_DIRECT_IN), registered on every GPU build so a quality
capture can prove which arm is installed.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_decomp.cells import F32Ptr, I32Ptr
from x_decomp.rr import RR_OFF_TPB
from x_decomp.device import (
    LU_SCAL_LEN,
    TPB,
    _blocks,
    _pj_off_blocks,
    launch_lu,
    lu_aux_clamp_kernel,
    lu_aux_fold_kernel,
    lu_aux_part_kernel,
    lu_perm_kernel,
    xd_ctx,
)
from x_decomp.resident import _id, _n, _ptr

#: HOLD-quality, opt-in only (`-D MOJOLEARN_LLE_FAST_DEV_LU`), measured source
#: e9d72edb5 (2026-10-04). M3 afc_ab_def, full board size, 1 run per arm: lle
#: taxi 1783.8 -> 1039.0 ms, but trustworthiness_k15 0.8662 -> 0.8410 vs FAST
#: main (arm A) is a real drop, not noise, so default OFF (FAST quality must
#: not go down vs FAST main and must match the best opponent).
#: 2026-10-05 re-check with LLE_FAST_NULL_CANON (M3 rab14, 1 run per arm), DEV_LU + CANON vs FAST main:
#:   lle istella 1803.9 -> 1048.9 ms (-41.9%), trustworthiness_k15 0.8851 -> 0.8615 (-2.7%, consistent across runs);
#:   lle taxi    1745.1 ->  968.0 ms (-44.5%), trustworthiness_k15 0.8105 -> 0.8420 (arm A itself ranges 0.81-0.87 run to run).
#:   sklearn-cpu trustworthiness: istella 0.8491, taxi 0.7708, so DEV_LU stays above the opponent on both.
#:   Andrew (2026-10-05): 2.7% on istella is a lot, so it stays OFF. Re-judge if the istella drop goes away.
comptime LLE_FAST_DEV_LU = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_LLE_FAST_DEV_LU"]()
)
#: lane/apple-fast-s-shap (2026-10-04), READY-AB, opt-in
#: (`-D MOJOLEARN_LLE_FAST_NULL_CANON`, FAST + Apple): a canonical answer when
#: LLE's null space is wider than n_components. Why LLE_FAST_DEV_LU moved
#: trustworthiness: taxi's kNN graph has several components (near-duplicate
#: rows), so F^'s numerical null space N has more than nc dimensions and ANY
#: nc of them is a correct answer (`_LLE_NULL_FLOOR`). Inside N the
#: shift-invert operator's values are 1 / sigma^2 of float32 rounding noise,
#: so WHICH nc directions the iteration settles on is decided by the LU's
#: last-bit rounding: a rounding-level LU change (LU_FAST_MMA's sum order,
#: the device LU's words, the trisolve order) re-draws the embedding. That
#: is how FAST main itself went 0.826 (scalar LU, digest 68a3bf15) -> 0.866
#: (MMA LU, 75f6322f) and DEV_LU 0.841 (45185107): three draws, not three
#: precisions (w4q-v1 on non-degenerate 3,000-row fixtures: trust |A - B|
#: 1.3e-5, angle 2.9e-5 rad). With this on, once the wanted Ritz values are
#: under the floor, `_lle_smallest` takes ALL of N (every Ritz value under
#: the floor, p widened when every column is null) and returns the nc
#: directions of N along which the input data varies most (top left singular
#: vectors of V_N^T X): a function of N and X only, the same on any LU.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-05,
#: verdicts batch 6): lle trustworthiness down on both datasets. DROPPED:
#: stays off (opt-in only).
comptime LLE_FAST_NULL_CANON = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_LLE_FAST_NULL_CANON"]()
)
#: FAST Apple default, measured source e9d72edb5 (2026-10-04).
#: M3 afc_ab_def, full board size, 1 run per arm: randomized-svd istella
#: 517.3 -> 501.3 ms, taxi 200.8 -> 198.5 ms; relative_reconstruction_error
#: equal (istella 2.3594584442703e-4 -> 2.3594584442301e-4, taxi
#: 0.0271967625509 -> 0.0271967625509). ACCEPT.
#: `-D MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF` restores `_M.from_input`'s host copy.
comptime RSVD_FAST_DIRECT_IN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_RSVD_FAST_DIRECT_IN_OFF"]()
)


def w4_flags_py() raises -> PythonObject:
    """Which w4 candidates this build compiled in (bit 1 LLE_FAST_DEV_LU,
    bit 2 RSVD_FAST_DIRECT_IN, bit 4 LLE_FAST_NULL_CANON)."""
    var f = 0
    comptime if LLE_FAST_DEV_LU:
        f |= 1
    comptime if RSVD_FAST_DIRECT_IN:
        f |= 2
    comptime if LLE_FAST_NULL_CANON:
        f |= 4
    return PythonObject(f)


def w4_copy_kernel(dst: F32Ptr, src: F32Ptr, n: Int32):
    """dst[i] = src[i], a thread per word."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        dst.unsafe_store(i, src.unsafe_load(i))


def dev_lu_aux_py(
    a: PythonObject,
    lu: PythonObject,
    pm: PythonObject,
    im: PythonObject,
    diag: PythonObject,
    st: PythonObject,
    p: PythonObject,
) raises -> PythonObject:
    """p = [n, clamp]. lu = the LU factor of device matrix a (a untouched;
    `launch_lu`, DevExec.lu's launches), then DevExec.lu_aux's launches on
    it: pm / im the swaps' row order and its inverse (n x 1), diag (1 x n),
    st = (max |u_ii|, zero pivots, negative pivots, swaps), and with clamp
    the pivots under eps max |u_jj| floored in lu. Waits once at the end
    (the scratch buffers are this call's). Returns n."""
    var n = _n(p, 0)
    var clamp = _n(p, 1)
    if n <= 0 or n * n > 2147483647:
        raise Error("x_decomp: dev_lu_aux needs 0 < n and n * n inside Int32")
    var pa = _ptr(_id(a), n * n)
    var pl = _ptr(_id(lu), n * n)
    var ppm = _ptr(_id(pm), n)
    var pim = _ptr(_id(im), n)
    var pd = _ptr(_id(diag), n)
    var pst = _ptr(_id(st), 4)
    with GILReleased(Python()):
        var ctx = xd_ctx()
        var nb = _pj_off_blocks(n)
        var dp = ctx.enqueue_create_buffer[DType.int32](n)
        var di = ctx.enqueue_create_buffer[DType.float32](1)
        var ds = ctx.enqueue_create_buffer[DType.float32](LU_SCAL_LEN)
        var dact = ctx.enqueue_create_buffer[DType.float32](n)
        var dpart = ctx.enqueue_create_buffer[DType.float32](4 * nb)
        ctx.enqueue_function[w4_copy_kernel](pl, pa, Int32(n * n), grid_dim=_blocks(n * n), block_dim=TPB)
        var pp = I32Ptr(unsafe_from_address=Int(dp.unsafe_ptr()))
        launch_lu(
            ctx, pl, pp, F32Ptr(unsafe_from_address=Int(di.unsafe_ptr())),
            F32Ptr(unsafe_from_address=Int(ds.unsafe_ptr())), F32Ptr(unsafe_from_address=Int(dact.unsafe_ptr())), n,
        )
        var ppart = F32Ptr(unsafe_from_address=Int(dpart.unsafe_ptr()))
        ctx.enqueue_function[lu_perm_kernel](pp, ppm, Int32(n), Int32(0), grid_dim=_blocks(n), block_dim=TPB)
        ctx.enqueue_function[lu_perm_kernel](pp, pim, Int32(n), Int32(1), grid_dim=_blocks(n), block_dim=TPB)
        ctx.enqueue_function[lu_aux_part_kernel](pl, pp, ppart, Int32(n), grid_dim=nb, block_dim=RR_OFF_TPB)
        ctx.enqueue_function[lu_aux_fold_kernel](ppart, pst, Int32(nb), grid_dim=1, block_dim=RR_OFF_TPB)
        ctx.enqueue_function[lu_aux_clamp_kernel](
            pl, pd, pst, Int32(n), Int32(clamp), grid_dim=_blocks(n), block_dim=TPB
        )
        ctx.synchronize()
        _ = dp^
        _ = di^
        _ = ds^
        _ = dact^
        _ = dpart^
    return PythonObject(n)
