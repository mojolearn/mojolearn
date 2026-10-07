# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE ON THE DEVICE (lane/algos-linear, 2026-09-27).

Pass 1: every fit is x_linear/dispatch.mojo's `fit_dispatch`, run by ONE
device thread, so the GPU executes the host's exact sequence of operations.
Speed phase: a fit `team_fit` names runs on ONE BLOCK of LINEAR_TPB threads
(x_linear/team.mojo): each stored value still comes from one thread's
one-thread sequence, so the bits are the host's.
Scoring is one thread per (row, output) pair. A parallel fit schedule with
the same fold order is pass 2's speed work.

Every entry runs on ONE process-lifetime DeviceContext (`linear_ctx`, the
x_cnn `_Global` pattern; CURRENT DIRECTIVES 2026-09-27: a context per call
exhausts Metal's per-process command queues, and x_cluster/x_neighbors hung
on the second GPU call of a process). Each entry's buffers are released
before it returns; the context stays.
"""
from std.gpu import block_idx, block_dim, thread_idx, MAX_THREADS_PER_BLOCK_METADATA
from std.utils import StaticTuple
from std.gpu.primitives.warp import shuffle_idx, shuffle_xor
from std.ffi import _Global
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, NUMERIC_FAST
from x_linear.ops import FP, IP
from x_linear.dispatch import fit_dispatch, decision_one, decision_code_row, team_fit, team_rows, team_own, ALGO_SGD, ALGO_LARS, ALGO_GLM, ALGO_RIDGE_KFOLD, ALGO_ENETCV, ALGO_RIDGE, ALGO_BAYES, ALGO_ARD
from x_linear.cd_grid import enetcv_fit_grid
from x_linear.moments_grid import MOMENTS_GRID, MG_NT, mg_means_kernel, mg_cross_kernel, mg_tiles
from x_linear.ops import ld, st, ldi, fd, i2f, fa, fs, fm, fabs, shuffle, fmad, flog, fill, copy, row_dot, mean_of
from x_linear.ops import sti, fmax, fmin, fsqrt
from std.atomic import Atomic
from x_linear.sgd import (
    sgd_loss, sgd_dloss, sgd_reg_block, _sgd_target, _clip_one,
    ws_mul, ws_div, ws_decay, ws_clip, WS_RESET, ff_add, oc_hinge, oc_offset,
    L_HINGE, LR_INVSCALING, P_NONE, P_EN,
)
from checks.numerics import identical_pow
from experiments.classical_identical_ideas.linear_controls import C13_FOLD_STATS, C17_OVR, C19_SGD_CHUNK, C16_GLM_FUSED
from x_linear.finite_device import XLIN_IDN_DEV_FINITE, xlin_finite_device, xlin_finite_host
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.glm_ydom import XLIN_GLM_DEV_YDOM, GLM_YDOM_REFUSED, glm_ydom_bad
from x_linear.sgd_end import sgd_ys_kernel, sgd_iota_kernel, sgd_perm_kernel, sgd_mb_end_kernel, sgd_mb_res_kernel, sgd_ps_end_kernel, sgd_ps_res_kernel, SGD_END_ST, SGD_END_TPB, SGD_MB_FLAGS, SGD_MB_WORDS
from x_linear.vfold import vscratch
from x_linear.sgd import LR_INVSCALING
from x_linear.sgd import (
    sgd_batch, mb_dot, mb_oc_mode, mb_oc_row, oc_delta, oc_count, oc_count_tie, oc_hinge_at, mb_oc_bias_step,
)
from x_linear.sgd import sgd_perc_avg_on, sgd_perc_avg_from
from x_linear.sgd_avg import sgd_avg_acc_kernel, sgd_avg_fin_kernel
from x_linear.sgd import sgd_mb_on, mb_sub_size, mb_dblk, mb_row, mb_row_dot, mb_rowsq, mb_block_dot, MB_DBLK, LR_PA1, LR_PA2, mb_part, mb_step, mb_bias_step, mb_subs, mb_eta, mb_optimal_init, mb_penalty, LR_OPTIMAL, LR_ADAPTIVE, P_L2, P_L1
from x_linear.bayes import bayes_wy_part, bayes_wx_part, bayes_wgram_part, bayes_wxty_part, bayes_wvar_part, bayes_coef_one
from x_linear.bayes import bayes_prep, bayes_coef, bayes_step, bayes_finish, _sse_part, bayes_eig_prep, bayes_yvar_part, GRAM_SSE_TRUST
from x_linear.classical_fold_stats import fold_stat_words, kfold_mean_cell, kfold_gram_cell, kfold_combine_unit
from x_linear.ridgecv import kf_start, kf_end, kf_mean, kf_cross, kf_solve, kf_pred, kf_score, kf_ff_solve, kf_blocks, kf_ysum_part, kf_sq_part, kf_score_final
from x_linear.ridge import ridge_ff_unit, ridge_ff_units, ridge_ff_solve
from x_linear.tops import t_fold_fa_staged, t_fold_fa_blocked, fold_parts, fold_blocks, FOLD_BLOCK
from x_linear.glm import (
    _unit, _unit_all, _glm_deriv_row, _glm_cell, _glm_slot_count, _glm_slot_cell, GLM_LINK_LOG, GLM_STALL_ITERS,
    glm_g_item, glm_h_item, glm_fwd_col, glm_back_col, glm_slope_part, glm_slope_blocks,
    _glm_cell_part, _glm_cell_store, glm_den, glm_start, glm_start_of,
)
from x_linear.tops import upper_cell, fold_fa, chain_cfmad, chain_fmad
from std.os import getenv
from x_linear.logcv_grid import logcv_fit_grid, lcv_fold_ids_device
from x_linear.huber_grid import huber_fit_grid
from x_linear.huber_fast import HUBER_DEVICE_LBFGS, huber_fit_fast
from x_linear.dispatch import ALGO_HUBER, ALGO_ENETCV
from x_linear.enetcv_fast import enetcv_fast
from x_linear.fast_gram import fast_gram_into, XL_RIDGE_FAST_GRAM
from x_linear.sgdoc_tail import sgdoc_centered
from x_linear.cls1_fast import (
    C1_TPB, C1_BATCH, C1_BAYES_STATE, BAYES_CLS1_STATS, BAYES_CLS1_PARTS, BAYES_CLS1_BATCH,
    RIDGE_CLS1_CODES, c1_sq_parts_kernel, c1_sum_parts_kernel, c1_dev_parts_kernel, c1_codes_targets_kernel,
)
from x_linear.dispatch import ALGO_LOGCV
from x_linear.team import LINEAR_TPB, team_work, device_team, solo, team_barrier
from x_linear.dispatch import ALGO_ISOTONIC, ALGO_ISOTONIC_PREDICT, ALGO_QUANTILE
from x_linear.quantile_grid import quantile_fit_grid
from x_linear.ard_grid import ard_fit_grid
from x_linear.ridge_grid import ridge_fit_grid
from x_linear.lars import lars_fit
from x_linear.isotonic import iso_predict_one, iso_oob_flag, OOB_RAISE, iso_gather_one, iso_group, ISO_CHUNK, iso_pava_chunk, iso_pava_merge, iso_pava_levels, iso_reverse_one, iso_clip_one, iso_keep
from std.memory import bitcast
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.kernel_matrix import TARGET_COLUMN, lib_smem_page_fits_for


struct _LinearContext(Defaultable, Movable):
    """The slot `linear_ctx` fills on first use; one per numeric tier so a
    FAST and an IDENTICAL .so in one process never share it."""
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.ctx = Optional[DeviceContext]()


comptime _CTX_NAME = "MojoXLinearContextIdentical" if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else "MojoXLinearContextFast"
comptime X_LINEAR_CONTEXT = _Global[StorageType=_LinearContext, name=_CTX_NAME, init_fn=_LinearContext.__init__]


def linear_ctx() raises -> DeviceContext:
    """The shared context, created on first use."""
    var slot = X_LINEAR_CONTEXT.get_or_create_ptr()
    if not slot[].ctx:
        slot[].ctx = DeviceContext()
    return slot[].ctx.value().copy()


#: FAST on Apple (lane/apple-fast-bayes, 2026-10-02): BayesianRidge's Gram
#: sse on the grid driver guarded by a reference row pass with an error
#: bound (`bayes_step_guard_kernel`), the guard x_linear/bayes.mojo
#: `bayes_ridge_fit` carries on the one-block fit. The FAST default since the
#: M3 A/B 2026-10-02 (istella: NaN -> finite, r2 equal to the row-pass arm);
#: `-D MOJOLEARN_BAYES_GRID_GUARD_OFF=1` is main's unguarded Gram sse, the A/B
#: arm. IDENTICAL and the other vendors never compile the branch.
comptime BAYES_GRID_GUARD = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_BAYES_GRID_GUARD_OFF"]()
)


def lars_path_kernel(
    x: FP, y: FP, d: Int32, ip: IP, fp: FP, res: FP, fw: FP, iw: IP, tw: FP, wf: IP, woff: Int32, nonce: Int32,
):
    """One block team: LARS's d x d path (x_linear/lars.mojo `lars_fit`, the
    moments already in fw; ip[6] the row count for the alpha scale). No row
    pass runs here (cgr-linear: the one-block fit kernel is gone)."""
    var t = device_team(tw, 0, 3, 0)
    lars_fit(t, x, y, ldi(ip, 6), Int(d), ip, fp, res, fw, iw)
    witness_end(wf, woff, nonce)


def _ridge_device(
    var ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """Ridge on the grid (x_linear/ridge_grid.mojo), then the float-float
    refit when the float32 factor was not trusted (status 1)."""
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if XLIN_IDN_DEV_FINITE:
        xlin_finite_device(ctx, dx, n_x, y, n_y)
    var dxp = FP(unsafe_from_address=Int(dx.unsafe_ptr()))
    var dyp = FP(unsafe_from_address=Int(dy.unsafe_ptr()))
    # ip[4] == 1 means y holds the n int32 class codes (RidgeClassifier),
    # then the n sample weights when ip[3]; the +-1 targets are built here on
    # the device (x_linear/cls1_fast.mojo), the weights copied after them
    # (every column since lane pyglue-numeric: Python built the targets)
    var dyt = ctx.enqueue_create_buffer[DType.float32](1)
    if len(ip) > 4 and Int(ip[4]) == 1 and n > 0:
        var t_c = Int(ip[0])
        var has_sw = Int(ip[3]) != 0
        dyt = ctx.enqueue_create_buffer[DType.float32](n * t_c + (n if has_sw else 0))
        ctx.enqueue_function[c1_codes_targets_kernel](
            dy.unsafe_ptr().bitcast[Int32](), Int32(n), Int32(t_c), dyt.unsafe_ptr(),
            grid_dim=(n + C1_TPB - 1) // C1_TPB, block_dim=C1_TPB,
        )
        if has_sw:
            ctx.enqueue_copy(dst_buf=dyt.create_sub_buffer[DType.float32](n * t_c, n),
                             src_buf=dy.create_sub_buffer[DType.float32](n, n))
        dyp = FP(unsafe_from_address=Int(dyt.unsafe_ptr()))
    ridge_fit_grid(ctx.copy(), dxp, dyp, n, d, ip, fp, n_out, res)
    # lane/neural-pass93: the float-float refit on the grid
    var t_n = Int(ip[0])
    var a_n = Int(ip[2])
    var sidx = t_n * d + t_n + 2 + a_n
    if n_out > sidx and res.unsafe_load(sidx) == Float32(1):
        _ridge_ff_grid(ctx, dxp, dyp, n, d, t_n, Int(ip[1]) != 0, len(ip) > 3 and Int(ip[3]) != 0,
                       res.unsafe_load(t_n * d + t_n), res, sidx)
    ctx.synchronize()
    _ = dx^
    _ = dy^
    _ = dyt^


def decision_kernel(x: FP, wb: FP, n: Int32, d: Int32, k: Int32, link: Int32, res: FP):
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < Int(n) * Int(k):
        var i = t // Int(k)
        var c = t % Int(k)
        res.unsafe_store(t, decision_one(x, i, Int(d), wb, c, Int(link)))


comptime XG_TPB = 256


def _xg_blocks(count: Int) -> Int:
    return (count + XG_TPB - 1) // XG_TPB


def xg_means_kernel(x: FP, n: Int32, d: Int32, fi: Int32, fw: FP, wf: IP, woff: Int32, nonce: Int32):
    """`t_col_means` as a grid: thread j folds column j ascending
    (`fold_fa`) and divides by n, the same statements; zeros without an
    intercept, as `lars_fit` fills them."""
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if j < Int(d):
        if fi != 0:
            st(fw, j, fd(fold_fa(x, j, Int(d), Int(n)), i2f(Int(n))))
        else:
            st(fw, j, Float32(0))
    witness_end(wf, woff, nonce)


def xg_gram_kernel(x: FP, n: Int32, d: Int32, fw: FP, lo: Int32, cnt: Int32, src: FP, dst: FP,
                   wf: IP, woff: Int32, nonce: Int32):
    """`t_centered_gram` as a grid (lane/neural-net-experiment, 2026-09-30,
    the classical pass): one thread per upper-triangle cell, each the same
    `chain_cfmad` over the rows ascending from the means in fw[0, d), into
    G at fw[d, d + d*d). The team form ran the same chains on ONE block of
    256 threads, 96 chains of a million rows per thread at 220 features:
    15.4 s on an L40S for `lars` on istella (bench_board 0.8.25) against
    cuML's 0.064. Per cell the chain is unchanged, so the bits are the
    team form's; only the thread that runs it differs."""
    var dd = Int(d)
    var cells = dd * (dd + 1) // 2
    var c = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if c < cells:
        var jk = upper_cell(c, dd)
        var j = jk[0]
        var k = jk[1]
        # rows [lo, lo + cnt) resuming the cell's chain from `src` (lane/
        # neural-pass131: the Apple slices; one slice of every row elsewhere)
        var init = ld(src, dd + j * dd + k) if lo > 0 else Float32(0)
        var acc = chain_cfmad(x, Int(lo) * dd + j, dd, ld(fw, j), x, Int(lo) * dd + k, dd, ld(fw, k), Int(cnt), init)
        st(dst, dd + j * dd + k, acc)
        st(dst, dd + k * dd + j, acc)
    witness_end(wf, woff, nonce)



# lane/neural-pass131 (2026-10-02): BayesianRidge's iterations with the sse
# on the grid. The fit kernel ran every iteration's sse (a row pass: each
# row's residual, then FOLD_BLOCK partials folded in order) on its one
# block: the peer's L40S, istella 1M x 220 standardized (300 iterations):
# 43.4 s, 141 ms an iteration. Here one block runs `bayes_prep` (the
# statistics, the eigendecomposition, the first coefficients); then per
# iteration a thread per row writes its residual (`_t_sse`'s statements),
# a thread per block its partial (`_sse_part`), and one thread folds them
# (`fold_parts`) and runs `bayes_step` (and the next coefficients); the
# host reads the stop word. `bayes_ridge_fit`'s statements in its order.
def bayes_yparts_kernel(y: FP, n: Int32, yparts: FP, state: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread b: row block b's target sum from zero (`fold_fa`, rows
    ascending), the partials `bayes_ymean` folds blocks ascending; thread 0
    also writes the row count (wsum) into state[3]."""
    var b = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if b < fold_blocks(nn):
        var lo = b * FOLD_BLOCK
        st(yparts, b, fold_fa(y, lo, 1, min(FOLD_BLOCK, nn - lo)))
    if b == 0:
        st(state, 3, i2f(nn))
    witness_end(wf, woff, nonce)

def bayes_yvar_parts_kernel(y: FP, n: Int32, yparts: FP, vparts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread b: row block b's squared deviations from the blocked mean
    (`bayes_yvar_part`), the partials `bayes_yvar` folds."""
    var b = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var nb = fold_blocks(nn)
    if b < nb:
        var m = fd(fold_parts(yparts, 0, nb), i2f(nn))
        var lo = b * FOLD_BLOCK
        st(vparts, b, bayes_yvar_part(y, m, lo, min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)

def bayes_xty_kernel(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, fw: FP, yparts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread j: X'y_j on centered data, `bayes_prep`'s team statement
    (`chain_cfmad` over the rows ascending, from the means at fw[0, d) and
    the blocked target mean), into fw[d + d*d + j]."""
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var nn = Int(n)
    if j < dd:
        var ym = fd(fold_parts(yparts, 0, fold_blocks(nn)), i2f(nn)) if fi != 0 else Float32(0)
        st(fw, dd + dd * dd + j, chain_cfmad(x, j, dd, ld(fw, j), y, 0, 1, ym, nn))
    witness_end(wf, woff, nonce)

# cgr-linear: weighted BayesianRidge's statistics on the grid, the blocked
# order of x_linear/bayes.mojo's host branch (w at y + n): row-block
# partials, a thread per (statistic, block), then a thread per statistic
# folding them blocks ascending.
def bayes_wparts_kernel(y: FP, n: Int32, yparts: FP, wparts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread b: block b's sum w y and sum w from zero."""
    var b = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if b < fold_blocks(nn):
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        st(yparts, b, bayes_wy_part(y, nn, lo, cnt))
        st(wparts, b, fold_fa(y, nn + lo, 1, cnt))
    witness_end(wf, woff, nonce)

def bayes_wvar_parts_kernel(y: FP, n: Int32, yparts: FP, wparts: FP, vparts: FP, state: FP,
                            wf: IP, woff: Int32, nonce: Int32):
    """Thread b: block b's sum w (y - m)^2 (m the weighted mean); thread 0
    also writes sum(w) into state[3]."""
    var b = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var nb = fold_blocks(nn)
    if b < nb:
        var wsum = fold_parts(wparts, 0, nb)
        var m = fd(fold_parts(yparts, 0, nb), wsum)
        var lo = b * FOLD_BLOCK
        st(vparts, b, bayes_wvar_part(y, nn, m, lo, min(FOLD_BLOCK, nn - lo)))
        if b == 0:
            st(state, 3, wsum)
    witness_end(wf, woff, nonce)

def bayes_wx_parts_kernel(x: FP, y: FP, n: Int32, d: Int32, mparts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (column, block): sum w x_j over the block from zero."""
    var t = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var dd = Int(d)
    var nb = fold_blocks(nn)
    if t < dd * nb:
        var j = t % dd
        var b = t // dd
        var lo = b * FOLD_BLOCK
        st(mparts, j * nb + b, bayes_wx_part(x, y, nn, dd, j, lo, min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)

def bayes_wmeans_kernel(n: Int32, d: Int32, fi: Int32, mparts: FP, wparts: FP, fw: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread j: the weighted mean of column j into fw[j] (0 without an intercept)."""
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nb = fold_blocks(Int(n))
    if j < Int(d):
        if fi != 0:
            st(fw, j, fd(fold_parts(mparts, j * nb, nb), fold_parts(wparts, 0, nb)))
        else:
            st(fw, j, Float32(0))
    witness_end(wf, woff, nonce)

def bayes_wgram_parts_kernel(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, fw: FP, yparts: FP, wparts: FP,
                             gparts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread (statistic, block): an upper cell of the weighted centered Gram,
    or (after the cells) a column of the weighted centered X'y."""
    var t = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var dd = Int(d)
    var nb = fold_blocks(nn)
    var cells = dd * (dd + 1) // 2
    var stats = cells + dd
    if t < stats * nb:
        var c = t % stats
        var b = t // stats
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        if c < cells:
            var jk = upper_cell(c, dd)
            st(gparts, c * nb + b, bayes_wgram_part(x, y, nn, dd, fw, jk[0], jk[1], lo, cnt))
        else:
            var ym = fd(fold_parts(yparts, 0, nb), fold_parts(wparts, 0, nb)) if fi != 0 else Float32(0)
            st(gparts, c * nb + b, bayes_wxty_part(x, y, nn, dd, fw, c - cells, ym, lo, cnt))
    witness_end(wf, woff, nonce)

def bayes_wgram_fin_kernel(n: Int32, d: Int32, gparts: FP, fw: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread per statistic: its partials folded blocks ascending, into G at
    fw[d, d + d*d) (both halves) or X'y at fw[d + d*d + j]."""
    var c = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var nb = fold_blocks(Int(n))
    var cells = dd * (dd + 1) // 2
    if c < cells + dd:
        var v = fold_parts(gparts, c * nb, nb)
        if c < cells:
            var jk = upper_cell(c, dd)
            st(fw, dd + jk[0] * dd + jk[1], v)
            st(fw, dd + jk[1] * dd + jk[0], v)
        else:
            st(fw, dd + dd * dd + (c - cells), v)
    witness_end(wf, woff, nonce)

def bayes_coef_kernel(fw: FP, res: FP, d: Int32, state: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread j: the next iteration's coefficient j (`bayes_coef_one`, lambda
    and alpha from state) unless the step stopped (its final update wrote them)."""
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if j < Int(d) and ld(state, 5) == Float32(0):
        st(res, j, bayes_coef_one(fw, Int(d), j, fd(ld(state, 1), ld(state, 0))))
    witness_end(wf, woff, nonce)

def bayes_eig_kernel(fw: FP, fp: FP, res: FP, ip: IP, d: Int32, nb: Int32, yparts: FP, vparts: FP, tw: FP,
                     state: FP, wf: IP, woff: Int32, nonce: Int32):
    """One block team: the d x d half of the prep (`bayes_eig_prep`: the
    Jacobi eigendecomposition of G, V'X'y, the starting alpha and lambda)
    and the first coefficients. Every row pass ran on the grid kernels
    above; this block folds only the nb row-block partials."""
    var a = ALGO_BAYES
    var dd = Int(d)
    var t = device_team(tw, 0, team_rows(a, ip), team_own(a, dd))
    var wsum = ld(state, 3)
    var yvar = fd(fold_parts(vparts, 0, Int(nb)), wsum)
    var al = bayes_eig_prep(t, fw, fp, dd, yvar)
    var ratio = fd(al[1], al[0])
    for j in range(t.tid, dd, t.nt):
        st(res, j, bayes_coef_one(fw, dd, j, ratio))
    if t.lead():
        var ym = fd(fold_parts(yparts, 0, Int(nb)), wsum) if ldi(ip, 1) != 0 else Float32(0)
        st(state, 0, al[0])
        st(state, 1, al[1])
        st(state, 2, ym)
        st(state, 5, Float32(0))
    witness_end(wf, woff, nonce)

def bayes_resid_kernel(x: FP, y: FP, n: Int32, d: Int32, fw: FP, res: FP, state: FP, rows: FP, wf: IP, woff: Int32, nonce: Int32):
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        var dd = Int(d)
        var p = Float32(0)
        for j in range(dd):
            p = fmad(fs(ld(x, i * dd + j), ld(fw, j)), ld(res, j), p)
        st(rows, i, fs(fs(ld(y, i), ld(state, 2)), p))
    witness_end(wf, woff, nonce)

def bayes_part_kernel(rows: FP, y: FP, n: Int32, sw: Int32, parts: FP, wf: IP, woff: Int32, nonce: Int32):
    var bk = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if bk < fold_blocks(nn):
        var lo = bk * FOLD_BLOCK
        st(parts, bk, _sse_part(rows, y, nn, sw != 0, lo, min(FOLD_BLOCK, nn - lo)))
    witness_end(wf, woff, nonce)

def bayes_step_kernel(fw: FP, res: FP, fp: FP, d: Int32, nb: Int32, parts: FP, state: FP, it: Int32, wf: IP, woff: Int32, nonce: Int32):
    """One thread: the nb row-block sse partials folded blocks ascending,
    then the scalar update (`bayes_step`, O(d) folds); `bayes_coef_kernel`
    then writes the next coefficients a thread each."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var sse = fold_parts(parts, 0, Int(nb))
        var r = bayes_step(fw, res, Int(d), fp, ld(state, 1), ld(state, 0), sse, ld(state, 3), Int(it))
        st(state, 0, r[1])
        st(state, 1, r[0])
        st(state, 5, Float32(r[2]))
    witness_end(wf, woff, nonce)

#: FAST on Apple (lane/apple-fast-classical, re-expressed on the grid
#: driver above): x_linear/bayes.mojo `X_LINEAR_GRAM_SSE`. The iteration's
#: sse from the normal equations in the eigenbasis (yy - sum_k (2 z_k vty_k
#: - ev_k z_k^2), z_k = vty_k / (ev_k+ + lam/alpha)) in place of the row
#: pass (`bayes_resid_kernel` + `bayes_part_kernel`): O(d) an iteration on
#: one thread. yy = |y - ym|^2 once, on the grid (`bayes_yy_parts_kernel`).
#: hip[5] turns it on (`MOJOLEARN_X_LINEAR_GRAM_SSE=0` is the A/B arm).
#: IDENTICAL and the other vendors never compile it.
def bayes_yy_parts_kernel(y: FP, n: Int32, state: FP, parts: FP, wf: IP, woff: Int32, nonce: Int32):
    """Thread b: row block b's |y - ym|^2 from zero (ym = state[2])."""
    var b = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if b < fold_blocks(nn):
        var ym = ld(state, 2)
        var lo = b * FOLD_BLOCK
        var acc = Float32(0)
        for i in range(lo, lo + min(FOLD_BLOCK, nn - lo)):
            var r = fs(ld(y, i), ym)
            acc = fmad(r, r, acc)
        st(parts, b, acc)
    witness_end(wf, woff, nonce)

def bayes_step_gram_kernel(fw: FP, res: FP, fp: FP, d: Int32, nb: Int32, yyparts: FP, state: FP, it: Int32,
                           wf: IP, woff: Int32, nonce: Int32):
    """One thread: the sse from the eigenbasis (the coefficients in res are
    `bayes_coef`'s from state's lambda and alpha), then `bayes_step_kernel`'s
    update."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        var dd = Int(d)
        var gg = dd
        var vty = gg + dd * dd + dd + dd * dd
        var tmp = vty + 2 * dd
        var yy = fold_parts(yyparts, 0, Int(nb))
        var lam = ld(state, 1)
        var alpha = ld(state, 0)
        var ratio = fd(lam, alpha)
        var acc = Float32(0)
        for k in range(dd):
            var v = ld(fw, vty + k)
            var z = fd(v, fa(ld(fw, tmp + k), ratio))
            acc = fa(acc, fs(fm(fm(Float32(2), z), v), fm(ld(fw, gg + k * dd + k), fm(z, z))))
        var sse = fmax(fs(yy, acc), Float32(0))
        var r = bayes_step(fw, res, dd, fp, lam, alpha, sse, ld(state, 3), Int(it))
        st(state, 0, r[1])
        st(state, 1, r[0])
        st(state, 5, Float32(r[2]))
        if r[2] == 0:
            bayes_coef(fw, res, dd, r[0], r[1])
    witness_end(wf, woff, nonce)

#: FAST on Apple, the default (`BAYES_GRID_GUARD`, lane/apple-fast-bayes;
#: `-D MOJOLEARN_BAYES_GRID_GUARD_OFF=1` turns it off):
#: `bayes_step_gram_kernel`'s sse relative to a REFERENCE row pass, as
#: x_linear/bayes.mojo `bayes_ridge_fit` guards the one-block fit. The plain yy - sum_k (2 z_k vty_k - ev_k z_k^2) cancelled
#: to below zero on istella (220 features, near-null Gram directions with
#: f32 noise eigenvalues, z huge along them): sse clamped to 0, alpha to inf,
#: coef NaN. With s0 the sse of the last row pass (`bayes_resid_kernel` +
#: `bayes_part_kernel`, state[4]) at z0 (the host's sse scratch
#: fw[3dd + 5d, +d), unused on the device, n >= d) the next iteration's sse
#: is s0 + sum_k dz_k (ev_k (z_k + z0_k) - 2 vty_k), dz = z - z0, with a
#: bound taking every eigenvalue off by 2^-12 (|ev_k| + max |ev|): when it
#: could move sse by more than 2^-8 of itself (or sse is not positive or
#: finite) the next iteration makes the row pass, which becomes the
#: reference. The step computes the candidate for the coefficients it
#: writes, so the host's existing read of state (the stop word) also carries
#: the verdict (state[6], 1 = trusted) and the value (state[7]): no
#: device-to-host read beyond the grid driver's. IDENTICAL and the other
#: vendors never compile it.
def bayes_step_guard_kernel(fw: FP, res: FP, fp: FP, d: Int32, nb: Int32, parts: FP, state: FP, it: Int32,
                            fresh: Int32, wf: IP, woff: Int32, nonce: Int32):
    """One thread: the sse (fresh != 0: the nb row-block partials folded
    blocks ascending, the new reference at this iteration's z; else state[7],
    the trusted candidate), `bayes_step_kernel`'s update, then the candidate
    for the new coefficients into state[6] (1 = trusted) and state[7]."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        _bayes_step_guard_body(fw, res, fp, d, nb, parts, state, it, fresh)
    witness_end(wf, woff, nonce)


def c1_bayes_step_kernel(fw: FP, res: FP, fp: FP, d: Int32, nb: Int32, parts: FP, state: FP,
                         wf: IP, woff: Int32, nonce: Int32):
    """lane/apple-fast-gap-cls1 BAYES_CLS1_BATCH: `bayes_step_guard_kernel`
    with the stop (state[5]), the row-pass verdict (state[6] == 0: fresh)
    and the iteration count (state[8]) read on the device, so iterations
    queue in batches; a stopped fit no-ops."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0 and ld(state, 5) == Float32(0):
        var it = Int(ld(state, 8))
        var fresh = Int32(1) if ld(state, 6) == Float32(0) else Int32(0)
        _bayes_step_guard_body(fw, res, fp, d, nb, parts, state, Int32(it), fresh)
        st(state, 8, i2f(it + 1))
    witness_end(wf, woff, nonce)


def c1_bayes_state_init_kernel(state: FP):
    """BAYES_CLS1_BATCH: the first iteration makes the row pass (state[6] =
    0), no reference yet, the iteration count 0."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        st(state, 6, Float32(0))
        st(state, 7, Float32(0))
        st(state, 8, Float32(0))


def c1_bayes_resid_kernel(x: FP, y: FP, n: Int32, d: Int32, fw: FP, res: FP, state: FP, rows: FP,
                          wf: IP, woff: Int32, nonce: Int32):
    """BAYES_CLS1_BATCH: `bayes_resid_kernel` gated on the device (live
    while not stopped and the next sse is not trusted)."""
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n) and ld(state, 5) == Float32(0) and ld(state, 6) == Float32(0):
        var dd = Int(d)
        var p = Float32(0)
        for j in range(dd):
            p = fmad(fs(ld(x, i * dd + j), ld(fw, j)), ld(res, j), p)
        st(rows, i, fs(fs(ld(y, i), ld(state, 2)), p))
    witness_end(wf, woff, nonce)


def c1_bayes_finish_kernel(fw: FP, res: FP, d: Int32, fi: Int32, state: FP, wf: IP, woff: Int32, nonce: Int32):
    """BAYES_CLS1_BATCH: `bayes_finish_kernel` with the count from state[8]."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        bayes_finish(fw, res, Int(d), fi != 0, ld(state, 2), ld(state, 0), ld(state, 1), Int(ld(state, 8)))
    witness_end(wf, woff, nonce)


@always_inline
def _bayes_step_guard_body(fw: FP, res: FP, fp: FP, d: Int32, nb: Int32, parts: FP, state: FP, it: Int32,
                           fresh: Int32):
    """`bayes_step_guard_kernel`'s statements (one thread)."""
    if True:
        var dd = Int(d)
        var gg = dd
        var vty = gg + dd * dd + dd + dd * dd
        var tmp = vty + 2 * dd
        var z0 = 3 * dd * dd + 5 * dd
        var lam = ld(state, 1)
        var alpha = ld(state, 0)
        var sse: Float32
        if fresh != 0:
            sse = fold_parts(parts, 0, Int(nb))
            var ratio = fd(lam, alpha)
            for k in range(dd):
                st(fw, z0 + k, fd(ld(fw, vty + k), fa(ld(fw, tmp + k), ratio)))
            st(state, 4, sse)
        else:
            sse = ld(state, 7)
        var r = bayes_step(fw, res, dd, fp, lam, alpha, sse, ld(state, 3), Int(it))
        st(state, 0, r[1])
        st(state, 1, r[0])
        st(state, 5, Float32(r[2]))
        var trusted = Float32(0)
        var s = Float32(0)
        if r[2] == 0:
            bayes_coef(fw, res, dd, r[0], r[1])
            var ratio = fd(r[0], r[1])
            var emax = Float32(0)
            for k in range(dd):
                emax = fmax(emax, fabs(ld(fw, gg + k * dd + k)))
            var acc = Float32(0)
            var mag = Float32(0)
            for k in range(dd):
                var v = ld(fw, vty + k)
                var e = ld(fw, gg + k * dd + k)
                var z = fd(v, fa(ld(fw, tmp + k), ratio))
                var zo = ld(fw, z0 + k)
                var dz = fs(z, zo)
                var zs = fa(z, zo)
                acc = fmad(dz, fs(fm(e, zs), fm(Float32(2), v)), acc)
                mag = fmad(fabs(dz), fa(fm(fa(fabs(e), emax), fabs(zs)), fm(Float32(2), fabs(v))), mag)
            s = fa(ld(state, 4), acc)
            if fm(mag, GRAM_SSE_TRUST) <= s:
                trusted = Float32(1)
        st(state, 6, trusted)
        st(state, 7, s)

def bayes_finish_kernel(fw: FP, res: FP, d: Int32, fi: Int32, state: FP, iters: Int32, wf: IP, woff: Int32, nonce: Int32):
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        bayes_finish(fw, res, Int(d), fi != 0, ld(state, 2), ld(state, 0), ld(state, 1), Int(iters))
    witness_end(wf, woff, nonce)

#: Apple: the longest a guarded x_linear launch runs, in chain steps (the
#: grid Gram's row slices; x_linear/witness.mojo, lane/neural-pass131).
comptime XL_APPLE_SLICE_MACS = 1 << 27


#: lane/apple-fast-gram (2026-10-02), FAST on Apple, build-time switches
#: (`-D MOJOLEARN_X_LINEAR_LARS_FAST_GRAM`, `-D MOJOLEARN_X_LINEAR_RIDGE_FAST_GRAM`;
#: no env read on the fit path). Both are the FAST + Apple default since the M3
#: A/B (lane/apple-fast-gram 47ab9b791) and the re-A/B on lane head c338b88dd
#: (n=1, taxi, quality the same: lars 59.1 -> 15.1 ms, lasso-lars 59.1 ->
#: 14.2 ms, ridge-clf 168 -> 122 ms, ridge-cv 3,614 -> 37 ms); `-D MOJOLEARN_X_LINEAR_LARS_FAST_GRAM_OFF` /
#: `-D MOJOLEARN_X_LINEAR_RIDGE_FAST_GRAM_OFF` restore main's path, and the old
#: on-defines stay harmless:
#: LARS_FAST_GRAM builds Lars / LassoLars' means, centered Gram, X'y and y mean
#: with x_linear/fast_gram.mojo (row chunks x 32 x 32 tiles on the grid) instead
#: of main's moments grid (one block per 16-column tile pair, one serial chain
#: per cell: ONE block at taxi's 16 features) or the sliced `xg_gram_kernel`;
#: RIDGE_FAST_GRAM does the same for RidgeClassifier / RidgeCV's means, centered
#: Gram and X'Y, and for k-fold RidgeCV's fold Grams instead of `kf_cells_kernel`
#: (one thread per cell walking the fold's rows). Unweighted fits only.
comptime XL_LARS_FAST_GRAM = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                              and not is_defined["MOJOLEARN_X_LINEAR_LARS_FAST_GRAM_OFF"]())


@always_inline
def _lars_fast_gram() -> Bool:
    return XL_LARS_FAST_GRAM


@always_inline
def _ridge_fast_gram() -> Bool:
    return XL_RIDGE_FAST_GRAM


# ------------------------------------------------ minibatch SGD on the grid (lane/neural-pass103)
# x_linear/sgd.mojo `sgd_mb_one` with each batch as three launches: the rows'
# loss derivatives (one thread a row), the gradient partials (one thread a
# (sub-block, column), sub-block-major so neighbouring threads read one row's
# words), the step (one thread a weight, the intercept and the objective).
# The epoch order (`sgd_perm_kernel`), the targets and the epoch end
# (x_linear/sgd_end.mojo) are the device's; the host enqueues an epoch's
# batches without a sync and reads two stop words.
# AFCL-L14: NEVER RUN — PENDING MEASUREMENT. Uncompiled/unverified, OFF.
# Four SIMD groups per independent row-prediction block expose twice as
# many schedulable blocks as the 256-thread baseline. Training examples,
# minibatches, seeded order, gradient folds and update/stopping policy do
# not change. Witness capacities/offsets use the same row-grid helper.
comptime AFCL_L14 = (GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
                    and is_defined["MOJOLEARN_AFCL_L14"]())
comptime SGD_ROW_TPB = 128 if AFCL_L14 else XG_TPB


@always_inline
def _sgd_row_blocks(rows: Int) -> Int:
    return max((rows + SGD_ROW_TPB - 1) // SGD_ROW_TPB, 1)


@always_inline
def _sgd_mb_rows_kernel_body(x: FP, ys: FP, idx: IP, start: Int32, bs: Int32, d: Int32, w: FP, bias: FP, loss: Int32,
                       eps: Float32, swp: FP, has_sw: Int32, wpos: Float32, wneg: Float32, has_cw: Int32,
                       dlv: FP, lv: FP, lr: Int32, eta0: Float32, dblk: Int32, ocm: Int32):
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if r < Int(bs):
        var i = Int(idx.unsafe_load(Int(start) + r))
        if ocm != 0:
            # one-class (SGD_OC_MB_IMPLICIT): the float-float intercept's row
            var oc = mb_oc_row(mb_dot(x, i, Int(d), w, 0, Int(dblk)), ld(bias, 0), ld(bias, 1), Int(ocm),
                               ld(swp, i) if has_sw != 0 else Float32(1), has_sw != 0)
            st(dlv, r, oc[0])
            st(lv, r, oc[1])
            return
        var o = mb_row(x, ys, i, Int(d), w, 0, ld(bias, 0), Int(loss), eps, swp, has_sw != 0, wpos, wneg, has_cw != 0,
                       Int(lr), eta0, Int(dblk))
        st(dlv, r, o[0])
        st(lv, r, o[1])


def sgd_mb_rows_kernel(x: FP, ys: FP, idx: IP, start: Int32, bs: Int32, d: Int32, w: FP, bias: FP, loss: Int32,
                       eps: Float32, swp: FP, has_sw: Int32, wpos: Float32, wneg: Float32, has_cw: Int32,
                       dlv: FP, lv: FP, lr: Int32, eta0: Float32, dblk: Int32, ocm: Int32, wf: IP, woff: Int32,
                       nonce: Int32):
    _sgd_mb_rows_kernel_body(x, ys, idx, start, bs, d, w, bias, loss, eps, swp, has_sw, wpos, wneg, has_cw, dlv, lv, lr, eta0,
                             dblk, ocm)
    witness_end(wf, woff, nonce)


@always_inline
def _oc_team_sum(cnt: IP, tid: Int, nt: Int, v: Int) -> Int:
    """The block's sum of each thread's count v (integers: exact in any
    order), the same on every thread."""
    sti(cnt, tid, v)
    team_barrier()
    var c = 0
    for u in range(nt):
        c += ldi(cnt, u)
    team_barrier()
    return c


@always_inline
def sgd_mb_oc_team(dlv: FP, lv: FP, bs: Int, eta: Float32, alpha: Float32, bsum: Bool, cnt: IP, tid: Int,
                   nt: Int) -> Float32:
    """x_linear/sgd.mojo `oc_solve` on one block (SGD_OC_MB_IMPLICIT, from
    lane/neural-pass139): each probe's count split over the threads (rows
    tid, tid + nt, ..; integer counts, so any split equals the host's),
    every thread folding the nt counts and taking the same branch; then
    each row's (dl, loss) at the step (`oc_hinge_at`). dlv holds the rows'
    margins on entry; each thread reads and rewrites only its own rows, so
    a caller whose rows pass used the same row-to-thread map needs no
    barrier before. Returns the step dhi. cnt: nt words."""
    var lo = 0
    var hi = bs
    while lo < hi:
        var mid = (lo + hi) // 2
        var c = _oc_team_sum(cnt, tid, nt, oc_count(dlv, tid, bs, nt, oc_delta(mid, bs, eta, alpha, bsum)))
        if mid >= c:
            hi = mid
        else:
            lo = mid + 1
    var dhi = oc_delta(lo, bs, eta, alpha, bsum)
    var dlo = dhi
    var cut = 0
    if lo > 0:
        dlo = oc_delta(lo - 1, bs, eta, alpha, bsum)
        var need = lo - _oc_team_sum(cnt, tid, nt, oc_count(dlv, tid, bs, nt, dhi))
        var a = 0
        var b = bs
        while a < b:
            var mid = (a + b) // 2
            if _oc_team_sum(cnt, tid, nt, oc_count_tie(dlv, tid, bs, nt, dhi, dlo, mid)) >= need:
                b = mid
            else:
                a = mid + 1
        cut = a
    for r in range(tid, bs, nt):
        var o = oc_hinge_at(ld(dlv, r), r, dhi, dlo, cut)
        st(dlv, r, o[0])
        st(lv, r, o[1])
    return dhi


def sgd_mb_oc_kernel(dlv: FP, lv: FP, bs: Int32, eta: Float32, alpha: Float32, bsum: Int32, cnt: IP, dlt: FP,
                     etp: FP, dev_eta: Int32, wf: IP, woff: Int32, nonce: Int32):
    """The one-class implicit intercept step of a batch (`sgd_mb_oc_team`,
    one block between the rows and the partials launches; the counts are
    bisection probes over bs rows): the step to dlt[0] for the step
    kernel. dev_eta: the rate is the device's epoch-end word etp[0], as in
    `sgd_mb_step_kernel`."""
    var tid = Int(thread_idx.x)
    var e = ld(etp, 0) if dev_eta != 0 else eta
    var d0 = sgd_mb_oc_team(dlv, lv, Int(bs), e, alpha, bsum != 0, cnt, tid, Int(block_dim.x))
    if tid == 0:
        st(dlt, 0, d0)
    witness_end(wf, woff, nonce)

@always_inline
def _sgd_mb_parts_kernel_body(x: FP, d: Int32, idx: IP, start: Int32, dlv: FP, lv: FP, bs: Int32, nsub: Int32, parts: FP,
                              sub: Int32):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var subs = mb_subs(Int(bs), Int(sub))
    var s = q // (dd + 2)
    var j = q - s * (dd + 2)
    if s < subs:
        st(parts, j * Int(nsub) + s, mb_part(x, dd, idx, Int(start), dlv, lv, j, s, Int(bs), Int(sub)))


def sgd_mb_parts_kernel(x: FP, d: Int32, idx: IP, start: Int32, dlv: FP, lv: FP, bs: Int32, nsub: Int32, parts: FP,
                        sub: Int32, wf: IP, woff: Int32, nonce: Int32):
    _sgd_mb_parts_kernel_body(x, d, idx, start, dlv, lv, bs, nsub, parts, sub)
    witness_end(wf, woff, nonce)

@always_inline
def _sgd_mb_step_kernel_body(parts: FP, nsub: Int32, bs: Int32, d: Int32, w: FP, bias: FP, obj: FP, eta: Float32,
                       alpha: Float32, l1r: Float32, penalty: Int32, fi: Int32, need_obj: Int32, one_class: Int32,
                       bsum: Int32, sub: Int32, ocm: Int32, dlt: FP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var subs = mb_subs(Int(bs), Int(sub))
    if j > dd + 1:
        return
    var g = Float32(0)
    for s in range(subs):
        g = fa(g, ld(parts, j * Int(nsub) + s))
    if j < dd:
        st(w, j, mb_step(ld(w, j), g, Int(bs), eta, alpha, l1r, Int(penalty), bsum != 0))
    elif j == dd:
        if fi != 0:
            if ocm != 0:
                var nb = mb_oc_bias_step(ld(bias, 0), ld(bias, 1), g, Int(bs), eta, alpha, bsum != 0, Int(ocm),
                                         ld(dlt, 0) if ocm == 2 else Float32(0))
                st(bias, 0, nb[0])
                st(bias, 1, nb[1])
            else:
                st(bias, 0, mb_bias_step(ld(bias, 0), g, Int(bs), eta, alpha, one_class != 0, bsum != 0))
    elif need_obj != 0:
        st(obj, 0, fa(ld(obj, 0), g))


def sgd_mb_step_kernel(parts: FP, nsub: Int32, bs: Int32, d: Int32, w: FP, bias: FP, obj: FP, eta: Float32,
                       alpha: Float32, l1r: Float32, penalty: Int32, fi: Int32, need_obj: Int32, one_class: Int32,
                       bsum: Int32, sub: Int32, etp: FP, dev_eta: Int32, ocm: Int32, dlt: FP, wf: IP, woff: Int32,
                       nonce: Int32):
    """dev_eta: the constant / adaptive rate is the device's epoch-end state
    (etp[0], x_linear/sgd_end.mojo); the others are the schedule's `eta`.
    ocm / dlt: the one-class intercept (SGD_OC_MB_IMPLICIT; dlt[0] the
    implicit step of `sgd_mb_oc_kernel` when ocm == 2)."""
    var e = ld(etp, 0) if dev_eta != 0 else eta
    _sgd_mb_step_kernel_body(parts, nsub, bs, d, w, bias, obj, e, alpha, l1r, penalty, fi, need_obj, one_class, bsum, sub,
                             ocm, dlt)
    witness_end(wf, woff, nonce)


# lane/neural-pass134 (2026-10-02): K batches in ONE launch on one block. The
# three per-batch kernels each ran as a single block at batch 256 (the peer's
# MI325X: perceptron istella 24.6 s against 7.7 on the L40S, launch-bound);
# here one block runs each batch's rows, partials and step in order with
# device-ordering barriers between them, the same helpers in the same order,
# and the rate of each batch from the same schedule (`mb_eta` with t
# counting as the host does). `MOJOLEARN_X_LINEAR_SGD_CHUNK=1` restores one
# batch a launch (the three kernels).
comptime SGD_CHUNK_DEFAULT = 64

# lane/idn-sgd-multiblock (2026-10-04): IDENTICAL on NVIDIA and AMD runs the
# chunk kernel's one block at 1024 threads instead of XG_TPB. Every task of a
# batch (a (row, column-block) dot, a row, a (sub-block, column) partial, a
# weight) writes its own slot and the kernel strides them over block_dim, so
# the words do not depend on the width: at batch 256 and istella's 220
# columns the 1,792 dots and 1,776 partials were 7 tasks a thread.
# `-D MOJOLEARN_SGD_IDN_CHUNK_WIDE_OFF` restores XG_TPB.
comptime SGD_IDN_CHUNK_WIDE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not has_apple_gpu_accelerator()
    and not (is_defined["MOJOLEARN_SGD_IDN_CHUNK_WIDE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
comptime SGD_CHUNK_TPB = 1024 if SGD_IDN_CHUNK_WIDE else XG_TPB

# lane/idn-sgd-multiblock: batches above the chunk kernel's (SGDClassifier /
# SGDRegressor at 4096) were three launches each (rows, partials, step). Here
# batch b's step and batch b + 1's rows are ONE launch: every block applies
# the step from the weights w0 into w1 (the same word from every block, so
# the stores agree), crosses a device-ordering barrier and runs its rows on
# w1; block 0 alone folds the objective. The weights alternate between two
# buffers so no block reads a word another block is replacing. The same
# helpers on the same operands in the same order: no bit moves.
# `-D MOJOLEARN_SGD_IDN_MB_FUSE_OFF` restores three launches a batch.
comptime SGD_IDN_MB_FUSE = (
    # I12 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_SGD_IDN_MB_FUSE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)


def sgd_mb_steprows_kernel(
    parts: FP, w0: FP, b0: FP, w1: FP, b1: FP, obj: FP, x: FP, ys: FP, idx: IP, swp: FP, dlv: FP, lv: FP,
    ci: IP, cf: FP, etp: FP, dlt: FP, start: Int32, bs_prev: Int32, bs: Int32, eta: Float32, dev_eta: Int32,
    wf: IP, woff: Int32, nonce: Int32,
):
    """ci / cf: `_sgd_mb_chunk_kernel_body`'s. The step of the batch of
    bs_prev rows (`sgd_mb_step_kernel`'s statements, w0 -> w1), then the rows
    [start, start + bs) of the next batch (`sgd_mb_rows_kernel`'s) on w1;
    bs == 0 is the step alone."""
    var dd = ldi(ci, 2)
    var loss = ldi(ci, 3)
    var has_sw = ldi(ci, 4) != 0
    var has_cw = ldi(ci, 5) != 0
    var lr = ldi(ci, 6)
    var nsub = ldi(ci, 7)
    var penalty = ldi(ci, 8)
    var fi = ldi(ci, 9) != 0
    var need_obj = ldi(ci, 10) != 0
    var one_class = ldi(ci, 11) != 0
    var bsum = ldi(ci, 12) != 0
    var sub = ldi(ci, 13)
    var dblk = ldi(ci, 14)
    var ocm = ldi(ci, 16)  # SGD_OC_MB_IMPLICIT; dlt[0] the previous batch's implicit step
    var eps = ld(cf, 0)
    var wpos = ld(cf, 1)
    var wneg = ld(cf, 2)
    var eta0 = ld(cf, 3)
    var alpha = ld(cf, 5)
    var l1r = ld(cf, 6)
    var e = ld(etp, 0) if dev_eta != 0 else eta
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var bp = Int(bs_prev)
    var subs = mb_subs(bp, sub)
    for j in range(tid, dd + 2, nt):
        var g = Float32(0)
        for s in range(subs):
            g = fa(g, ld(parts, j * nsub + s))
        if j < dd:
            st(w1, j, mb_step(ld(w0, j), g, bp, e, alpha, l1r, penalty, bsum))
        elif j == dd:
            if ocm != 0:
                if fi:
                    var nb = mb_oc_bias_step(ld(b0, 0), ld(b0, 1), g, bp, e, alpha, bsum, ocm,
                                             ld(dlt, 0) if ocm == 2 else Float32(0))
                    st(b1, 0, nb[0])
                    st(b1, 1, nb[1])
                else:
                    st(b1, 0, ld(b0, 0))
                    st(b1, 1, ld(b0, 1))
            elif fi:
                st(b1, 0, mb_bias_step(ld(b0, 0), g, bp, e, alpha, one_class, bsum))
            else:
                st(b1, 0, ld(b0, 0))
        elif need_obj and Int(block_idx.x) == 0:
            st(obj, 0, fa(ld(obj, 0), g))
    team_barrier()
    var r = Int(block_idx.x) * nt + tid
    if r < Int(bs):
        var i = Int(idx.unsafe_load(Int(start) + r))
        if ocm != 0:
            var oc = mb_oc_row(mb_dot(x, i, dd, w1, 0, dblk), ld(b1, 0), ld(b1, 1), ocm,
                               ld(swp, i) if has_sw else Float32(1), has_sw)
            st(dlv, r, oc[0])
            st(lv, r, oc[1])
        else:
            var o = mb_row(x, ys, i, dd, w1, 0, ld(b1, 0), loss, eps, swp, has_sw, wpos, wneg, has_cw, lr, eta0, dblk)
            st(dlv, r, o[0])
            st(lv, r, o[1])
    witness_end(wf, woff, nonce)


# lane/idn-sgd-multiblock: the fit's finiteness check on the device
# (IDENTICAL). The binding walked all n x d host words one at a time before
# every fit (bindings/_mojolearn_x_linear.mojo `_finite`); here one thread
# tests SGD_FIN_RUN words of the uploaded X (and of y) and a bad word raises
# the flag: the same verdict (exponent all ones = NaN or infinity), the same
# error. `-D MOJOLEARN_SGD_IDN_DEV_FINITE_OFF` restores the walk.
# Lane idn-all (2026-10-04): Apple runs the device check too. No Metal limit
# prevents it; the one hazard (macOS cutting a command buffer would leave the
# flag clear) is closed by the completion witness (x_linear/witness.mojo):
# every block reports, the check is idempotent, a cut launch reruns and after
# WITNESS_TRIES the fit raises. `-D MOJOLEARN_SGD_IDN_DEV_FINITE_APPLE_OFF`
# restores the host walk on Apple only.
comptime SGD_IDN_DEV_FINITE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not (is_defined["MOJOLEARN_SGD_IDN_DEV_FINITE_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
    and not (has_apple_gpu_accelerator() and is_defined["MOJOLEARN_SGD_IDN_DEV_FINITE_APPLE_OFF"]())
)
comptime SGD_FIN_RUN = 64


def sgd_finite_kernel(p: FP, count: Int32, flag: IP, slot: Int32, wf: IP, woff: Int32, nonce: Int32):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var lo = q * SGD_FIN_RUN
    var hi = min(lo + SGD_FIN_RUN, Int(count))
    var bad = False
    for i in range(lo, hi):
        var bits = bitcast[DType.uint32](ld(p, i))
        if (bits & UInt32(0x7F800000)) == UInt32(0x7F800000):
            bad = True
    if bad:
        sti(flag, Int(slot), 1)
    witness_end(wf, woff, nonce)


def _sgd_finite_device(mut ctx: DeviceContext, dxp: FP, n_x: Int, y: FP, n_y: Int) raises:
    """Raises the binding's error when the uploaded X (dxp, n_x words) or the
    host y block (n_y words: labels, then sample weights) holds a NaN or an
    infinity; X is named first, as the host walk names it. One witness-
    guarded unit (the flags are rebuilt from inputs the launches do not
    write, so a cut Apple launch reruns)."""
    var dfl = ctx.enqueue_create_buffer[DType.int32](2)
    var dyf = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var gx = _xg_blocks((n_x + SGD_FIN_RUN - 1) // SGD_FIN_RUN) if n_x > 0 else 0
    var gy = _xg_blocks((n_y + SGD_FIN_RUN - 1) // SGD_FIN_RUN) if n_y > 0 else 0
    var wit = Witness(ctx, max(gx + gy, 1))
    var hfl = List[Int32](length=2, fill=Int32(0))
    var tries = 0
    while True:
        var nonce = wit.begin()
        dfl.enqueue_fill(Int32(0))
        if n_x > 0:
            ctx.enqueue_function[sgd_finite_kernel](
                dxp, Int32(n_x), dfl.unsafe_ptr(), Int32(0), wit.p(), Int32(0), nonce,
                grid_dim=gx, block_dim=XG_TPB,
            )
        if n_y > 0:
            ctx.enqueue_copy(dst_buf=dyf, src_ptr=y)
            ctx.enqueue_function[sgd_finite_kernel](
                dyf.unsafe_ptr(), Int32(n_y), dfl.unsafe_ptr(), Int32(1), wit.p(), Int32(gx), nonce,
                grid_dim=gy, block_dim=XG_TPB,
            )
        ctx.enqueue_copy(dst_ptr=hfl.unsafe_ptr(), src_buf=dfl)
        ctx.synchronize()
        if wit.ok(ctx, gx + gy, "SGD finite check"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    var bx = hfl[0] != 0
    var by = hfl[1] != 0
    _ = hfl^
    _ = dfl^
    _ = dyf^
    _ = wit^
    if bx:
        raise Error("mojolearn: X contains NaN or infinity")
    if by:
        raise Error("mojolearn: y contains NaN or infinity")


def sgd_rowsq_kernel(x: FP, n: Int32, d: Int32, sqp: FP, wf: IP, woff: Int32, nonce: Int32):
    """sqp[i] = `mb_rowsq` of row i: the PA rates' |x_i|^2, the chain
    `mb_row` walks, once an epoch instead of once a visit."""
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        st(sqp, i, mb_rowsq(x, i, Int(d)))
    witness_end(wf, woff, nonce)


@always_inline
def _sgd_mb_chunk_kernel_body(
    x: FP, ys: FP, idx: IP, w: FP, bias: FP, swp: FP, dlv: FP, lv: FP, parts: FP, obj: FP, ci: IP, cf: FP,
    dotp: FP, sqp: FP, cnt: IP, ocm: Int,
    start0: Int32, nbat: Int32, t0: Int32,
):
    """ci: n, batch, d, loss, has_sw, has_cw, lr, nsub, penalty, fi, need_obj,
    one_class, bsum, sub, dblk, has_sq; ocm the one-class mode
    (SGD_OC_MB_IMPLICIT, ci[16] in `_sgd_mb_grid`), cnt block_dim words; cf: eps, wpos, wneg, eta0, eta, alpha, l1r, power_t,
    opt_init (Metal takes at most 31 kernel arguments)."""
    var nn = ldi(ci, 0)
    var batch = ldi(ci, 1)
    var dd = ldi(ci, 2)
    var loss = ldi(ci, 3)
    var has_sw = ldi(ci, 4) != 0
    var has_cw = ldi(ci, 5) != 0
    var lr = ldi(ci, 6)
    var nsub = ldi(ci, 7)
    var penalty = ldi(ci, 8)
    var fi = ldi(ci, 9) != 0
    var need_obj = ldi(ci, 10) != 0
    var one_class = ldi(ci, 11) != 0
    var bsum = ldi(ci, 12) != 0
    var sub = ldi(ci, 13)
    var dblk = ldi(ci, 14)
    var has_sq = ldi(ci, 15) != 0
    var eps = ld(cf, 0)
    var wpos = ld(cf, 1)
    var wneg = ld(cf, 2)
    var eta0 = ld(cf, 3)
    var eta = ld(cf, 4)
    var alpha = ld(cf, 5)
    var l1r = ld(cf, 6)
    var power_t = ld(cf, 7)
    var opt_init = ld(cf, 8)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var start = Int(start0)
    var t = Int(t0)
    for _q in range(Int(nbat)):
        var bs = min(batch, nn - start)
        if bs <= 0:
            break
        var et = mb_eta(lr, eta, eta0, alpha, power_t, opt_init, t)
        if dblk == MB_DBLK:
            # one task per (row, MB_DBLK-column block), its loads all in
            # flight before its chain, then each row's blocks folded
            # ascending: `mb_dot`'s words (lane/neural-pass132: one thread
            # walking a whole random row missed cache every step on the
            # MI325X, istella 220 columns)
            var nb = (dd + MB_DBLK - 1) // MB_DBLK
            for q in range(tid, bs * nb, nt):
                var r = q // nb
                st(dotp, q, mb_block_dot(x, Int(idx.unsafe_load(start + r)), dd, w, 0, q - r * nb))
            team_barrier()
            for r in range(tid, bs, nt):
                var i = Int(idx.unsafe_load(start + r))
                var dot = Float32(0)
                for bb in range(nb):
                    dot = fa(dot, ld(dotp, r * nb + bb))
                if ocm != 0:
                    var oc = mb_oc_row(dot, ld(bias, 0), ld(bias, 1), ocm, ld(swp, i) if has_sw else Float32(1), has_sw)
                    st(dlv, r, oc[0])
                    st(lv, r, oc[1])
                    continue
                var o = mb_row_dot(x, ys, i, dd, dot, ld(bias, 0), loss, eps, swp, has_sw, wpos, wneg, has_cw, lr, eta0,
                                   sqp, has_sq)
                st(dlv, r, o[0])
                st(lv, r, o[1])
        else:
            for r in range(tid, bs, nt):
                var i = Int(idx.unsafe_load(start + r))
                if ocm != 0:
                    var oc = mb_oc_row(mb_dot(x, i, dd, w, 0, dblk), ld(bias, 0), ld(bias, 1), ocm,
                                       ld(swp, i) if has_sw else Float32(1), has_sw)
                    st(dlv, r, oc[0])
                    st(lv, r, oc[1])
                    continue
                var o = mb_row(x, ys, i, dd, w, 0, ld(bias, 0), loss, eps, swp, has_sw, wpos, wneg, has_cw, lr, eta0, dblk)
                st(dlv, r, o[0])
                st(lv, r, o[1])
        var dlt = Float32(0)
        if ocm == 2:
            # the one-class implicit intercept step (`oc_solve`): the rows'
            # margins above sit at the rows this thread owns
            dlt = sgd_mb_oc_team(dlv, lv, bs, et, alpha, bsum, cnt, tid, nt)
        team_barrier()
        var subs = mb_subs(bs, sub)
        for q in range(tid, (dd + 2) * subs, nt):
            var sb = q // (dd + 2)
            var j = q - sb * (dd + 2)
            st(parts, j * nsub + sb, mb_part(x, dd, idx, start, dlv, lv, j, sb, bs, sub))
        team_barrier()
        for j in range(tid, dd + 2, nt):
            var g = Float32(0)
            for sb in range(subs):
                g = fa(g, ld(parts, j * nsub + sb))
            if j < dd:
                st(w, j, mb_step(ld(w, j), g, bs, et, alpha, l1r, penalty, bsum))
            elif j == dd:
                if fi:
                    if ocm != 0:
                        var nb = mb_oc_bias_step(ld(bias, 0), ld(bias, 1), g, bs, et, alpha, bsum, ocm, dlt)
                        st(bias, 0, nb[0])
                        st(bias, 1, nb[1])
                    else:
                        st(bias, 0, mb_bias_step(ld(bias, 0), g, bs, et, alpha, one_class, bsum))
            elif need_obj:
                st(obj, 0, fa(ld(obj, 0), g))
        team_barrier()
        t += bs if bsum else 1
        start += bs


# The launch width also constrains register allocation: without this bound
# the 1024-thread IDENTICAL launch can exceed NVIDIA registers per block
# (Blackwell CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES). This declares the existing
# launch, preserving the work order and the CHUNK_WIDE_OFF comparison arm.
@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SGD_CHUNK_TPB)))
def sgd_mb_chunk_kernel(
    x: FP, ys: FP, idx: IP, w: FP, bias: FP, swp: FP, dlv: FP, lv: FP, parts: FP, obj: FP, ci: IP, cf: FP,
    dotp: FP, sqp: FP, cnt: IP, start0: Int32, nbat: Int32, t0: Int32, wf: IP, woff: Int32, nonce: Int32,
):
    _sgd_mb_chunk_kernel_body(x, ys, idx, w, bias, swp, dlv, lv, parts, obj, ci, cf, dotp, sqp, cnt, ldi(ci, 16), start0,
                              nbat, t0)
    witness_end(wf, woff, nonce)


# lane fam2-linear (2026-10-04): SGD_IDN_OVR_PAR. The minibatch driver ran the
# one-vs-rest problems one after another (`for c in range(problems)` in
# `_sgd_mb_grid`), every chunk launch ONE block: Perceptron and the
# passive-aggressive classifier at batch 256 used one block of the GPU for
# C x epochs x chunks launches. The problems are independent fits, so here a
# chunk launch runs one block a problem (grid_dim = problems): problem c's
# targets, order, weights, bias, scratch, objective and rate words sit at
# offset c of buffers `problems` times as long, and a stopped problem's
# block returns at once (`act`). The epoch end is the same kernel, one small
# launch a live problem, and ONE read brings every problem's stop words
# home. Each problem's statements are the sequential driver's on its own
# words: NO BIT MOVES, the host column is untouched.
# `-D MOJOLEARN_SGD_IDN_OVR_PAR_OFF` (or `MOJOLEARN_IDN_ALL_OFF`) restores
# the sequential problems.
comptime SGD_IDN_OVR_PAR = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    # I12 current experiment: NEVER RUN — PENDING MEASUREMENT; existing defaults preserved.
    and not (is_defined["MOJOLEARN_SGD_IDN_OVR_PAR_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)
#: `cf` words a problem (`_sgd_mb_chunk_kernel_body`'s nine)
comptime SGD_OVR_CF = 9
#: the per-problem targets and orders are problems x n words each
comptime SGD_OVR_MAX_CELLS = 1 << 28


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SGD_CHUNK_TPB)))
def sgd_mb_chunk_ovr_kernel(
    x: FP, ys: FP, idx: IP, w: FP, bias: FP, swp: FP, dlv: FP, lv: FP, parts: FP, obj: FP, ci: IP, cf: FP,
    dotp: FP, sqp: FP, act: IP, start0: Int32, nbat: Int32, t0: Int32, wf: IP, woff: Int32, nonce: Int32,
    problem_start: Int32,
):
    """`sgd_mb_chunk_kernel`, block c on problem c's words (see
    SGD_IDN_OVR_PAR). `act[c] == 0` is a stopped problem: nothing runs."""
    var c = Int(block_idx.x) + Int(problem_start)
    if ldi(act, c) != 0:
        var nn = ldi(ci, 0)
        var batch = ldi(ci, 1)
        var dd = ldi(ci, 2)
        var nsub = ldi(ci, 7)
        var nbk = (dd + MB_DBLK - 1) // MB_DBLK
        _sgd_mb_chunk_kernel_body(
            x, ys + c * nn, idx + c * nn, w + c * dd, bias + c, swp, dlv + c * batch, lv + c * batch,
            parts + c * (dd + 2) * nsub, obj + c, ci, cf + c * SGD_OVR_CF, dotp + c * batch * nbk, sqp,
            act, 0, start0, nbat, t0,
        )
    witness_end(wf, woff, nonce)


def _sgd_mb_ovr_applies(problems: Int, n: Int, d: Int, batch: Int, chunk: Int) -> Bool:
    """The parallel one-vs-rest driver serves this fit: more than one
    problem, the chunk kernel's shape (`_sgd_mb_grid`'s own test), and
    per-problem buffers of a bounded size."""
    comptime if SGD_IDN_OVR_PAR:
        return (
            problems > 1 and n > 0 and d >= 1 and chunk > 1 and batch <= XG_TPB and d + 2 <= XG_TPB
            and problems * n <= SGD_OVR_MAX_CELLS
        )
    return False


def _sgd_mb_ovr_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                     n_out: Int, res: FP) raises:
    """`_sgd_mb_grid`'s chunk arm with the one-vs-rest problems side by side
    (SGD_IDN_OVR_PAR): the same statements per problem, the same words."""
    var ctx = linear_ctx()
    var k = Int(ip[0])
    var loss = Int(ip[1])
    var penalty = Int(ip[2])
    var lr = Int(ip[3])
    var fi = Int(ip[4]) != 0
    var max_iter = Int(ip[5])
    var nic = Int(ip[6])
    var do_shuffle = Int(ip[7]) != 0
    var seed = (UInt64(UInt32(ip[9])) << 32) | UInt64(UInt32(ip[8]))
    var has_sw = Int(ip[10]) != 0
    var has_cw = Int(ip[11]) != 0
    var batch = sgd_batch(Int(ip[12]), k)
    var bsum = len(ip) > 13 and Int(ip[13]) != 0
    var alpha = fp[0]
    var l1r = fp[1]
    var eta0 = fp[2]
    var power_t = fp[3]
    var eps = fp[4]
    var tol = fp[5]
    if penalty == P_L2:
        l1r = Float32(0)
    elif penalty == P_L1:
        l1r = Float32(1)
    var problems = k
    var sub = mb_sub_size(batch)
    var dblk = mb_dblk(batch)
    var nsub = mb_subs(batch, sub)
    var nbk = (d + MB_DBLK - 1) // MB_DBLK
    var chunk = _sgd_chunk()
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var need_obj = tol > Float32(-3.0e38)
    var opt_init = mb_optimal_init(loss, alpha, eps) if lr == LR_OPTIMAL else Float32(0)
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](n)
    var dys = ctx.enqueue_create_buffer[DType.float32](problems * n)
    var dsw = ctx.enqueue_create_buffer[DType.float32](n)
    var didx = ctx.enqueue_create_buffer[DType.int32](problems * n)
    var ddl = ctx.enqueue_create_buffer[DType.float32](problems * batch)
    var dlv = ctx.enqueue_create_buffer[DType.float32](problems * batch)
    var dparts = ctx.enqueue_create_buffer[DType.float32](problems * (d + 2) * nsub)
    var dw = ctx.enqueue_create_buffer[DType.float32](problems * d)
    var dbias = ctx.enqueue_create_buffer[DType.float32](problems)
    var dobj = ctx.enqueue_create_buffer[DType.float32](problems)
    var dws = ctx.enqueue_create_buffer[DType.float32](problems * d)
    var dbs = ctx.enqueue_create_buffer[DType.float32](problems)
    var ddotp = ctx.enqueue_create_buffer[DType.float32](max(problems * batch * nbk, 1))
    var dsq = ctx.enqueue_create_buffer[DType.float32](n if pa_rate else 1)
    var dci = ctx.enqueue_create_buffer[DType.int32](16)
    var dcf = ctx.enqueue_create_buffer[DType.float32](problems * SGD_OVR_CF)
    var dstt = ctx.enqueue_create_buffer[DType.float32](problems * SGD_MB_WORDS)
    var dvp = ctx.enqueue_create_buffer[DType.float32](vscratch(d) + 16)
    var dact = ctx.enqueue_create_buffer[DType.int32](problems)
    var dres = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    dres.enqueue_fill(Float32(0))
    # chunk launches an epoch (each reports one word a problem), the PA
    # row norms, the epoch end's one word a problem
    var n_batches = (n + batch - 1) // batch
    var launches = (n_batches + chunk - 1) // chunk
    var wit = Witness(ctx, max(launches * problems + _xg_blocks(n) + problems, 1))
    var hci = List[Int32](length=16, fill=Int32(0))
    hci[0] = Int32(n)
    hci[1] = Int32(batch)
    hci[2] = Int32(d)
    hci[3] = Int32(loss)
    hci[4] = Int32(1 if has_sw else 0)
    hci[5] = Int32(1 if has_cw else 0)
    hci[6] = Int32(lr)
    hci[7] = Int32(nsub)
    hci[8] = Int32(penalty)
    hci[9] = Int32(1 if fi else 0)
    hci[10] = Int32(1 if need_obj else 0)
    hci[11] = Int32(0)
    hci[12] = Int32(1 if bsum else 0)
    hci[13] = Int32(sub)
    hci[14] = Int32(dblk)
    hci[15] = Int32(1 if pa_rate else 0)
    var hcf = List[Float32](length=problems * SGD_OVR_CF, fill=Float32(0))
    var hstt = List[Float32](length=problems * SGD_MB_WORDS, fill=Float32(0))
    var hact = List[Int32](length=problems, fill=Int32(1))
    var active = List[Bool](length=problems, fill=True)
    var failed = List[Bool](length=problems, fill=False)
    var epochs = List[Int](length=problems, fill=0)
    for c in range(problems):  # small-loop(problems: OvR problems): per-problem launch constants, one row of parameters a class
        var o = c * SGD_OVR_CF
        hcf[o] = eps
        hcf[o + 1] = fp[6 + c] if has_cw else Float32(1)
        hcf[o + 2] = fp[6 + problems + c] if has_cw else Float32(1)
        hcf[o + 3] = eta0
        hcf[o + 4] = eta0
        hcf[o + 5] = alpha
        hcf[o + 6] = l1r
        hcf[o + 7] = power_t
        hcf[o + 8] = opt_init
        hstt[c * SGD_MB_WORDS] = Float32(3.0e38)
        hstt[c * SGD_MB_WORDS + 1] = Float32(0)
        hstt[c * SGD_MB_WORDS + 2] = eta0
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if has_sw:
        ctx.enqueue_copy(dst_buf=dsw, src_ptr=y + n)
    ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if SGD_IDN_DEV_FINITE:
        _sgd_finite_device(ctx, FP(unsafe_from_address=Int(dx.unsafe_ptr())), n_x, y, n_y)
    ctx.enqueue_copy(dst_buf=dci, src_ptr=hci.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dcf, src_ptr=hcf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dstt, src_ptr=hstt.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dact, src_ptr=hact.unsafe_ptr())
    dw.enqueue_fill(Float32(0))
    dbias.enqueue_fill(Float32(0))
    var seed_lo = Int32(Int(UInt32(seed & UInt64(0xFFFFFFFF))))
    var seed_hi = Int32(Int(UInt32(seed >> 32)))
    # host-side base addresses of the per-problem buffers
    var a_ys = Int(dys.unsafe_ptr())
    var a_idx = Int(didx.unsafe_ptr())
    var a_w = Int(dw.unsafe_ptr())
    var a_bias = Int(dbias.unsafe_ptr())
    var a_obj = Int(dobj.unsafe_ptr())
    var a_stt = Int(dstt.unsafe_ptr())
    var a_cf = Int(dcf.unsafe_ptr())
    for c in range(problems):
        # problem c's targets and identity order (sgd_fit's statements)
        ctx.enqueue_function[sgd_ys_kernel](
            dy.unsafe_ptr(), FP(unsafe_from_address=a_ys) + c * n, IP(unsafe_from_address=a_idx) + c * n,
            Int32(n), Int32(k), Int32(c), Int32(0), grid_dim=_xg_blocks(n), block_dim=XG_TPB,
        )
    var t = 1
    var live = problems
    for epoch in range(max_iter):
        if live == 0:
            break
        for c in range(problems):  # small-loop(problems: OvR problems): per-problem epoch counters, bookkeeping only
            if active[c]:
                epochs[c] = epoch + 1
        var par = epoch % 2
        # the epoch as ONE guarded unit: a cut epoch restores every problem's
        # weights and replays the same batches
        ctx.enqueue_copy(dst_buf=dws, src_buf=dw)
        ctx.enqueue_copy(dst_buf=dbs, src_buf=dbias)
        var t_start = t
        var tries = 0
        while True:
            var nonce = wit.begin()
            var wo = 0
            t = t_start
            if do_shuffle:
                ctx.enqueue_function[sgd_perm_kernel](
                    didx.unsafe_ptr(), Int32(n), seed_lo, seed_hi, Int32(epoch), Int32(0), Int32(problems),
                    grid_dim=_xg_blocks(problems * n), block_dim=XG_TPB,
                )
            dobj.enqueue_fill(Float32(0))
            if pa_rate:
                ctx.enqueue_function[sgd_rowsq_kernel](
                    dx.unsafe_ptr(), Int32(n), Int32(d), dsq.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                    grid_dim=_xg_blocks(n), block_dim=XG_TPB,
                )
                wo += _xg_blocks(n)
            var start = 0
            while start < n:
                var nbt = 0
                var tt = t
                var s1 = start
                while nbt < chunk and s1 < n:
                    var bs1 = min(batch, n - s1)
                    tt += bs1 if bsum else 1
                    s1 += bs1
                    nbt += 1
                # C17: bounded class waves reduce simultaneously live model
                # state. Four is a scheduling budget, never a dataset route.
                # NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
                var wave = 4 if C17_OVR else problems
                for first_problem in range(0, problems, wave):
                    var active_problems = min(wave, problems - first_problem)
                    ctx.enqueue_function[sgd_mb_chunk_ovr_kernel](
                        dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), dw.unsafe_ptr(), dbias.unsafe_ptr(),
                        dsw.unsafe_ptr(), ddl.unsafe_ptr(), dlv.unsafe_ptr(), dparts.unsafe_ptr(), dobj.unsafe_ptr(),
                        dci.unsafe_ptr(), dcf.unsafe_ptr(), ddotp.unsafe_ptr(), dsq.unsafe_ptr(), dact.unsafe_ptr(),
                        Int32(start), Int32(nbt), Int32(t),
                        wit.p(), Int32(wo), nonce, Int32(first_problem), grid_dim=active_problems, block_dim=SGD_CHUNK_TPB,
                    )
                    wo += active_problems
                t = tt
                start = s1
            if wit.ok(ctx, wo, "SGD one-vs-rest epoch"):
                break
            tries += 1
            if tries >= WITNESS_TRIES:
                wit.fail()
            ctx.enqueue_copy(dst_buf=dw, src_buf=dws)
            ctx.enqueue_copy(dst_buf=dbias, src_buf=dbs)
        # the epoch end: `sgd_mb_end_kernel` on each live problem's words (one
        # block each, reads parity par, writes 1 - par), every problem's stop
        # words home behind one synchronize
        tries = 0
        while True:
            var nonce = wit.begin()
            var we = 0
            for c in range(problems):
                if active[c]:
                    ctx.enqueue_function[sgd_mb_end_kernel](  # small-launch(d: weights): one block strides the d weights and two vfold sums; n only divides the objective
                        FP(unsafe_from_address=a_w) + c * d, FP(unsafe_from_address=a_bias) + c,
                        FP(unsafe_from_address=a_obj) + c, FP(unsafe_from_address=a_stt) + c * SGD_MB_WORDS,
                        FP(unsafe_from_address=a_cf) + c * SGD_OVR_CF,
                        dvp.unsafe_ptr(), Int32(d), Int32(n), alpha, l1r, Int32(penalty), tol, Int32(nic), Int32(lr),
                        Int32(1 if need_obj else 0), Int32(0), Int32(par), wit.p(), Int32(we), nonce,
                        grid_dim=1, block_dim=SGD_END_TPB,
                    )
                    we += 1
            ctx.enqueue_copy(dst_ptr=hstt.unsafe_ptr(), src_buf=dstt)
            if wit.ok(ctx, we, "SGD one-vs-rest epoch end"):
                break
            tries += 1
            if tries >= WITNESS_TRIES:
                wit.fail()
        ctx.synchronize()
        var changed = False
        for c in range(problems):  # small-loop(problems: OvR problems): reads each problem's two stop words, the epoch's scalar decision
            if active[c]:
                if hstt[c * SGD_MB_WORDS + SGD_MB_FLAGS + 1] != Float32(0):
                    failed[c] = True
                    active[c] = False
                elif hstt[c * SGD_MB_WORDS + SGD_MB_FLAGS] != Float32(0):
                    active[c] = False
                if not active[c]:
                    hact[c] = Int32(0)
                    live -= 1
                    changed = True
        if changed:
            ctx.enqueue_copy(dst_buf=dact, src_ptr=hact.unsafe_ptr())
            ctx.synchronize()
    var max_epochs = 0
    var status = 0
    for c in range(problems):
        ctx.enqueue_function[sgd_mb_res_kernel](
            FP(unsafe_from_address=a_w) + c * d, FP(unsafe_from_address=a_bias) + c, dres.unsafe_ptr(),
            Int32(c), Int32(d), Int32(problems), Int32(0), Int32(1 if failed[c] else 0),
            grid_dim=_xg_blocks(d + 1), block_dim=XG_TPB,
        )
        if failed[c]:
            status = -1
        elif epochs[c] > max_epochs:
            max_epochs = epochs[c]
    ctx.enqueue_copy(dst_ptr=res, src_buf=dres)
    ctx.synchronize()
    res.unsafe_store(problems * d + problems, i2f(max_epochs))
    res.unsafe_store(problems * d + problems + 1, i2f(status))
    _ = hci^
    _ = hcf^
    _ = hstt^
    _ = hact^
    _ = dx^
    _ = dy^
    _ = dys^
    _ = dsw^
    _ = didx^
    _ = ddl^
    _ = dlv^
    _ = dparts^
    _ = dw^
    _ = dbias^
    _ = dobj^
    _ = dws^
    _ = dbs^
    _ = ddotp^
    _ = dsq^
    _ = dci^
    _ = dcf^
    _ = dstt^
    _ = dvp^
    _ = dact^
    _ = dres^
    _ = wit^


def _sgd_chunk() -> Int:
    var v = String(getenv("MOJOLEARN_X_LINEAR_SGD_CHUNK"))
    if v == "":
        return SGD_CHUNK_DEFAULT
    try:
        return max(1, Int(v))
    except:
        return SGD_CHUNK_DEFAULT

def _sgd_mb_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                 n_out: Int, res: FP) raises:
    var ctx = linear_ctx()
    var k = Int(ip[0])
    var loss = Int(ip[1])
    var penalty = Int(ip[2])
    var lr = Int(ip[3])
    var fi = Int(ip[4]) != 0
    var max_iter = Int(ip[5])
    var nic = Int(ip[6])
    var do_shuffle = Int(ip[7]) != 0
    var seed = (UInt64(UInt32(ip[9])) << 32) | UInt64(UInt32(ip[8]))
    var has_sw = Int(ip[10]) != 0
    var has_cw = Int(ip[11]) != 0
    var batch = sgd_batch(Int(ip[12]), Int(ip[0]))
    var bsum = len(ip) > 13 and Int(ip[13]) != 0
    var alpha = fp[0]
    var l1r = fp[1]
    var eta0 = fp[2]
    var power_t = fp[3]
    var eps = fp[4]
    var tol = fp[5]
    if penalty == P_L2:
        l1r = Float32(0)
    elif penalty == P_L1:
        l1r = Float32(1)
    var problems = k if k > 2 else 1
    var one_class = k == 1
    # lane fam2-linear: the one-vs-rest problems side by side (SGD_IDN_OVR_PAR)
    if _sgd_mb_ovr_applies(problems, n, d, batch, _sgd_chunk()):
        _sgd_mb_ovr_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    var sub = mb_sub_size(batch)
    var dblk = mb_dblk(batch)
    var nsub = mb_subs(batch, sub)
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dys = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dsw = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var didx = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var ddl = ctx.enqueue_create_buffer[DType.float32](batch)
    var dlv = ctx.enqueue_create_buffer[DType.float32](batch)
    var dparts = ctx.enqueue_create_buffer[DType.float32]((d + 2) * nsub)
    var dw = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    # the intercept as (hi, lo): the one-class float-float (SGD_OC_MB_IMPLICIT;
    # the low word stays 0 and unread with ocm 0)
    var dbias = ctx.enqueue_create_buffer[DType.float32](2)
    var dobj = ctx.enqueue_create_buffer[DType.float32](1)
    var dws = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dbs = ctx.enqueue_create_buffer[DType.float32](2)
    var ocm = mb_oc_mode(one_class, fi, has_sw)
    # the implicit step's per-thread counts (the chunk kernel's width or a
    # XG_TPB block) and the step word
    var dcnt = ctx.enqueue_create_buffer[DType.int32](max(SGD_CHUNK_TPB, XG_TPB))
    var ddlt = ctx.enqueue_create_buffer[DType.float32](1)
    ddlt.enqueue_fill(Float32(0))
    var hb0 = List[Float32](length=2, fill=Float32(0))
    hb0[0] = Float32(1) if one_class else Float32(0)
    # SGD_IDN_MB_FUSE: the pair the fused step writes while it reads dw / dbias
    var dw2 = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dbias2 = ctx.enqueue_create_buffer[DType.float32](2)
    var wcap = 0
    var ws0 = 0
    while ws0 < n:
        var wbs = min(batch, n - ws0)
        wcap += _sgd_row_blocks(wbs) + _xg_blocks((d + 2) * mb_subs(wbs, sub)) + _xg_blocks(d + 2) + (1 if ocm == 2 else 0)
        ws0 += wbs
    wcap += _xg_blocks(n)
    var wit = Witness(ctx, max(wcap, 1))
    var chunk = _sgd_chunk()
    var dci = ctx.enqueue_create_buffer[DType.int32](17)
    var ddotp = ctx.enqueue_create_buffer[DType.float32](max(batch * ((d + MB_DBLK - 1) // MB_DBLK), 1))
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var dsq = ctx.enqueue_create_buffer[DType.float32](max(n, 1) if pa_rate else 1)
    var dcf = ctx.enqueue_create_buffer[DType.float32](9)
    var hci = List[Int32](length=17, fill=Int32(0))
    var hcf = List[Float32](length=9, fill=Float32(0))
    hci[0] = Int32(n)
    hci[1] = Int32(batch)
    hci[2] = Int32(d)
    hci[3] = Int32(loss)
    hci[4] = Int32(1 if has_sw else 0)
    hci[5] = Int32(1 if has_cw else 0)
    hci[6] = Int32(lr)
    hci[7] = Int32(nsub)
    hci[8] = Int32(penalty)
    hci[9] = Int32(1 if fi else 0)
    hci[12] = Int32(1 if bsum else 0)
    hci[13] = Int32(sub)
    hci[14] = Int32(dblk)
    hci[15] = Int32(1 if pa_rate else 0)
    hci[16] = Int32(ocm)
    hcf[0] = eps
    hcf[3] = eta0
    hcf[5] = alpha
    hcf[6] = l1r
    hcf[7] = power_t
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if has_sw:
        ctx.enqueue_copy(dst_buf=dsw, src_ptr=y + n)
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    if not one_class and n > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if SGD_IDN_DEV_FINITE:
        _sgd_finite_device(ctx, FP(unsafe_from_address=Int(dx.unsafe_ptr())), n_x, y, n_y)
    var seed_lo = Int32(Int(UInt32(seed & UInt64(0xFFFFFFFF))))
    var seed_hi = Int32(Int(UInt32(seed >> 32)))
    # the epoch end on the device (x_linear/sgd_end.mojo): its state, the
    # penalty fold's scratch, the result words
    var dstt = ctx.enqueue_create_buffer[DType.float32](SGD_MB_WORDS)
    var dvp = ctx.enqueue_create_buffer[DType.float32](vscratch(d) + 16)
    var dres = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    dres.enqueue_fill(Float32(0))
    var hst = List[Float32](length=SGD_MB_WORDS, fill=Float32(0))
    var hfl = List[Float32](length=2, fill=Float32(0))
    var dev_eta = 0 if (lr == LR_PA1 or lr == LR_PA2 or lr == LR_OPTIMAL or lr == LR_INVSCALING) else 1
    var need_obj = tol > Float32(-3.0e38)
    var max_epochs = 0
    var status = 0
    # SGD_PERC_AVG (x_linear/sgd_avg.mojo): the epoch-end iterate sums
    var avg = sgd_perc_avg_on(loss, lr, k, max_iter)
    var avg_from = sgd_perc_avg_from(max_iter)
    var dacc = ctx.enqueue_create_buffer[DType.float32](d + 1)
    for c in range(problems):
        var navg = 0
        if avg:
            dacc.enqueue_fill(Float32(0))
        # the targets and the identity order on the device (sgd_fit's statements)
        ctx.enqueue_function[sgd_ys_kernel](
            dy.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), Int32(n), Int32(k), Int32(c),
            Int32(1 if one_class else 0), grid_dim=_xg_blocks(n), block_dim=XG_TPB,
        )
        dw.enqueue_fill(Float32(0))
        ctx.enqueue_copy(dst_buf=dbias, src_ptr=hb0.unsafe_ptr())
        var wpos = fp[6 + c] if has_cw else Float32(1)
        var wneg = fp[6 + problems + c] if has_cw else Float32(1)
        var eta = eta0
        var opt_init = mb_optimal_init(loss, alpha, eps) if lr == LR_OPTIMAL else Float32(0)
        var t = 1
        var epochs = 0
        var failed = False
        hst[0] = Float32(3.0e38)
        hst[1] = Float32(0)
        hst[2] = eta0
        ctx.enqueue_copy(dst_buf=dstt, src_ptr=hst.unsafe_ptr())
        hcf[4] = eta0
        hcf[1] = wpos
        hcf[2] = wneg
        hcf[8] = opt_init
        hci[10] = Int32(1 if need_obj else 0)
        hci[11] = Int32(1 if one_class else 0)
        ctx.enqueue_copy(dst_buf=dci, src_ptr=hci.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dcf, src_ptr=hcf.unsafe_ptr())
        for epoch in range(max_iter):
            epochs = epoch + 1
            var par = epoch % 2
            # the epoch as ONE guarded unit (x_linear/witness.mojo): its steps
            # update the weights in place, so a cut epoch restores the weights
            # it started from and replays the same batches
            ctx.enqueue_copy(dst_buf=dws, src_buf=dw)
            ctx.enqueue_copy(dst_buf=dbs, src_buf=dbias)
            var t_start = t
            var tries = 0
            while True:
                var nonce = wit.begin()
                var wo = 0
                t = t_start
                if do_shuffle:
                    # the epoch's order on the device (a thread a row; idempotent)
                    ctx.enqueue_function[sgd_perm_kernel](
                        didx.unsafe_ptr(), Int32(n), seed_lo, seed_hi, Int32(epoch), Int32(c), Int32(1),
                        grid_dim=_xg_blocks(n), block_dim=XG_TPB,
                    )
                dobj.enqueue_fill(Float32(0))
                var start = 0
                if chunk > 1 and batch <= XG_TPB and d + 2 <= XG_TPB:
                    if pa_rate:
                        ctx.enqueue_function[sgd_rowsq_kernel](
                            dx.unsafe_ptr(), Int32(n), Int32(d), dsq.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                            grid_dim=_xg_blocks(n), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(n)
                    while start < n:
                        var nbt = 0
                        var tt = t
                        var s1 = start
                        while nbt < chunk and s1 < n:
                            var bs1 = min(batch, n - s1)
                            tt += bs1 if bsum else 1
                            s1 += bs1
                            nbt += 1
                        ctx.enqueue_function[sgd_mb_chunk_kernel](
                            dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), dw.unsafe_ptr(), dbias.unsafe_ptr(),
                            dsw.unsafe_ptr(), ddl.unsafe_ptr(), dlv.unsafe_ptr(), dparts.unsafe_ptr(), dobj.unsafe_ptr(),
                            dci.unsafe_ptr(), dcf.unsafe_ptr(), ddotp.unsafe_ptr(), dsq.unsafe_ptr(), dcnt.unsafe_ptr(),
                            Int32(start), Int32(nbt), Int32(t),
                            wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=SGD_CHUNK_TPB,
                        )
                        wo += 1
                        t = tt
                        start = s1
                elif SGD_IDN_MB_FUSE:
                    # the weights are in dw / dbias at the epoch's start (and
                    # after a restore); cur names the pair the last step wrote
                    var cur = 0
                    var bs_prev = 0
                    var et_prev = eta0
                    var etp = FP(unsafe_from_address=Int(dstt.unsafe_ptr())) + 4 * par + 2
                    # host-side addresses (the two pairs swap roles a batch)
                    var wa = Int(dw.unsafe_ptr())
                    var wb = Int(dw2.unsafe_ptr())
                    var ba = Int(dbias.unsafe_ptr())
                    var bb = Int(dbias2.unsafe_ptr())
                    while start < n:
                        var bs = min(batch, n - start)
                        var et = mb_eta(lr, eta0, eta0, alpha, power_t, opt_init, t)
                        if bs_prev == 0:
                            ctx.enqueue_function[sgd_mb_rows_kernel](
                                dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), Int32(start), Int32(bs), Int32(d),
                                dw.unsafe_ptr(), dbias.unsafe_ptr(), Int32(loss), eps, dsw.unsafe_ptr(), Int32(1 if has_sw else 0),
                                wpos, wneg, Int32(1 if has_cw else 0), ddl.unsafe_ptr(), dlv.unsafe_ptr(), Int32(lr), eta0, Int32(dblk),
                                Int32(ocm), wit.p(), Int32(wo), nonce, grid_dim=_sgd_row_blocks(bs), block_dim=SGD_ROW_TPB,
                            )
                        else:
                            var w_src = FP(unsafe_from_address=wa if cur == 0 else wb)
                            var b_src = FP(unsafe_from_address=ba if cur == 0 else bb)
                            var w_dst = FP(unsafe_from_address=wb if cur == 0 else wa)
                            var b_dst = FP(unsafe_from_address=bb if cur == 0 else ba)
                            ctx.enqueue_function[sgd_mb_steprows_kernel](
                                dparts.unsafe_ptr(), w_src, b_src, w_dst, b_dst,
                                dobj.unsafe_ptr(), dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), dsw.unsafe_ptr(),
                                ddl.unsafe_ptr(), dlv.unsafe_ptr(), dci.unsafe_ptr(), dcf.unsafe_ptr(), etp, ddlt.unsafe_ptr(),
                                Int32(start), Int32(bs_prev), Int32(bs), et_prev, Int32(dev_eta),
                                wit.p(), Int32(wo), nonce, grid_dim=_sgd_row_blocks(bs), block_dim=SGD_ROW_TPB,
                            )
                            cur = 1 - cur
                        wo += _sgd_row_blocks(bs)
                        if ocm == 2:
                            # the implicit step of this batch (its rows' margins
                            # above), dlt[0] for the step the next launch takes
                            ctx.enqueue_function[sgd_mb_oc_kernel](
                                ddl.unsafe_ptr(), dlv.unsafe_ptr(), Int32(bs), et, alpha, Int32(1 if bsum else 0),
                                dcnt.unsafe_ptr(), ddlt.unsafe_ptr(), etp, Int32(dev_eta), wit.p(), Int32(wo), nonce,
                                grid_dim=1, block_dim=XG_TPB,
                            )
                            wo += 1
                        ctx.enqueue_function[sgd_mb_parts_kernel](
                            dx.unsafe_ptr(), Int32(d), didx.unsafe_ptr(), Int32(start), ddl.unsafe_ptr(), dlv.unsafe_ptr(),
                            Int32(bs), Int32(nsub), dparts.unsafe_ptr(), Int32(sub), wit.p(), Int32(wo), nonce,
                            grid_dim=_xg_blocks((d + 2) * mb_subs(bs, sub)), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks((d + 2) * mb_subs(bs, sub))
                        bs_prev = bs
                        et_prev = et
                        t += bs if bsum else 1
                        start += bs
                    # the last batch's step, landing in dw / dbias: in place
                    # when the weights are there, else the step-only launch
                    if bs_prev > 0 and cur == 0:
                        ctx.enqueue_function[sgd_mb_step_kernel](
                            dparts.unsafe_ptr(), Int32(nsub), Int32(bs_prev), Int32(d), dw.unsafe_ptr(), dbias.unsafe_ptr(),
                            dobj.unsafe_ptr(), et_prev, alpha, l1r, Int32(penalty), Int32(1 if fi else 0), Int32(1 if need_obj else 0),
                            Int32(1 if one_class else 0), Int32(1 if bsum else 0), Int32(sub),
                            etp, Int32(dev_eta), Int32(ocm), ddlt.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                            grid_dim=_xg_blocks(d + 2), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(d + 2)
                    elif bs_prev > 0:
                        ctx.enqueue_function[sgd_mb_steprows_kernel](
                            dparts.unsafe_ptr(), dw2.unsafe_ptr(), dbias2.unsafe_ptr(), dw.unsafe_ptr(), dbias.unsafe_ptr(),
                            dobj.unsafe_ptr(), dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), dsw.unsafe_ptr(),
                            ddl.unsafe_ptr(), dlv.unsafe_ptr(), dci.unsafe_ptr(), dcf.unsafe_ptr(), etp, ddlt.unsafe_ptr(),
                            Int32(start), Int32(bs_prev), Int32(0), et_prev, Int32(dev_eta),
                            wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=XG_TPB,
                        )
                        wo += 1
                else:
                    while start < n:
                        var bs = min(batch, n - start)
                        var et = mb_eta(lr, eta0, eta0, alpha, power_t, opt_init, t)
                        ctx.enqueue_function[sgd_mb_rows_kernel](
                            dx.unsafe_ptr(), dys.unsafe_ptr(), didx.unsafe_ptr(), Int32(start), Int32(bs), Int32(d),
                            dw.unsafe_ptr(), dbias.unsafe_ptr(), Int32(loss), eps, dsw.unsafe_ptr(), Int32(1 if has_sw else 0),
                            wpos, wneg, Int32(1 if has_cw else 0), ddl.unsafe_ptr(), dlv.unsafe_ptr(), Int32(lr), eta0, Int32(dblk),
                            Int32(ocm), wit.p(), Int32(wo), nonce, grid_dim=_sgd_row_blocks(bs), block_dim=SGD_ROW_TPB,
                        )
                        wo += _sgd_row_blocks(bs)
                        if ocm == 2:
                            ctx.enqueue_function[sgd_mb_oc_kernel](
                                ddl.unsafe_ptr(), dlv.unsafe_ptr(), Int32(bs), et, alpha, Int32(1 if bsum else 0),
                                dcnt.unsafe_ptr(), ddlt.unsafe_ptr(), FP(unsafe_from_address=Int(dstt.unsafe_ptr())) + 4 * par + 2,
                                Int32(dev_eta), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=XG_TPB,
                            )
                            wo += 1
                        ctx.enqueue_function[sgd_mb_parts_kernel](
                            dx.unsafe_ptr(), Int32(d), didx.unsafe_ptr(), Int32(start), ddl.unsafe_ptr(), dlv.unsafe_ptr(),
                            Int32(bs), Int32(nsub), dparts.unsafe_ptr(), Int32(sub), wit.p(), Int32(wo), nonce,
                            grid_dim=_xg_blocks((d + 2) * mb_subs(bs, sub)), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks((d + 2) * mb_subs(bs, sub))
                        ctx.enqueue_function[sgd_mb_step_kernel](
                            dparts.unsafe_ptr(), Int32(nsub), Int32(bs), Int32(d), dw.unsafe_ptr(), dbias.unsafe_ptr(),
                            dobj.unsafe_ptr(), et, alpha, l1r, Int32(penalty), Int32(1 if fi else 0), Int32(1 if need_obj else 0),
                            Int32(1 if one_class else 0), Int32(1 if bsum else 0), Int32(sub),
                            FP(unsafe_from_address=Int(dstt.unsafe_ptr())) + 4 * par + 2, Int32(dev_eta), Int32(ocm),
                            ddlt.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                            grid_dim=_xg_blocks(d + 2), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(d + 2)
                        t += bs if bsum else 1
                        start += bs
                if wit.ok(ctx, wo, "SGD epoch"):
                    break
                tries += 1
                if tries >= WITNESS_TRIES:
                    wit.fail()
                ctx.enqueue_copy(dst_buf=dw, src_buf=dws)
                ctx.enqueue_copy(dst_buf=dbias, src_buf=dbs)
            # the epoch end (one block, its own guarded unit; reads parity
            # par, writes 1 - par): two words home
            tries = 0
            while True:
                var nonce = wit.begin()
                ctx.enqueue_function[sgd_mb_end_kernel](  # small-launch(d: weights): one block strides the d weights and two vfold sums; n only divides the objective
                    dw.unsafe_ptr(), dbias.unsafe_ptr(), dobj.unsafe_ptr(), dstt.unsafe_ptr(), dcf.unsafe_ptr(),
                    dvp.unsafe_ptr(), Int32(d), Int32(n), alpha, l1r, Int32(penalty), tol, Int32(nic), Int32(lr),
                    Int32(1 if need_obj else 0), Int32(1 if one_class else 0), Int32(par), wit.p(), Int32(0), nonce,
                    grid_dim=1, block_dim=SGD_END_TPB,
                )
                ctx.enqueue_copy(dst_ptr=hfl.unsafe_ptr(), src_buf=dstt.create_sub_buffer[DType.float32](SGD_MB_FLAGS, 2))
                if wit.ok(ctx, 1, "SGD epoch end"):
                    break
                tries += 1
                if tries >= WITNESS_TRIES:
                    wit.fail()
            ctx.synchronize()
            if hfl[1] != Float32(0):
                failed = True
                break
            if avg and epoch >= avg_from:
                ctx.enqueue_function[sgd_avg_acc_kernel](
                    dw.unsafe_ptr(), dbias.unsafe_ptr(), dacc.unsafe_ptr(), Int32(d),
                    grid_dim=_xg_blocks(d + 1), block_dim=XG_TPB,
                )
                navg += 1
            if hfl[0] != Float32(0):
                break
        if navg > 0 and not failed:
            ctx.enqueue_function[sgd_avg_fin_kernel](
                dw.unsafe_ptr(), dbias.unsafe_ptr(), dacc.unsafe_ptr(), Int32(d), Float32(1) / Float32(navg),
                grid_dim=_xg_blocks(d + 1), block_dim=XG_TPB,
            )
        ctx.enqueue_function[sgd_mb_res_kernel](
            dw.unsafe_ptr(), dbias.unsafe_ptr(), dres.unsafe_ptr(), Int32(c), Int32(d), Int32(problems),
            Int32((2 if ocm != 0 else 1) if one_class else 0), Int32(1 if failed else 0), grid_dim=_xg_blocks(d + 1),
            block_dim=XG_TPB,
        )
        if failed:
            status = -1
        elif epochs > max_epochs:
            max_epochs = epochs
    ctx.enqueue_copy(dst_ptr=res, src_buf=dres)
    ctx.synchronize()
    res.unsafe_store(problems * d + problems, i2f(max_epochs))
    res.unsafe_store(problems * d + problems + 1, i2f(status))
    _ = dy^
    _ = hst^
    _ = hfl^
    _ = dstt^
    _ = dvp^
    _ = dres^
    _ = dx^
    _ = dys^
    _ = dsw^
    _ = didx^
    _ = ddl^
    _ = dlv^
    _ = dparts^
    _ = dw^
    _ = dbias^
    _ = dobj^
    _ = dws^
    _ = dacc^
    _ = dbs^
    _ = dw2^
    _ = dbias2^
    _ = dcnt^
    _ = ddlt^
    _ = hb0^
    _ = wit^



# ------------------------------------------------ per-sample SGD on the grid (lane/neural-pass139)
# x_linear/sgd.mojo `sgd_one` (their plain SGD: one sample after the next, the
# order is the algorithm) with each sample spread over a block (GPU-only
# rule: the one-thread team form took 630 s for a million istella rows on an
# L40S, so the device binding ran it on the host). One block a one-vs-rest
# problem (grid_dim = P); per sample, in order:
#   A. one task a MB_DBLK-column block of the predictor (`mb_block_dot`, its
#      loads all in flight before its chain) and, when `tol` reads the
#      objective, one a block of the penalty norms (`sgd_reg_block`);
#      the partials to device words, then a device-ordering barrier;
#   B. EVERY thread folds the blocks ascending from zero (`mb_dot`'s words)
#      and runs sgd_one's scalar statements (the rate, the loss, the PA step
#      from the |x_i|^2 of `mb_rowsq`, the clip, the class and sample
#      weights, the decay factor, the intercept and the cumulative L1 u) in
#      registers: the same words in every thread, so no barrier is needed
#      before C and the control flow is uniform;
#   C. thread j applies sgd_one's per-weight statements to v_j (the fold
#      `ws_mul(v_j, hi, lo)` when wscale fell below 1e-9, the step
#      `fmad(update / wscale, x_ij, v_j)`, Tsuruoka's clip `ws_clip` with
#      q_j), then a barrier.
# The weights are their WeightVector's v with wscale = (hi, lo) a float-float
# in the scalar state (x_linear/sgd.mojo `ws_decay`); w = wscale * v leaves
# the device only at the finite check and the result.
# A launch runs SGD_PS_CHUNK samples of the epoch; the epoch is ONE witness-
# guarded unit (x_linear/witness.mojo): w, q, the scalar state (wscale
# included) and t are
# restored from their epoch-start copies and the epoch replays on a rerun.
# The epoch order (`sgd_perm_kernel`), the finite check, the stopping and
# the adaptive rate are the device's (x_linear/sgd_end.mojo, sgd_one's
# statements); the host reads the live problem count once an epoch.
# C19: scheduling only; preserve sample order/time index and per-sample updates.
# Fixed bounded launch budgets, independent of dataset dimensions.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
comptime SGD_PS_CHUNK = C19_SGD_CHUNK
# the per-problem scalar state ps[SGD_PS_ST c ..]: intercept, u, objective,
# the one-class intercept's low word, wscale hi, wscale lo
comptime SGD_PS_ST = 6


def _sgd_ps_kernel_body(
    x: FP, lab: FP, swp: FP, idx: IP, w: FP, q: FP, sqp: FP, part: FP,
    ps: FP, pt: IP, act: IP, pf: FP, ci: IP, cf: FP, start: Int32, cnt: Int32,
):
    """ci: n, d, loss, penalty, lr, fit_intercept, need_obj, one_class, has_sw,
    has_cw, k, has_sq; cf: alpha, l1_ratio (the penalty's), eta0, power_t,
    eps, optimal_init, decay_factor; per problem c: pf[3c..] eta, wpos, wneg;
    ps[SGD_PS_ST c..] intercept, u, objective, intercept lo (one-class), wscale hi, lo; pt[c] t; act[c] 1 while it trains."""
    var c = Int(block_idx.x)
    if ldi(act, c) == 0:
        return
    var nn = ldi(ci, 0)
    var d = ldi(ci, 1)
    var loss = ldi(ci, 2)
    var penalty = ldi(ci, 3)
    var lr = ldi(ci, 4)
    var fi = ldi(ci, 5) != 0
    var need_obj = ldi(ci, 6) != 0
    var one_class = ldi(ci, 7) != 0
    var has_sw = ldi(ci, 8) != 0
    var has_cw = ldi(ci, 9) != 0
    var k = ldi(ci, 10)
    var alpha = ld(cf, 0)
    var l1_ratio = ld(cf, 1)
    var eta0 = ld(cf, 2)
    var power_t = ld(cf, 3)
    var eps = ld(cf, 4)
    var optimal_init = ld(cf, 5)
    var decay_factor = ld(cf, 6)
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var nb = (d + MB_DBLK - 1) // MB_DBLK
    var need_reg = need_obj and penalty != P_NONE and not pa_rate
    var ntask = 2 * nb if need_reg else nb
    var woff = c * d
    var poff = c * 3 * nb
    var do_decay = penalty == P_L2 or penalty == P_EN
    var do_l1 = penalty == P_L1 or penalty == P_EN
    var intercept = ld(ps, SGD_PS_ST * c)
    var u = ld(ps, SGD_PS_ST * c + 1)
    var objective = Float32(0) if Int(start) == 0 else ld(ps, SGD_PS_ST * c + 2)
    var il = ld(ps, SGD_PS_ST * c + 3)
    var whi = ld(ps, SGD_PS_ST * c + 4)
    var wlo = ld(ps, SGD_PS_ST * c + 5)
    var t = ldi(pt, c)
    var eta = ld(pf, 3 * c)
    var wpos = ld(pf, 3 * c + 1)
    var wneg = ld(pf, 3 * c + 2)
    for r in range(Int(start), Int(start) + Int(cnt)):
        var i = ldi(idx, c * nn + r)
        # A: the blocks
        for qq in range(tid, ntask, nt):
            if qq < nb:
                st(part, poff + qq, mb_block_dot(x, i, d, w, woff, qq))
            else:
                var rb = sgd_reg_block(w, woff, d, qq - nb)
                st(part, poff + qq, rb[0])
                st(part, poff + nb + qq, rb[1])
        team_barrier()
        # B: sgd_one's scalar statements, in every thread
        var dot = Float32(0)
        for b in range(nb):
            dot = fa(dot, ld(part, poff + b))
        var y = _sgd_target(k, c, ld(lab, i))
        var dotw = ws_mul(dot, whi, wlo)
        var p = fa(fa(dotw, il), intercept) if one_class else fa(dotw, intercept)
        if lr == LR_OPTIMAL:
            eta = fd(Float32(1), fm(alpha, fs(fa(optimal_init, i2f(t)), Float32(1))))
        elif lr == LR_INVSCALING:
            eta = fd(eta0, identical_pow(i2f(t), power_t))
        var oc = one_class and loss == L_HINGE
        var och = oc_hinge(dotw, intercept, il)
        var cur = och[0] if oc else sgd_loss(loss, y, p, eps)
        if need_obj:
            objective = fa(objective, cur)
            if not pa_rate:
                if penalty != P_NONE:
                    var n2 = Float32(0)
                    var n1 = Float32(0)
                    for b in range(nb):
                        n2 = fa(n2, ld(part, poff + nb + b))
                        n1 = fa(n1, ld(part, poff + 2 * nb + b))
                    n2 = fm(fm(whi, whi), n2)
                    n1 = fm(whi, n1)
                    var reg = fa(fm(fm(fs(Float32(1), l1_ratio), Float32(0.5)), n2), fm(l1_ratio, n1))
                    objective = fa(objective, fm(alpha, reg))
                if one_class:
                    objective = fa(objective, fm(intercept, alpha))
        var skip = False
        var update = Float32(0)
        if pa_rate:
            var sq = ld(sqp, i)
            if lr == LR_PA1:
                if sq == 0:
                    skip = True
                else:
                    update = fmin(eta0, fd(cur, sq))
            else:
                update = fd(cur, fa(sq, fd(Float32(0.5), eta0)))
            if not skip:
                if loss == L_HINGE:
                    update = fm(update, y)
                elif fs(y, p) < 0:
                    update = -update
        else:
            var dl = och[1] if oc else sgd_dloss(loss, y, p, eps)
            if dl < Float32(-1e12):
                dl = Float32(-1e12)
            elif dl > Float32(1e12):
                dl = Float32(1e12)
            update = fm(-eta, dl)
        if not skip:
            if has_cw or has_sw:
                var cwv = Float32(1)
                if has_cw:
                    cwv = wpos if y > 0 else wneg
                var swi = ld(swp, i) if has_sw else Float32(1)
                update = fm(update, fm(cwv, swi))
            # their w.scale on wscale, the fold into v below 1e-9 (C)
            var fold = False
            var fhi = Float32(1)
            var flo = Float32(0)
            if do_decay:
                var ws = ws_decay(whi, wlo, fm(decay_factor, eta))
                whi = ws[0]
                wlo = ws[1]
                if whi < WS_RESET:
                    fold = True
                    fhi = whi
                    flo = wlo
                    whi = Float32(1)
                    wlo = Float32(0)
            var cu = Float32(0)
            if update != 0:
                cu = ws_div(update, whi, wlo)
            if fi:
                var iu = update
                if one_class:
                    iu = fs(iu, fm(eta, alpha))
                if iu != 0:
                    if one_class:
                        var ia = ff_add(intercept, il, iu)
                        intercept = ia[0]
                        il = ia[1]
                    else:
                        intercept = fa(intercept, iu)
            if do_l1:
                u = fa(u, fm(fm(l1_ratio, eta), alpha))
            # C: the weights, thread j
            var ixd = i * d
            for j in range(tid, d, nt):
                var wj = ld(w, woff + j)
                if fold:
                    wj = ws_mul(wj, fhi, flo)
                if update != 0:
                    wj = fmad(cu, ld(x, ixd + j), wj)
                if do_l1:
                    var rq = ws_clip(wj, u, ld(q, woff + j), whi)
                    st(q, woff + j, rq[1])
                    wj = rq[0]
                st(w, woff + j, wj)
            t += 1
        team_barrier()
    if tid == 0:
        st(ps, SGD_PS_ST * c, intercept)
        st(ps, SGD_PS_ST * c + 1, u)
        st(ps, SGD_PS_ST * c + 2, objective)
        st(ps, SGD_PS_ST * c + 3, il)
        st(ps, SGD_PS_ST * c + 4, whi)
        st(ps, SGD_PS_ST * c + 5, wlo)
        sti(pt, c, t)


# lane/apple-fast-gap-clus3 (2026-10-03): SGD_FAST_PS_SIMD, FAST + Apple default
# for d <= SPS_MAX_D = SPS_W * SPS_MAXC, the register bound (off:
# -D MOJOLEARN_SGD_FAST_PS_SIMD_OFF; old d <= 32 gate behind
# MOJOLEARN_LEGACY_NARROW_SGD_PS_SIMD). M3 A/Bs (n=1):
# sgd-ocsvm taxi (d 11) 58,334 -> 42,252 ms (-27.6%), fraction_flagged .05557
# identical (clus3-sgdoc-simd-taxi); sgd-ocsvm istella (d 220) 75,506 ->
# 106,443 ms (+41%, clus3-sgdoc-simd-istella), hence the old small-d gate. Cause: the per-sample fit above runs a
# whole block per problem and, per SAMPLE, writes the MB_DBLK block partials
# to device memory, crosses a device-memory `team_barrier`, re-reads them and
# reloads/stores every weight from device memory: ~3.7 us a sample on the M3,
# 75 s for SGDOneClassSVM's 20 epochs of a million Istella rows (0.8.34's
# warp form, since deleted, was ~1.8x scikit-learn). Here a problem is ONE
# simdgroup (SPS_W threads): lane l keeps weights l, l + 32, ... (and their
# q) in registers for the whole launch, the predictor and the penalty norms
# are a butterfly over the simdgroup with lane 0's word broadcast (uniform
# scalars in every lane, no barrier: a lane only touches its own columns),
# and the weights go back to device memory once a launch. d <= SPS_W *
# SPS_MAXC, else the block form. FAST: the predictor's fold order changes.
# SGDOC_FAST_TAIL: FAST+Apple DEFAULT (rollback -D MOJOLEARN_SGDOC_FAST_TAIL_OFF;
# -D MOJOLEARN_SGDOC_FAST_TAIL_LONG for the 4x tail). SGDOneClassSVM at
# learning_rate='optimal', tol=None on centered data: main's per-sample
# kernels over the LAST SGDOC_TAIL_K steps of the real schedule from w = 0,
# intercept = 1 (the 1/t schedule makes the final state a stationary chain's
# draw; derivation and gate in x_linear/sgdoc_tail.mojo). Non-centered data
# keeps the full run bit for bit. M3, source 97ea50da3, one run per arm:
# taxi 42142.4 -> 153.7 ms, istella 75533.9 -> 407.0 ms (w2-sgdoc-tail-*);
# w2-sgdoc-q PASS (flag fraction, objective, |w|, score spread, Jaccard vs
# main's own seed spread). Still main's serial per-sample chain, 300x
# shorter: a parallel SGD one-class algorithm remains owed.
comptime SGDOC_FAST_TAIL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SGDOC_FAST_TAIL_OFF"]()
)
comptime SGDOC_TAIL_K = 262144 if is_defined["MOJOLEARN_SGDOC_FAST_TAIL_LONG"]() else 65536
# The tail must exceed the (S, R) chain's mixing time, which grows with d
# (x_linear/sgdoc_tail.mojo: tens of steps at d ~ 10, up to thousands at
# d ~ 200, i.e. at most ~20 steps per feature). A fixed K was only checked at
# those two widths, so the tail is max(SGDOC_TAIL_K, SGDOC_TAIL_K_PER_D * d):
# a >= 10x margin over that per-feature mixing at every d. At d <= 256 this is
# SGDOC_TAIL_K exactly (no change); wider fits get a proportionally longer
# tail (FAST only; longer tail = closer to main's full run, never worse).
comptime SGDOC_TAIL_K_PER_D = 256


def sgdoc_tail_k(d: Int) -> Int:
    return max(SGDOC_TAIL_K, SGDOC_TAIL_K_PER_D * d)


comptime SGD_FAST_PS_SIMD = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SGD_FAST_PS_SIMD_OFF"]()
)
comptime SPS_W = 32
comptime SPS_MAXC = 8
#: Kernel limit (register footprint): each of the SPS_W lanes holds SPS_MAXC
#: weights and their q in registers, so d <= SPS_W * SPS_MAXC.
comptime SPS_MAX_D = SPS_W * SPS_MAXC
#: LEGACY, default OFF: the old gate admitted only d <= 32, chosen between
#: taxi (d 11, -27.6%) and istella (d 220, +41%). Removed as benchmark-tuned
#: on 2026-10-04; the register-bound replacement is UNMEASURED.
comptime SPS_LEGACY_NARROW = is_defined["MOJOLEARN_LEGACY_NARROW_SGD_PS_SIMD"]()
comptime SPS_LEGACY_MAX_D = 32


@always_inline
def _sps_sum(v: Float32) -> Float32:
    var a = v
    comptime for k in range(5):
        a = fa(a, shuffle_xor(a, UInt32(16 >> k)))
    return shuffle_idx(a, UInt32(0))


def _sgd_ps_simd_body(
    x: FP, lab: FP, swp: FP, idx: IP, w: FP, q: FP, sqp: FP, part: FP,
    ps: FP, pt: IP, act: IP, pf: FP, ci: IP, cf: FP, start: Int32, cnt: Int32,
):
    """ci: n, d, loss, penalty, lr, fit_intercept, need_obj, one_class, has_sw,
    has_cw, k, has_sq; cf: alpha, l1_ratio (the penalty's), eta0, power_t,
    eps, optimal_init, decay_factor; per problem c: pf[3c..] eta, wpos, wneg;
    ps[SGD_PS_ST c..] intercept, u, objective, intercept lo (one-class), wscale hi, lo; pt[c] t; act[c] 1 while it trains."""
    var c = Int(block_idx.x)
    if ldi(act, c) == 0:
        return
    var nn = ldi(ci, 0)
    var d = ldi(ci, 1)
    var loss = ldi(ci, 2)
    var penalty = ldi(ci, 3)
    var lr = ldi(ci, 4)
    var fi = ldi(ci, 5) != 0
    var need_obj = ldi(ci, 6) != 0
    var one_class = ldi(ci, 7) != 0
    var has_sw = ldi(ci, 8) != 0
    var has_cw = ldi(ci, 9) != 0
    var k = ldi(ci, 10)
    var alpha = ld(cf, 0)
    var l1_ratio = ld(cf, 1)
    var eta0 = ld(cf, 2)
    var power_t = ld(cf, 3)
    var eps = ld(cf, 4)
    var optimal_init = ld(cf, 5)
    var decay_factor = ld(cf, 6)
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var nb = (d + MB_DBLK - 1) // MB_DBLK
    var need_reg = need_obj and penalty != P_NONE and not pa_rate
    var ntask = 2 * nb if need_reg else nb
    var woff = c * d
    var poff = c * 3 * nb
    var do_decay = penalty == P_L2 or penalty == P_EN
    var do_l1 = penalty == P_L1 or penalty == P_EN
    var intercept = ld(ps, SGD_PS_ST * c)
    var u = ld(ps, SGD_PS_ST * c + 1)
    var objective = Float32(0) if Int(start) == 0 else ld(ps, SGD_PS_ST * c + 2)
    var il = ld(ps, SGD_PS_ST * c + 3)
    var whi = ld(ps, SGD_PS_ST * c + 4)
    var wlo = ld(ps, SGD_PS_ST * c + 5)
    var t = ldi(pt, c)
    var eta = ld(pf, 3 * c)
    var wpos = ld(pf, 3 * c + 1)
    var wneg = ld(pf, 3 * c + 2)
    var wv = SIMD[DType.float32, SPS_MAXC](0)
    var qv = SIMD[DType.float32, SPS_MAXC](0)
    comptime for uu in range(SPS_MAXC):
        var j = tid + uu * SPS_W
        if j < d:
            wv[uu] = ld(w, woff + j)
            if do_l1:
                qv[uu] = ld(q, woff + j)
    for r in range(Int(start), Int(start) + Int(cnt)):
        var i = ldi(idx, c * nn + r)
        # A: lane l's columns l, l + 32, ... from the registers, the row's
        # loads coalesced; the simdgroup's butterfly, lane 0's word broadcast
        var ixd = i * d
        var accd = Float32(0)
        comptime for ua in range(SPS_MAXC):
            var j = tid + ua * SPS_W
            if j < d:
                accd = fmad(ld(x, ixd + j), wv[ua], accd)
        var dot = _sps_sum(accd)
        var y = _sgd_target(k, c, ld(lab, i))
        var dotw = ws_mul(dot, whi, wlo)
        var p = fa(fa(dotw, il), intercept) if one_class else fa(dotw, intercept)
        if lr == LR_OPTIMAL:
            eta = fd(Float32(1), fm(alpha, fs(fa(optimal_init, i2f(t)), Float32(1))))
        elif lr == LR_INVSCALING:
            eta = fd(eta0, identical_pow(i2f(t), power_t))
        var oc = one_class and loss == L_HINGE
        var och = oc_hinge(dotw, intercept, il)
        var cur = och[0] if oc else sgd_loss(loss, y, p, eps)
        if need_obj:
            objective = fa(objective, cur)
            if not pa_rate:
                if penalty != P_NONE:
                    var a2 = Float32(0)
                    var a1 = Float32(0)
                    comptime for ub in range(SPS_MAXC):
                        if tid + ub * SPS_W < d:
                            a2 = fmad(wv[ub], wv[ub], a2)
                            a1 = fa(a1, fabs(wv[ub]))
                    var n2 = _sps_sum(a2)
                    var n1 = _sps_sum(a1)
                    n2 = fm(fm(whi, whi), n2)
                    n1 = fm(whi, n1)
                    var reg = fa(fm(fm(fs(Float32(1), l1_ratio), Float32(0.5)), n2), fm(l1_ratio, n1))
                    objective = fa(objective, fm(alpha, reg))
                if one_class:
                    objective = fa(objective, fm(intercept, alpha))
        var skip = False
        var update = Float32(0)
        if pa_rate:
            var sq = ld(sqp, i)
            if lr == LR_PA1:
                if sq == 0:
                    skip = True
                else:
                    update = fmin(eta0, fd(cur, sq))
            else:
                update = fd(cur, fa(sq, fd(Float32(0.5), eta0)))
            if not skip:
                if loss == L_HINGE:
                    update = fm(update, y)
                elif fs(y, p) < 0:
                    update = -update
        else:
            var dl = och[1] if oc else sgd_dloss(loss, y, p, eps)
            if dl < Float32(-1e12):
                dl = Float32(-1e12)
            elif dl > Float32(1e12):
                dl = Float32(1e12)
            update = fm(-eta, dl)
        if not skip:
            if has_cw or has_sw:
                var cwv = Float32(1)
                if has_cw:
                    cwv = wpos if y > 0 else wneg
                var swi = ld(swp, i) if has_sw else Float32(1)
                update = fm(update, fm(cwv, swi))
            # their w.scale on wscale, the fold into v below 1e-9 (C)
            var fold = False
            var fhi = Float32(1)
            var flo = Float32(0)
            if do_decay:
                var ws = ws_decay(whi, wlo, fm(decay_factor, eta))
                whi = ws[0]
                wlo = ws[1]
                if whi < WS_RESET:
                    fold = True
                    fhi = whi
                    flo = wlo
                    whi = Float32(1)
                    wlo = Float32(0)
            var cu = Float32(0)
            if update != 0:
                cu = ws_div(update, whi, wlo)
            if fi:
                var iu = update
                if one_class:
                    iu = fs(iu, fm(eta, alpha))
                if iu != 0:
                    if one_class:
                        var ia = ff_add(intercept, il, iu)
                        intercept = ia[0]
                        il = ia[1]
                    else:
                        intercept = fa(intercept, iu)
            if do_l1:
                u = fa(u, fm(fm(l1_ratio, eta), alpha))
            # C: the weights, thread j
            comptime for uu in range(SPS_MAXC):
                var j = tid + uu * SPS_W
                if j < d:
                    var wj = wv[uu]
                    if fold:
                        wj = ws_mul(wj, fhi, flo)
                    if update != 0:
                        wj = fmad(cu, ld(x, ixd + j), wj)
                    if do_l1:
                        var rq = ws_clip(wj, u, qv[uu], whi)
                        qv[uu] = rq[1]
                        wj = rq[0]
                    wv[uu] = wj
            t += 1
    comptime for uu in range(SPS_MAXC):
        var j = tid + uu * SPS_W
        if j < d:
            st(w, woff + j, wv[uu])
            if do_l1:
                st(q, woff + j, qv[uu])
    if tid == 0:
        st(ps, SGD_PS_ST * c, intercept)
        st(ps, SGD_PS_ST * c + 1, u)
        st(ps, SGD_PS_ST * c + 2, objective)
        st(ps, SGD_PS_ST * c + 3, il)
        st(ps, SGD_PS_ST * c + 4, whi)
        st(ps, SGD_PS_ST * c + 5, wlo)
        sti(pt, c, t)


# lane/idn-sgd-multiblock (2026-10-04): SGD_IDN_PS_WARP, IDENTICAL on NVIDIA
# and AMD for d <= SPS_W * MB_DBLK (off: -D MOJOLEARN_SGD_IDN_PS_WARP_OFF).
# The block form above pays, per SAMPLE, the partials' stores, two device-
# ordering barriers and a device load and store of every weight. Here a
# problem is ONE warp of SPS_W lanes: lane l keeps the weights (and q) of
# column block l (MB_DBLK columns) in registers for the whole launch and
# runs that block's chain (`mb_block_dot`'s, `sgd_reg_block`'s); every lane
# then folds the blocks' partials ascending from zero, each read from its
# lane (`shuffle_idx`), so the scalars are uniform without a barrier, and
# updates its own block's weights. The same chains folded in the same order
# as the block form and the host's `sgd_one`: no bit moves.
comptime SGD_IDN_PS_WARP = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not has_apple_gpu_accelerator()
    and not (is_defined["MOJOLEARN_SGD_IDN_PS_WARP_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())
)


@always_inline
def _spw_fold(v: Float32, nb: Int) -> Float32:
    """fa-fold of lanes 0 .. nb - 1's words, ascending from zero."""
    var acc = Float32(0)
    for b in range(nb):
        acc = fa(acc, shuffle_idx(v, UInt32(b)))
    return acc


def _sgd_ps_warp_body(
    x: FP, lab: FP, swp: FP, idx: IP, w: FP, q: FP, sqp: FP, part: FP,
    ps: FP, pt: IP, act: IP, pf: FP, ci: IP, cf: FP, start: Int32, cnt: Int32,
):
    """ci: n, d, loss, penalty, lr, fit_intercept, need_obj, one_class, has_sw,
    has_cw, k, has_sq; cf: alpha, l1_ratio (the penalty's), eta0, power_t,
    eps, optimal_init, decay_factor; per problem c: pf[3c..] eta, wpos, wneg;
    ps[SGD_PS_ST c..] intercept, u, objective, intercept lo (one-class), wscale hi, lo; pt[c] t; act[c] 1 while it trains."""
    var c = Int(block_idx.x)
    if ldi(act, c) == 0:
        return
    var nn = ldi(ci, 0)
    var d = ldi(ci, 1)
    var loss = ldi(ci, 2)
    var penalty = ldi(ci, 3)
    var lr = ldi(ci, 4)
    var fi = ldi(ci, 5) != 0
    var need_obj = ldi(ci, 6) != 0
    var one_class = ldi(ci, 7) != 0
    var has_sw = ldi(ci, 8) != 0
    var has_cw = ldi(ci, 9) != 0
    var k = ldi(ci, 10)
    var alpha = ld(cf, 0)
    var l1_ratio = ld(cf, 1)
    var eta0 = ld(cf, 2)
    var power_t = ld(cf, 3)
    var eps = ld(cf, 4)
    var optimal_init = ld(cf, 5)
    var decay_factor = ld(cf, 6)
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var nb = (d + MB_DBLK - 1) // MB_DBLK
    var need_reg = need_obj and penalty != P_NONE and not pa_rate
    var ntask = 2 * nb if need_reg else nb
    var woff = c * d
    var j0 = tid * MB_DBLK
    var poff = c * 3 * nb
    var do_decay = penalty == P_L2 or penalty == P_EN
    var do_l1 = penalty == P_L1 or penalty == P_EN
    var intercept = ld(ps, SGD_PS_ST * c)
    var u = ld(ps, SGD_PS_ST * c + 1)
    var objective = Float32(0) if Int(start) == 0 else ld(ps, SGD_PS_ST * c + 2)
    var il = ld(ps, SGD_PS_ST * c + 3)
    var whi = ld(ps, SGD_PS_ST * c + 4)
    var wlo = ld(ps, SGD_PS_ST * c + 5)
    var t = ldi(pt, c)
    var eta = ld(pf, 3 * c)
    var wpos = ld(pf, 3 * c + 1)
    var wneg = ld(pf, 3 * c + 2)
    var wv = SIMD[DType.float32, MB_DBLK](0)
    var qv = SIMD[DType.float32, MB_DBLK](0)
    comptime for uu in range(MB_DBLK):
        var j = j0 + uu
        if j < d:
            wv[uu] = ld(w, woff + j)
            if do_l1:
                qv[uu] = ld(q, woff + j)
    for r in range(Int(start), Int(start) + Int(cnt)):
        var i = ldi(idx, c * nn + r)
        # A: lane l's block (`mb_block_dot`'s chain, the weights from the
        # registers); the blocks' partials folded ascending in every lane
        var ixd = i * d
        var accd = Float32(0)
        comptime for ua in range(MB_DBLK):
            var j = j0 + ua
            if j < d:
                accd = fmad(ld(x, ixd + j), wv[ua], accd)
        var dot = _spw_fold(accd, nb)
        var y = _sgd_target(k, c, ld(lab, i))
        var dotw = ws_mul(dot, whi, wlo)
        var p = fa(fa(dotw, il), intercept) if one_class else fa(dotw, intercept)
        if lr == LR_OPTIMAL:
            eta = fd(Float32(1), fm(alpha, fs(fa(optimal_init, i2f(t)), Float32(1))))
        elif lr == LR_INVSCALING:
            eta = fd(eta0, identical_pow(i2f(t), power_t))
        var oc = one_class and loss == L_HINGE
        var och = oc_hinge(dotw, intercept, il)
        var cur = och[0] if oc else sgd_loss(loss, y, p, eps)
        if need_obj:
            objective = fa(objective, cur)
            if not pa_rate:
                if penalty != P_NONE:
                    var a2 = Float32(0)
                    var a1 = Float32(0)
                    comptime for ub in range(MB_DBLK):
                        if j0 + ub < d:
                            a2 = fmad(wv[ub], wv[ub], a2)
                            a1 = fa(a1, fabs(wv[ub]))
                    var n2 = _spw_fold(a2, nb)
                    var n1 = _spw_fold(a1, nb)
                    n2 = fm(fm(whi, whi), n2)
                    n1 = fm(whi, n1)
                    var reg = fa(fm(fm(fs(Float32(1), l1_ratio), Float32(0.5)), n2), fm(l1_ratio, n1))
                    objective = fa(objective, fm(alpha, reg))
                if one_class:
                    objective = fa(objective, fm(intercept, alpha))
        var skip = False
        var update = Float32(0)
        if pa_rate:
            var sq = ld(sqp, i)
            if lr == LR_PA1:
                if sq == 0:
                    skip = True
                else:
                    update = fmin(eta0, fd(cur, sq))
            else:
                update = fd(cur, fa(sq, fd(Float32(0.5), eta0)))
            if not skip:
                if loss == L_HINGE:
                    update = fm(update, y)
                elif fs(y, p) < 0:
                    update = -update
        else:
            var dl = och[1] if oc else sgd_dloss(loss, y, p, eps)
            if dl < Float32(-1e12):
                dl = Float32(-1e12)
            elif dl > Float32(1e12):
                dl = Float32(1e12)
            update = fm(-eta, dl)
        if not skip:
            if has_cw or has_sw:
                var cwv = Float32(1)
                if has_cw:
                    cwv = wpos if y > 0 else wneg
                var swi = ld(swp, i) if has_sw else Float32(1)
                update = fm(update, fm(cwv, swi))
            # their w.scale on wscale, the fold into v below 1e-9 (C)
            var fold = False
            var fhi = Float32(1)
            var flo = Float32(0)
            if do_decay:
                var ws = ws_decay(whi, wlo, fm(decay_factor, eta))
                whi = ws[0]
                wlo = ws[1]
                if whi < WS_RESET:
                    fold = True
                    fhi = whi
                    flo = wlo
                    whi = Float32(1)
                    wlo = Float32(0)
            var cu = Float32(0)
            if update != 0:
                cu = ws_div(update, whi, wlo)
            if fi:
                var iu = update
                if one_class:
                    iu = fs(iu, fm(eta, alpha))
                if iu != 0:
                    if one_class:
                        var ia = ff_add(intercept, il, iu)
                        intercept = ia[0]
                        il = ia[1]
                    else:
                        intercept = fa(intercept, iu)
            if do_l1:
                u = fa(u, fm(fm(l1_ratio, eta), alpha))
            # C: the weights, thread j
            comptime for uu in range(MB_DBLK):
                var j = j0 + uu
                if j < d:
                    var wj = wv[uu]
                    if fold:
                        wj = ws_mul(wj, fhi, flo)
                    if update != 0:
                        wj = fmad(cu, ld(x, ixd + j), wj)
                    if do_l1:
                        var rq = ws_clip(wj, u, qv[uu], whi)
                        qv[uu] = rq[1]
                        wj = rq[0]
                    wv[uu] = wj
            t += 1
    comptime for uu in range(MB_DBLK):
        var j = j0 + uu
        if j < d:
            st(w, woff + j, wv[uu])
            if do_l1:
                st(q, woff + j, qv[uu])
    if tid == 0:
        st(ps, SGD_PS_ST * c, intercept)
        st(ps, SGD_PS_ST * c + 1, u)
        st(ps, SGD_PS_ST * c + 2, objective)
        st(ps, SGD_PS_ST * c + 3, il)
        st(ps, SGD_PS_ST * c + 4, whi)
        st(ps, SGD_PS_ST * c + 5, wlo)
        sti(pt, c, t)


def sgd_ps_warp_kernel(
    x: FP, lab: FP, swp: FP, idx: IP, w: FP, q: FP, sqp: FP, part: FP,
    ps: FP, pt: IP, act: IP, pf: FP, ci: IP, cf: FP, start: Int32, cnt: Int32,
    wf: IP, woff: Int32, nonce: Int32,
):
    _sgd_ps_warp_body(x, lab, swp, idx, w, q, sqp, part, ps, pt, act, pf, ci, cf, start, cnt)
    witness_end(wf, woff, nonce)


def sgd_ps_simd_kernel(
    x: FP, lab: FP, swp: FP, idx: IP, w: FP, q: FP, sqp: FP, part: FP,
    ps: FP, pt: IP, act: IP, pf: FP, ci: IP, cf: FP, start: Int32, cnt: Int32,
    wf: IP, woff: Int32, nonce: Int32,
):
    _sgd_ps_simd_body(x, lab, swp, idx, w, q, sqp, part, ps, pt, act, pf, ci, cf, start, cnt)
    witness_end(wf, woff, nonce)


def sgd_ps_kernel(
    x: FP, lab: FP, swp: FP, idx: IP, w: FP, q: FP, sqp: FP, part: FP,
    ps: FP, pt: IP, act: IP, pf: FP, ci: IP, cf: FP, start: Int32, cnt: Int32,
    wf: IP, woff: Int32, nonce: Int32,
):
    _sgd_ps_kernel_body(x, lab, swp, idx, w, q, sqp, part, ps, pt, act, pf, ci, cf, start, cnt)
    witness_end(wf, woff, nonce)


def _sgd_ps_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                 n_out: Int, res: FP) raises:
    """The per-sample fit (batch_size 0) on the device: `sgd_fit` with every
    problem's `sgd_one` as `sgd_ps_kernel` launches; the same words."""
    var ctx = linear_ctx()
    var k = Int(ip[0])
    var loss = Int(ip[1])
    var penalty = Int(ip[2])
    var lr = Int(ip[3])
    var fi = Int(ip[4]) != 0
    var max_iter = Int(ip[5])
    var nic = Int(ip[6])
    var do_shuffle = Int(ip[7]) != 0
    var seed = (UInt64(UInt32(ip[9])) << 32) | UInt64(UInt32(ip[8]))
    var has_sw = Int(ip[10]) != 0
    var has_cw = Int(ip[11]) != 0
    var alpha = fp[0]
    var l1_ratio = fp[1]
    var eta0 = fp[2]
    var power_t = fp[3]
    var eps = fp[4]
    var tol = fp[5]
    if penalty == P_L2:
        l1_ratio = Float32(0)
    elif penalty == P_L1:
        l1_ratio = Float32(1)
    var problems = k if k > 2 else 1
    var one_class = k == 1
    var pa_rate = lr == LR_PA1 or lr == LR_PA2
    var need_obj = tol > Float32(-3.0e38)
    var nb = (d + MB_DBLK - 1) // MB_DBLK
    var optimal_init = mb_optimal_init(loss, alpha, eps) if lr == LR_OPTIMAL else Float32(0)
    var decay_factor = fm(fs(Float32(1), l1_ratio), alpha)
    var chunk = SGD_PS_CHUNK
    var sps = False
    comptime if SGD_FAST_PS_SIMD:
        sps = d >= 1 and d <= SPS_MAX_D
        comptime if SPS_LEGACY_NARROW:
            sps = sps and d <= SPS_LEGACY_MAX_D
    var spw = False
    comptime if SGD_IDN_PS_WARP:
        spw = d >= 1 and nb <= SPS_W
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dlab = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dsw = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var didx = ctx.enqueue_create_buffer[DType.int32](max(problems * n, 1))
    var dw = ctx.enqueue_create_buffer[DType.float32](max(problems * d, 1))
    var dq = ctx.enqueue_create_buffer[DType.float32](max(problems * d, 1))
    var dsq = ctx.enqueue_create_buffer[DType.float32](max(n, 1) if pa_rate else 1)
    var dpart = ctx.enqueue_create_buffer[DType.float32](max(problems * 3 * nb, 1))
    var dps = ctx.enqueue_create_buffer[DType.float32](SGD_PS_ST * problems)
    var dpt = ctx.enqueue_create_buffer[DType.int32](problems)
    var dact = ctx.enqueue_create_buffer[DType.int32](problems)
    var dpf = ctx.enqueue_create_buffer[DType.float32](3 * problems)
    var dci = ctx.enqueue_create_buffer[DType.int32](12)
    var dcf = ctx.enqueue_create_buffer[DType.float32](7)
    var dws = ctx.enqueue_create_buffer[DType.float32](max(problems * d, 1))
    var dqs = ctx.enqueue_create_buffer[DType.float32](max(problems * d, 1))
    var dpss = ctx.enqueue_create_buffer[DType.float32](SGD_PS_ST * problems)
    var dpts = ctx.enqueue_create_buffer[DType.int32](problems)
    var launches = (n + chunk - 1) // chunk
    var wit = Witness(ctx, max(launches * problems + (_xg_blocks(n) if pa_rate else 0), 1))
    var hci = List[Int32](length=12, fill=Int32(0))
    hci[0] = Int32(n)
    hci[1] = Int32(d)
    hci[2] = Int32(loss)
    hci[3] = Int32(penalty)
    hci[4] = Int32(lr)
    hci[5] = Int32(1 if fi else 0)
    hci[6] = Int32(1 if need_obj else 0)
    hci[7] = Int32(1 if one_class else 0)
    hci[8] = Int32(1 if has_sw else 0)
    hci[9] = Int32(1 if has_cw else 0)
    hci[10] = Int32(k)
    hci[11] = Int32(1 if pa_rate else 0)
    var hcf = List[Float32](length=7, fill=Float32(0))
    hcf[0] = alpha
    hcf[1] = l1_ratio
    hcf[2] = eta0
    hcf[3] = power_t
    hcf[4] = eps
    hcf[5] = optimal_init
    hcf[6] = decay_factor
    var seed_lo = Int32(Int(UInt32(seed & UInt64(0xFFFFFFFF))))
    var seed_hi = Int32(Int(UInt32(seed >> 32)))
    var hps = List[Float32](length=SGD_PS_ST * problems, fill=Float32(0))
    var hpt = List[Int32](length=problems, fill=Int32(1))
    var hact = List[Int32](length=problems, fill=Int32(1))
    var hpf = List[Float32](length=3 * problems, fill=Float32(0))
    # the epoch end's state on the device (x_linear/sgd_end.mojo), parity 0
    # at the start: best, no-improve, eta, active, epochs, failed
    var hst = List[Float32](length=2 * problems * SGD_END_ST, fill=Float32(0))
    var dstt = ctx.enqueue_create_buffer[DType.float32](2 * problems * SGD_END_ST)
    var dlive = ctx.enqueue_create_buffer[DType.int32](1)
    var hlive = List[Int32](length=1, fill=Int32(0))
    var dres = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    dres.enqueue_fill(Float32(0))
    for c in range(problems):  # small-loop(problems: OvR problems): per-problem launch constants and start state, a few words a class
        hps[SGD_PS_ST * c] = Float32(1) if one_class else Float32(0)
        hps[SGD_PS_ST * c + 4] = Float32(1)
        hpf[3 * c] = eta0
        hpf[3 * c + 1] = fp[6 + c] if has_cw else Float32(1)
        hpf[3 * c + 2] = fp[6 + problems + c] if has_cw else Float32(1)
        hst[c * SGD_END_ST] = Float32(3.0e38)
        hst[c * SGD_END_ST + 2] = eta0
        hst[c * SGD_END_ST + 3] = bitcast[DType.float32](Int32(1))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    # SGDOC_FAST_TAIL: the first epoch e0 and its first position r0 of the
    # last SGDOC_TAIL_K steps, t of that step (0, 0 and t = 1: the full run)
    var e0 = 0
    var r0 = 0
    comptime if SGDOC_FAST_TAIL:
        if (one_class and lr == LR_OPTIMAL and penalty == P_L2 and fi and do_shuffle and not need_obj
                and not has_sw and max_iter * n > sgdoc_tail_k(d)):
            if sgdoc_centered(ctx, FP(unsafe_from_address=Int(dx.unsafe_ptr())), n, d):
                var s0 = max_iter * n - sgdoc_tail_k(d)
                e0 = s0 // n
                r0 = s0 - e0 * n
                hpt[0] = Int32(s0 + 1)
    ctx.enqueue_copy(dst_buf=dlab, src_ptr=y)
    if has_sw:
        ctx.enqueue_copy(dst_buf=dsw, src_ptr=y + n)
    comptime if SGD_IDN_DEV_FINITE:
        _sgd_finite_device(ctx, FP(unsafe_from_address=Int(dx.unsafe_ptr())), n_x, y, n_y)
    ctx.enqueue_copy(dst_buf=dci, src_ptr=hci.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dcf, src_ptr=hcf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dps, src_ptr=hps.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dpt, src_ptr=hpt.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dact, src_ptr=hact.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dpf, src_ptr=hpf.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dstt, src_ptr=hst.unsafe_ptr())
    dw.enqueue_fill(Float32(0))
    dq.enqueue_fill(Float32(0))
    ctx.enqueue_function[sgd_iota_kernel](didx.unsafe_ptr(), Int32(n), Int32(problems),
                                          grid_dim=_xg_blocks(problems * n), block_dim=XG_TPB)
    var fin_par = 0
    for epoch in range(e0, max_iter):
        # which problems still run is the device's (dact); every problem's
        # order is written (a stopped one's is never read). The stop state's
        # parity counts from e0 (parity 0 holds the initial words).
        var par = (epoch - e0) % 2
        # the epoch as ONE guarded unit: it updates w, q, the scalar state
        # and t in place, so a cut epoch restores them and replays
        ctx.enqueue_copy(dst_buf=dws, src_buf=dw)
        ctx.enqueue_copy(dst_buf=dqs, src_buf=dq)
        ctx.enqueue_copy(dst_buf=dpss, src_buf=dps)
        ctx.enqueue_copy(dst_buf=dpts, src_buf=dpt)
        var tries = 0
        while True:
            var nonce = wit.begin()
            var wo = 0
            if do_shuffle:
                ctx.enqueue_function[sgd_perm_kernel](
                    didx.unsafe_ptr(), Int32(n), seed_lo, seed_hi, Int32(epoch), Int32(0), Int32(problems),
                    grid_dim=_xg_blocks(problems * n), block_dim=XG_TPB,
                )
            if pa_rate:
                ctx.enqueue_function[sgd_rowsq_kernel](
                    dx.unsafe_ptr(), Int32(n), Int32(d), dsq.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                    grid_dim=_xg_blocks(n), block_dim=XG_TPB,
                )
                wo += _xg_blocks(n)
            var start = r0 if epoch == e0 else 0
            while start < n:
                var cnt = min(chunk, n - start)
                if spw:
                    ctx.enqueue_function[sgd_ps_warp_kernel](
                        dx.unsafe_ptr(), dlab.unsafe_ptr(), dsw.unsafe_ptr(), didx.unsafe_ptr(), dw.unsafe_ptr(),
                        dq.unsafe_ptr(), dsq.unsafe_ptr(), dpart.unsafe_ptr(), dps.unsafe_ptr(), dpt.unsafe_ptr(),
                        dact.unsafe_ptr(), dpf.unsafe_ptr(), dci.unsafe_ptr(), dcf.unsafe_ptr(), Int32(start), Int32(cnt),
                        wit.p(), Int32(wo), nonce, grid_dim=problems, block_dim=SPS_W,
                    )
                elif sps:
                    ctx.enqueue_function[sgd_ps_simd_kernel](
                        dx.unsafe_ptr(), dlab.unsafe_ptr(), dsw.unsafe_ptr(), didx.unsafe_ptr(), dw.unsafe_ptr(),
                        dq.unsafe_ptr(), dsq.unsafe_ptr(), dpart.unsafe_ptr(), dps.unsafe_ptr(), dpt.unsafe_ptr(),
                        dact.unsafe_ptr(), dpf.unsafe_ptr(), dci.unsafe_ptr(), dcf.unsafe_ptr(), Int32(start), Int32(cnt),
                        wit.p(), Int32(wo), nonce, grid_dim=problems, block_dim=SPS_W,
                    )
                else:
                    ctx.enqueue_function[sgd_ps_kernel](
                        dx.unsafe_ptr(), dlab.unsafe_ptr(), dsw.unsafe_ptr(), didx.unsafe_ptr(), dw.unsafe_ptr(),
                        dq.unsafe_ptr(), dsq.unsafe_ptr(), dpart.unsafe_ptr(), dps.unsafe_ptr(), dpt.unsafe_ptr(),
                        dact.unsafe_ptr(), dpf.unsafe_ptr(), dci.unsafe_ptr(), dcf.unsafe_ptr(), Int32(start), Int32(cnt),
                        wit.p(), Int32(wo), nonce, grid_dim=problems, block_dim=XG_TPB,
                    )
                wo += problems
                start += cnt
            if wit.ok(ctx, wo, "SGD per-sample epoch"):
                break
            tries += 1
            if tries >= WITNESS_TRIES:
                wit.fail()
            ctx.enqueue_copy(dst_buf=dw, src_buf=dws)
            ctx.enqueue_copy(dst_buf=dq, src_buf=dqs)
            ctx.enqueue_copy(dst_buf=dps, src_buf=dpss)
            ctx.enqueue_copy(dst_buf=dpt, src_buf=dpts)
        # the epoch end (a block a problem, its own guarded unit; reads
        # parity par, writes 1 - par): the live count home
        tries = 0
        while True:
            var nonce = wit.begin()
            dlive.enqueue_fill(Int32(0))
            ctx.enqueue_function[sgd_ps_end_kernel](
                dw.unsafe_ptr(), dps.unsafe_ptr(), dact.unsafe_ptr(), dpf.unsafe_ptr(), dstt.unsafe_ptr(),
                dlive.unsafe_ptr(), Int32(d), Int32(n), Int32(problems), tol, Int32(nic), Int32(lr),
                Int32(1 if need_obj else 0), Int32(par), Int32(epoch), Int32(SGD_PS_ST), wit.p(), Int32(0), nonce,
                grid_dim=problems, block_dim=SGD_END_TPB,
            )
            ctx.enqueue_copy(dst_ptr=hlive.unsafe_ptr(), src_buf=dlive)
            if wit.ok(ctx, problems, "SGD per-sample epoch end"):
                break
            tries += 1
            if tries >= WITNESS_TRIES:
                wit.fail()
        ctx.synchronize()
        fin_par = 1 - par
        if hlive[0] == 0:
            break
    ctx.enqueue_function[sgd_ps_res_kernel](
        dw.unsafe_ptr(), dps.unsafe_ptr(), dstt.unsafe_ptr(), dres.unsafe_ptr(), Int32(d), Int32(problems),
        Int32(1 if one_class else 0), Int32(fin_par), Int32(SGD_PS_ST),
        grid_dim=_xg_blocks(problems * d + problems + 1), block_dim=XG_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res, src_buf=dres)
    ctx.synchronize()

    _ = hci^
    _ = hcf^
    _ = hst^
    _ = hlive^
    _ = dstt^
    _ = dlive^
    _ = dres^
    _ = hps^
    _ = hpt^
    _ = hact^
    _ = hpf^
    _ = dx^
    _ = dlab^
    _ = dsw^
    _ = didx^
    _ = dw^
    _ = dq^
    _ = dsq^
    _ = dpart^
    _ = dps^
    _ = dpt^
    _ = dact^
    _ = dpf^
    _ = dci^
    _ = dcf^
    _ = dws^
    _ = dqs^
    _ = dpss^
    _ = dpts^
    _ = wit^


# ------------------------------------------------ GLM on the grid (lane/neural-pass89)
# The GLM Newton loop driven from the host, each pass a kernel over the whole
# grid: the rows' derivatives and loss terms (one thread a row) and the
# gradient / Hessian cells (one thread a cell chain over the rows ascending,
# in glm_fit's warp-uniform slots). The folds that define the objective run on
# one block (`t_fold_fa_staged`), the m x m step on one thread
# (`_glm_step`), and the control (the line search, the stall count) on the
# host with the same fa / fm words. Every value is glm_fit's: the same
# helpers, the same order. glm_fit ran all of it on ONE block: istella's
# 24,531 cells were ~96 million-row chains a thread (the L40S board timed
# out). The one-block fit is no longer reachable (cpu-gpu-cleanup c-linear).


# lane/gap-serial-gpu (2026-10-02): the prologue on the grid. One thread a
# FOLD_BLOCK row block: its weight-sum and y-sum partials from zero (glm_den /
# glm_start's blocks); then one thread folds the partials ascending, the same
# words as glm_fit's blocked folds. The one-thread prologue walked all n rows
# twice (removed, cpu-gpu-cleanup c-linear).
def glm_ydom_kernel(y: FP, n: Int32, flag: IP, wf: IP, woff: Int32, nonce: Int32):
    """The targets' range facts (x_linear/glm_ydom.mojo), one thread a row,
    by bits: flag[0] some y < 0, flag[1] some y > 0, flag[2] some y == 0.
    Every writer stores the same 1; idempotent."""
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        var b = bitcast[DType.uint32](ld(y, i))
        if (b & UInt32(0x7FFFFFFF)) == UInt32(0):
            sti(flag, 2, 1)
        elif (b >> 31) != UInt32(0):
            sti(flag, 0, 1)
        else:
            sti(flag, 1, 1)
    witness_end(wf, woff, nonce)


def glm_init_parts_kernel(y: FP, n: Int32, nb: Int32, sw: Int32, parts: FP, wf: IP, woff: Int32, nonce: Int32):
    """parts[b] = the block's weight sum (sw), parts[nb + b] = its y sum (sum w y with sw)."""
    var b = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var nbb = Int(nb)
    if b < nbb:
        var lo = b * FOLD_BLOCK
        var cnt = min(FOLD_BLOCK, nn - lo)
        if sw != 0:
            st(parts, b, fold_fa(y, nn + lo, 1, cnt))
            st(parts, nbb + b, chain_fmad(y, nn + lo, 1, y, lo, 1, cnt))
        else:
            st(parts, nbb + b, fold_fa(y, lo, 1, cnt))
    witness_end(wf, woff, nonce)

def glm_init_finish_kernel(parts: FP, nf: Float32, nb: Int32, d: Int32, fi: Int32, link: Int32, sw: Int32,
                           res: FP, sc: FP, wf: IP, woff: Int32, nonce: Int32):
    """The partials folded ascending: den (sc[0]) and the intercept start."""
    var nbb = Int(nb)
    var dd = Int(d)
    var den = fold_parts(parts, 0, nbb) if sw != 0 else nf
    st(sc, 0, den)
    fill(res, 0, dd + 3, Float32(0))
    if fi != 0:
        st(res, dd, glm_start_of(fold_parts(parts, nbb, nbb), nf, den, Int(link), sw != 0))
    witness_end(wf, woff, nonce)

def glm_obj_map_kernel(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, power: Float32, link: Int32, sw: Int32,
                       theta: FP, eta: FP, lt: FP, gr: FP, hr: FP, wf: IP, woff: Int32, nonce: Int32):
    """`_objective_team`'s row statements, one thread a row."""
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if i < nn:
        var dd = Int(d)
        var b = ld(theta, dd) if fi != 0 else Float32(0)
        var e = fa(row_dot(x, i, dd, theta, 0), b)
        st(eta, i, e)
        var l = Float32(0)
        comptime if C16_GLM_FUSED:
            var values = _unit_all(power, Int(link), ld(y, i), e)
            l = values[0]
            var gi = values[1]
            var hi = fmax(Float32(0), values[2])
            if sw != 0:
                gi = fm(ld(y, nn + i), gi)
                hi = fm(ld(y, nn + i), hi)
            st(gr, i, gi)
            st(hr, i, hi)
        else:
            l = _unit(power, Int(link), ld(y, i), e, 0)
        if sw != 0:
            l = fm(ld(y, nn + i), l)
        st(lt, i, l)
    witness_end(wf, woff, nonce)

# lane/gap-serial-gpu (2026-10-02): the objective fold's FOLD_BLOCK partials
# one thread a block over the grid (t_fold_fa_blocked ran them on one block
# of LINEAR_TPB threads: nb / 256 blocks a thread past 1M rows), then one
# thread folds them ascending: t_fold_fa_blocked's words.
# (the one-block fold was removed, cpu-gpu-cleanup c-linear).
def glm_obj_parts_kernel(lt: FP, n: Int32, nb: Int32, parts: FP, wf: IP, woff: Int32, nonce: Int32):
    var b = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if b < Int(nb):
        var lo = b * FOLD_BLOCK
        st(parts, b, fold_fa(lt, lo, 1, min(FOLD_BLOCK, Int(n) - lo)))
    witness_end(wf, woff, nonce)

def glm_obj_finish_kernel(parts: FP, nb: Int32, d: Int32, theta: FP, alpha: Float32, sc: FP, slot: Int32,
                          wf: IP, woff: Int32, nonce: Int32):
    """`glm_obj_fold_kernel`'s value from the partials: sc[slot] = f."""
    var acc = fold_parts(parts, 0, Int(nb))
    var reg = Float32(0)
    for j in range(Int(d)):
        var w = ld(theta, j)
        reg = fmad(w, w, reg)
    st(sc, Int(slot), fa(fd(acc, ld(sc, 0)), fm(fm(Float32(0.5), alpha), reg)))
    witness_end(wf, woff, nonce)

def glm_deriv_kernel(y: FP, n: Int32, power: Float32, link: Int32, sw: Int32, eta: FP, gr: FP, hr: FP, wf: IP, woff: Int32, nonce: Int32):
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        _glm_deriv_row(y, Int(n), i, power, Int(link), eta, sw != 0, gr, hr)
    witness_end(wf, woff, nonce)

#: Apple: macOS silently aborts a command buffer that holds the GPU for
#: seconds and leaves its output partly stale. The GLM cells run in row
#: slices of at most GLM_APPLE_SLICE_MACS chain steps a launch, each waited
#: on (the same words).
comptime GLM_APPLE_SLICE_MACS = 1 << 29

def _glm_rows_slice(n: Int, slots: Int) -> Int:
    comptime if has_apple_gpu_accelerator():
        return max(64, min(n, (GLM_APPLE_SLICE_MACS // max(slots, 1)) // 64 * 64))
    return n


def _glm_blocks_slice(nb: Int, slots: Int) -> Int:
    """The row blocks a parts launch takes: all of them off Apple."""
    comptime if has_apple_gpu_accelerator():
        return max(1, min(nb, GLM_APPLE_SLICE_MACS // (max(slots, 1) * FOLD_BLOCK)))
    return nb


def glm_cell_parts_kernel(x: FP, gr: FP, hr: FP, n: Int32, d: Int32, m: Int32, nb: Int32, b0: Int32, bcnt: Int32,
                          parts: FP, wf: IP, woff: Int32, nonce: Int32):
    """lane/neural-pass97: one thread a (slot, row block): the slot's cell
    chain over the block from zero into parts[slot * nb + block]; the row
    blocks [b0, b0 + bcnt) only (the Apple slices, `_glm_blocks_slice`)."""
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var mm = Int(m)
    var nbb = Int(nb)
    # block-major: neighbouring threads are neighbouring slots over the same
    # rows, so their x words share cache lines (and a warp keeps one kind)
    var slots = _glm_slot_count(dd, mm)
    var bq = q // slots
    var sl = q - bq * slots
    var b = Int(b0) + bq
    if bq < Int(bcnt) and b < nbb:
        var c = _glm_slot_cell(sl, dd, mm)
        if c >= 0:
            var lo = b * FOLD_BLOCK
            st(parts, sl * nbb + b, _glm_cell_part(c, x, gr, hr, dd, mm, lo, min(FOLD_BLOCK, Int(n) - lo)))
    witness_end(wf, woff, nonce)

def glm_cell_combine_kernel(parts: FP, d: Int32, m: Int32, nb: Int32, g: FP, h: FP, wf: IP, woff: Int32, nonce: Int32):
    """One thread a slot: its partials folded blocks ascending."""
    var sl = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var mm = Int(m)
    if sl < _glm_slot_count(dd, mm):
        var c = _glm_slot_cell(sl, dd, mm)
        if c >= 0:
            _glm_cell_store(c, fold_parts(parts, sl * Int(nb), Int(nb)), g, h, mm)
    witness_end(wf, woff, nonce)

# cpu-gpu-cleanup c-linear (2026-10-02): `_glm_step` (x_linear/glm.mojo) as
# parallel launches, the host column's statements: the gradient cells and
# the order-free gmax (an integer max over the non-negative float bits),
# the Hessian cells, the Cholesky one launch per column (out of place, every
# row recomputing the pivot's chain), the column-form substitutions one
# launch per column, the slope in GLM_SLOPE_BLK partials. sc: [den, f, flag,
# slope, chol ok]. Replaces the one-thread `glm_step_kernel`.
@always_inline
def _gt() -> Int:
    return Int(block_idx.x) * XG_TPB + Int(thread_idx.x)


def glm_sg_kernel(g: FP, step: FP, res: FP, m: Int32, d: Int32, alpha: Float32, sc: FP, gm: IP):
    var j = _gt()
    if j < Int(m):
        var term = glm_g_item(j, g, step, res, Int(d), alpha, fd(Float32(1), ld(sc, 0)))
        _ = Atomic[DType.int32].max(gm, Int32(Int(bitcast[DType.uint32](term))))


def glm_flag_kernel(gm: IP, sc: FP, tol: Float32):
    if _gt() == 0:
        var gmax = bitcast[DType.float32](UInt32(Int(gm.unsafe_load(0))))
        st(sc, 2, Float32(1) if gmax <= tol else Float32(0))
        st(sc, 3, Float32(0))
        st(sc, 4, Float32(1))


def glm_sh_kernel(h: FP, m: Int32, d: Int32, alpha: Float32, sc: FP):
    var t = _gt()
    if t < Int(m) * Int(m):
        glm_h_item(t, h, Int(m), Int(d), alpha, fd(Float32(1), ld(sc, 0)))


def glm_chol_col_kernel(h: FP, l: FP, m: Int32, j: Int32, sc: FP):
    """`cholesky`'s column j (x_linear/ops.mojo), out of place into l, row
    i = j + t; row j clears sc[4] on a pivot that is not positive."""
    var mm = Int(m)
    var jj = Int(j)
    var i = jj + _gt()
    if i >= mm:
        return
    var s = ld(h, jj * mm + jj)
    for k in range(jj):
        var lv = ld(l, jj * mm + k)
        s = fs(s, fm(lv, lv))
    if i == jj and not (s > 0):
        st(sc, 4, Float32(0))
    var r = fsqrt(s)
    if i == jj:
        st(l, jj * mm + jj, r)
        return
    var t = ld(h, i * mm + jj)
    for k in range(jj):
        t = fs(t, fm(ld(l, i * mm + k), ld(l, jj * mm + k)))
    st(l, i * mm + jj, fd(t, r))


def glm_fwd_kernel(l: FP, m: Int32, j: Int32, b: FP, y: FP, sc: FP):
    var i = Int(j) + _gt()
    if i < Int(m) and ld(sc, 4) != Float32(0):
        glm_fwd_col(l, Int(m), Int(j), i, b, y)


def glm_back_kernel(l: FP, m: Int32, j: Int32, c: FP, x: FP, sc: FP):
    var i = _gt()
    if i <= Int(j) and ld(sc, 4) != Float32(0):
        glm_back_col(l, Int(m), Int(j), i, c, x)


def glm_slope_parts_kernel(g: FP, step: FP, m: Int32, parts: FP):
    var b = _gt()
    if b < glm_slope_blocks(Int(m)):
        st(parts, b, glm_slope_part(g, step, Int(m), b))


def glm_slope_fin_kernel(parts: FP, nb: Int32, sc: FP):
    if _gt() == 0 and ld(sc, 2) == Float32(0):
        var acc = Float32(0)
        for b in range(Int(nb)):
            acc = fa(acc, ld(parts, b))
        st(sc, 3, acc)
        if not (acc < 0):
            st(sc, 2, Float32(2))


def _glm_step_device(
    ctx: DeviceContext, g: FP, h: FP, step: FP, res: FP, l: FP, yv: FP, parts: FP, gm: IP, sc: FP,
    m: Int, d: Int, alpha: Float32, tol: Float32,
) raises:
    """`_glm_step` on the device (enqueued): sc[2] flag, sc[3] slope."""
    ctx.enqueue_function[glm_fill_i_kernel](gm, grid_dim=1, block_dim=1)
    ctx.enqueue_function[glm_sg_kernel](g, step, res, Int32(m), Int32(d), alpha, sc, gm,
                                        grid_dim=_xg_blocks(m), block_dim=XG_TPB)
    ctx.enqueue_function[glm_flag_kernel](gm, sc, tol, grid_dim=1, block_dim=1)
    ctx.enqueue_function[glm_sh_kernel](h, Int32(m), Int32(d), alpha, sc, grid_dim=_xg_blocks(m * m), block_dim=XG_TPB)
    for j in range(m):
        ctx.enqueue_function[glm_chol_col_kernel](h, l, Int32(m), Int32(j), sc, grid_dim=_xg_blocks(m - j), block_dim=XG_TPB)
    for j in range(m):
        ctx.enqueue_function[glm_fwd_kernel](l, Int32(m), Int32(j), step, yv, sc, grid_dim=_xg_blocks(m - j), block_dim=XG_TPB)
    var j = m - 1
    while j >= 0:
        ctx.enqueue_function[glm_back_kernel](l, Int32(m), Int32(j), yv, step, sc, grid_dim=_xg_blocks(j + 1), block_dim=XG_TPB)
        j -= 1
    var nbs = glm_slope_blocks(m)
    ctx.enqueue_function[glm_slope_parts_kernel](g, step, Int32(m), parts, grid_dim=_xg_blocks(nbs), block_dim=XG_TPB)
    ctx.enqueue_function[glm_slope_fin_kernel](parts, Int32(nbs), sc, grid_dim=1, block_dim=1)


def glm_fill_i_kernel(gm: IP):
    if _gt() == 0:
        gm.unsafe_store(0, Int32(0))


def glm_trial_kernel(step: FP, res: FP, trial: FP, m: Int32, tt: Float32):
    """One thread a coefficient (was one thread over all m)."""
    var j = _gt()
    if j < Int(m):
        st(trial, j, fmad(tt, ld(step, j), ld(res, j)))


def glm_accept_kernel(res: FP, trial: FP, m: Int32):
    var j = _gt()
    if j < Int(m):
        st(res, j, ld(trial, j))


def _glm_grid_objective(
    mut ctx: DeviceContext, theta: FP, x: FP, y: FP, eta: FP, lt: FP, tw: FP, fparts: FP, sc: FP,
    sc_buf: DeviceBuffer[DType.float32], hsc: FP,
    n: Int, d: Int, fi: Int, power: Float32, link: Int, sw: Int, alpha: Float32, rows_grid: Int,
    mut wit: Witness, gr: FP, hr: FP,
) raises -> Float32:
    """`_objective_team` at theta on the grid (map) and one block (fold); eta
    holds its linear predictor afterwards."""
    var tries = 0
    while True:
        var nonce = wit.begin()
        ctx.enqueue_function[glm_obj_map_kernel](x, y, Int32(n), Int32(d), Int32(fi), power, Int32(link), Int32(sw),
                                                 theta, eta, lt, gr, hr, wit.p(), Int32(0), nonce,
                                                 grid_dim=rows_grid, block_dim=XG_TPB)
        var nb = fold_blocks(n)
        ctx.enqueue_function[glm_obj_parts_kernel](lt, Int32(n), Int32(nb), fparts, wit.p(), Int32(rows_grid), nonce,
                                                   grid_dim=_xg_blocks(nb), block_dim=XG_TPB)
        ctx.enqueue_function[glm_obj_finish_kernel](fparts, Int32(nb), Int32(d), theta, alpha, sc, Int32(1),
                                                    wit.p(), Int32(rows_grid + _xg_blocks(nb)), nonce,
                                                    grid_dim=1, block_dim=1)
        var count = rows_grid + _xg_blocks(nb) + 1
        if wit.ok(ctx, count, "GLM objective"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    var v = Float32(0)
    ctx.enqueue_copy(dst_ptr=hsc, src_buf=sc_buf)
    ctx.synchronize()
    v = hsc.unsafe_load(1)
    return v


def _glm_fit_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                  n_out: Int, res: FP) raises:
    var ctx = linear_ctx()
    var max_iter = Int(ip[0])
    var fi = Int(ip[1])
    var link = Int(ip[2])
    var sw = Int(ip[3])
    var power = fp[0]
    var alpha = fp[1]
    var tol = fp[2]
    var m = d + 1 if fi != 0 else d
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dres = ctx.enqueue_create_buffer[DType.float32](max(n_out, d + 3))
    var deta = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dlt = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dgr = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dhr = ctx.enqueue_create_buffer[DType.float32](max(n, 1))
    var dg = ctx.enqueue_create_buffer[DType.float32](m)
    var dh = ctx.enqueue_create_buffer[DType.float32](m * m)
    var dstep = ctx.enqueue_create_buffer[DType.float32](m)
    var dtrial = ctx.enqueue_create_buffer[DType.float32](m)
    var dsc = ctx.enqueue_create_buffer[DType.float32](8)
    var dl_ = ctx.enqueue_create_buffer[DType.float32](m * m)
    var dyv = ctx.enqueue_create_buffer[DType.float32](m)
    var dslp = ctx.enqueue_create_buffer[DType.float32](glm_slope_blocks(m))
    var dgm = ctx.enqueue_create_buffer[DType.int32](1)
    var dtw = ctx.enqueue_create_buffer[DType.float32](team_work(n, 3, 0))
    var dfparts = ctx.enqueue_create_buffer[DType.float32](max(2 * fold_blocks(n), 1))
    var hsc_l = List[Float32](length=8, fill=Float32(0))
    var hscp = FP(unsafe_from_address=Int(hsc_l.unsafe_ptr()))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if XLIN_IDN_DEV_FINITE:
        xlin_finite_device(ctx, dx, n_x, y, n_y)
    dsc.enqueue_fill(Float32(0))
    dh.enqueue_fill(Float32(0))
    dtw.enqueue_fill(Float32(0))
    var rows_grid = _xg_blocks(n)
    var slot_ub = m + m * (m + 1) // 2 + 4 * 64
    var slot_grid = _xg_blocks(slot_ub)
    var nb = fold_blocks(n)
    var dparts = ctx.enqueue_create_buffer[DType.float32](max(slot_ub * nb, 1))
    var rows_slice = _glm_rows_slice(n, _glm_slot_count(d, m))
    var blocks_slice = _glm_blocks_slice(nb, _glm_slot_count(d, m))
    var wit = Witness(ctx, max(rows_grid + _xg_blocks(nb) + 1, max(_xg_blocks(slot_ub * blocks_slice), slot_grid + 1)))
    var dydom = ctx.enqueue_create_buffer[DType.int32](4)
    dydom.enqueue_fill(Int32(0))
    var tries0 = 0
    while True:
        var nonce = wit.begin()
        var nbi = fold_blocks(n)
        ctx.enqueue_function[glm_init_parts_kernel](dy.unsafe_ptr(), Int32(n), Int32(nbi), Int32(sw), dfparts.unsafe_ptr(),
                                                    wit.p(), Int32(0), nonce, grid_dim=_xg_blocks(nbi), block_dim=XG_TPB)
        ctx.enqueue_function[glm_init_finish_kernel](dfparts.unsafe_ptr(), i2f(n), Int32(nbi), Int32(d), Int32(fi),
                                                     Int32(link), Int32(sw), dres.unsafe_ptr(), dsc.unsafe_ptr(),
                                                     wit.p(), Int32(_xg_blocks(nbi)), nonce, grid_dim=1, block_dim=1)
        var count0 = _xg_blocks(nbi) + 1
        comptime if XLIN_GLM_DEV_YDOM:
            # lane fam2-linear: the targets' range on the device, in the same
            # guarded unit (the Python layer no longer walks y)
            ctx.enqueue_function[glm_ydom_kernel](dy.unsafe_ptr(), Int32(n), dydom.unsafe_ptr(),
                                                  wit.p(), Int32(count0), nonce, grid_dim=rows_grid, block_dim=XG_TPB)
            count0 += rows_grid
        if wit.ok(ctx, count0, "GLM init"):
            break
        tries0 += 1
        if tries0 >= WITNESS_TRIES:
            wit.fail()
    comptime if XLIN_GLM_DEV_YDOM:
        var hyd = List[Int32](length=4, fill=Int32(0))
        ctx.enqueue_copy(dst_ptr=hyd.unsafe_ptr(), src_buf=dydom)
        ctx.synchronize()
        var yd_bad = glm_ydom_bad(power, hyd[0] != Int32(0), hyd[1] != Int32(0), hyd[2] != Int32(0))
        _ = hyd^
        if yd_bad:
            for i in range(d + 2):
                res.unsafe_store(i, Float32(0))
            res.unsafe_store(d + 2, GLM_YDOM_REFUSED)
            return

    var iters = 0
    var converged = False
    var f = _glm_grid_objective(ctx, FP(unsafe_from_address=Int(dres.unsafe_ptr())), FP(unsafe_from_address=Int(dx.unsafe_ptr())), FP(unsafe_from_address=Int(dy.unsafe_ptr())), FP(unsafe_from_address=Int(deta.unsafe_ptr())), FP(unsafe_from_address=Int(dlt.unsafe_ptr())), FP(unsafe_from_address=Int(dtw.unsafe_ptr())), FP(unsafe_from_address=Int(dfparts.unsafe_ptr())), FP(unsafe_from_address=Int(dsc.unsafe_ptr())), dsc, hscp, n, d, fi, power, link, sw, alpha, rows_grid, wit, FP(unsafe_from_address=Int(dgr.unsafe_ptr())), FP(unsafe_from_address=Int(dhr.unsafe_ptr())))
    var stall = 0
    for it in range(max_iter):
        comptime if not C16_GLM_FUSED:
            var tries1 = 0
            while True:
                var nonce = wit.begin()
                ctx.enqueue_function[glm_deriv_kernel](dy.unsafe_ptr(), Int32(n), power, Int32(link), Int32(sw), deta.unsafe_ptr(),
                                                       dgr.unsafe_ptr(), dhr.unsafe_ptr(), wit.p(), Int32(0), nonce,
                                                       grid_dim=rows_grid, block_dim=XG_TPB)
                if wit.ok(ctx, rows_grid, "GLM derivatives"):
                    break
                tries1 += 1
                if tries1 >= WITNESS_TRIES:
                    wit.fail()
        var b0 = 0
        while b0 < nb:
            var bc = min(blocks_slice, nb - b0)
            var tries2 = 0
            while True:
                var nonce = wit.begin()
                ctx.enqueue_function[glm_cell_parts_kernel](dx.unsafe_ptr(), dgr.unsafe_ptr(), dhr.unsafe_ptr(), Int32(n), Int32(d),
                                                            Int32(m), Int32(nb), Int32(b0), Int32(bc), dparts.unsafe_ptr(),
                                                            wit.p(), Int32(0), nonce,
                                                            grid_dim=_xg_blocks(slot_ub * bc), block_dim=XG_TPB)
                if wit.ok(ctx, _xg_blocks(slot_ub * bc), "GLM cell parts"):
                    break
                tries2 += 1
                if tries2 >= WITNESS_TRIES:
                    wit.fail()
            b0 += bc
            if b0 < nb and blocks_slice < nb:
                ctx.synchronize()
        # the combine rebuilds g and h from the parts; the step (which scales
        # them in place) runs once the combine is witnessed
        var tries3 = 0
        while True:
            var nonce = wit.begin()
            ctx.enqueue_function[glm_cell_combine_kernel](dparts.unsafe_ptr(), Int32(d), Int32(m), Int32(nb), dg.unsafe_ptr(),
                                                          dh.unsafe_ptr(), wit.p(), Int32(0), nonce,
                                                          grid_dim=slot_grid, block_dim=XG_TPB)
            if wit.ok(ctx, slot_grid, "GLM combine"):
                break
            tries3 += 1
            if tries3 >= WITNESS_TRIES:
                wit.fail()
        _glm_step_device(ctx, FP(unsafe_from_address=Int(dg.unsafe_ptr())), FP(unsafe_from_address=Int(dh.unsafe_ptr())),
                         FP(unsafe_from_address=Int(dstep.unsafe_ptr())), FP(unsafe_from_address=Int(dres.unsafe_ptr())),
                         FP(unsafe_from_address=Int(dl_.unsafe_ptr())), FP(unsafe_from_address=Int(dyv.unsafe_ptr())),
                         FP(unsafe_from_address=Int(dslp.unsafe_ptr())), IP(unsafe_from_address=Int(dgm.unsafe_ptr())),
                         FP(unsafe_from_address=Int(dsc.unsafe_ptr())), m, d, alpha, tol)
        ctx.enqueue_copy(dst_ptr=hscp, src_buf=dsc)
        ctx.synchronize()
        var flag = Int(hscp.unsafe_load(2))
        var slope = hscp.unsafe_load(3)
        if flag == 1:
            converged = True
            break
        iters = it + 1
        if flag == 2:
            break
        var tt = Float32(1)
        var accepted = False
        for _ in range(40):
            ctx.enqueue_function[glm_trial_kernel](dstep.unsafe_ptr(), dres.unsafe_ptr(), dtrial.unsafe_ptr(), Int32(m), tt,
                                                   grid_dim=_xg_blocks(m), block_dim=XG_TPB)
            var ft = _glm_grid_objective(ctx, FP(unsafe_from_address=Int(dtrial.unsafe_ptr())), FP(unsafe_from_address=Int(dx.unsafe_ptr())), FP(unsafe_from_address=Int(dy.unsafe_ptr())), FP(unsafe_from_address=Int(deta.unsafe_ptr())), FP(unsafe_from_address=Int(dlt.unsafe_ptr())), FP(unsafe_from_address=Int(dtw.unsafe_ptr())), FP(unsafe_from_address=Int(dfparts.unsafe_ptr())), FP(unsafe_from_address=Int(dsc.unsafe_ptr())), dsc, hscp, n, d, fi, power, link, sw, alpha, rows_grid, wit, FP(unsafe_from_address=Int(dgr.unsafe_ptr())), FP(unsafe_from_address=Int(dhr.unsafe_ptr())))
            if ft == ft and ft <= fa(f, fm(fm(Float32(1e-4), tt), slope)):
                ctx.enqueue_function[glm_accept_kernel](dres.unsafe_ptr(), dtrial.unsafe_ptr(), Int32(m),
                                                        grid_dim=_xg_blocks(m), block_dim=XG_TPB)
                if ft == f:
                    stall += 1
                else:
                    stall = 0
                f = ft
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        if not accepted:
            f = _glm_grid_objective(ctx, FP(unsafe_from_address=Int(dres.unsafe_ptr())), FP(unsafe_from_address=Int(dx.unsafe_ptr())), FP(unsafe_from_address=Int(dy.unsafe_ptr())), FP(unsafe_from_address=Int(deta.unsafe_ptr())), FP(unsafe_from_address=Int(dlt.unsafe_ptr())), FP(unsafe_from_address=Int(dtw.unsafe_ptr())), FP(unsafe_from_address=Int(dfparts.unsafe_ptr())), FP(unsafe_from_address=Int(dsc.unsafe_ptr())), dsc, hscp, n, d, fi, power, link, sw, alpha, rows_grid, wit, FP(unsafe_from_address=Int(dgr.unsafe_ptr())), FP(unsafe_from_address=Int(dhr.unsafe_ptr())))
            break
        if stall >= GLM_STALL_ITERS:
            converged = True
            break
    ctx.enqueue_copy(dst_ptr=res, src_buf=dres)
    ctx.synchronize()
    if fi == 0:
        res.unsafe_store(d, Float32(0))
    res.unsafe_store(d + 1, i2f(iters))
    res.unsafe_store(d + 2, Float32(1) if converged else Float32(0))
    _ = dx^
    _ = dy^
    _ = dres^
    _ = deta^
    _ = dlt^
    _ = dgr^
    _ = dhr^
    _ = dg^
    _ = dh^
    _ = dstep^
    _ = dtrial^
    _ = dl_^
    _ = dyv^
    _ = dslp^
    _ = dgm^
    _ = dsc^
    _ = dtw^
    _ = dparts^
    _ = hsc_l^
    _ = wit^



# ------------------------------------------------ k-fold RidgeCV on the grid (lane/neural-pass91)
# x_linear/ridgecv.mojo's chains, one thread each: the training means (d + 1
# threads), the centered Gram cells and X'y (one thread a cell), the solves
# (one thread an alpha, its own scratch), the held-out predictions (one
# thread a row and alpha) and the fold scores (one thread an alpha, summed
# folds ascending). The same helpers as the host fit, so the same words.
@always_inline
def _kf_means_kernel_body(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, fi: Int32, xm: FP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    if j < dd:
        st(xm, j, kf_mean(x, dd, j, Int(n), Int(s), Int(e)) if fi != 0 else Float32(0))
    elif j == dd:
        st(xm, dd, kf_mean(y, 1, 0, Int(n), Int(s), Int(e)) if fi != 0 else Float32(0))


def kf_means_kernel(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, fi: Int32, xm: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_means_kernel_body(x, y, n, d, s, e, fi, xm)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_cells_kernel_body(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, xm: FP, g: FP, xty: FP):
    var c = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    var cells = dd * (dd + 1) // 2
    if c < cells:
        var jk = upper_cell(c, dd)
        var j = jk[0]
        var k = jk[1]
        var v = kf_cross(x, dd, j, ld(xm, j), x, dd, k, ld(xm, k), Int(n), Int(s), Int(e))
        st(g, j * dd + k, v)
        st(g, k * dd + j, v)
    elif c < cells + dd:
        var j = c - cells
        st(xty, j, kf_cross(x, dd, j, ld(xm, j), y, 1, 0, ld(xm, dd), Int(n), Int(s), Int(e)))


def kf_cells_kernel(x: FP, y: FP, n: Int32, d: Int32, s: Int32, e: Int32, xm: FP, g: FP, xty: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_cells_kernel_body(x, y, n, d, s, e, xm, g, xty)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_solve_kernel_body(g: FP, xty: FP, xm: FP, d: Int32, alphas: FP, na: Int32, fi: Int32, aw: FP, w: FP, b: FP,
                    trust: FP):
    var a = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var dd = Int(d)
    if a < Int(na):
        var r = kf_solve(g, xty, xm, ld(xm, dd), dd, ld(alphas, a), fi != 0, aw + a * dd * dd, w + a * dd)
        st(b, a, r[0])
        st(trust, a, Float32(1) if r[1] else Float32(0))


def kf_solve_kernel(g: FP, xty: FP, xm: FP, d: Int32, alphas: FP, na: Int32, fi: Int32, aw: FP, w: FP, b: FP,
                    trust: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_solve_kernel_body(g, xty, xm, d, alphas, na, fi, aw, w, b, trust)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_ff_unit_kernel_body(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, s: Int32, e: Int32, u0: Int32, count: Int32,
                      sh: FP, sl: FP):
    var u = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if u < Int(count):
        ridge_ff_unit(Int(u0) + u, x, y, Int(n), Int(d), 1, fi != 0, False, Int(n), sh, sl, Int(s), Int(e))


def kf_ff_unit_kernel(x: FP, y: FP, n: Int32, d: Int32, fi: Int32, s: Int32, e: Int32, u0: Int32, count: Int32,
                      sh: FP, sl: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_ff_unit_kernel_body(x, y, n, d, fi, s, e, u0, count, sh, sl)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_ff_solve_kernel_body(d: Int32, fi: Int32, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, fh: FP, fl: FP, tmp: FP,
                       w: FP, b: FP, a: Int32):
    var dd = Int(d)
    st(b, Int(a), kf_ff_solve(dd, fi != 0, alpha, sh, sl, bh, bl, fh, fl, tmp, w + Int(a) * dd))


def kf_ff_solve_kernel(d: Int32, fi: Int32, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, fh: FP, fl: FP, tmp: FP,
                       w: FP, b: FP, a: Int32, wf: IP, woff: Int32, nonce: Int32):
    _kf_ff_solve_kernel_body(d, fi, alpha, sh, sl, bh, bl, fh, fl, tmp, w, b, a)
    witness_end(wf, woff, nonce)

@always_inline
def _kf_pred_kernel_body(x: FP, d: Int32, s: Int32, e: Int32, na: Int32, w: FP, b: FP, p: FP, stride: Int32):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nt = Int(e) - Int(s)
    if q < nt * Int(na):
        var a = q // nt
        var r = q - a * nt
        st(p, a * Int(stride) + r, kf_pred(x, Int(s) + r, Int(d), w + a * Int(d), ld(b, a)))


def kf_pred_kernel(x: FP, d: Int32, s: Int32, e: Int32, na: Int32, w: FP, b: FP, p: FP, stride: Int32, wf: IP, woff: Int32, nonce: Int32):
    _kf_pred_kernel_body(x, d, s, e, na, w, b, p, stride)
    witness_end(wf, woff, nonce)

# the held-out r2 on the grid in x_linear/ridgecv.mojo `kf_score`'s blocked
# order (cgr-linear): the target's block sums, then a thread per (alpha,
# block) for the two squared sums, then a thread per alpha
def kf_ysum_kernel(y: FP, s: Int32, e: Int32, yp: FP, wf: IP, woff: Int32, nonce: Int32):
    var b = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if b < kf_blocks(Int(s), Int(e)):
        st(yp, b, kf_ysum_part(y, Int(s), Int(e), b))
    witness_end(wf, woff, nonce)


def kf_sq_kernel(y: FP, p: FP, s: Int32, e: Int32, na: Int32, stride: Int32, yp: FP, sp: FP,
                 wf: IP, woff: Int32, nonce: Int32):
    var t = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var si = Int(s)
    var ei = Int(e)
    var nb = kf_blocks(si, ei)
    if t < Int(na) * nb:
        var a = t // nb
        var b = t % nb
        var mt = fd(fold_parts(yp, 0, nb), i2f(ei - si))
        var q = kf_sq_part(y, p + a * Int(stride), si, ei, mt, b)
        st(sp, (2 * a) * nb + b, q[0])
        st(sp, (2 * a + 1) * nb + b, q[1])
    witness_end(wf, woff, nonce)


@always_inline
def _kf_score_kernel_body(s: Int32, e: Int32, na: Int32, sp: FP, sums: FP):
    var a = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if a < Int(na):
        var nb = kf_blocks(Int(s), Int(e))
        var sc = kf_score_final(fold_parts(sp, (2 * a) * nb, nb), fold_parts(sp, (2 * a + 1) * nb, nb))
        st(sums, a, fa(ld(sums, a), sc))


def kf_score_kernel(s: Int32, e: Int32, na: Int32, sp: FP, sums: FP, wf: IP, woff: Int32, nonce: Int32):
    _kf_score_kernel_body(s, e, na, sp, sums)
    witness_end(wf, woff, nonce)

def classical_kf_means_kernel(x: FP, y: FP, n: Int32, d: Int32, folds: Int32, fi: Int32, cache: FP,
                               wf: IP, woff: Int32, nonce: Int32):
    var u = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var width = Int(d) + 1
    if u < Int(folds) * width:
        kfold_mean_cell(x, y, Int(n), Int(d), Int(folds), u // width, u % width, fi != 0, cache)
    witness_end(wf, woff, nonce)


def classical_kf_grams_kernel(x: FP, y: FP, n: Int32, d: Int32, folds: Int32, cache: FP,
                               wf: IP, woff: Int32, nonce: Int32):
    var u = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var width = Int(d) + 1
    if u < Int(folds) * width * width:
        var f = u // (width * width)
        var i = (u // width) % width
        var j = u % width
        if j >= i:
            kfold_gram_cell(x, y, Int(n), Int(d), Int(folds), f, i, j, cache)
    witness_end(wf, woff, nonce)


def classical_kf_combine_kernel(cache: FP, d: Int32, folds: Int32, held: Int32, xm: FP, g: FP, xty: FP,
                                 wf: IP, woff: Int32, nonce: Int32):
    var u = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if u < Int(d) + 1 + Int(d) * Int(d) + Int(d):
        kfold_combine_unit(u, cache, Int(d), Int(folds), Int(held), xm, g, xty)
    witness_end(wf, woff, nonce)


def _ridge_kfold_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int, ip: List[Int32], fp: List[Float32],
                      res: FP) raises:
    var ctx = linear_ctx()
    var k = Int(ip[0])
    var fi = Int(ip[1])
    var na = Int(ip[2])
    var stride = n // k + 1
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dcache = ctx.enqueue_create_buffer[DType.float32](k * fold_stat_words(d) if C13_FOLD_STATS else 1)
    var dal = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    var dxm = ctx.enqueue_create_buffer[DType.float32](d + 1)
    var dg = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    var dxty = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var daw = ctx.enqueue_create_buffer[DType.float32](max(na * d * d, 1))
    var dw = ctx.enqueue_create_buffer[DType.float32](max(na * d, 1))
    var db = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    var dp = ctx.enqueue_create_buffer[DType.float32](max(na * stride, 1))
    var dsum = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    var kbm = (stride + FOLD_BLOCK - 1) // FOLD_BLOCK
    var dkyp = ctx.enqueue_create_buffer[DType.float32](max(kbm, 1))
    var dksp = ctx.enqueue_create_buffer[DType.float32](max(2 * na * kbm, 1))
    var dtr = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    var htr = List[Float32](length=max(na, 1), fill=Float32(0))
    var ffw = d + 1 + d * d + d
    var ff_units = ridge_ff_units(n, d, 1)
    var dsh = ctx.enqueue_create_buffer[DType.float32](ffw)
    var dsl = ctx.enqueue_create_buffer[DType.float32](ffw)
    var dbh = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dbl = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dfh = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    var dfl = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    var dtmp = ctx.enqueue_create_buffer[DType.float32](d + 1)
    var hfp = fp.copy()
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if XLIN_IDN_DEV_FINITE:
        xlin_finite_device(ctx, dx, n_x, y, n_y)
    ctx.enqueue_copy(dst_buf=dal, src_ptr=hfp.unsafe_ptr())
    dsum.enqueue_fill(Float32(0))
    var cells = d * (d + 1) // 2
    # lane/neural-pass93 + the Metal witness (x_linear/witness.mojo): per
    # fold, unit A (means, Gram, solves and the trust flags home) rebuilds
    # from the data; unit B (float-float re-solves, predictions, the score
    # added into the sums) reruns from the sums as they stood before it
    var nt_max = n // k + 1
    var wcap = max(_xg_blocks(d + 1) + _xg_blocks(cells + d) + _xg_blocks(na),
                   _xg_blocks(d + 1) + _xg_blocks(max(ff_units - (d + 1), 1)) + na + _xg_blocks(nt_max * na) + _xg_blocks(na)
                   + _xg_blocks(kbm) + _xg_blocks(na * kbm))
    comptime if C13_FOLD_STATS:
        wcap = max(wcap, _xg_blocks(d + 1 + d * d + d) + _xg_blocks(na))
        var prep_wit = Witness(ctx, _xg_blocks(k * (d + 1)) + _xg_blocks(k * (d + 1) * (d + 1)))
        var attempt = 0
        while True:
            var nonce = prep_wit.begin()
            var wo = 0
            ctx.enqueue_function[classical_kf_means_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(k),
                Int32(fi), dcache.unsafe_ptr(), prep_wit.p(), Int32(wo), nonce,
                grid_dim=_xg_blocks(k * (d + 1)), block_dim=XG_TPB)
            wo += _xg_blocks(k * (d + 1))
            ctx.enqueue_function[classical_kf_grams_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(k),
                dcache.unsafe_ptr(), prep_wit.p(), Int32(wo), nonce,
                grid_dim=_xg_blocks(k * (d + 1) * (d + 1)), block_dim=XG_TPB)
            wo += _xg_blocks(k * (d + 1) * (d + 1))
            if prep_wit.ok(ctx, wo, "RidgeCV disjoint-fold cache"):
                break
            attempt += 1
            if attempt >= WITNESS_TRIES:
                prep_wit.fail()
    var wit = Witness(ctx, wcap)
    var dsum_save = ctx.enqueue_create_buffer[DType.float32](max(na, 1))
    # lane/apple-fast-gram (2026-10-02), FAST on Apple (default since the M3 A/B;
    # `-D MOJOLEARN_X_LINEAR_RIDGE_FAST_GRAM_OFF` restores `kf_cells_kernel`)
    # builds each fold's means, centered Gram and X'y with the shared grid
    # Gram (x_linear/fast_gram.mojo; the same dxm / dg / dxty words) instead
    # of `kf_cells_kernel`, one thread per cell walking the fold's rows
    # (136 threads at taxi's 16 features). Its launches are not witnessed
    # (it waits for them itself); the unit's solve that follows is.
    var fold_fast_gram = False
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
        fold_fast_gram = _ridge_fast_gram() and d > 0
    for f in range(k):
        var s = Int32(kf_start(n, k, f))
        var e = Int32(kf_end(n, k, f))
        var ta = 0
        while True:
            var nonce = wit.begin()
            var wo = 0
            if C13_FOLD_STATS:
                ctx.enqueue_function[classical_kf_combine_kernel](dcache.unsafe_ptr(), Int32(d), Int32(k), Int32(f),
                    dxm.unsafe_ptr(), dg.unsafe_ptr(), dxty.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                    grid_dim=_xg_blocks(d + 1 + d * d + d), block_dim=XG_TPB)
                wo += _xg_blocks(d + 1 + d * d + d)
            elif fold_fast_gram:
                var pxm = FP(unsafe_from_address=Int(dxm.unsafe_ptr()))
                fast_gram_into(ctx, FP(unsafe_from_address=Int(dx.unsafe_ptr())), FP(unsafe_from_address=Int(dy.unsafe_ptr())),
                               Int(s), Int(e) - Int(s), d, 1, fi != 0, pxm, pxm + d,
                               FP(unsafe_from_address=Int(dg.unsafe_ptr())), FP(unsafe_from_address=Int(dxty.unsafe_ptr())))
            else:
                ctx.enqueue_function[kf_means_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), s, e, Int32(fi),
                                                      dxm.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                      grid_dim=_xg_blocks(d + 1), block_dim=XG_TPB)
                wo += _xg_blocks(d + 1)
                ctx.enqueue_function[kf_cells_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), s, e, dxm.unsafe_ptr(),
                                                      dg.unsafe_ptr(), dxty.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                      grid_dim=_xg_blocks(cells + d), block_dim=XG_TPB)
                wo += _xg_blocks(cells + d)
            ctx.enqueue_function[kf_solve_kernel](dg.unsafe_ptr(), dxty.unsafe_ptr(), dxm.unsafe_ptr(), Int32(d), dal.unsafe_ptr(),
                                                  Int32(na), Int32(fi), daw.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(),
                                                  dtr.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                  grid_dim=_xg_blocks(na), block_dim=XG_TPB)
            wo += _xg_blocks(na)
            # lane/neural-pass93: the alphas whose float32 factor is not trusted, in float-float
            ctx.enqueue_copy(dst_ptr=htr.unsafe_ptr(), src_buf=dtr)
            if wit.ok(ctx, wo, "RidgeCV fold"):
                break
            ta += 1
            if ta >= WITNESS_TRIES:
                wit.fail()
        ctx.synchronize()
        ctx.enqueue_copy(dst_buf=dsum_save, src_buf=dsum)
        var tb = 0
        while True:
            var nonce = wit.begin()
            var wo = 0
            var have_ff = False
            for a in range(na):
                if htr[a] == Float32(1):
                    continue
                if not have_ff:
                    ctx.enqueue_function[kf_ff_unit_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(fi), s, e,
                                                            Int32(0), Int32(d + 1), dsh.unsafe_ptr(), dsl.unsafe_ptr(),
                                                            wit.p(), Int32(wo), nonce,
                                                            grid_dim=_xg_blocks(d + 1), block_dim=XG_TPB)
                    wo += _xg_blocks(d + 1)
                    ctx.enqueue_function[kf_ff_unit_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(fi), s, e,
                                                            Int32(d + 1), Int32(ff_units - (d + 1)), dsh.unsafe_ptr(), dsl.unsafe_ptr(),
                                                            wit.p(), Int32(wo), nonce,
                                                            grid_dim=_xg_blocks(ff_units - (d + 1)), block_dim=XG_TPB)
                    wo += _xg_blocks(ff_units - (d + 1))
                    have_ff = True
                ctx.enqueue_function[kf_ff_solve_kernel](Int32(d), Int32(fi), hfp[a], dsh.unsafe_ptr(), dsl.unsafe_ptr(),
                                                         dbh.unsafe_ptr(), dbl.unsafe_ptr(), dfh.unsafe_ptr(), dfl.unsafe_ptr(),
                                                         dtmp.unsafe_ptr(), dw.unsafe_ptr(), db.unsafe_ptr(), Int32(a),
                                                         wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=1)
                wo += 1
            var nt = Int(e) - Int(s)
            ctx.enqueue_function[kf_pred_kernel](dx.unsafe_ptr(), Int32(d), s, e, Int32(na), dw.unsafe_ptr(), db.unsafe_ptr(),
                                                 dp.unsafe_ptr(), Int32(stride), wit.p(), Int32(wo), nonce,
                                                 grid_dim=_xg_blocks(nt * na), block_dim=XG_TPB)
            wo += _xg_blocks(nt * na)
            var kb = kf_blocks(Int(s), Int(e))
            ctx.enqueue_function[kf_ysum_kernel](dy.unsafe_ptr(), s, e, dkyp.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                 grid_dim=_xg_blocks(kb), block_dim=XG_TPB)
            wo += _xg_blocks(kb)
            ctx.enqueue_function[kf_sq_kernel](dy.unsafe_ptr(), dp.unsafe_ptr(), s, e, Int32(na), Int32(stride),
                                               dkyp.unsafe_ptr(), dksp.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                               grid_dim=_xg_blocks(na * kb), block_dim=XG_TPB)
            wo += _xg_blocks(na * kb)
            ctx.enqueue_function[kf_score_kernel](s, e, Int32(na), dksp.unsafe_ptr(), dsum.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                  grid_dim=_xg_blocks(na), block_dim=XG_TPB)
            wo += _xg_blocks(na)
            if wit.ok(ctx, wo, "RidgeCV scores"):
                break
            tb += 1
            if tb >= WITNESS_TRIES:
                wit.fail()
            ctx.enqueue_copy(dst_buf=dsum, src_buf=dsum_save)
    var hs = List[Float32](length=max(na, 1), fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=hs.unsafe_ptr(), src_buf=dsum)
    ctx.synchronize()
    for a in range(na):
        res.unsafe_store(a, fd(hs[a], i2f(k)))
    _ = hfp^
    _ = hs^
    _ = dx^
    _ = dy^
    _ = dal^
    _ = dcache^
    _ = dxm^
    _ = dg^
    _ = dxty^
    _ = daw^
    _ = dw^
    _ = db^
    _ = dp^
    _ = dsum^
    _ = dkyp^
    _ = dksp^
    _ = dtr^
    _ = htr^
    _ = dsh^
    _ = dsl^
    _ = dbh^
    _ = wit^
    _ = dsum_save^
    _ = dbl^
    _ = dfh^
    _ = dfl^
    _ = dtmp^



# ------------------------------------------------ Ridge's float-float refit on the grid (lane/neural-pass93)
def ridge_ff_unit_kernel(x: FP, y: FP, n: Int32, d: Int32, t_n: Int32, fi: Int32, sw: Int32, u0: Int32, count: Int32,
                         sh: FP, sl: FP, wf: IP, woff: Int32, nonce: Int32):
    var u = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if u < Int(count):
        ridge_ff_unit(Int(u0) + u, x, y, Int(n), Int(d), Int(t_n), fi != 0, sw != 0, Int(n) * Int(t_n), sh, sl)
    witness_end(wf, woff, nonce)


def ridge_ff_solve_kernel(d: Int32, t_n: Int32, fi: Int32, alpha: Float32, sh: FP, sl: FP, bh: FP, bl: FP, dst: FP,
                          fh: FP, fl: FP, wf: IP, woff: Int32, nonce: Int32):
    """dst: coef T*d | intercept T | ok (1 / 0)."""
    var ok = ridge_ff_solve(Int(d), Int(t_n), fi != 0, alpha, sh, sl, bh, bl, dst, fh, fl)
    st(dst, Int(t_n) * Int(d) + Int(t_n), Float32(1) if ok else Float32(0))
    witness_end(wf, woff, nonce)


def _ridge_ff_grid(mut ctx: DeviceContext, x: FP, y: FP, n: Int, d: Int, t_n: Int, fi: Bool, sw: Bool, alpha: Float32,
                   res: FP, sidx: Int) raises:
    var nm = d + t_n
    var units = ridge_ff_units(n, d, t_n)
    var words = d + t_n + d * d + d * t_n
    var dsh = ctx.enqueue_create_buffer[DType.float32](words)
    var dsl = ctx.enqueue_create_buffer[DType.float32](words)
    var dbh = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dbl = ctx.enqueue_create_buffer[DType.float32](max(d, 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](t_n * d + t_n + 1)
    var dfh = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    var dfl = ctx.enqueue_create_buffer[DType.float32](max(d * d, 1))
    # lane/neural-xlw: the refit as ONE guarded unit from zeroed sums (the M2
    # under load returned an all-zero RidgeClassifier from a cut unit launch)
    var h = List[Float32](length=t_n * d + t_n + 1, fill=Float32(0))
    var b1 = _xg_blocks(nm)
    var b2 = _xg_blocks(units - nm)
    var wit = Witness(ctx, b1 + b2 + 1)
    var tries = 0
    while True:
        var nonce = wit.begin()
        dsh.enqueue_fill(Float32(0))
        dsl.enqueue_fill(Float32(0))
        ctx.enqueue_function[ridge_ff_unit_kernel](x, y, Int32(n), Int32(d), Int32(t_n), Int32(1 if fi else 0),
                                                   Int32(1 if sw else 0), Int32(0), Int32(nm), dsh.unsafe_ptr(),
                                                   dsl.unsafe_ptr(), wit.p(), Int32(0), nonce,
                                                   grid_dim=b1, block_dim=XG_TPB)
        ctx.enqueue_function[ridge_ff_unit_kernel](x, y, Int32(n), Int32(d), Int32(t_n), Int32(1 if fi else 0),
                                                   Int32(1 if sw else 0), Int32(nm), Int32(units - nm), dsh.unsafe_ptr(),
                                                   dsl.unsafe_ptr(), wit.p(), Int32(b1), nonce,
                                                   grid_dim=b2, block_dim=XG_TPB)
        ctx.enqueue_function[ridge_ff_solve_kernel](Int32(d), Int32(t_n), Int32(1 if fi else 0), alpha, dsh.unsafe_ptr(),
                                                    dsl.unsafe_ptr(), dbh.unsafe_ptr(), dbl.unsafe_ptr(), dout.unsafe_ptr(),
                                                    dfh.unsafe_ptr(), dfl.unsafe_ptr(), wit.p(), Int32(b1 + b2), nonce,
                                                    grid_dim=1, block_dim=1)
        ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=dout)
        if wit.ok(ctx, b1 + b2 + 1, "Ridge float-float refit"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    _ = wit^
    var ok = h[t_n * d + t_n] == Float32(1)
    if ok:
        for i in range(t_n * d + t_n):
            res.unsafe_store(i, h[i])
    res.unsafe_store(sidx, Float32(0) if ok else Float32(2))
    _ = h^
    _ = dsh^
    _ = dsl^
    _ = dbh^
    _ = dbl^
    _ = dout^
    _ = dfh^
    _ = dfl^


# ------------------------------------------------ isotonic on the grid (lane/neural-pass107)
# The fit's sort as an LSD radix sort on the device: the rows start in index
# order and are stably sorted by y's key, then by x's key, then (a weighted
# fit) by a one-bit key that puts the rows of weight <= 0 last. The keys
# order exactly as the host's comparison (`_iso_less`: x, then y, then the
# row): IEEE order with -0 folded onto +0 (the comparison has -0 == +0).
# The order (x, y, row) is total, so the first nk rows of the permutation
# are the host sort's (x_linear/isotonic_host.mojo, the rows of positive
# weight) whatever the algorithm, and so is every word after it. The group
# bounds, the groups, PAVA (chunked then merged, cgr-linear) and the trim are
# grid launches. Predict: one thread a query.
#
# cpu-gpu-cleanup c-linear (2026-10-02): every scan is a parallel block scan
# (the one-thread scans of the block totals are gone), the 4096-row tile
# sort and its `MOJOLEARN_X_LINEAR_ISO_BLOCK_RADIX` switch are deleted (the
# block radix's page fits every GPU column), the group bounds are a pointer
# jumping pass over the x change points (no one-thread walk), and a weighted
# fit runs here too (it used to fall through to the team fit, which no
# longer runs isotonic on a device). No word changes: the sort's
# permutation, the starts and the counts are integers, and the float chains
# (`iso_group`, `iso_after_unique`) are the host's statements.


@always_inline
def _iso_key(v: Float32) -> UInt32:
    var b = bitcast[DType.uint32](v)
    if (b & UInt32(0x7FFFFFFF)) == UInt32(0):
        b = UInt32(0)
    if (b & UInt32(0x80000000)) != UInt32(0):
        return ~b
    return b | UInt32(0x80000000)


@always_inline
def _iso_kept(y: FP, n: Int, has_w: Int, i: Int) -> Bool:
    """Row i takes part in the fit: every row unweighted, a row of positive
    weight weighted (the host's filter, x_linear/isotonic_host.mojo)."""
    return has_w == 0 or y.unsafe_load(n + i) > Float32(0)


@always_inline
def _iso_keys_kernel_body(x: FP, y: FP, kx: IP, ky: IP, kw: IP, perm: IP, n: Int32, has_w: Int32):
    var i = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if i < Int(n):
        kx.unsafe_store(i, bitcast[DType.int32](_iso_key(x.unsafe_load(i))))
        ky.unsafe_store(i, bitcast[DType.int32](_iso_key(y.unsafe_load(i))))
        kw.unsafe_store(i, Int32(0) if _iso_kept(y, Int(n), Int(has_w), i) else Int32(1))
        perm.unsafe_store(i, Int32(i))


def iso_keys_kernel(x: FP, y: FP, kx: IP, ky: IP, kw: IP, perm: IP, n: Int32, has_w: Int32,
                    wf: IP, woff: Int32, nonce: Int32):
    _iso_keys_kernel_body(x, y, kx, ky, kw, perm, n, has_w)
    witness_end(wf, woff, nonce)


from x_linear.scan import SC_NT, _sc_block_excl, _sc_block_sum


@always_inline
def _iso_scan1_kernel_body(cnt: IP, nb: Int32, tot: IP):
    """Exclusive prefix of cnt[0:nb] in place; the total at cnt[nb] and tot[0]."""
    var part = stack_allocation[SC_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var total = _sc_block_excl(cnt, cnt, 0, Int(nb), Int32(0), part)
    if Int(thread_idx.x) == 0:
        cnt.unsafe_store(Int(nb), total)
        tot.unsafe_store(0, total)


def iso_scan1_kernel(cnt: IP, nb: Int32, tot: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_scan1_kernel_body(cnt, nb, tot)
    witness_end(wf, woff, nonce)


# ---------------------------------------------- block radix (lane/neural-pass118)
# A pass is three launches: block b (RS_NT threads x RS_IPT contiguous rows)
# counts each thread's rows per 4-bit digit in threadgroup memory and
# publishes the block's per-digit totals; block d of the scan launch turns
# digit d's totals (digit-major, block-minor) into global offsets (the
# counts of every earlier digit, then a block scan of its own row) and notes
# whether digit d holds every row; block b recounts, scans its thread
# counters (digit-major, thread-minor) and writes each row to its digit's
# global offset + its thread's offset + its rank within the thread: the
# stable order, so the LSD passes give the (x, y, row) order. A one-digit
# pass is a copy.
comptime RS_BITS = 4
comptime RS_D = 1 << RS_BITS
comptime RS_NT = 256
comptime RS_IPT = 16
comptime RS_TILE = RS_NT * RS_IPT
comptime RS_BYTES = (RS_D * RS_NT + RS_NT) * 4


@always_inline
def _rdig(keys: IP, row: Int, shift: Int) -> Int:
    return Int((bitcast[DType.uint32](keys.unsafe_load(row)) >> UInt32(shift)) & UInt32(RS_D - 1))


@always_inline
def _rs_count_kernel_body(keys: IP, src: IP, n: Int32, shift: Int32, btot: IP, nb: Int32):
    var cnt = stack_allocation[RS_D * RS_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var tid = Int(thread_idx.x)
    for d in range(RS_D):
        cnt[d * RS_NT + tid] = Int32(0)
    var lo0 = Int(block_idx.x) * RS_TILE + tid * RS_IPT
    for u in range(RS_IPT):
        var i = lo0 + u
        if i < Int(n):
            var d = _rdig(keys, Int(src.unsafe_load(i)), Int(shift))
            cnt[d * RS_NT + tid] = cnt[d * RS_NT + tid] + 1
    barrier()
    if tid < RS_D:
        var c = Int32(0)
        for t in range(RS_NT):
            c += cnt[tid * RS_NT + t]
        btot.unsafe_store(tid * Int(nb) + Int(block_idx.x), c)


def rs_count_kernel(keys: IP, src: IP, n: Int32, shift: Int32, btot: IP, nb: Int32, wf: IP, woff: Int32, nonce: Int32):
    _rs_count_kernel_body(keys, src, n, shift, btot, nb)
    witness_end(wf, woff, nonce)

@always_inline
def _rs_scan_kernel_body(btot: IP, nb: Int32, n: Int32, boff: IP, flag: IP):
    """Block d (RS_D blocks): boff's row d = the exclusive prefix of btot
    (digit-major, block-minor) over that row; flag[d] = 1 when digit d holds
    every row (the pass is a copy). btot is read only, so the blocks never
    race on it."""
    var part = stack_allocation[SC_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var d = Int(block_idx.x)
    var nbi = Int(nb)
    var base = _sc_block_sum(btot, d * nbi, part)
    var dt = _sc_block_excl(btot, boff, d * nbi, nbi, base, part)
    if Int(thread_idx.x) == 0:
        flag.unsafe_store(d, Int32(1) if dt == n else Int32(0))


def rs_scan_kernel(btot: IP, nb: Int32, n: Int32, boff: IP, flag: IP, wf: IP, woff: Int32, nonce: Int32):
    _rs_scan_kernel_body(btot, nb, n, boff, flag)
    witness_end(wf, woff, nonce)

@always_inline
def _rs_scatter_kernel_body(keys: IP, src: IP, dst: IP, n: Int32, shift: Int32, btot: IP, nb: Int32, flag: IP):
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var lo = Int(block_idx.x) * RS_TILE + tid * RS_IPT
    var one = False
    for q in range(RS_D):
        if flag.unsafe_load(q) != 0:
            one = True
    if one:
        for u in range(RS_IPT):
            var i = lo + u
            if i < nn:
                dst.unsafe_store(i, src.unsafe_load(i))
        return
    var cnt = stack_allocation[RS_D * RS_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var part = stack_allocation[RS_NT, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    for d in range(RS_D):
        cnt[d * RS_NT + tid] = Int32(0)
    var lo0 = Int(block_idx.x) * RS_TILE + tid * RS_IPT
    for u in range(RS_IPT):
        var i = lo0 + u
        if i < Int(n):
            var d = _rdig(keys, Int(src.unsafe_load(i)), Int(shift))
            cnt[d * RS_NT + tid] = cnt[d * RS_NT + tid] + 1
    barrier()
    # exclusive scan of cnt (digit-major, thread-minor) within the block:
    # thread t owns the RS_D consecutive entries [t * RS_D, t * RS_D + RS_D)
    var own = SIMD[DType.int32, RS_D]()
    var s = Int32(0)
    comptime for q in range(RS_D):
        own[q] = cnt[tid * RS_D + q]
        s += own[q]
    part[tid] = s
    barrier()
    var off = 1
    while off < RS_NT:
        var v = part[tid] + (part[tid - off] if tid >= off else Int32(0))
        barrier()
        part[tid] = v
        barrier()
        off *= 2
    var base = part[tid] - s
    barrier()
    comptime for q in range(RS_D):
        cnt[tid * RS_D + q] = base
        base += own[q]
    barrier()
    # the block's in-tile offset of (digit d, thread t) is cnt[d * RS_NT + t]
    # minus the block's first offset of digit d, which is cnt[d * RS_NT]
    var run = SIMD[DType.int32, RS_D](0)
    for u in range(RS_IPT):
        var i = lo + u
        if i < nn:
            var r = src.unsafe_load(i)
            var d = _rdig(keys, Int(r), Int(shift))
            var at = btot.unsafe_load(d * Int(nb) + Int(block_idx.x)) + (cnt[d * RS_NT + tid] - cnt[d * RS_NT]) + run[d]
            dst.unsafe_store(Int(at), r)
            run[d] += 1


def rs_scatter_kernel(keys: IP, src: IP, dst: IP, n: Int32, shift: Int32, btot: IP, nb: Int32, flag: IP, wf: IP, woff: Int32, nonce: Int32):
    _rs_scatter_kernel_body(keys, src, dst, n, shift, btot, nb, flag)
    witness_end(wf, woff, nonce)


# ---------------------------------------------- the rows that take part (nk)
@always_inline
def _iso_kept_count_kernel_body(y: FP, n: Int32, has_w: Int32, bcnt: IP):
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    sh[tid] = Int32(1) if (j < nn and _iso_kept(y, nn, Int(has_w), j)) else Int32(0)
    barrier()
    if tid == 0:
        var c = Int32(0)
        for u in range(XG_TPB):
            c += sh[u]
        bcnt.unsafe_store(Int(block_idx.x), c)


def iso_kept_count_kernel(y: FP, n: Int32, has_w: Int32, bcnt: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_kept_count_kernel_body(y, n, has_w, bcnt)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_gather_kernel_body(x: FP, y: FP, n: Int32, has_w: Int32, perm: IP, fw: FP, nkp: IP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if j < Int(nkp.unsafe_load(0)):
        iso_gather_one(j, x, y, nn, has_w != 0, perm, fw + 3 * nn, fw + 4 * nn, fw + 5 * nn)


def iso_gather_kernel(x: FP, y: FP, n: Int32, has_w: Int32, perm: IP, fw: FP, nkp: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_gather_kernel_body(x, y, n, has_w, perm, fw, nkp)
    witness_end(wf, woff, nonce)

# The group bounds (lane/neural-pass117, cpu-gpu-cleanup c-linear). A row
# whose x equals the row before never starts a group, so only the rows where
# x changes (flagged and compacted in order by the grid) are candidates.
# Node 0 is row 0 and node q + 1 the q-th candidate; next(k) is the first
# node after k at least 1e-6 above k's x (a binary search: the nodes' x
# ascend and `fs` is monotone), N (= the node count) when none is. The
# groups start at the nodes the chain 0, next(0), next(next(0)), ... visits,
# which is the host walk `iso_bounds` exactly. The chain is marked by
# pointer jumping: round r marks P(k) for every marked k with
# P = next^(2^r), then P = P o P; after R rounds (2^R > N) every node of
# the chain is marked. A mark is only ever written onto a node of the chain,
# so the races inside a round are benign and the set is the same on every
# run. The marked nodes compact, in order, to the starts.
@always_inline
def _iso_changed(xs: FP, j: Int, nk: Int) -> Int:
    if j >= 1 and j < nk and xs.unsafe_load(j) != xs.unsafe_load(j - 1):
        return 1
    return 0


@always_inline
def _iso_flag_count_kernel_body(fw: FP, n: Int32, bcnt: IP, nkp: IP):
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    sh[tid] = Int32(_iso_changed(fw + 3 * nn, j, Int(nkp.unsafe_load(0))))
    barrier()
    if tid == 0:
        var c = Int32(0)
        for u in range(XG_TPB):
            c += sh[u]
        bcnt.unsafe_store(Int(block_idx.x), c)


def iso_flag_count_kernel(fw: FP, n: Int32, bcnt: IP, nkp: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_flag_count_kernel_body(fw, n, bcnt, nkp)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_flag_write_kernel_body(fw: FP, n: Int32, bcnt: IP, cand: IP, nkp: IP):
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var f = _iso_changed(fw + 3 * nn, j, Int(nkp.unsafe_load(0)))
    sh[tid] = Int32(f)
    barrier()
    if f != 0:
        var r = Int32(0)
        for u in range(tid):
            r += sh[u]
        cand.unsafe_store(Int(bcnt.unsafe_load(Int(block_idx.x)) + r), Int32(j))


def iso_flag_write_kernel(fw: FP, n: Int32, bcnt: IP, cand: IP, nkp: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_flag_write_kernel_body(fw, n, bcnt, cand, nkp)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_nodes(ncp: IP, nkp: IP) -> Int:
    """The node count N: row 0 and the candidates (none when no row takes part)."""
    if Int(nkp.unsafe_load(0)) < 1:
        return 0
    return Int(ncp.unsafe_load(0)) + 1


@always_inline
def _iso_node_row(cand: IP, k: Int) -> Int:
    if k == 0:
        return 0
    return Int(cand.unsafe_load(k - 1))


@always_inline
def _iso_next_kernel_body(fw: FP, n: Int32, cand: IP, ncp: IP, nkp: IP, nxt: IP, mk: IP):
    var k = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var nodes = _iso_nodes(ncp, nkp)
    var xs = fw + 3 * nn
    if k < nodes:
        var xk = xs.unsafe_load(_iso_node_row(cand, k))
        var lo = k + 1
        var hi = nodes
        while lo < hi:
            var mid = (lo + hi) // 2
            if fs(xs.unsafe_load(_iso_node_row(cand, mid)), xk) >= Float32(1e-6):
                hi = mid
            else:
                lo = mid + 1
        nxt.unsafe_store(k, Int32(lo))
        mk.unsafe_store(k, Int32(1) if k == 0 else Int32(0))
    elif k == nodes:
        nxt.unsafe_store(k, Int32(nodes))
        mk.unsafe_store(k, Int32(0))


def iso_next_kernel(fw: FP, n: Int32, cand: IP, ncp: IP, nkp: IP, nxt: IP, mk: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_next_kernel_body(fw, n, cand, ncp, nkp, nxt, mk)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_mark_kernel_body(ncp: IP, nkp: IP, p: IP, mk: IP):
    var k = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nodes = _iso_nodes(ncp, nkp)
    if k < nodes and mk.unsafe_load(k) != 0:
        var t = Int(p.unsafe_load(k))
        if t < nodes:
            mk.unsafe_store(t, Int32(1))


def iso_mark_kernel(ncp: IP, nkp: IP, p: IP, mk: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_mark_kernel_body(ncp, nkp, p, mk)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_jump_kernel_body(ncp: IP, nkp: IP, p: IP, q: IP):
    var k = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if k <= _iso_nodes(ncp, nkp):
        q.unsafe_store(k, p.unsafe_load(Int(p.unsafe_load(k))))


def iso_jump_kernel(ncp: IP, nkp: IP, p: IP, q: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_jump_kernel_body(ncp, nkp, p, q)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_mark_count_kernel_body(ncp: IP, nkp: IP, mk: IP, bcnt: IP):
    var tid = Int(thread_idx.x)
    var k = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    sh[tid] = Int32(1) if (k < _iso_nodes(ncp, nkp) and mk.unsafe_load(k) != 0) else Int32(0)
    barrier()
    if tid == 0:
        var c = Int32(0)
        for u in range(XG_TPB):
            c += sh[u]
        bcnt.unsafe_store(Int(block_idx.x), c)


def iso_mark_count_kernel(ncp: IP, nkp: IP, mk: IP, bcnt: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_mark_count_kernel_body(ncp, nkp, mk, bcnt)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_starts_kernel_body(n: Int32, cand: IP, ncp: IP, nkp: IP, mk: IP, bcnt: IP, nb: Int32, iw: IP, mslot: IP):
    """starts (iw + n) = the rows of the marked nodes in order, then nk at
    starts[m]; mslot[0] = m (`iso_bounds`' layout and answer)."""
    var nn = Int(n)
    var tid = Int(thread_idx.x)
    var k = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var f = 1 if (k < _iso_nodes(ncp, nkp) and mk.unsafe_load(k) != 0) else 0
    sh[tid] = Int32(f)
    barrier()
    var starts = iw + nn
    if f != 0:
        var r = Int32(0)
        for u in range(tid):
            r += sh[u]
        starts.unsafe_store(Int(bcnt.unsafe_load(Int(block_idx.x)) + r), Int32(_iso_node_row(cand, k)))
    if k == 0:
        var m = bcnt.unsafe_load(Int(nb))
        starts.unsafe_store(Int(m), nkp.unsafe_load(0))
        mslot.unsafe_store(0, m)


def iso_starts_kernel(n: Int32, cand: IP, ncp: IP, nkp: IP, mk: IP, bcnt: IP, nb: Int32, iw: IP, mslot: IP,
                      wf: IP, woff: Int32, nonce: Int32):
    _iso_starts_kernel_body(n, cand, ncp, nkp, mk, bcnt, nb, iw, mslot)
    witness_end(wf, woff, nonce)


def iso_rounds(n: Int) -> Int:
    """Pointer-jumping rounds for up to n + 1 nodes: 2^R > n + 1."""
    var r = 0
    while (1 << r) <= n + 1:
        r += 1
    return r


@always_inline
def _iso_group_kernel_body(fw: FP, n: Int32, iw: IP, mslot: IP):
    var g = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if g < Int(mslot.unsafe_load(0)):
        iso_group(g, fw + 3 * nn, fw + 4 * nn, fw + 5 * nn, iw + nn, fw, nn)


def iso_group_kernel(fw: FP, n: Int32, iw: IP, mslot: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_group_kernel_body(fw, n, iw, mslot)
    witness_end(wf, woff, nonce)

# PAVA on the grid (cgr-linear): x_linear/isotonic.mojo's chunked-then-merged
# order, a thread a chunk, then a thread a segment pair per level; each group
# finds its block's start by pointer jumping over the start flags (iw + n);
# the trim compacts through a block scan. Every launch reads m from mslot and
# sizes itself by n.
@always_inline
def _iso_m(mslot: IP) -> Int:
    return Int(mslot.unsafe_load(0))


@always_inline
def _iso_rev_kernel_body(n: Int32, fw: FP, mslot: IP, both: Int32):
    var a = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var m = _iso_m(mslot)
    if a < m // 2:
        iso_reverse_one(a, m, Int(n), fw, both != 0)


def iso_rev_kernel(n: Int32, fw: FP, mslot: IP, both: Int32, wf: IP, woff: Int32, nonce: Int32):
    """Thread a: groups a and m - 1 - a swapped (uy, and uw with both)."""
    _iso_rev_kernel_body(n, fw, mslot, both)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_chunk_kernel_body(n: Int32, fw: FP, iw: IP, mslot: IP):
    var c = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var m = _iso_m(mslot)
    if c * ISO_CHUNK < m:
        iso_pava_chunk(c, m, Int(n), fw, iw)


def iso_chunk_kernel(n: Int32, fw: FP, iw: IP, mslot: IP, wf: IP, woff: Int32, nonce: Int32):
    """Thread c: PAVA over chunk c's ISO_CHUNK groups."""
    _iso_chunk_kernel_body(n, fw, iw, mslot)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_merge_kernel_body(lvl: Int32, n: Int32, fw: FP, iw: IP, mslot: IP):
    var p = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var m = _iso_m(mslot)
    if p * (2 * ISO_CHUNK << Int(lvl)) < m:
        iso_pava_merge(Int(lvl), p, m, Int(n), fw, iw)


def iso_merge_kernel(lvl: Int32, n: Int32, fw: FP, iw: IP, mslot: IP, wf: IP, woff: Int32, nonce: Int32):
    """Thread p: segment pair p of level lvl pooled at its seam."""
    _iso_merge_kernel_body(lvl, n, fw, iw, mslot)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_start_init_kernel_body(n: Int32, iw: IP, mslot: IP, pt: IP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if j < _iso_m(mslot):
        pt.unsafe_store(j, Int32(j) if iw.unsafe_load(Int(n) + j) != 0 else Int32(j - 1))


def iso_start_init_kernel(n: Int32, iw: IP, mslot: IP, pt: IP, wf: IP, woff: Int32, nonce: Int32):
    """Thread j: itself when it starts a block, else the group before it."""
    _iso_start_init_kernel_body(n, iw, mslot, pt)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_start_jump_kernel_body(mslot: IP, pc: IP, pn: IP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if j < _iso_m(mslot):
        pn.unsafe_store(j, pc.unsafe_load(Int(pc.unsafe_load(j))))


def iso_start_jump_kernel(mslot: IP, pc: IP, pn: IP, wf: IP, woff: Int32, nonce: Int32):
    """P = P o P: after enough rounds every group points at its block's start."""
    _iso_start_jump_kernel_body(mslot, pc, pn)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_fill_kernel_body(n: Int32, fw: FP, iw: IP, mslot: IP, pt: IP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if j < _iso_m(mslot) and iw.unsafe_load(nn + j) == 0:
        st(fw, nn + j, ld(fw, nn + Int(pt.unsafe_load(j))))


def iso_fill_kernel(n: Int32, fw: FP, iw: IP, mslot: IP, pt: IP, wf: IP, woff: Int32, nonce: Int32):
    """Thread j (not a start): its block's pooled value."""
    _iso_fill_kernel_body(n, fw, iw, mslot, pt)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_clip_kernel_body(n: Int32, ip: IP, fp: FP, fw: FP, mslot: IP):
    var j = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if j < _iso_m(mslot):
        iso_clip_one(j, Int(n), ip, fp, fw)


def iso_clip_kernel(n: Int32, ip: IP, fp: FP, fw: FP, mslot: IP, wf: IP, woff: Int32, nonce: Int32):
    _iso_clip_kernel_body(n, ip, fp, fw, mslot)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_keep_flag(n: Int, fw: FP, m: Int, j: Int) -> Int:
    return 1 if (j < m and iso_keep(j, m, n, fw)) else 0


@always_inline
def _iso_keep_count_kernel_body(n: Int32, fw: FP, mslot: IP, bcnt: IP):
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    sh[tid] = Int32(_iso_keep_flag(Int(n), fw, _iso_m(mslot), j))
    barrier()
    if tid == 0:
        var c = Int32(0)
        for u in range(XG_TPB):
            c += sh[u]
        bcnt.unsafe_store(Int(block_idx.x), c)


def iso_keep_count_kernel(n: Int32, fw: FP, mslot: IP, bcnt: IP, wf: IP, woff: Int32, nonce: Int32):
    """Block b: how many of its groups the trim keeps."""
    _iso_keep_count_kernel_body(n, fw, mslot, bcnt)
    witness_end(wf, woff, nonce)


@always_inline
def _iso_keep_write_kernel_body(n: Int32, fw: FP, mslot: IP, bcnt: IP, nbk: Int32, res: FP):
    var nn = Int(n)
    var m = _iso_m(mslot)
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x) * XG_TPB + tid
    var sh = stack_allocation[XG_TPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var f = _iso_keep_flag(nn, fw, m, j)
    sh[tid] = Int32(f)
    barrier()
    if f != 0:
        var r = Int32(0)
        for u in range(tid):
            r += sh[u]
        var at = Int(bcnt.unsafe_load(Int(block_idx.x)) + r)
        st(res, 3 + at, ld(fw, j))
        st(res, 3 + nn + at, ld(fw, nn + j))
    if j == 0:
        if m > 0:
            st(res, 0, i2f(Int(bcnt.unsafe_load(Int(nbk)))))
            st(res, 1, ld(fw, 0))
            st(res, 2, ld(fw, m - 1))
        else:
            st(res, 0, Float32(0))
            st(res, 1, Float32(0))
            st(res, 2, Float32(0))


def iso_keep_write_kernel(n: Int32, fw: FP, mslot: IP, bcnt: IP, nbk: Int32, res: FP, wf: IP, woff: Int32, nonce: Int32):
    """The kept groups' x and value at their scanned slots; res[0:3]."""
    _iso_keep_write_kernel_body(n, fw, mslot, bcnt, nbk, res)
    witness_end(wf, woff, nonce)

@always_inline
def _iso_predict_kernel_body(x: FP, thr: FP, n: Int32, m: Int32, oob: Int32, fp: FP, res: FP):
    var q = Int(block_idx.x) * XG_TPB + Int(thread_idx.x)
    if q < Int(n):
        iso_predict_one(q, x, thr, Int(m), Int(oob), fp, res)
        iso_oob_flag(q, Int(n), x, Int(oob), fp, res)


def iso_predict_kernel(x: FP, thr: FP, n: Int32, m: Int32, oob: Int32, fp: FP, res: FP, wf: IP, woff: Int32, nonce: Int32):
    _iso_predict_kernel_body(x, thr, n, m, oob, fp, res)
    witness_end(wf, woff, nonce)

def ISO_WIT_CAP(n: Int) -> Int:
    """The witness words of one isotonic fit (an upper bound)."""
    var nb = max((n + RS_TILE - 1) // RS_TILE, 1)
    var passes = 2 * (32 // RS_BITS) + 1
    var nodes = _xg_blocks(n + 1)
    var chunks = max((n + ISO_CHUNK - 1) // ISO_CHUNK, 1)
    var pava = (iso_rounds(n) + 6) * _xg_blocks(n) + (iso_pava_levels(n) + 1) * _xg_blocks(chunks) + 2 * _xg_blocks(max(n // 2, 1)) + 1
    return passes * (2 * nb + RS_D) + 6 * _xg_blocks(n) + (2 * iso_rounds(n) + 3) * nodes + pava + 16


def _iso_fit_grid(x: FP, n_x: Int, y: FP, n_y: Int, n: Int, ip: List[Int32], fp: List[Float32], n_out: Int,
                  n_fw: Int, n_iw: Int, res: FP) raises:
    comptime assert lib_smem_page_fits_for[TARGET_COLUMN, RS_BYTES](), "the isotonic block radix page must fit"
    comptime assert RS_NT == SC_NT, "the radix scan runs the shared block scan"
    var ctx = linear_ctx()
    var hip = ip.copy()
    var hfp = fp.copy()
    var has_w = Int32(1) if (len(hip) > 3 and Int(hip[3]) != 0) else Int32(0)
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dip = ctx.enqueue_create_buffer[DType.int32](max(len(hip), 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(hfp), 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var dfw = ctx.enqueue_create_buffer[DType.float32](max(n_fw, 1))
    var diw = ctx.enqueue_create_buffer[DType.int32](max(n_iw, 1))
    var dkx = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dky = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dkw = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dpa = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dpb = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if XLIN_IDN_DEV_FINITE:
        xlin_finite_device(ctx, dx, n_x, y, n_y)
    ctx.enqueue_copy(dst_buf=dip, src_ptr=hip.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    var dm = ctx.enqueue_create_buffer[DType.int32](1)
    var nbk = _xg_blocks(n)
    var nbn = _xg_blocks(n + 1)
    # block counts: the rows that take part (dkc), the x change points (dbc)
    var dkc = ctx.enqueue_create_buffer[DType.int32](nbk + 1)
    var dbc = ctx.enqueue_create_buffer[DType.int32](nbk + 1)
    var dcand = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    var dna = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var dnb = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var dmk = ctx.enqueue_create_buffer[DType.int32](n + 1)
    var dmc = ctx.enqueue_create_buffer[DType.int32](nbn + 1)
    var nb = max((n + RS_TILE - 1) // RS_TILE, 1)
    var dbt = ctx.enqueue_create_buffer[DType.int32](RS_D * nb)
    var dbo = ctx.enqueue_create_buffer[DType.int32](RS_D * nb)
    var dfl = ctx.enqueue_create_buffer[DType.int32](RS_D)
    # nk and the candidate count in their own words (a kernel argument is a
    # buffer's own pointer, never an offset into one)
    var dnk = ctx.enqueue_create_buffer[DType.int32](1)
    var dnc = ctx.enqueue_create_buffer[DType.int32](1)
    var dmt = ctx.enqueue_create_buffer[DType.int32](1)
    var nkp = dnk.unsafe_ptr()
    var ncp = dnc.unsafe_ptr()
    var rounds = iso_rounds(n)
    var inc = len(hip) < 1 or Int(hip[0]) != 0
    var clip = len(hip) > 2 and (Int(hip[1]) != 0 or Int(hip[2]) != 0)
    var chunks = max((n + ISO_CHUNK - 1) // ISO_CHUNK, 1)
    var levels = iso_pava_levels(n)
    var nbh = _xg_blocks(max(n // 2, 1))
    var nbc = _xg_blocks(chunks)
    # the whole isotonic fit as ONE guarded unit from zeroed scratch (x_linear/witness.mojo)
    var wit = Witness(ctx, ISO_WIT_CAP(n))
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        dout.enqueue_fill(Float32(0))
        dfw.enqueue_fill(Float32(0))
        diw.enqueue_fill(Int32(0))
        ctx.enqueue_function[iso_keys_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), dkx.unsafe_ptr(), dky.unsafe_ptr(),
                                              dkw.unsafe_ptr(), dpa.unsafe_ptr(), Int32(n), has_w,
                                              wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        ctx.enqueue_function[iso_kept_count_kernel](dy.unsafe_ptr(), Int32(n), has_w, dkc.unsafe_ptr(),
                                                    wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        ctx.enqueue_function[iso_scan1_kernel](dkc.unsafe_ptr(), Int32(nbk), nkp, wit.p(), Int32(wo), nonce,
                                               grid_dim=1, block_dim=SC_NT)
        wo += 1
        # y's key, then x's, then (weighted) the kept bit: the most significant last
        var cur_a = True
        for key in range(3 if has_w != 0 else 2):
            var npass = 32 // RS_BITS if key < 2 else 1
            for pas in range(npass):
                var shift = Int32(RS_BITS * pas)
                var kaddr = Int(dky.unsafe_ptr()) if key == 0 else (Int(dkx.unsafe_ptr()) if key == 1 else Int(dkw.unsafe_ptr()))
                var kp = IP(unsafe_from_address=kaddr)
                var src = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
                var dst = IP(unsafe_from_address=Int(dpb.unsafe_ptr()) if cur_a else Int(dpa.unsafe_ptr()))
                ctx.enqueue_function[rs_count_kernel](kp, src, Int32(n), shift, dbt.unsafe_ptr(), Int32(nb),
                                                      wit.p(), Int32(wo), nonce, grid_dim=nb, block_dim=RS_NT)
                wo += nb
                ctx.enqueue_function[rs_scan_kernel](dbt.unsafe_ptr(), Int32(nb), Int32(n), dbo.unsafe_ptr(), dfl.unsafe_ptr(),
                                                     wit.p(), Int32(wo), nonce, grid_dim=RS_D, block_dim=SC_NT)
                wo += RS_D
                ctx.enqueue_function[rs_scatter_kernel](kp, src, dst, Int32(n), shift, dbo.unsafe_ptr(), Int32(nb),
                                                        dfl.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=nb, block_dim=RS_NT)
                wo += nb
                cur_a = not cur_a
        var pp = IP(unsafe_from_address=Int(dpa.unsafe_ptr()) if cur_a else Int(dpb.unsafe_ptr()))
        ctx.enqueue_function[iso_gather_kernel](dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), has_w, pp, dfw.unsafe_ptr(), nkp,
                                                wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        ctx.enqueue_function[iso_flag_count_kernel](dfw.unsafe_ptr(), Int32(n), dbc.unsafe_ptr(), nkp,
                                                    wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        ctx.enqueue_function[iso_scan1_kernel](dbc.unsafe_ptr(), Int32(nbk), ncp, wit.p(), Int32(wo), nonce,
                                               grid_dim=1, block_dim=SC_NT)
        wo += 1
        ctx.enqueue_function[iso_flag_write_kernel](dfw.unsafe_ptr(), Int32(n), dbc.unsafe_ptr(), dcand.unsafe_ptr(), nkp,
                                                    wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        # the group starts: next() by binary search, then pointer jumping
        ctx.enqueue_function[iso_next_kernel](dfw.unsafe_ptr(), Int32(n), dcand.unsafe_ptr(), ncp, nkp,
                                              dna.unsafe_ptr(), dmk.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                              grid_dim=nbn, block_dim=XG_TPB)
        wo += nbn
        var p_a = True
        for r in range(rounds):
            var pcur = IP(unsafe_from_address=Int(dna.unsafe_ptr()) if p_a else Int(dnb.unsafe_ptr()))
            var pnext = IP(unsafe_from_address=Int(dnb.unsafe_ptr()) if p_a else Int(dna.unsafe_ptr()))
            ctx.enqueue_function[iso_mark_kernel](ncp, nkp, pcur, dmk.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                  grid_dim=nbn, block_dim=XG_TPB)
            wo += nbn
            if r + 1 < rounds:
                ctx.enqueue_function[iso_jump_kernel](ncp, nkp, pcur, pnext, wit.p(), Int32(wo), nonce,
                                                      grid_dim=nbn, block_dim=XG_TPB)
                wo += nbn
                p_a = not p_a
        ctx.enqueue_function[iso_mark_count_kernel](ncp, nkp, dmk.unsafe_ptr(), dmc.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                    grid_dim=nbn, block_dim=XG_TPB)
        wo += nbn
        ctx.enqueue_function[iso_scan1_kernel](dmc.unsafe_ptr(), Int32(nbn), dmt.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                               grid_dim=1, block_dim=SC_NT)
        wo += 1
        ctx.enqueue_function[iso_starts_kernel](Int32(n), dcand.unsafe_ptr(), ncp, nkp, dmk.unsafe_ptr(), dmc.unsafe_ptr(),
                                                Int32(nbn), diw.unsafe_ptr(), dm.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                grid_dim=nbn, block_dim=XG_TPB)
        wo += nbn
        ctx.enqueue_function[iso_group_kernel](dfw.unsafe_ptr(), Int32(n), diw.unsafe_ptr(), dm.unsafe_ptr(),
                                               wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        # PAVA, chunked then merged (x_linear/isotonic.mojo), then the fill,
        # the clip and the trim, every step a grid launch
        var fwq = dfw.unsafe_ptr()
        var iwq = diw.unsafe_ptr()
        if not inc:
            ctx.enqueue_function[iso_rev_kernel](Int32(n), fwq, dm.unsafe_ptr(), Int32(1), wit.p(), Int32(wo), nonce,
                                                 grid_dim=nbh, block_dim=XG_TPB)
            wo += nbh
        ctx.enqueue_function[iso_chunk_kernel](Int32(n), fwq, iwq, dm.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                               grid_dim=nbc, block_dim=XG_TPB)
        wo += nbc
        for l in range(levels):
            var g = _xg_blocks((chunks + (2 << l) - 1) // (2 << l))
            ctx.enqueue_function[iso_merge_kernel](Int32(l), Int32(n), fwq, iwq, dm.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                   grid_dim=g, block_dim=XG_TPB)
            wo += g
        ctx.enqueue_function[iso_start_init_kernel](Int32(n), iwq, dm.unsafe_ptr(), dna.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                    grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        var s_a = True
        for _ in range(rounds):
            var pcur = IP(unsafe_from_address=Int(dna.unsafe_ptr()) if s_a else Int(dnb.unsafe_ptr()))
            var pnext = IP(unsafe_from_address=Int(dnb.unsafe_ptr()) if s_a else Int(dna.unsafe_ptr()))
            ctx.enqueue_function[iso_start_jump_kernel](dm.unsafe_ptr(), pcur, pnext, wit.p(), Int32(wo), nonce,
                                                        grid_dim=nbk, block_dim=XG_TPB)
            wo += nbk
            s_a = not s_a
        var pfin = IP(unsafe_from_address=Int(dna.unsafe_ptr()) if s_a else Int(dnb.unsafe_ptr()))
        ctx.enqueue_function[iso_fill_kernel](Int32(n), fwq, iwq, dm.unsafe_ptr(), pfin, wit.p(), Int32(wo), nonce,
                                              grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        if not inc:
            ctx.enqueue_function[iso_rev_kernel](Int32(n), fwq, dm.unsafe_ptr(), Int32(0), wit.p(), Int32(wo), nonce,
                                                 grid_dim=nbh, block_dim=XG_TPB)
            wo += nbh
        if clip:
            ctx.enqueue_function[iso_clip_kernel](Int32(n), dip.unsafe_ptr(), dfp.unsafe_ptr(), fwq, dm.unsafe_ptr(),
                                                  wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
            wo += nbk
        ctx.enqueue_function[iso_keep_count_kernel](Int32(n), fwq, dm.unsafe_ptr(), dkc.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                                    grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        ctx.enqueue_function[iso_scan1_kernel](dkc.unsafe_ptr(), Int32(nbk), dmt.unsafe_ptr(), wit.p(), Int32(wo), nonce,
                                               grid_dim=1, block_dim=SC_NT)
        wo += 1
        ctx.enqueue_function[iso_keep_write_kernel](Int32(n), fwq, dm.unsafe_ptr(), dkc.unsafe_ptr(), Int32(nbk),
                                                    dout.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=nbk, block_dim=XG_TPB)
        wo += nbk
        if n_out > 0:
            ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
        if wit.ok(ctx, wo, "isotonic fit"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    _ = hip^
    _ = hfp^
    _ = dx^
    _ = dy^
    _ = dip^
    _ = dfp^
    _ = dout^
    _ = dfw^
    _ = diw^
    _ = dkx^
    _ = dky^
    _ = dkw^
    _ = dpa^
    _ = dpb^
    _ = dkc^
    _ = dbc^
    _ = dcand^
    _ = dna^
    _ = dnb^
    _ = dmk^
    _ = dmc^
    _ = dbt^
    _ = dbo^
    _ = dfl^
    _ = dnk^
    _ = dnc^
    _ = dmt^
    _ = dm^
    _ = wit^


def _iso_predict_grid(x: FP, n_x: Int, thr: FP, n_thr: Int, n: Int, ip: List[Int32], fp: List[Float32], res: FP) raises:
    var ctx = linear_ctx()
    var hfp = fp.copy()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dt = ctx.enqueue_create_buffer[DType.float32](max(n_thr, 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(hfp), 1))
    # OOB_RAISE: one more word, res[n], the out-of-bounds flag (zeroed here)
    var n_res = n + 1 if (len(ip) > 1 and Int(ip[1]) == OOB_RAISE) else n
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_res, 1))
    dout.enqueue_fill(Float32(0))
    ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dt, src_ptr=thr)
    comptime if XLIN_IDN_DEV_FINITE:
        xlin_finite_device(ctx, dx, n_x, thr, n_thr)
    ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    # the whole isotonic predict as ONE guarded unit from zeroed scratch (x_linear/witness.mojo)
    var wit = Witness(ctx, _xg_blocks(n))
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wo = 0
        ctx.enqueue_function[iso_predict_kernel](dx.unsafe_ptr(), dt.unsafe_ptr(), Int32(n), ip[0], ip[1], dfp.unsafe_ptr(),
                                                 dout.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(n), block_dim=XG_TPB)
        wo += _xg_blocks(n)
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
        if wit.ok(ctx, wo, "isotonic predict"):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    _ = hfp^
    _ = dx^
    _ = dt^
    _ = dfp^
    _ = dout^
    _ = wit^


def fit_device(
    algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, n_fw: Int, n_iw: Int, res: FP,
) raises:
    # lane/neural-pass107: isotonic on the grid (the radix sort, the bounds,
    # the groups and PAVA on the grid; one thread a predicted query);
    # weighted fits too (cpu-gpu-cleanup c-linear)
    if algo == ALGO_ISOTONIC and n > 0:
        _iso_fit_grid(x, n_x, y, n_y, n, ip, fp, n_out, n_fw, n_iw, res)
        return
    if algo == ALGO_ISOTONIC_PREDICT and n > 0:
        _iso_predict_grid(x, n_x, y, n_y, n, ip, fp, res)
        return
    if algo == ALGO_GLM and n > 0:
        _glm_fit_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    if algo == ALGO_RIDGE_KFOLD:
        _ridge_kfold_grid(x, n_x, y, n_y, n, d, ip, fp, res)
        return
    if algo == ALGO_QUANTILE and n > 0:
        # cgr-linear: QuantileRegressor's ADMM with every row pass on the grid
        # (x_linear/quantile_grid.mojo), no longer the one-block fit kernel
        quantile_fit_grid(linear_ctx(), x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    if algo == ALGO_SGD and len(ip) > 12 and sgd_mb_on(sgd_batch(Int(ip[12]), Int(ip[0])), Int(ip[0]), Int(ip[3])):
        _sgd_mb_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    if algo == ALGO_SGD:
        _sgd_ps_grid(x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    var ctx = linear_ctx()
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
        # LassoCV / ElasticNetCV with every row pass on the grid
        # (x_linear/enetcv_fast.mojo, lane/apple-fast-classical); `=0` is the A/B arm
        if algo == ALGO_ENETCV and n > 0 and String(getenv("MOJOLEARN_X_LINEAR_ENETCV_FAST")) != "0":
            if enetcv_fast(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, res):
                return
    if algo == ALGO_LOGCV and n > 0:
        logcv_fit_grid(ctx, algo, x, n_x, y, n_y, n, d, ip, fp, n_out, n_fw, n_iw, res)
        return
    # HuberRegressor with the line search batched on the device
    # (x_linear/huber_fast.mojo, lane/apple-fast-robust recovered):
    # -D MOJOLEARN_HUBER_DEVICE_LBFGS (or HUBER_FAST_BLOCK512), FAST + Apple only
    comptime if HUBER_DEVICE_LBFGS:
        if algo == ALGO_HUBER and n > 0:
            huber_fit_fast(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, res)
            return
    if algo == ALGO_HUBER and n > 0:
        huber_fit_grid(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, n_fw, n_iw, res)
        return
    if algo == ALGO_ENETCV and d > 0 and len(ip) >= 7:
        enetcv_fit_grid(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    # cgr-linear: ARD and Ridge on their grid drivers; LARS and BayesianRidge
    # below; nothing runs on a one-block fit kernel any more
    if algo == ALGO_ARD and n > 0 and d > 0:
        ard_fit_grid(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, n_fw, n_iw, res)
        return
    if algo == ALGO_RIDGE and n > 0 and d > 0:
        _ridge_device(ctx, x, n_x, y, n_y, n, d, ip, fp, n_out, res)
        return
    if not ((algo == ALGO_LARS or algo == ALGO_BAYES) and n > 0 and d > 0):
        # no grid uploads X on this path: the binding's walk, so the
        # NaN refusal still comes first
        comptime if XLIN_IDN_DEV_FINITE:
            xlin_finite_host(x, n_x, "X")
            xlin_finite_host(y, n_y, "y")
        raise Error("x_linear: no device route for fit " + String(algo) + " at this shape")
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(fp), 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var dfw = ctx.enqueue_create_buffer[DType.float32](max(n_fw, 1))
    var diw = ctx.enqueue_create_buffer[DType.int32](max(n_iw, 1))
    var hip = ip.copy()
    comptime assert MOMENTS_GRID, "the moments grid's page must fit every GPU column"
    var grid_gram = algo == ALGO_LARS
    # lane/neural-pass87 (2026-10-01): BayesianRidge (unweighted) and ARD read
    # the same layout (xm at 0, G at d, ip[1] fit_intercept) and the same
    # centered Gram chains, which the team ran on ONE block (24,310 chains
    # of every row at 220 features over 256 threads).
    var bayes_like = algo == ALGO_BAYES and len(ip) > 2 and ip[2] == 0
    if bayes_like:
        grid_gram = True
    # LARS: the moments of [X | y] (means, Gram, X'y, y's mean) on the grid,
    # then the path on one block team (`lars_path_kernel`, which reads the
    # row count from ip[6])
    var lars_pre = algo == ALGO_LARS
    if algo == ALGO_LARS:
        while len(hip) < 7:
            hip.append(Int32(0))
        hip[6] = Int32(n)
    var fast_gram = False
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
        # x_linear/bayes.mojo `X_LINEAR_GRAM_SSE`: ip[5], the sse from the
        # normal equations (lane/apple-fast-classical); `=0` is the A/B arm
        if bayes_like:
            while len(hip) < 6:
                hip.append(Int32(0))
            hip[5] = Int32(0 if String(getenv("MOJOLEARN_X_LINEAR_GRAM_SSE")) == "0" else 1)
        # lane/apple-fast-gram: the shared grid Gram (x_linear/fast_gram.mojo,
        # row chunks across the grid) fills the words main's moments grid
        # fills on one block per tile pair (taxi's 16 features: ONE block);
        # `lars_path_kernel` reads the same words, so lars_pre is off.
        # (Ridge's switch lives in x_linear/ridge_grid.mojo `ridge_fit_grid`.)
        if algo == ALGO_LARS and d > 0 and n > 0:
            fast_gram = _lars_fast_gram()
            if fast_gram:
                grid_gram = False
                lars_pre = False
    # lane/apple-fast-kernel (2026-10-02), FAST on Apple only: the means, the
    # centered Gram, X'y from x_linear/fast_gram.mojo (the gram lane's shared
    # grid Gram, chunked tiles on the grid) instead of `xg_means_kernel` +
    # `xg_gram_kernel` (one serial million-row chain per cell) and the team's
    # own X'y / mean / variance passes on one block (after the 2026-10-02
    # merge: in place of main's moments grid, one block per 16-column tile
    # pair with one serial chain per cell, and main's `bayes_xty_kernel`; the
    # y partials stay main's). Taken under BAYES_CLS1_STATS (the KEPT
    # lane/apple-fast-gap-cls1 default); its own opt-in define
    # MOJOLEARN_KERNEL_FAST_BAYES_STATS and the Jacobi threshold arm
    # MOJOLEARN_KERNEL_FAST_BAYES_JACOBI were dropped (lane/apple-fast-kernel
    # @ 9e851777c).
    var kstats = False
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
        if bayes_like:
            # STATS: BayesianRidge (unweighted) on main's grid driver only;
            # ARD keeps main's moments grid (its X'y pass is the team's)
            # lane/apple-fast-gap-cls1: BAYES_CLS1_STATS (FAST + Apple default) turns the same path on
            if algo == ALGO_BAYES and n > 0 and d > 0 and BAYES_CLS1_STATS:
                kstats = True
                grid_gram = True
                hip[4] = Int32(1)
    var dip = ctx.enqueue_create_buffer[DType.int32](max(len(hip), 1))
    var dtw = ctx.enqueue_create_buffer[DType.float32](team_work(0, 3, 0))
    var hfp = fp.copy()
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    comptime if XLIN_IDN_DEV_FINITE:
        xlin_finite_device(ctx, dx, n_x, y, n_y)
    if algo == ALGO_LOGCV and n > 0:
        # the StratifiedKFold ids from the device labels (x_linear/logcv_grid.mojo)
        lcv_fold_ids_device(ctx.copy(), FP(unsafe_from_address=Int(dy.unsafe_ptr())), n, max(Int(ip[2]), 2), Int(ip[4]))
    if len(hip) > 0:
        ctx.enqueue_copy(dst_buf=dip, src_ptr=hip.unsafe_ptr())
    if len(hfp) > 0:
        ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    # lane/neural-pass131 (2026-10-02): the setup (zeroed scratch, the grid
    # Gram, then the prep or the whole one-block fit) runs under the Metal
    # completion witness (x_linear/witness.mojo) as ONE unit: the fit and the
    # prep work in place, so a cut launch reruns the unit from its zeros.
    var cells = d * (d + 1) // 2
    var rows_slice = n
    comptime if has_apple_gpu_accelerator():
        rows_slice = max(64, min(n, (XL_APPLE_SLICE_MACS // max(cells, 1)) // 64 * 64))
    var nslices = (n + rows_slice - 1) // rows_slice if n > 0 else 1
    var dgs = ctx.enqueue_create_buffer[DType.float32](max(d + d * d, 1) if grid_gram and nslices > 1 else 1)
    var bayes_grid = False
    # unweighted, with the grid Gram: every row pass on grid kernels
    bayes_grid = algo == ALGO_BAYES and n > 0 and d > 0 and grid_gram
    # cgr-linear: a weighted BayesianRidge on the grid too (its statistics in
    # the blocked order, then the same eigen prep and iterations)
    var bayes_w = False
    bayes_w = algo == ALGO_BAYES and n > 0 and d > 0 and len(ip) > 2 and ip[2] != 0
    if bayes_w:
        bayes_grid = True
    # lane/apple-fast-kernel BAYES_STATS: unweighted BayesianRidge on the grid only
    kstats = kstats and bayes_grid and not bayes_w
    var ynb = fold_blocks(n)
    var prep_blocks = 2 * _xg_blocks(ynb) + _xg_blocks(d) + 1
    comptime if BAYES_CLS1_PARTS:
        prep_blocks = 2 * ynb + _xg_blocks(d) + 1
    if bayes_w:
        prep_blocks = 2 * _xg_blocks(ynb) + _xg_blocks(d * ynb) + _xg_blocks(d) + _xg_blocks((cells + d) * ynb) + _xg_blocks(cells + d) + 1
    var dwparts = ctx.enqueue_create_buffer[DType.float32](max(ynb, 1) if bayes_w else 1)
    var dmparts = ctx.enqueue_create_buffer[DType.float32](max(d * ynb, 1) if bayes_w else 1)
    var dgparts = ctx.enqueue_create_buffer[DType.float32](max((cells + d) * ynb, 1) if bayes_w else 1)
    var wit = Witness(ctx, max(max(_xg_blocks(max(cells, d)), 1) + 1, prep_blocks))
    var dstate = ctx.enqueue_create_buffer[DType.float32](C1_BAYES_STATE)
    var dyparts = ctx.enqueue_create_buffer[DType.float32](max(ynb, 1) if bayes_grid else 1)
    var dvparts = ctx.enqueue_create_buffer[DType.float32](max(ynb, 1) if bayes_grid else 1)
    var setup_tries = 0
    while True:
        dout.enqueue_fill(Float32(0))
        dfw.enqueue_fill(Float32(0))
        diw.enqueue_fill(Int32(0))
        dtw.enqueue_fill(Float32(0))
        var good = True
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
            if fast_gram:
                # lane/apple-fast-gram: the shared grid Gram (row chunks x 32 x
                # 32 tiles, x_linear/fast_gram.mojo) fills the words the moments
                # grid would (lars_pre is off); it waits for its
                # own launches (unwitnessed), the witnessed fit follows
                var pfw = FP(unsafe_from_address=Int(dfw.unsafe_ptr()))
                var px = FP(unsafe_from_address=Int(dx.unsafe_ptr()))
                var py = FP(unsafe_from_address=Int(dy.unsafe_ptr()))
                # lars_fit's fw: xm 0 | G d | xty d + d*d | prev = 2d + d*d
                # holds the y mean (lars_fit zeroes it)
                var xty_o = d + d * d
                fast_gram_into(ctx, px, py, 0, n, d, 1, hip[1] != 0, pfw, pfw + (xty_o + d), pfw + d, pfw + xty_o)
            elif kstats:
                # BayesianRidge on main's grid driver: xm at 0, G at d, X'y at
                # d + d*d; the y mean parked at the `old` offset (3d + 2d^2),
                # scratch until the first step (the driver's own y partials
                # give it ym). Waits for its own launches (unwitnessed); the
                # witnessed eig unit follows
                var pfw = FP(unsafe_from_address=Int(dfw.unsafe_ptr()))
                var px = FP(unsafe_from_address=Int(dx.unsafe_ptr()))
                var py = FP(unsafe_from_address=Int(dy.unsafe_ptr()))
                fast_gram_into(ctx, px, py, 0, n, d, 1, hip[1] != 0, pfw, pfw + (3 * d + 2 * d * d), pfw + d, pfw + (d + d * d))
        if lars_pre:
            # lane/neural-pass120's moments of [X | y] (fw: xm 0, G d, X'y
            # d + d*d, y's mean parked in prev = 2d + d*d)
            ctx.enqueue_function[mg_means_kernel](
                dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(1), Int32(hip[1]), dfw.unsafe_ptr(),
                Int32(0), Int32(2 * d + d * d), grid_dim=mg_tiles(d, 1), block_dim=MG_NT,
            )
            var tl = mg_tiles(d, 1)
            ctx.enqueue_function[mg_cross_kernel](
                dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(1), dfw.unsafe_ptr(),
                Int32(0), Int32(2 * d + d * d), Int32(d), Int32(d + d * d), grid_dim=tl * (tl + 1) // 2, block_dim=MG_NT,
            )
        elif kstats:
            pass
        elif grid_gram and bayes_like:
            # lane/neural-pass130: BayesianRidge / ARD's means and centered Gram
            # from the staged moments kernels (no Y columns), into the layout
            # `xg_gram_kernel` fills (xm at 0, G at d): the same chains, staged
            var tlb = mg_tiles(d, 0)
            ctx.enqueue_function[mg_means_kernel](
                dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(0), Int32(hip[1]), dfw.unsafe_ptr(),
                Int32(0), Int32(0), grid_dim=tlb, block_dim=MG_NT,
            )
            ctx.enqueue_function[mg_cross_kernel](
                dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(0), dfw.unsafe_ptr(),
                Int32(0), Int32(0), Int32(d), Int32(0), grid_dim=tlb * (tlb + 1) // 2, block_dim=MG_NT,
            )
        elif grid_gram:
            # The means then the centered Gram into fw[0, d + d*d), the layout
            # `lars_fit` reads (xm at 0, G at d); the team recomputes the means
            # itself (the same statements, the same values) and skips the Gram.
            var nonce = wit.begin()
            ctx.enqueue_function[xg_means_kernel](
                dx.unsafe_ptr(), Int32(n), Int32(d), Int32(hip[1]), dfw.unsafe_ptr(), wit.p(), Int32(0), nonce,
                grid_dim=_xg_blocks(d), block_dim=XG_TPB,
            )
            good = wit.ok(ctx, _xg_blocks(d), "means")
            # row slices ping-pong between fw and dgs so a rerun slice reads
            # what its predecessor wrote; the last one writes fw
            var si = 0
            var lo = 0
            while good and lo < n:
                var cnt = min(rows_slice, n - lo)
                var to_fw = (nslices - 1 - si) % 2 == 0
                var dst = dfw.unsafe_ptr() if to_fw else dgs.unsafe_ptr()
                var src = dgs.unsafe_ptr() if to_fw else dfw.unsafe_ptr()
                var t = 0
                while True:
                    var nc = wit.begin()
                    ctx.enqueue_function[xg_gram_kernel](
                        dx.unsafe_ptr(), Int32(n), Int32(d), dfw.unsafe_ptr(), Int32(lo), Int32(cnt),
                        FP(unsafe_from_address=Int(src)), FP(unsafe_from_address=Int(dst)), wit.p(), Int32(0), nc,
                        grid_dim=_xg_blocks(cells), block_dim=XG_TPB,
                    )
                    if wit.ok(ctx, _xg_blocks(cells), "Gram slice"):
                        break
                    t += 1
                    if t >= WITNESS_TRIES:
                        wit.fail()
                lo += cnt
                si += 1
        if good:
            var nonce = wit.begin()
            var units = 1
            if bayes_grid:
                # the prep's row passes on the grid (the target's blocked sum
                # and variance partials, X'y one column a thread), then its
                # d x d half on one block team
                var wo = 0
                if bayes_w:
                    ctx.enqueue_function[bayes_wparts_kernel](
                        dy.unsafe_ptr(), Int32(n), dyparts.unsafe_ptr(), dwparts.unsafe_ptr(),
                        wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(ynb), block_dim=XG_TPB,
                    )
                    wo += _xg_blocks(ynb)
                    ctx.enqueue_function[bayes_wvar_parts_kernel](
                        dy.unsafe_ptr(), Int32(n), dyparts.unsafe_ptr(), dwparts.unsafe_ptr(), dvparts.unsafe_ptr(),
                        dstate.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(ynb), block_dim=XG_TPB,
                    )
                    wo += _xg_blocks(ynb)
                    ctx.enqueue_function[bayes_wx_parts_kernel](
                        dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dmparts.unsafe_ptr(),
                        wit.p(), Int32(wo), nonce, grid_dim=max(_xg_blocks(d * ynb), 1), block_dim=XG_TPB,
                    )
                    wo += max(_xg_blocks(d * ynb), 1)
                    ctx.enqueue_function[bayes_wmeans_kernel](
                        Int32(n), Int32(d), Int32(hip[1]), dmparts.unsafe_ptr(), dwparts.unsafe_ptr(), dfw.unsafe_ptr(),
                        wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(d), block_dim=XG_TPB,
                    )
                    wo += _xg_blocks(d)
                    ctx.enqueue_function[bayes_wgram_parts_kernel](
                        dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(hip[1]), dfw.unsafe_ptr(),
                        dyparts.unsafe_ptr(), dwparts.unsafe_ptr(), dgparts.unsafe_ptr(),
                        wit.p(), Int32(wo), nonce, grid_dim=max(_xg_blocks((cells + d) * ynb), 1), block_dim=XG_TPB,
                    )
                    wo += max(_xg_blocks((cells + d) * ynb), 1)
                    ctx.enqueue_function[bayes_wgram_fin_kernel](
                        Int32(n), Int32(d), dgparts.unsafe_ptr(), dfw.unsafe_ptr(),
                        wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(cells + d), block_dim=XG_TPB,
                    )
                    wo += _xg_blocks(cells + d)
                else:
                    var c1p = False
                    comptime if BAYES_CLS1_PARTS:
                        # lane/apple-fast-gap-cls1: a block per row block
                        c1p = True
                        ctx.enqueue_function[c1_sum_parts_kernel](
                            dy.unsafe_ptr(), Int32(n), dyparts.unsafe_ptr(), dstate.unsafe_ptr(), Int32(3),
                            wit.p(), Int32(wo), nonce, grid_dim=ynb, block_dim=C1_TPB,
                        )
                        wo += ynb
                        ctx.enqueue_function[c1_dev_parts_kernel](
                            dy.unsafe_ptr(), Int32(n), dyparts.unsafe_ptr(), dvparts.unsafe_ptr(),
                            wit.p(), Int32(wo), nonce, grid_dim=ynb, block_dim=C1_TPB,
                        )
                        wo += ynb
                    if not c1p:
                        ctx.enqueue_function[bayes_yparts_kernel](
                            dy.unsafe_ptr(), Int32(n), dyparts.unsafe_ptr(), dstate.unsafe_ptr(),
                            wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(ynb), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(ynb)
                        ctx.enqueue_function[bayes_yvar_parts_kernel](
                            dy.unsafe_ptr(), Int32(n), dyparts.unsafe_ptr(), dvparts.unsafe_ptr(),
                            wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(ynb), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(ynb)
                    if not kstats:
                        # BAYES_STATS: X'y came from the fast grid Gram
                        ctx.enqueue_function[bayes_xty_kernel](
                            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(hip[1]), dfw.unsafe_ptr(),
                            dyparts.unsafe_ptr(), wit.p(), Int32(wo), nonce, grid_dim=_xg_blocks(d), block_dim=XG_TPB,
                        )
                        wo += _xg_blocks(d)
                ctx.enqueue_function[bayes_eig_kernel](
                    dfw.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(), dip.unsafe_ptr(), Int32(d), Int32(ynb),
                    dyparts.unsafe_ptr(), dvparts.unsafe_ptr(), dtw.unsafe_ptr(), dstate.unsafe_ptr(),
                    wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=LINEAR_TPB,
                )
                units = wo + 1
            else:
                ctx.enqueue_function[lars_path_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(d), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(),
                    dfw.unsafe_ptr(), diw.unsafe_ptr(), dtw.unsafe_ptr(), wit.p(), Int32(0), nonce,
                    grid_dim=1, block_dim=LINEAR_TPB,
                )
            good = wit.ok(ctx, units, "fit")
        if good:
            break
        setup_tries += 1
        if setup_tries >= WITNESS_TRIES:
            wit.fail()
    if bayes_grid:
        var drows = ctx.enqueue_create_buffer[DType.float32](n)
        var dparts = ctx.enqueue_create_buffer[DType.float32](max(fold_blocks(n), 1))
        var hst = List[Float32](length=C1_BAYES_STATE, fill=Float32(0))
        # lane/apple-fast-gap-cls1 BAYES_CLS1_PARTS: the iteration's partials
        # a block per row block (unweighted fits)
        var c1parts = False
        comptime if BAYES_CLS1_PARTS:
            c1parts = Int(hip[2]) == 0 if len(hip) > 2 else True
        var pblocks = fold_blocks(n) if c1parts else _xg_blocks(fold_blocks(n))
        var wit2 = Witness(ctx, max(_xg_blocks(n) + pblocks, _xg_blocks(d)) + 1)
        var max_iter = Int(hip[0])
        var sw = Int(hip[2]) if len(hip) > 2 else 0
        var gram_sse = False
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator():
            gram_sse = sw == 0 and len(hip) > 5 and Int(hip[5]) != 0
        # FAST on Apple, the default (lane/apple-fast-bayes; `_GUARD_OFF` turns it off):
        # the Gram sse guarded by a reference row pass (`bayes_step_guard_kernel`)
        # takes over gram_sse (no yy pass, no `bayes_step_gram_kernel`); n >= d
        # for z0 in the sse scratch
        var guard = False
        comptime if BAYES_GRID_GUARD:
            guard = gram_sse and n >= d
            if guard:
                gram_sse = False
        if gram_sse:
            # FAST on Apple: yy once on the grid (`bayes_step_gram_kernel`)
            var tr = 0
            while True:
                var nonce = wit2.begin()
                ctx.enqueue_function[bayes_yy_parts_kernel](
                    dy.unsafe_ptr(), Int32(n), dstate.unsafe_ptr(), dparts.unsafe_ptr(),
                    wit2.p(), Int32(0), nonce, grid_dim=_xg_blocks(fold_blocks(n)), block_dim=XG_TPB,
                )
                if wit2.ok(ctx, _xg_blocks(fold_blocks(n)), "Bayes yy"):
                    break
                tr += 1
                if tr >= WITNESS_TRIES:
                    wit2.fail()
        var iters = 0
        var dev_iters = False
        comptime if BAYES_CLS1_BATCH:
            if guard:
                # lane/apple-fast-gap-cls1: C1_BATCH guarded iterations per
                # witness check; the row pass, the step and the count gate on
                # the device's state words; one synchronize per batch (the
                # state read rides on the witness wait). A cut launch raises
                # (the step updates in place), as a cut step does on main.
                var g_r = _xg_blocks(n)
                var g_p = fold_blocks(n)
                var per = g_r + g_p + 1
                var wit3 = Witness(ctx, C1_BATCH * per + 1)
                ctx.enqueue_function[c1_bayes_state_init_kernel](dstate.unsafe_ptr(), grid_dim=1, block_dim=1)
                var done_it = 0
                while done_it < max_iter:
                    var nbt = min(C1_BATCH, max_iter - done_it)
                    var nc = wit3.begin()
                    var wo = 0
                    for _b in range(nbt):
                        ctx.enqueue_function[c1_bayes_resid_kernel](
                            dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dfw.unsafe_ptr(), dout.unsafe_ptr(),
                            dstate.unsafe_ptr(), drows.unsafe_ptr(), wit3.p(), Int32(wo), nc,
                            grid_dim=g_r, block_dim=XG_TPB,
                        )
                        wo += g_r
                        ctx.enqueue_function[c1_sq_parts_kernel](
                            drows.unsafe_ptr(), Int32(n), dstate.unsafe_ptr(), Int32(5), Int32(6), Int32(0),
                            dparts.unsafe_ptr(), wit3.p(), Int32(wo), nc, grid_dim=g_p, block_dim=C1_TPB,
                        )
                        wo += g_p
                        ctx.enqueue_function[c1_bayes_step_kernel](
                            dfw.unsafe_ptr(), dout.unsafe_ptr(), dfp.unsafe_ptr(), Int32(d), Int32(ynb), dparts.unsafe_ptr(),
                            dstate.unsafe_ptr(), wit3.p(), Int32(wo), nc, grid_dim=1, block_dim=1,
                        )
                        wo += 1
                    ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=dstate.create_sub_buffer[DType.float32](0, 8))
                    if not wit3.ok(ctx, wo, "Bayes batch"):
                        wit3.fail()
                    done_it += nbt
                    if hst[5] != Float32(0):
                        break
                _ = wit3^
                dev_iters = True
                max_iter = 0  # the loops below ran here
        comptime if BAYES_GRID_GUARD:
            if guard and not dev_iters:
                # the first iteration makes the row pass (no reference yet);
                # after each step the stop word's read also brings the verdict
                # on the next iteration's Gram sse (state[6])
                var fresh = True
                for it in range(max_iter):
                    iters = it + 1
                    if fresh:
                        var tr = 0
                        while True:
                            var nonce = wit2.begin()
                            ctx.enqueue_function[bayes_resid_kernel](
                                dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dfw.unsafe_ptr(), dout.unsafe_ptr(),
                                dstate.unsafe_ptr(), drows.unsafe_ptr(), wit2.p(), Int32(0), nonce,
                                grid_dim=_xg_blocks(n), block_dim=XG_TPB,
                            )
                            if c1parts:
                                ctx.enqueue_function[c1_sq_parts_kernel](
                                    drows.unsafe_ptr(), Int32(n), dstate.unsafe_ptr(), Int32(-1), Int32(0), Int32(0),
                                    dparts.unsafe_ptr(), wit2.p(), Int32(_xg_blocks(n)), nonce,
                                    grid_dim=fold_blocks(n), block_dim=C1_TPB,
                                )
                            else:
                                ctx.enqueue_function[bayes_part_kernel](
                                    drows.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(sw), dparts.unsafe_ptr(),
                                    wit2.p(), Int32(_xg_blocks(n)), nonce, grid_dim=_xg_blocks(fold_blocks(n)), block_dim=XG_TPB,
                                )
                            if wit2.ok(ctx, _xg_blocks(n) + pblocks, "Bayes residuals"):
                                break
                            tr += 1
                            if tr >= WITNESS_TRIES:
                                wit2.fail()
                    var ng = wit2.begin()
                    ctx.enqueue_function[bayes_step_guard_kernel](
                        dfw.unsafe_ptr(), dout.unsafe_ptr(), dfp.unsafe_ptr(), Int32(d), Int32(ynb), dparts.unsafe_ptr(),
                        dstate.unsafe_ptr(), Int32(it), Int32(1 if fresh else 0), wit2.p(), Int32(0), ng,
                        grid_dim=1, block_dim=1,
                    )
                    if not wit2.ok(ctx, 1, "Bayes step"):
                        wit2.fail()
                    ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=dstate)
                    ctx.synchronize()
                    if hst[5] != Float32(0):
                        break
                    fresh = hst[6] == Float32(0)
                max_iter = 0  # the loop below ran here
        for it in range(max_iter):
            iters = it + 1
            if gram_sse:
                var ng = wit2.begin()
                ctx.enqueue_function[bayes_step_gram_kernel](
                    dfw.unsafe_ptr(), dout.unsafe_ptr(), dfp.unsafe_ptr(), Int32(d), Int32(ynb), dparts.unsafe_ptr(),
                    dstate.unsafe_ptr(), Int32(it), wit2.p(), Int32(0), ng, grid_dim=1, block_dim=1,
                )
                if not wit2.ok(ctx, 1, "Bayes step"):
                    wit2.fail()
                ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=dstate)
                ctx.synchronize()
                if hst[5] != Float32(0):
                    break
                continue
            # the residuals and partials rebuild from the coefficients: rerun
            # on a cut; the step updates in place: a cut raises
            var tr = 0
            while True:
                var nonce = wit2.begin()
                ctx.enqueue_function[bayes_resid_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), dfw.unsafe_ptr(), dout.unsafe_ptr(),
                    dstate.unsafe_ptr(), drows.unsafe_ptr(), wit2.p(), Int32(0), nonce,
                    grid_dim=_xg_blocks(n), block_dim=XG_TPB,
                )
                if c1parts:
                    ctx.enqueue_function[c1_sq_parts_kernel](
                        drows.unsafe_ptr(), Int32(n), dstate.unsafe_ptr(), Int32(-1), Int32(0), Int32(0),
                        dparts.unsafe_ptr(), wit2.p(), Int32(_xg_blocks(n)), nonce,
                        grid_dim=fold_blocks(n), block_dim=C1_TPB,
                    )
                else:
                    ctx.enqueue_function[bayes_part_kernel](
                        drows.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(sw), dparts.unsafe_ptr(),
                        wit2.p(), Int32(_xg_blocks(n)), nonce, grid_dim=_xg_blocks(fold_blocks(n)), block_dim=XG_TPB,
                    )
                if wit2.ok(ctx, _xg_blocks(n) + pblocks, "Bayes residuals"):
                    break
                tr += 1
                if tr >= WITNESS_TRIES:
                    wit2.fail()
            var ns = wit2.begin()
            ctx.enqueue_function[bayes_step_kernel](
                dfw.unsafe_ptr(), dout.unsafe_ptr(), dfp.unsafe_ptr(), Int32(d), Int32(ynb), dparts.unsafe_ptr(),
                dstate.unsafe_ptr(), Int32(it), wit2.p(), Int32(0), ns, grid_dim=1, block_dim=1,
            )
            ctx.enqueue_function[bayes_coef_kernel](
                dfw.unsafe_ptr(), dout.unsafe_ptr(), Int32(d), dstate.unsafe_ptr(), wit2.p(), Int32(1), ns,
                grid_dim=_xg_blocks(d), block_dim=XG_TPB,
            )
            if not wit2.ok(ctx, 1 + _xg_blocks(d), "Bayes step"):
                wit2.fail()
            ctx.enqueue_copy(dst_ptr=hst.unsafe_ptr(), src_buf=dstate)
            ctx.synchronize()
            if hst[5] != Float32(0):
                break
        var nf = wit2.begin()
        if dev_iters:
            ctx.enqueue_function[c1_bayes_finish_kernel](
                dfw.unsafe_ptr(), dout.unsafe_ptr(), Int32(d), Int32(hip[1]), dstate.unsafe_ptr(),
                wit2.p(), Int32(0), nf, grid_dim=1, block_dim=1,
            )
        else:
            ctx.enqueue_function[bayes_finish_kernel](
                dfw.unsafe_ptr(), dout.unsafe_ptr(), Int32(d), Int32(hip[1]), dstate.unsafe_ptr(), Int32(iters),
                wit2.p(), Int32(0), nf, grid_dim=1, block_dim=1,
            )
        if not wit2.ok(ctx, 1, "Bayes finish"):
            wit2.fail()
        ctx.synchronize()
        _ = wit2^
        _ = drows^
        _ = dparts^
        _ = hst^
    if n_out > 0:
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.synchronize()
    _ = hip^
    _ = hfp^
    _ = dx^
    _ = dy^
    _ = dip^
    _ = dfp^
    _ = dout^
    _ = dfw^
    _ = diw^
    _ = dtw^
    _ = dgs^
    _ = dstate^
    _ = dyparts^
    _ = dvparts^
    _ = dwparts^
    _ = dmparts^
    _ = dgparts^
    _ = wit^
    _ = ctx^


def decision_codes_kernel(s: FP, n: Int32, k: Int32, strict: Int32, below: Int32, above: Int32, codes: IP):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n):
        codes.unsafe_store(i, decision_code_row(s, i, Int(k), Int(strict), Int(below), Int(above)))


def decision_codes_device(
    x: FP, wb: FP, n: Int, d: Int, k: Int, link: Int, strict: Int, below: Int, above: Int, codes: IP
) raises:
    """`decision_device`'s block, then each row's class code on the device
    (`decision_code_row`); only the n int32 codes come down."""
    var ctx = linear_ctx()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n * d, 1))
    var dwb = ctx.enqueue_create_buffer[DType.float32](k * (d + 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n * k, 1))
    var dc = ctx.enqueue_create_buffer[DType.int32](max(n, 1))
    if n * d > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dwb, src_ptr=wb)
    if n > 0:
        ctx.enqueue_function[decision_kernel](
            dx.unsafe_ptr(), dwb.unsafe_ptr(), Int32(n), Int32(d), Int32(k), Int32(link), dout.unsafe_ptr(),
            grid_dim=(n * k + 127) // 128, block_dim=128,
        )
        ctx.enqueue_function[decision_codes_kernel](
            dout.unsafe_ptr(), Int32(n), Int32(k), Int32(strict), Int32(below), Int32(above), dc.unsafe_ptr(),
            grid_dim=(n + 127) // 128, block_dim=128,
        )
        ctx.enqueue_copy(dst_ptr=codes, src_buf=dc.create_sub_buffer[DType.int32](0, n))
    ctx.synchronize()
    _ = dx^
    _ = dwb^
    _ = dout^
    _ = dc^
    _ = ctx^


def decision_device(x: FP, wb: FP, n: Int, d: Int, k: Int, link: Int, res: FP) raises:
    var ctx = linear_ctx()
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n * d, 1))
    var dwb = ctx.enqueue_create_buffer[DType.float32](k * (d + 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n * k, 1))
    if n * d > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    ctx.enqueue_copy(dst_buf=dwb, src_ptr=wb)
    var total = n * k
    if total > 0:
        ctx.enqueue_function[decision_kernel](
            dx.unsafe_ptr(), dwb.unsafe_ptr(), Int32(n), Int32(d), Int32(k), Int32(link), dout.unsafe_ptr(),
            grid_dim=(total + 127) // 128, block_dim=128,
        )
        ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.synchronize()
    _ = dx^
    _ = dwb^
    _ = dout^
    _ = ctx^
