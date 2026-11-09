# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SpectralClustering's discretize on the device in float-float (lane
n9b-device-svd-qr, 2026-11-09): x_cluster/spectral_df64.mojo's bodies as
kernels, the same order as its host column `discretize_df64_host`.

Every step is a grid over its independent units (cells, rows, row chunks,
class-sum cells); the k x k SVD with its stop test and the next rotation is
ONE block (`sdk_step_kernel`: each cyclic rotation reads the previous one's
words; the rows of a rotation and the cells of V U^T run across the block).
The iterations are gated on the device's done word, so the driver enqueues
SD_CHUNK of them behind ONE read of the four state words (done, n_iter,
status, argmin); the prep and the greedy start read nothing.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace

from core.device_zero import enqueue_fill
from x_cluster.bodies import SplitMix64
from x_cluster.spectral_df64 import (
    SD_AMIN, SD_BAD_BASIS, SD_DONE, SD_ITER, SD_ROWS, SD_STATE_WORDS, SD_STATUS, SD_SVD_SWEEPS,
    sd_apply_row, sd_better_min, sd_cacc_row, sd_chunks, sd_class_fold, sd_class_part, sd_colnorm, sd_colscale_cell,
    sd_colsign_cell, sd_colsq_part, sd_complete_u, sd_dot_cols, sd_ld, sd_label_row, sd_load_cell, sd_ncut_step,
    sd_norm_col, sd_of_int, sd_polar_cell, sd_raise_status, sd_rotation, sd_rownorm_row, sd_st, sd_take_row_cell,
    sd_zero, sd_label_abs_row, cq_apply_col, cq_chunk_max, cq_colnorm_cell, cq_gather_cell, cq_pivot_step,
    cq_raise_status,
)
from x_decomp.jacobi2 import dev_barrier
from x_linear.ff import FF, ff_sqrt
from x_linear.ops import FP, IP

comptime SD_TPB = 256
comptime SD_CHUNK = 8
"""Iterations enqueued behind one read of the state words."""


@always_inline
def _g() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


@always_inline
def _blocks(count: Int) -> Int:
    return (count + SD_TPB - 1) // SD_TPB if count > 0 else 1


@always_inline
def _fp(buf: DeviceBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _ip(buf: DeviceBuffer[DType.int32]) -> IP:
    return IP(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _done(state: IP) -> Bool:
    return state.unsafe_load(SD_DONE) != Int32(0)


# ---------------------------------------------------------------- the prep


def sdk_load_kernel(x: FP, vh: FP, vl: FP, count: Int32):
    var t = _g()
    if t < Int(count):
        sd_load_cell(x, vh, vl, t)


def sdk_colsq_kernel(vh: FP, vl: FP, n_in: Int32, k_in: Int32, ph: FP, pl: FP, cells: Int32):
    var t = _g()
    var k = Int(k_in)
    if t < Int(cells):
        sd_colsq_part(vh, vl, Int(n_in), k, t // k, t - (t // k) * k, ph, pl)


def sdk_colnorm_kernel(ph: FP, pl: FP, nch: Int32, k_in: Int32, cnh: FP, cnl: FP, state: IP):
    var c = _g()
    if c < Int(k_in):
        sd_colnorm(ph, pl, Int(nch), Int(k_in), c, cnh, cnl, state)


def sdk_colscale_kernel(vh: FP, vl: FP, k_in: Int32, cnh: FP, cnl: FP, rh: Float32, rl: Float32, sgn: FP, count: Int32):
    var t = _g()
    if t < Int(count):
        sd_colscale_cell(vh, vl, Int(k_in), cnh, cnl, FF(rh, rl), sgn, t)


def sdk_colsign_kernel(vh: FP, vl: FP, k_in: Int32, sgn: FP, count: Int32):
    var t = _g()
    if t < Int(count):
        sd_colsign_cell(vh, vl, Int(k_in), sgn, t)


def sdk_rownorm_kernel(vh: FP, vl: FP, k_in: Int32, state: IP, n_in: Int32):
    var i = _g()
    if i < Int(n_in):
        sd_rownorm_row(vh, vl, Int(k_in), i, state)


def sdk_fail_done_kernel(state: IP):
    """One thread: a refused embedding (status != 0) marks the fit done, so
    no iteration runs on it; the driver raises at its first read."""
    if _g() == 0 and state.unsafe_load(SD_STATUS) != Int32(0):
        state.unsafe_store(SD_DONE, Int32(1))


# ---------------------------------------------------------------- the greedy start


def sdk_take_row_kernel(vh: FP, vl: FP, k_in: Int32, row_in: Int32, state: IP, roth: FP, rotl: FP, col: Int32):
    """rot[:, col] = vec[row, :]; row_in < 0: the argmin word's row."""
    var r = _g()
    if r >= Int(k_in):
        return
    var row = Int(row_in) if row_in >= Int32(0) else Int(state.unsafe_load(SD_AMIN))
    sd_take_row_cell(vh, vl, Int(k_in), row, roth, rotl, Int(col), r)


def sdk_cacc_kernel(vh: FP, vl: FP, k_in: Int32, roth: FP, rotl: FP, col: Int32, cah: FP, cal: FP, n_in: Int32):
    var i = _g()
    if i < Int(n_in):
        sd_cacc_row(vh, vl, Int(k_in), roth, rotl, Int(col), cah, cal, i)


def sdk_chunk_min_kernel(cah: FP, cal: FP, n_in: Int32, nch: Int32, bmh: FP, bml: FP, bmi: IP):
    """Thread ch: chunk ch's (value, index) minimum, rows ascending (the
    host column's `cmin`)."""
    var ch = _g()
    if ch >= Int(nch):
        return
    var n = Int(n_in)
    var bi = -1
    var bv = sd_zero()
    for i in range(ch * SD_ROWS, min(ch * SD_ROWS + SD_ROWS, n)):
        var v = sd_ld(cah, cal, i)
        if sd_better_min(v, i, bv, bi):
            bv = v
            bi = i
    sd_st(bmh, bml, ch, bv)
    bmi.unsafe_store(ch, Int32(bi))


def sdk_min_fold_kernel(bmh: FP, bml: FP, bmi: IP, nch_in: Int32, state: IP):
    """One thread: the chunk minima to the argmin word (exact compares)."""
    if _g() != 0:
        return
    var bi = -1
    var bv = sd_zero()
    for ch in range(Int(nch_in)):
        var v = sd_ld(bmh, bml, ch)
        var vi = Int(bmi.unsafe_load(ch))
        if sd_better_min(v, vi, bv, bi):
            bv = v
            bi = vi
    state.unsafe_store(SD_AMIN, Int32(bi))


# ---------------------------------------------------------------- the iterations


def sdk_label_kernel(vh: FP, vl: FP, k_in: Int32, roth: FP, rotl: FP, lab: IP, state: IP, n_in: Int32):
    var i = _g()
    if i >= Int(n_in) or _done(state):
        return
    lab.unsafe_store(i, sd_label_row(vh, vl, Int(k_in), roth, rotl, i))


def sdk_class_part_kernel(vh: FP, vl: FP, lab: IP, n_in: Int32, k_in: Int32, ph: FP, pl: FP, state: IP, cells: Int32):
    var t = _g()
    if t >= Int(cells) or _done(state):
        return
    var kk = Int(k_in) * Int(k_in)
    sd_class_part(vh, vl, lab, Int(n_in), Int(k_in), t // kk, t - (t // kk) * kk, ph, pl)


def sdk_class_fold_kernel(ph: FP, pl: FP, nch: Int32, kk_in: Int32, mh: FP, ml: FP, state: IP):
    var t = _g()
    if t >= Int(kk_in) or _done(state):
        return
    sd_class_fold(ph, pl, Int(nch), Int(kk_in), t, mh, ml)


@always_inline
def _sdk_svd_block(
    mh: FP, ml: FP, vvh: FP, vvl: FP, uh: FP, ul: FP, svh: FP, svl: FP, filled: IP, k: Int, tid: Int
):
    """The one-block one-sided Jacobi SVD of m (k x k, rotated in place; v =
    I rotated) in `_sd_svd_host`'s order (the pair's three sums on threads
    0..2, each its rows ascending; the rows of the rotation across the
    block), then sv and U's filled columns; every thread past the last
    barrier. Shared by `sdk_step_kernel` (discretize) and
    `cqk_svd_kernel` (cluster_qr): a launched kernel is never called from
    another kernel."""
    var kk = k * k
    var sh = stack_allocation[8, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var flag = stack_allocation[1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var t = tid
    while t < kk:
        var i = t // k
        sd_st(vvh, vvl, t, FF(Float32(1) if i * k + i == t else Float32(0), Float32(0)))
        t += SD_TPB
    dev_barrier()
    for _sweep in range(SD_SVD_SWEEPS):
        if tid == 0:
            flag[0] = Int32(0)
        dev_barrier()
        for p in range(k - 1):
            for q in range(p + 1, k):
                if tid < 3:
                    var a = p if tid != 1 else q
                    var b = p if tid == 0 else q
                    var d = sd_dot_cols(mh, ml, k, a, b)
                    sh[2 * tid] = d.hi
                    sh[2 * tid + 1] = d.lo
                dev_barrier()
                var rot = sd_rotation(FF(sh[0], sh[1]), FF(sh[2], sh[3]), FF(sh[4], sh[5]))
                if rot[0]:
                    var r = tid
                    while r < k:
                        sd_apply_row(mh, ml, vvh, vvl, k, p, q, rot[1], rot[2], r)
                        r += SD_TPB
                    if tid == 0:
                        flag[0] = Int32(1)
                dev_barrier()
        var rotated = flag[0] != Int32(0)
        dev_barrier()
        if not rotated:
            break
    var j = tid
    while j < k:
        sd_norm_col(mh, ml, uh, ul, svh, svl, filled, k, j)
        j += SD_TPB
    dev_barrier()


def sdk_step_kernel(
    mh: FP, ml: FP, vvh: FP, vvl: FP, uh: FP, ul: FP, svh: FP, svl: FP, filled: IP, ch: FP, cl: FP,
    lasth: FP, roth: FP, rotl: FP, state: IP, k_in: Int32, n_in: Int32, n_iter_max: Int32,
):
    """ONE block: n_iter + 1, the one-sided Jacobi SVD of the class sums m
    (k x k, rotated in place; v = I rotated) in `_sd_svd_host`'s order (the
    pair's three sums on threads 0..2, each its rows ascending; the rows of
    the rotation across the block), sv and U, the completion on thread 0,
    the ncut stop test on thread 0, then the next rotation V U^T across the
    block. Returns at once once the fit is done."""
    if _done(state):
        return
    var k = Int(k_in)
    var kk = k * k
    var tid = Int(thread_idx.x)
    var flag = stack_allocation[2, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    if tid == 0:
        state.unsafe_store(SD_ITER, state.unsafe_load(SD_ITER) + 1)
    _sdk_svd_block(mh, ml, vvh, vvl, uh, ul, svh, svl, filled, k, tid)
    if tid == 0:
        var stop = False
        if not sd_complete_u(uh, ul, filled, ch, cl, k):
            state.unsafe_store(SD_STATUS, Int32(SD_BAD_BASIS))
            state.unsafe_store(SD_DONE, Int32(1))
            stop = True
        else:
            stop = sd_ncut_step(svh, svl, k, Int(n_in), lasth, lasth + 1, state, Int(n_iter_max))
        flag[1] = Int32(1) if stop else Int32(0)
    dev_barrier()
    if flag[1] != Int32(0):
        return
    var c = tid
    while c < kk:
        sd_polar_cell(vvh, vvl, uh, ul, k, c, roth, rotl)
        c += SD_TPB


# ---------------------------------------------------------------- the driver


def discretize_df64_device(
    ctx: DeviceContext, x: List[Float32], n: Int, k: Int, seed: UInt64, max_svd_restarts: Int, n_iter_max: Int,
    mut n_iter_out: Int,
) raises -> List[Int32]:
    """`discretize_df64_host` on the device: labels (n) and n_iter. One
    upload of the embedding; the state words come home once per SD_CHUNK
    iterations, the labels once at the end."""
    if n < 1 or k < 1 or len(x) != n * k:
        raise Error("SpectralClustering discretize: the embedding is not n x k")
    if max_svd_restarts < 1:
        raise Error("SpectralClustering discretize: SVD did not converge")
    if n >= 1 << 31 or n * k >= 1 << 31:
        raise Error("SpectralClustering discretize: the embedding exceeds the device's int32 indices")
    var nk = n * k
    var kk = k * k
    var nch = sd_chunks(n)
    var xl = x.copy()
    var dx = ctx.enqueue_create_buffer[DType.float32](nk)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=FP(unsafe_from_address=Int(xl.unsafe_ptr())))
    var vh = ctx.enqueue_create_buffer[DType.float32](nk)
    var vl = ctx.enqueue_create_buffer[DType.float32](nk)
    var np = max(nch * kk, nch * k)
    var ph = ctx.enqueue_create_buffer[DType.float32](np)
    var pl = ctx.enqueue_create_buffer[DType.float32](np)
    var cnh = ctx.enqueue_create_buffer[DType.float32](k)
    var cnl = ctx.enqueue_create_buffer[DType.float32](k)
    var sgn = ctx.enqueue_create_buffer[DType.float32](k)
    var state = ctx.enqueue_create_buffer[DType.int32](SD_STATE_WORDS)
    enqueue_fill(ctx, state, Int32(0))
    ctx.enqueue_function[sdk_load_kernel](_fp(dx), _fp(vh), _fp(vl), Int32(nk), grid_dim=_blocks(nk), block_dim=SD_TPB)
    ctx.enqueue_function[sdk_colsq_kernel](
        _fp(vh), _fp(vl), Int32(n), Int32(k), _fp(ph), _fp(pl), Int32(nch * k), grid_dim=_blocks(nch * k),
        block_dim=SD_TPB,
    )
    ctx.enqueue_function[sdk_colnorm_kernel](
        _fp(ph), _fp(pl), Int32(nch), Int32(k), _fp(cnh), _fp(cnl), _ip(state), grid_dim=_blocks(k), block_dim=SD_TPB
    )
    var rootn = ff_sqrt(sd_of_int(n))
    ctx.enqueue_function[sdk_colscale_kernel](
        _fp(vh), _fp(vl), Int32(k), _fp(cnh), _fp(cnl), rootn.hi, rootn.lo, _fp(sgn), Int32(nk),
        grid_dim=_blocks(nk), block_dim=SD_TPB,
    )
    ctx.enqueue_function[sdk_colsign_kernel](
        _fp(vh), _fp(vl), Int32(k), _fp(sgn), Int32(nk), grid_dim=_blocks(nk), block_dim=SD_TPB
    )
    ctx.enqueue_function[sdk_rownorm_kernel](
        _fp(vh), _fp(vl), Int32(k), _ip(state), Int32(n), grid_dim=_blocks(n), block_dim=SD_TPB
    )
    ctx.enqueue_function[sdk_fail_done_kernel](_ip(state), grid_dim=1, block_dim=1)  # small-launch(state: one word): the refusal's done mark
    # the greedy start (its row from the lane's host stream, as the host column)
    var roth = ctx.enqueue_create_buffer[DType.float32](kk)
    var rotl = ctx.enqueue_create_buffer[DType.float32](kk)
    var cah = ctx.enqueue_create_buffer[DType.float32](n)
    var cal = ctx.enqueue_create_buffer[DType.float32](n)
    enqueue_fill(ctx, cah, Float32(0))
    enqueue_fill(ctx, cal, Float32(0))
    var bmh = ctx.enqueue_create_buffer[DType.float32](nch)
    var bml = ctx.enqueue_create_buffer[DType.float32](nch)
    var bmi = ctx.enqueue_create_buffer[DType.int32](nch)
    var rng = SplitMix64(state=seed)
    var r0 = rng.below(n)
    ctx.enqueue_function[sdk_take_row_kernel](
        _fp(vh), _fp(vl), Int32(k), Int32(r0), _ip(state), _fp(roth), _fp(rotl), Int32(0), grid_dim=_blocks(k),
        block_dim=SD_TPB,
    )
    for j in range(1, k):  # driver-loop: over the start's k - 1 greedy columns (model structure)
        ctx.enqueue_function[sdk_cacc_kernel](
            _fp(vh), _fp(vl), Int32(k), _fp(roth), _fp(rotl), Int32(j - 1), _fp(cah), _fp(cal), Int32(n),
            grid_dim=_blocks(n), block_dim=SD_TPB,
        )
        ctx.enqueue_function[sdk_chunk_min_kernel](
            _fp(cah), _fp(cal), Int32(n), Int32(nch), _fp(bmh), _fp(bml), _ip(bmi), grid_dim=_blocks(nch),
            block_dim=SD_TPB,
        )
        ctx.enqueue_function[sdk_min_fold_kernel](  # small-launch(nch: row chunks): the chunk minima's fold, exact compares
            _fp(bmh), _fp(bml), _ip(bmi), Int32(nch), _ip(state), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_function[sdk_take_row_kernel](
            _fp(vh), _fp(vl), Int32(k), Int32(-1), _ip(state), _fp(roth), _fp(rotl), Int32(j), grid_dim=_blocks(k),
            block_dim=SD_TPB,
        )
    # the iterations, SD_CHUNK behind one read
    var lab = ctx.enqueue_create_buffer[DType.int32](n)
    var mh = ctx.enqueue_create_buffer[DType.float32](kk)
    var ml = ctx.enqueue_create_buffer[DType.float32](kk)
    var vvh = ctx.enqueue_create_buffer[DType.float32](kk)
    var vvl = ctx.enqueue_create_buffer[DType.float32](kk)
    var uh = ctx.enqueue_create_buffer[DType.float32](kk)
    var ul = ctx.enqueue_create_buffer[DType.float32](kk)
    var svh = ctx.enqueue_create_buffer[DType.float32](k)
    var svl = ctx.enqueue_create_buffer[DType.float32](k)
    var cch = ctx.enqueue_create_buffer[DType.float32](k)
    var ccl = ctx.enqueue_create_buffer[DType.float32](k)
    var filled = ctx.enqueue_create_buffer[DType.int32](k)
    var last = ctx.enqueue_create_buffer[DType.float32](2)
    enqueue_fill(ctx, last, Float32(0))
    var hs = List[Int32](length=SD_STATE_WORDS, fill=Int32(0))
    while True:  # driver-loop: over the iteration chunks (at most (n_iter_max + 1) / SD_CHUNK + 1)
        for _c in range(SD_CHUNK):  # driver-loop: over the chunk's iterations (SD_CHUNK, gated on the done word)
            ctx.enqueue_function[sdk_label_kernel](
                _fp(vh), _fp(vl), Int32(k), _fp(roth), _fp(rotl), _ip(lab), _ip(state), Int32(n),
                grid_dim=_blocks(n), block_dim=SD_TPB,
            )
            ctx.enqueue_function[sdk_class_part_kernel](
                _fp(vh), _fp(vl), _ip(lab), Int32(n), Int32(k), _fp(ph), _fp(pl), _ip(state), Int32(nch * kk),
                grid_dim=_blocks(nch * kk), block_dim=SD_TPB,
            )
            ctx.enqueue_function[sdk_class_fold_kernel](
                _fp(ph), _fp(pl), Int32(nch), Int32(kk), _fp(mh), _fp(ml), _ip(state), grid_dim=_blocks(kk),
                block_dim=SD_TPB,
            )
            ctx.enqueue_function[sdk_step_kernel](  # small-launch(k: clusters): the k x k SVD's cyclic rotations, rows across the block
                _fp(mh), _fp(ml), _fp(vvh), _fp(vvl), _fp(uh), _fp(ul), _fp(svh), _fp(svl), _ip(filled), _fp(cch),
                _fp(ccl), _fp(last), _fp(roth), _fp(rotl), _ip(state), Int32(k), Int32(n), Int32(n_iter_max),
                grid_dim=1, block_dim=SD_TPB,
            )
        ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=Int(hs.unsafe_ptr())), src_buf=state)
        ctx.synchronize()
        sd_raise_status(Int(hs[SD_STATUS]))
        if hs[SD_DONE] != Int32(0):
            break
    n_iter_out = Int(hs[SD_ITER])
    var labels = List[Int32](length=n, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=Int(labels.unsafe_ptr())), src_buf=lab)
    ctx.synchronize()
    _ = xl^
    _ = dx^
    _ = vh^
    _ = vl^
    _ = ph^
    _ = pl^
    _ = cnh^
    _ = cnl^
    _ = sgn^
    _ = state^
    _ = roth^
    _ = rotl^
    _ = cah^
    _ = cal^
    _ = bmh^
    _ = bml^
    _ = bmi^
    _ = lab^
    _ = mh^
    _ = ml^
    _ = vvh^
    _ = vvl^
    _ = uh^
    _ = ul^
    _ = svh^
    _ = svl^
    _ = cch^
    _ = ccl^
    _ = filled^
    _ = last^
    return labels^


# ---------------------------------------------------------------- cluster_qr
# Lane m3-linalg-and-housekeeping (2026-11-09): assign_labels='cluster_qr'
# on the device in float-float (x_cluster/spectral_df64.mojo's cq_* bodies;
# host column `cluster_qr_df64_host`). Was binary64 on the host
# (x_cluster/spectral_assign.mojo): the fit's last host step after the
# embedding came home.


def cqk_load_kernel(x: FP, vh: FP, vl: FP, ah: FP, al: FP, count: Int32):
    var t = _g()
    if t < Int(count):
        sd_load_cell(x, vh, vl, t)
        sd_load_cell(x, ah, al, t)


def cqk_colnorm_kernel(ah: FP, al: FP, used: IP, k_in: Int32, j_in: Int32, nh: FP, nl: FP, n_in: Int32):
    var c = _g()
    if c < Int(n_in):
        cq_colnorm_cell(ah, al, used, Int(k_in), Int(j_in), c, nh, nl)


def cqk_chunk_max_kernel(nh: FP, nl: FP, used: IP, n_in: Int32, nch: Int32, bmh: FP, bml: FP, bmi: IP):
    var ch = _g()
    if ch < Int(nch):
        cq_chunk_max(nh, nl, used, Int(n_in), ch, bmh, bml, bmi)


def cqk_pivot_kernel(
    ah: FP, al: FP, used: IP, piv: IP, k_in: Int32, j_in: Int32, bmh: FP, bml: FP, bmi: IP, nch: Int32, hvh: FP,
    hvl: FP, hb: FP, state: IP,
):
    """One thread: step j's pivot and Householder vector; state[SD_AMIN] =
    1 when the reflector applies, else 0 (the apply kernel reads it)."""
    if _g() != 0:
        return
    var go = cq_pivot_step(ah, al, used, piv, Int(k_in), Int(j_in), bmh, bml, bmi, Int(nch), hvh, hvl, hb, hb + 1)
    state.unsafe_store(SD_AMIN, Int32(go))


def cqk_apply_kernel(ah: FP, al: FP, used: IP, k_in: Int32, j_in: Int32, hvh: FP, hvl: FP, hb: FP, state: IP, n_in: Int32):
    var c = _g()
    if c >= Int(n_in) or state.unsafe_load(SD_AMIN) == Int32(0):
        return
    cq_apply_col(ah, al, used, Int(k_in), Int(j_in), hvh, hvl, sd_ld(hb, hb + 1, 0), c)


def cqk_gather_kernel(vh: FP, vl: FP, piv: IP, k_in: Int32, mh: FP, ml: FP, count: Int32):
    var t = _g()
    if t < Int(count):
        cq_gather_cell(vh, vl, piv, Int(k_in), t, mh, ml)


def cqk_svd_kernel(
    mh: FP, ml: FP, vvh: FP, vvl: FP, uh: FP, ul: FP, svh: FP, svl: FP, filled: IP, ch: FP, cl: FP, roth: FP,
    rotl: FP, state: IP, k_in: Int32,
):
    """ONE block: the SVD of M = V[piv] (`_sdk_svd_block`), the completion
    on thread 0 (SD_BAD_BASIS in the status word when it fails), then Q = V
    U^T across the block."""
    var k = Int(k_in)
    var kk = k * k
    var tid = Int(thread_idx.x)
    var flag = stack_allocation[1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    _sdk_svd_block(mh, ml, vvh, vvl, uh, ul, svh, svl, filled, k, tid)
    if tid == 0:
        var ok = sd_complete_u(uh, ul, filled, ch, cl, k)
        if not ok:
            state.unsafe_store(SD_STATUS, Int32(SD_BAD_BASIS))
        flag[0] = Int32(1) if ok else Int32(0)
    dev_barrier()
    if flag[0] == Int32(0):
        return
    var c = tid
    while c < kk:
        sd_polar_cell(vvh, vvl, uh, ul, k, c, roth, rotl)
        c += SD_TPB


def cqk_label_kernel(vh: FP, vl: FP, k_in: Int32, roth: FP, rotl: FP, lab: IP, n_in: Int32):
    var i = _g()
    if i < Int(n_in):
        lab.unsafe_store(i, sd_label_abs_row(vh, vl, Int(k_in), roth, rotl, i))


def cluster_qr_df64_device(ctx: DeviceContext, x: List[Float32], n: Int, k: Int) raises -> List[Int32]:
    """SpectralClustering assign_labels='cluster_qr' on the device in
    float-float: labels (n). One upload of the embedding, the k pivot steps
    enqueued with no read (each a grid over the n columns plus one thread for
    the pivot), then the k x k SVD in one block, the labels across the rows;
    ONE read (the status word and the labels) at the end."""
    if n < 1 or k < 1 or k > n or len(x) != n * k:
        raise Error("SpectralClustering cluster_qr: the embedding is not n x k with k <= n")
    if n >= 1 << 31 or n * k >= 1 << 31:
        raise Error("SpectralClustering cluster_qr: the embedding exceeds the device's int32 indices")
    var nk = n * k
    var kk = k * k
    var nch = sd_chunks(n)
    var xl = x.copy()
    var dx = ctx.enqueue_create_buffer[DType.float32](nk)
    ctx.enqueue_copy(dst_buf=dx, src_ptr=FP(unsafe_from_address=Int(xl.unsafe_ptr())))
    var vh = ctx.enqueue_create_buffer[DType.float32](nk)
    var vl = ctx.enqueue_create_buffer[DType.float32](nk)
    var ah = ctx.enqueue_create_buffer[DType.float32](nk)
    var al = ctx.enqueue_create_buffer[DType.float32](nk)
    var nh = ctx.enqueue_create_buffer[DType.float32](n)
    var nl = ctx.enqueue_create_buffer[DType.float32](n)
    var used = ctx.enqueue_create_buffer[DType.int32](n)
    var piv = ctx.enqueue_create_buffer[DType.int32](k)
    var bmh = ctx.enqueue_create_buffer[DType.float32](nch)
    var bml = ctx.enqueue_create_buffer[DType.float32](nch)
    var bmi = ctx.enqueue_create_buffer[DType.int32](nch)
    var hvh = ctx.enqueue_create_buffer[DType.float32](k)
    var hvl = ctx.enqueue_create_buffer[DType.float32](k)
    var hb = ctx.enqueue_create_buffer[DType.float32](2)
    var state = ctx.enqueue_create_buffer[DType.int32](SD_STATE_WORDS)
    enqueue_fill(ctx, used, Int32(0))
    enqueue_fill(ctx, nh, Float32(0))
    enqueue_fill(ctx, nl, Float32(0))
    enqueue_fill(ctx, state, Int32(0))
    ctx.enqueue_function[cqk_load_kernel](
        _fp(dx), _fp(vh), _fp(vl), _fp(ah), _fp(al), Int32(nk), grid_dim=_blocks(nk), block_dim=SD_TPB
    )
    for j in range(k):  # driver-loop: over the QR's k pivot steps (model structure)
        ctx.enqueue_function[cqk_colnorm_kernel](
            _fp(ah), _fp(al), _ip(used), Int32(k), Int32(j), _fp(nh), _fp(nl), Int32(n), grid_dim=_blocks(n),
            block_dim=SD_TPB,
        )
        ctx.enqueue_function[cqk_chunk_max_kernel](
            _fp(nh), _fp(nl), _ip(used), Int32(n), Int32(nch), _fp(bmh), _fp(bml), _ip(bmi), grid_dim=_blocks(nch),
            block_dim=SD_TPB,
        )
        ctx.enqueue_function[cqk_pivot_kernel](  # small-launch(nch: row chunks, then k: the reflector): the step's pivot
            _fp(ah), _fp(al), _ip(used), _ip(piv), Int32(k), Int32(j), _fp(bmh), _fp(bml), _ip(bmi), Int32(nch),
            _fp(hvh), _fp(hvl), _fp(hb), _ip(state), grid_dim=1, block_dim=1,
        )
        ctx.enqueue_function[cqk_apply_kernel](
            _fp(ah), _fp(al), _ip(used), Int32(k), Int32(j), _fp(hvh), _fp(hvl), _fp(hb), _ip(state), Int32(n),
            grid_dim=_blocks(n), block_dim=SD_TPB,
        )
    var mh = ctx.enqueue_create_buffer[DType.float32](kk)
    var ml = ctx.enqueue_create_buffer[DType.float32](kk)
    var vvh = ctx.enqueue_create_buffer[DType.float32](kk)
    var vvl = ctx.enqueue_create_buffer[DType.float32](kk)
    var uh = ctx.enqueue_create_buffer[DType.float32](kk)
    var ul = ctx.enqueue_create_buffer[DType.float32](kk)
    var svh = ctx.enqueue_create_buffer[DType.float32](k)
    var svl = ctx.enqueue_create_buffer[DType.float32](k)
    var cch = ctx.enqueue_create_buffer[DType.float32](k)
    var ccl = ctx.enqueue_create_buffer[DType.float32](k)
    var filled = ctx.enqueue_create_buffer[DType.int32](k)
    var roth = ctx.enqueue_create_buffer[DType.float32](kk)
    var rotl = ctx.enqueue_create_buffer[DType.float32](kk)
    var lab = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_function[cqk_gather_kernel](
        _fp(vh), _fp(vl), _ip(piv), Int32(k), _fp(mh), _fp(ml), Int32(kk), grid_dim=_blocks(kk), block_dim=SD_TPB
    )
    ctx.enqueue_function[cqk_svd_kernel](  # small-launch(k: clusters): the k x k SVD's cyclic rotations, rows across the block
        _fp(mh), _fp(ml), _fp(vvh), _fp(vvl), _fp(uh), _fp(ul), _fp(svh), _fp(svl), _ip(filled), _fp(cch), _fp(ccl),
        _fp(roth), _fp(rotl), _ip(state), Int32(k), grid_dim=1, block_dim=SD_TPB,
    )
    ctx.enqueue_function[cqk_label_kernel](
        _fp(vh), _fp(vl), Int32(k), _fp(roth), _fp(rotl), _ip(lab), Int32(n), grid_dim=_blocks(n), block_dim=SD_TPB
    )
    var hs = List[Int32](length=SD_STATE_WORDS, fill=Int32(0))
    var labels = List[Int32](length=n, fill=Int32(0))
    ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=Int(hs.unsafe_ptr())), src_buf=state)
    ctx.enqueue_copy(dst_ptr=IP(unsafe_from_address=Int(labels.unsafe_ptr())), src_buf=lab)
    ctx.synchronize()
    cq_raise_status(Int(hs[SD_STATUS]))
    _ = xl^
    _ = dx^
    _ = vh^
    _ = vl^
    _ = ah^
    _ = al^
    _ = nh^
    _ = nl^
    _ = used^
    _ = piv^
    _ = bmh^
    _ = bml^
    _ = bmi^
    _ = hvh^
    _ = hvl^
    _ = hb^
    _ = state^
    _ = mh^
    _ = ml^
    _ = vvh^
    _ = vvl^
    _ = uh^
    _ = ul^
    _ = svh^
    _ = svl^
    _ = cch^
    _ = ccl^
    _ = filled^
    _ = roth^
    _ = rotl^
    _ = lab^
    return labels^
