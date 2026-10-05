# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HuberRegressor with the L-BFGS line search batched on the device
(lane/apple-fast-robust@cfdb95e48, 2026-10-02; recovered onto main
2026-10-04 by lane/apple-fast-rec-fa-robust; FAST + Apple only, behind
the FAST + Apple default since 2026-10-04; rollback
`-D MOJOLEARN_HUBER_DEVICE_LBFGS_OFF`).

Main's x_linear/huber_grid.mojo now runs the minimizer on the device too
(x_linear/lbfgs_device.mojo), but the host still drives the line search:
one unit per trial and two words read home (accepted, stop) after each,
a host wait per evaluation. Here the line-search decision, the tt halving
and the stop test also run on the device, so the host reads the stop word
once per HF_BATCH evaluations instead of once per evaluation. Here the
minimizer's state (theta, the trial point, both gradients, the direction,
the pair ring, f, slope, the step) lives in one device buffer, and each
evaluation is a fixed unit of four launches: the map (a thread per row,
`huber_map_row`), the block partials (a thread per (block, task),
`_huber_part`), the fold (`hg_fold_kernel`, a thread per task) and
`hf_step_kernel`, one block that finishes the objective (huber_finish's
statements), takes the line-search decision, updates the pair ring, builds
the next direction and writes the next trial point. The host enqueues
HF_BATCH units at a time and reads the stop word once per batch; units
after the stop are no-ops. The same objective and the same FOLD_BLOCK
partials as the grid path, with the source's ascending dots on the lead
(`_hf_dot`). A batch is one
guarded unit under the Metal witness (x_linear/witness.mojo): the state is
snapshotted before it and restored before a rerun. Recovery note: main's
device L-BFGS moved its P-vector sums to the vfold order; this fit keeps
the source's ascending dots (FAST bits only, never IDENTICAL), so its words
differ from main's FAST Huber fit and the A/B's quality check decides.
HUBER_FAST_BLOCK512 (the FAST + Apple default since rab19-huber512; rollback
`-D MOJOLEARN_HUBER_FAST_BLOCK512_OFF`) folds the partials
over 512-row blocks instead of FOLD_BLOCK: eight times the threads of the
partials launch (taxi: 21 tasks x 366 blocks under FOLD_BLOCK), a
different fold order (FAST bits only).
"""
from std.gpu import block_idx, block_dim, thread_idx
from std.memory import bitcast
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from x_linear.ops import FP, IP, ld, st, ldi, sti, fexp, fm, fd, fa, fs, fmad, fabs, fmax, fsqrt, i2f
from x_linear.team import LINEAR_TPB, team_barrier
from x_linear.tops import FOLD_BLOCK
from x_linear.witness import Witness, witness_end, WITNESS_TRIES
from x_linear.huber import huber_map_row, _huber_part
from x_linear.huber_grid import hg_fold_kernel, HG_TPB
from x_linear.lbfgs import LBFGS_M

#: (FAST + Apple, default ON since 2026-10-04) HuberRegressor's fit with the line search,
#: the pair ring and the stop test in one device step kernel per evaluation
#: (`hf_step_kernel`), HF_BATCH evaluations enqueued per host read of the
#: stop word, units after the stop no-ops. Source lane/apple-fast-robust@
#: cfdb95e48. Prior M3: huber taxi 242 -> 225 ms (n=1, against the source's
#: main, whose minimizer then ran on the host); EXPERIMENTS L181 OPEN.
#: Main has since moved the minimizer to the device (lbfgs_device.mojo, one
#: host read per trial), so the old gain is partly taken: the A/B measures
#: what batching the reads still buys. Recovery fixes: lbfgs's `_dot` is
#: gone from main (local `_hf_dot`, the same ascending fmad chain), the
#: X_LINEAR_SERIAL_FOLDS guard is gone (that define was deleted), and the
#: result also carries theta after n_iter (main's huber_fit layout).
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-04, lane/
#: apple-fast-rec-ab3 @ 0ca521cc5): huber taxi 219.8 -> 189.9 ms; r2/rmse
#: identical, output digest identical. KEEP: the FAST + Apple default since
#: then; rollback -D MOJOLEARN_HUBER_DEVICE_LBFGS_OFF (the old -D name is
#: harmless; HUBER_FAST_BLOCK512 still forces it on).
comptime HUBER_DEVICE_LBFGS = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
    and (not is_defined["MOJOLEARN_HUBER_DEVICE_LBFGS_OFF"]() or is_defined["MOJOLEARN_HUBER_FAST_BLOCK512"]())
)
#: (FAST + Apple, default ON since 2026-10-05; needs HUBER_DEVICE_LBFGS) the partials of
#: the fast fit over 512-row blocks instead of FOLD_BLOCK: eight times the
#: threads in the partials launch (taxi: 21 tasks x 366 blocks under
#: FOLD_BLOCK), a shorter serial chain per thread and a different fold order
#: (FAST bits only). Source lane/apple-fast-robust@cfdb95e48. Prior M3 (with
#: DEVICE_LBFGS): huber taxi 226 -> 196 ms (-13%), EXPERIMENTS L182 OPEN.
#: No failure recorded.
#: OUTCOME (M3 afc_ab_def, full board size, 1 run per arm, 2026-10-05,
#: rab19-huber512): huber taxi 192.0 -> 161.6 ms, r2 equal 0.900215. KEEP:
#: the FAST + Apple default; rollback -D MOJOLEARN_HUBER_FAST_BLOCK512_OFF
#: (the old -D name still forces HUBER_DEVICE_LBFGS on, harmless).
comptime HUBER_FAST_BLOCK512 = HUBER_DEVICE_LBFGS and not is_defined["MOJOLEARN_HUBER_FAST_BLOCK512_OFF"]()
#: Rows per partial of the fast fit: FOLD_BLOCK (the grid fit's order) or
#: 512 under HUBER_FAST_BLOCK512.
comptime HF_FOLD = 512 if HUBER_FAST_BLOCK512 else FOLD_BLOCK
#: Evaluation units enqueued between two reads of the stop word.
comptime HF_BATCH = 16
#: x_linear/lbfgs.mojo's line search: the step halved up to 40 times.
comptime HF_TRIES = 40
#: Int32 state words: flag (0 running, 1 stopped, 2 max_iter), it, tries,
#: count, head, mode (0 the first evaluation, 1 a line-search trial),
#: decision (the step kernel's own slot), spare.
comptime HF_ST = 8
#: Float32 scalars after the pair ring: f, slope, tt, fnew.
comptime HF_SC = 4


@always_inline
def _hf_dot(a: FP, ia: Int, b: FP, ib: Int, p: Int) -> Float32:
    """The source's lbfgs `_dot`: one ascending fmad chain (the lead's
    scalar sums over P, P = d + 2 parameters)."""
    var acc = Float32(0)
    for j in range(p):
        acc = fmad(ld(a, ia + j), ld(b, ib + j), acc)
    return acc


def hf_words(p: Int) -> Int:
    """The minimizer's float32 words: theta p | tn p | g p | gn p | dir p
    | S m*p | Y m*p | rho m | al m | scalars HF_SC."""
    return 5 * p + 2 * LBFGS_M * p + 2 * LBFGS_M + HF_SC


def hf_fold_blocks(n: Int) -> Int:
    return (n + HF_FOLD - 1) // HF_FOLD


def _hf_blocks(count: Int) -> Int:
    return max((count + HG_TPB - 1) // HG_TPB, 1)


def hf_init_kernel(lb: FP, ls: IP, total: Int32, wf: IP, woff: Int32, nonce: Int32):
    """theta = tn = 0 (huber_fit's start), the ring and scalars zero, the
    state words zero (mode 0: the next unit evaluates the start)."""
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    for j in range(tid, Int(total), nt):
        st(lb, j, Float32(0))
    if tid < HF_ST:
        sti(ls, tid, 0)
    witness_end(wf, woff, nonce)


def hf_map_kernel(x: FP, y: FP, n: Int32, d: Int32, p: Int32, lb: FP, ls: IP, eps: Float32, fi: Int32,
                  sw: Int32, rows: FP, wf: IP, woff: Int32, nonce: Int32):
    """Row i's residual and gradient coefficient at the trial point lb[p:2p]
    (the team map's statements); a no-op once the fit stopped."""
    var i = Int(block_idx.x) * HG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    if ldi(ls, 0) == 0 and i < nn:
        var dd = Int(d)
        var pp = Int(p)
        var sigma = fexp(ld(lb, pp + pp - 1))
        var b = ld(lb, pp + dd) if fi != 0 else Float32(0)
        huber_map_row(i, x, y, nn, dd, lb, pp, b, fm(eps, sigma), fd(Float32(2), sigma), fm(Float32(2), eps),
                      sw != 0, rows, rows + nn)
    witness_end(wf, woff, nonce)


def hf_part_kernel(x: FP, y: FP, n: Int32, d: Int32, p: Int32, cells: Int32, lb: FP, ls: IP, eps: Float32,
                   sw: Int32, rows: FP, scr: FP, nbk: Int32, wf: IP, woff: Int32, nonce: Int32):
    """Task o of block bk from zero (`_huber_part`) into scr[o * nbk + bk]."""
    var q = Int(block_idx.x) * HG_TPB + Int(thread_idx.x)
    var nn = Int(n)
    var nt = Int(cells) + 5
    var nb = Int(nbk)
    if ldi(ls, 0) == 0 and q < nt * nb:
        var pp = Int(p)
        var thr = fm(eps, fexp(ld(lb, pp + pp - 1)))
        var bk = q // nt
        var o = q - bk * nt
        var lo = bk * HF_FOLD
        st(scr, o * nb + bk, _huber_part(o, Int(cells), Int(d), x, y, nn, rows, rows + nn, thr, sw != 0, lo,
                                         min(HF_FOLD, nn - lo)))
    witness_end(wf, woff, nonce)


def hf_step_kernel(lb: FP, ls: IP, gc: FP, sums: FP, nf: Float32, d: Int32, p: Int32, cells: Int32, fi: Int32,
                   sw: Int32, eps: Float32, alpha: Float32, tol: Float32, max_iter: Int32,
                   wf: IP, woff: Int32, nonce: Int32):
    """One block: huber_finish on the folded cells and sums (the gradient at
    the trial point into gn, f into fnew), then x_linear/lbfgs.mojo's
    decision: mode 0 takes the start as the current point; a trial is
    accepted by the Armijo test or the noise + curvature test (a new pair,
    theta = tn, g = gn) or rejected (tt halved; the 40th rejection stops
    the fit); after an accepted point the max_iter and tol checks, the
    two-loop direction on the lead thread (ascending dots), the slope and
    its steepest-descent fallback; last, tn = theta + tt * dir across the
    block. The lead's scalar sequence is lbfgs's; the element-wise loops
    are dealt across the block (one thread per element, the same bits).
    nf: the row count as a float32 (huber_finish's i2f(n)); this block
    never walks the rows."""
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var lead = tid == 0
    var run = ldi(ls, 0) == 0
    team_barrier()
    if run:
        var dd = Int(d)
        var pp = Int(p)
        var cc = Int(cells)
        var m = LBFGS_M
        var TN = pp
        var G = 2 * pp
        var GN = 3 * pp
        var DR = 4 * pp
        var SS = 5 * pp
        var SY = SS + m * pp
        var RHO = SY + m * pp
        var AL = RHO + m
        var SC = AL + m
        var swb = sw != 0
        var mode = ldi(ls, 5)
        var it = ldi(ls, 1)
        var tries = ldi(ls, 2)
        var count = ldi(ls, 3)
        var head = ldi(ls, 4)
        var sigma = fexp(ld(lb, TN + pp - 1))
        # huber_finish: the gradient cells (the L2 term on the coefficients)
        for j in range(tid, cc, nt):
            var v = ld(gc, j)
            if j < dd:
                v = fmad(fm(Float32(2), alpha), ld(lb, TN + j), v)
            st(lb, GN + j, v)
        team_barrier()
        if lead:
            var wn = Float32(0)
            for j in range(dd):
                var w = ld(lb, TN + j)
                wn = fmad(w, w, wn)
            var sq = ld(sums, 0)
            var out_abs = ld(sums, 1)
            var w_out = ld(sums, 2)
            var w_all = ld(sums, 3) if swb else Float32(0)
            var n_out = Int(bitcast[DType.int32](ld(sums, 4)))
            var two_eps = fm(Float32(2), eps)
            var squared_loss = fd(sq, sigma)
            var eps2 = fm(eps, eps)
            var cnt_out = w_out if swb else i2f(n_out)
            var cnt = w_all if swb else nf
            var outlier_loss = fs(fm(two_eps, out_abs), fm(fm(sigma, cnt_out), eps2))
            var gsigma = fs(fs(cnt, fm(cnt_out, eps2)), fd(squared_loss, sigma))
            st(lb, GN + pp - 1, fm(gsigma, sigma))
            var fnew = fa(fa(fa(fm(cnt, sigma), squared_loss), outlier_loss), fm(alpha, wn))
            st(lb, SC + 3, fnew)
            var dec = 0  # 0 rejected, 1 accepted, 2 the start
            if mode == 0:
                dec = 2
            else:
                var f = ld(lb, SC)
                var slope = ld(lb, SC + 1)
                var tt = ld(lb, SC + 2)
                if fnew == fnew and fnew <= fa(f, fm(fm(Float32(1e-4), tt), slope)):
                    dec = 1
                elif fnew == fnew and fnew <= fa(f, fa(fm(Float32(1e-6), fabs(f)), Float32(1e-30))):
                    var dg = _hf_dot(lb, GN, lb, DR, pp)
                    if fabs(dg) <= fm(Float32(0.9), fabs(slope)):
                        dec = 1
            sti(ls, 6, dec)
        team_barrier()
        var decision = ldi(ls, 6)
        if decision == 1:
            # the new pair at head, then theta = tn and g = gn
            var k = head
            for j in range(tid, pp, nt):
                var tnj = ld(lb, TN + j)
                var thj = ld(lb, j)
                var gnj = ld(lb, GN + j)
                var gj = ld(lb, G + j)
                st(lb, SS + k * pp + j, fs(tnj, thj))
                st(lb, SY + k * pp + j, fs(gnj, gj))
                st(lb, j, tnj)
                st(lb, G + j, gnj)
        elif decision == 2:
            for j in range(tid, pp, nt):
                st(lb, G + j, ld(lb, GN + j))
        team_barrier()
        if lead:
            if decision == 0:
                tries += 1
                if tries >= HF_TRIES:
                    # the line search cannot decrease f at float32 resolution
                    sti(ls, 0, 1)
                else:
                    st(lb, SC + 2, fm(ld(lb, SC + 2), Float32(0.5)))
                sti(ls, 2, tries)
            else:
                if decision == 1:
                    it += 1
                    var k = head
                    var sy = _hf_dot(lb, SS + k * pp, lb, SY + k * pp, pp)
                    var ss = _hf_dot(lb, SS + k * pp, lb, SS + k * pp, pp)
                    var yy = _hf_dot(lb, SY + k * pp, lb, SY + k * pp, pp)
                    if sy > fm(Float32(1e-10), fsqrt(fm(ss, yy))) and sy > 0:
                        st(lb, RHO + k, fd(Float32(1), sy))
                        head = (head + 1) % m
                        if count < m:
                            count += 1
                    sti(ls, 1, it)
                st(lb, SC, ld(lb, SC + 3))
                # lbfgs's loop head: max_iter, tol, the two-loop direction
                if it >= Int(max_iter):
                    sti(ls, 0, 2)
                else:
                    var gmax = Float32(0)
                    for j in range(pp):
                        gmax = fmax(gmax, fabs(ld(lb, G + j)))
                    if gmax <= tol:
                        sti(ls, 0, 1)
                    else:
                        for j in range(pp):
                            st(lb, DR + j, ld(lb, G + j))
                        for kk in range(count):
                            var k = (head - 1 - kk + 2 * m) % m
                            var a = fm(ld(lb, RHO + k), _hf_dot(lb, SS + k * pp, lb, DR, pp))
                            st(lb, AL + k, a)
                            for j in range(pp):
                                st(lb, DR + j, fs(ld(lb, DR + j), fm(a, ld(lb, SY + k * pp + j))))
                        var gamma: Float32
                        if count > 0:
                            var k = (head - 1 + m) % m
                            gamma = fd(_hf_dot(lb, SS + k * pp, lb, SY + k * pp, pp),
                                       _hf_dot(lb, SY + k * pp, lb, SY + k * pp, pp))
                        else:
                            gamma = fd(Float32(1), fmax(Float32(1), fsqrt(_hf_dot(lb, G, lb, G, pp))))
                        for j in range(pp):
                            st(lb, DR + j, fm(gamma, ld(lb, DR + j)))
                        for kk in range(count):
                            var k = (head - count + kk + 2 * m) % m
                            var bb = fm(ld(lb, RHO + k), _hf_dot(lb, SY + k * pp, lb, DR, pp))
                            var c = fs(ld(lb, AL + k), bb)
                            for j in range(pp):
                                st(lb, DR + j, fmad(c, ld(lb, SS + k * pp + j), ld(lb, DR + j)))
                        for j in range(pp):
                            st(lb, DR + j, -ld(lb, DR + j))
                        var slope = _hf_dot(lb, G, lb, DR, pp)
                        if not (slope < 0):
                            count = 0
                            head = 0
                            for j in range(pp):
                                st(lb, DR + j, -ld(lb, G + j))
                            slope = _hf_dot(lb, G, lb, DR, pp)
                            if not (slope < 0):
                                sti(ls, 0, 1)
                        st(lb, SC + 1, slope)
                        st(lb, SC + 2, Float32(1))
                        sti(ls, 2, 0)
                        sti(ls, 5, 1)
                sti(ls, 3, count)
                sti(ls, 4, head)
        team_barrier()
        if ldi(ls, 0) == 0:
            # the next trial point: tn = theta + tt * dir
            var tt = ld(lb, SC + 2)
            for j in range(tid, pp, nt):
                st(lb, TN + j, fmad(tt, ld(lb, DR + j), ld(lb, j)))
    witness_end(wf, woff, nonce)


def huber_fit_fast(
    ctx0: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """`huber_fit` with the minimizer on the device. ip: [max_iter,
    fit_intercept, sample_weight]; fp: [epsilon, alpha, tol]; res: coef d,
    intercept, scale, n_iter (huber_fit's layout, the rest zero)."""
    # the witness takes the context mutably (x_linear/huber_grid.mojo's wctx)
    var ctx = ctx0.copy()
    var fi = Int(ip[1]) != 0
    var sw = Int(ip[2]) != 0
    var max_iter = Int(ip[0])
    var eps = fp[0]
    var alpha = fp[1]
    var tol = fp[2]
    var p = d + 2 if fi else d + 1
    var cells = d + 1 if fi else d
    var nbk = hf_fold_blocks(n)
    var words = hf_words(p)
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var drows = ctx.enqueue_create_buffer[DType.float32](max(2 * n, 1))
    var dscr = ctx.enqueue_create_buffer[DType.float32](max((cells + 5) * nbk, 1))
    var dgc = ctx.enqueue_create_buffer[DType.float32](max(cells, 1))
    var dsums = ctx.enqueue_create_buffer[DType.float32](5)
    var dlb = ctx.enqueue_create_buffer[DType.float32](words)
    var dls = ctx.enqueue_create_buffer[DType.int32](HF_ST)
    var dlbs = ctx.enqueue_create_buffer[DType.float32](words)
    var dlss = ctx.enqueue_create_buffer[DType.int32](HF_ST)
    if n_x > 0:
        ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    var b1 = _hf_blocks(n)
    var b2 = _hf_blocks((cells + 5) * nbk)
    var b3 = _hf_blocks(cells + 5)
    var unit = b1 + b2 + b3 + 1
    var wit = Witness(ctx, HF_BATCH * unit)
    var tr = 0
    while True:
        var nonce = wit.begin()
        ctx.enqueue_function[hf_init_kernel](
            dlb.unsafe_ptr(), dls.unsafe_ptr(), Int32(words), wit.p(), Int32(0), nonce,
            grid_dim=1, block_dim=LINEAR_TPB,
        )  # small-launch(p: parameters): one block zeroes the minimizer's P-sized state
        if wit.ok(ctx, 1, "HuberRegressor L-BFGS start"):
            break
        tr += 1
        if tr >= WITNESS_TRIES:
            wit.fail()
    var hls = List[Int32](length=HF_ST, fill=Int32(0))
    var nf = i2f(n)
    var units = 0
    # lbfgs evaluates the start, then at most HF_TRIES trials per iteration
    var max_units = 1 + HF_TRIES * max(max_iter, 0) + HF_BATCH
    while True:
        # a batch as ONE guarded unit: the step kernel updates the state in
        # place, so a cut batch restores it and replays
        ctx.enqueue_copy(dst_buf=dlbs, src_buf=dlb)
        ctx.enqueue_copy(dst_buf=dlss, src_buf=dls)
        tr = 0
        while True:
            var nonce = wit.begin()
            var wo = 0
            for _ in range(HF_BATCH):
                ctx.enqueue_function[hf_map_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(p), dlb.unsafe_ptr(), dls.unsafe_ptr(),
                    eps, Int32(1 if fi else 0), Int32(1 if sw else 0), drows.unsafe_ptr(),
                    wit.p(), Int32(wo), nonce, grid_dim=b1, block_dim=HG_TPB,
                )
                wo += b1
                ctx.enqueue_function[hf_part_kernel](
                    dx.unsafe_ptr(), dy.unsafe_ptr(), Int32(n), Int32(d), Int32(p), Int32(cells), dlb.unsafe_ptr(),
                    dls.unsafe_ptr(), eps, Int32(1 if sw else 0), drows.unsafe_ptr(), dscr.unsafe_ptr(), Int32(nbk),
                    wit.p(), Int32(wo), nonce, grid_dim=b2, block_dim=HG_TPB,
                )
                wo += b2
                ctx.enqueue_function[hg_fold_kernel](
                    dscr.unsafe_ptr(), Int32(nbk), Int32(cells), dgc.unsafe_ptr(), dsums.unsafe_ptr(),
                    wit.p(), Int32(wo), nonce, grid_dim=b3, block_dim=HG_TPB,
                )
                wo += b3
                ctx.enqueue_function[hf_step_kernel](
                    dlb.unsafe_ptr(), dls.unsafe_ptr(), dgc.unsafe_ptr(), dsums.unsafe_ptr(), nf, Int32(d),
                    Int32(p), Int32(cells), Int32(1 if fi else 0), Int32(1 if sw else 0), eps, alpha, tol,
                    Int32(max_iter), wit.p(), Int32(wo), nonce, grid_dim=1, block_dim=LINEAR_TPB,
                )  # small-launch(p: parameters): one block strides the P-vector line-search step and two-loop recursion
                wo += 1
            ctx.enqueue_copy(dst_ptr=hls.unsafe_ptr(), src_buf=dls)
            if wit.ok(ctx, wo, "HuberRegressor L-BFGS batch"):
                break
            tr += 1
            if tr >= WITNESS_TRIES:
                wit.fail()
            ctx.enqueue_copy(dst_buf=dlb, src_buf=dlbs)
            ctx.enqueue_copy(dst_buf=dls, src_buf=dlss)
        ctx.synchronize()
        units += HF_BATCH
        if Int(hls[0]) != 0 or units >= max_units:
            break
    var hlb = List[Float32](length=words, fill=Float32(0))
    ctx.enqueue_copy(dst_ptr=hlb.unsafe_ptr(), src_buf=dlb)
    ctx.synchronize()
    for i in range(n_out):
        res.unsafe_store(i, Float32(0))
    for j in range(d):
        res.unsafe_store(j, hlb[j])
    res.unsafe_store(d, hlb[d] if fi else Float32(0))
    res.unsafe_store(d + 1, fexp(hlb[p - 1]))
    var it = Int(hls[1])
    res.unsafe_store(d + 2, i2f(it if it >= 0 else -it))
    # main's huber_fit words: theta P after n_iter (hg_result_kernel)
    for j in range(p):
        if d + 4 + j < n_out:
            res.unsafe_store(d + 4 + j, hlb[j])
    _ = hlb^
    _ = hls^
    _ = dx^
    _ = dy^
    _ = drows^
    _ = dscr^
    _ = dgc^
    _ = dsums^
    _ = dlb^
    _ = dls^
    _ = dlbs^
    _ = dlss^
    _ = wit^
