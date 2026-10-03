# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DEVICE L-BFGS (lane cgr4-device-optim, 2026-10-03): x_linear/lbfgs.mojo's
minimizer with its state resident on the device. Before this lane the grid
fits (HuberRegressor, LogisticRegressionCV) ran `lbfgs` on a host team: every
objective's gradient came home and the two-loop recursion, the line search
and the pair updates ran on the CPU.

Here the state lives in one float32 device buffer (layout below), the
objective is enqueued on the device by the caller (`LbObjective`) and the
vector phases are lbfgs.mojo's own team functions (`lb_direction`,
`lb_accept`, `lb_pair`) run by one block of LBD_TPB threads, every entry
dealt across the block, every P-vector sum the vfold order
(x_linear/vfold.mojo). The host column runs the same functions as a team of
one: the same words. Per line-search trial the host enqueues one unit

    trial   theta[o] = theta[c] + tt dir[c]        (a thread an entry)
    f, g    the objective at theta[o]               (the caller's kernels)
    step    accept?  then the pair and the next direction into parity o

and reads TWO words home: accepted, stop. Nothing else crosses until the
fit's result. The state is double-buffered by iteration parity (c, o = 1 - c),
so a unit reads only parity c and writes only parity o: it is idempotent and
the Metal witness may rerun it (x_linear/witness.mojo). tt = 2^-k is the
host's exact sequence, passed in.

Layout (float32 words, P = p): theta[2] P | g[2] P | dir[2] P | S M*P | Y M*P |
rho M | parts vscratch(P) | sc[2] 4 (f, slope, count bits, head bits) | flags 2.
"""
from std.gpu import block_idx, thread_idx, block_dim
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from x_linear.ops import FP, IP, ld, st, fm, fmad
from x_linear.team import Team, team_at
from x_linear.lbfgs import LBFGS_M, lb_direction, lb_accept, lb_pair
from x_linear.vfold import vscratch
from x_linear.witness import Witness, witness_end, WITNESS_TRIES

comptime LBD_TPB = 256


@always_inline
def lbd_th(p: Int, c: Int) -> Int:
    return c * p


@always_inline
def lbd_g(p: Int, c: Int) -> Int:
    return (2 + c) * p


@always_inline
def lbd_dr(p: Int, c: Int) -> Int:
    return (4 + c) * p


@always_inline
def lbd_s(p: Int) -> Int:
    return 6 * p


@always_inline
def lbd_y(p: Int) -> Int:
    return 6 * p + LBFGS_M * p


@always_inline
def lbd_rho(p: Int) -> Int:
    return 6 * p + 2 * LBFGS_M * p


@always_inline
def lbd_parts(p: Int) -> Int:
    return lbd_rho(p) + LBFGS_M


@always_inline
def lbd_sc(p: Int, c: Int) -> Int:
    return lbd_parts(p) + vscratch(p) + 4 * c


@always_inline
def lbd_flags(p: Int) -> Int:
    return lbd_parts(p) + vscratch(p) + 8


def lbd_words(p: Int) -> Int:
    return lbd_flags(p) + 2


def lbd_blocks(count: Int) -> Int:
    return max((count + LBD_TPB - 1) // LBD_TPB, 1)


@always_inline
def _team(lw: FP) -> Team:
    return team_at(Int(thread_idx.x), Int(block_dim.x), lw, 0, 0, 0)


@always_inline
def _bits(v: Int) -> Float32:
    return bitcast[DType.float32](Int32(v))


@always_inline
def _int(v: Float32) -> Int:
    return Int(bitcast[DType.int32](v))


def lbd_start_kernel(lw: FP, p: Int32, tol: Float32, wf: IP, woff: Int32, nonce: Int32):
    """The first direction (an empty history) at parity 0."""
    var t = _team(lw)
    var pp = Int(p)
    var r = lb_direction(t, lw, lbd_g(pp, 0), lbd_dr(pp, 0), lbd_s(pp), lbd_y(pp), lbd_rho(pp), pp, 0, 0, tol,
                         lw + lbd_parts(pp))
    if t.lead():
        var sc = lbd_sc(pp, 0)
        st(lw, sc + 1, r[1])
        st(lw, sc + 2, _bits(r[2]))
        st(lw, sc + 3, _bits(r[3]))
        st(lw, lbd_flags(pp), Float32(1))
        st(lw, lbd_flags(pp) + 1, Float32(r[0]))
    witness_end(wf, woff, nonce)


def lbd_trial_kernel(lw: FP, p: Int32, c: Int32, tt: Float32, wf: IP, woff: Int32, nonce: Int32):
    """theta[o] = tt dir[c] + theta[c], a thread an entry."""
    var j = Int(block_idx.x) * LBD_TPB + Int(thread_idx.x)
    var pp = Int(p)
    var cc = Int(c)
    if j < pp:
        st(lw, lbd_th(pp, 1 - cc) + j, fmad(tt, ld(lw, lbd_dr(pp, cc) + j), ld(lw, lbd_th(pp, cc) + j)))
    witness_end(wf, woff, nonce)


def lbd_step_kernel(lw: FP, p: Int32, c: Int32, tt: Float32, tol: Float32, wf: IP, woff: Int32, nonce: Int32):
    """Accept or not; on accept the pair and the next direction at parity o."""
    var t = _team(lw)
    var pp = Int(p)
    var cc = Int(c)
    var o = 1 - cc
    var parts = lw + lbd_parts(pp)
    var scc = lbd_sc(pp, cc)
    var sco = lbd_sc(pp, o)
    var f = ld(lw, scc)
    var fnew = ld(lw, sco)
    var slope = ld(lw, scc + 1)
    var ok = lb_accept(t, lw, lbd_g(pp, o), lbd_dr(pp, cc), pp, f, fnew, tt, slope, parts)
    if ok == 1:
        var r = lb_pair(t, lw, lbd_th(pp, cc), lw, lbd_th(pp, o), lw, lbd_g(pp, cc), lbd_g(pp, o), lbd_s(pp),
                        lbd_y(pp), lbd_rho(pp), pp, _int(ld(lw, scc + 2)), _int(ld(lw, scc + 3)), parts)
        var dd = lb_direction(t, lw, lbd_g(pp, o), lbd_dr(pp, o), lbd_s(pp), lbd_y(pp), lbd_rho(pp), pp, r[0], r[1],
                              tol, parts)
        if t.lead():
            st(lw, sco + 1, dd[1])
            st(lw, sco + 2, _bits(dd[2]))
            st(lw, sco + 3, _bits(dd[3]))
            st(lw, lbd_flags(pp) + 1, Float32(dd[0]))
    if t.lead():
        st(lw, lbd_flags(pp), Float32(ok))
    witness_end(wf, woff, nonce)


trait LbObjective(Movable):
    def blocks(self) -> Int:
        """Witness words one evaluation's kernels write."""
        ...

    def enqueue(mut self, mut ctx: DeviceContext, th: FP, g: FP, f: FP, wf: IP, woff: Int, nonce: Int32) raises:
        """f[0] and g[0, P) at the device theta th, enqueued; every kernel
        ends with witness_end at its words from woff. Idempotent: it reads
        only th and data the fit does not change."""
        ...


def lbd_witness_words(ob: Int, p: Int) -> Int:
    """Witness capacity a `lbfgs_device` fit needs."""
    return lbd_blocks(p) + ob + 1


def lbfgs_device[O: LbObjective](
    mut ctx: DeviceContext, mut obj: O, mut wit: Witness, lw: DeviceBuffer[DType.float32], p: Int,
    max_iter: Int, tol: Float32, what: String,
) raises -> Tuple[Int, Int]:
    """Minimizes from theta[0] (lbd_th(p, 0), set by the caller). Returns
    (lbfgs's iteration count, negative on max_iter; the parity holding
    the final theta)."""
    var lp = lw.unsafe_ptr()
    var ob = obj.blocks()
    var bt = lbd_blocks(p)
    var flags = List[Float32](length=2, fill=Float32(0))
    var fl = lw.create_sub_buffer[DType.float32](lbd_flags(p), 2)
    var tries = 0
    while True:
        var nonce = wit.begin()
        var wf = wit.p()
        obj.enqueue(ctx, lp + lbd_th(p, 0), lp + lbd_g(p, 0), lp + lbd_sc(p, 0), wf, 0, nonce)
        ctx.enqueue_function[lbd_start_kernel](lp, Int32(p), tol, wf, Int32(ob), nonce, grid_dim=1, block_dim=LBD_TPB)
        ctx.enqueue_copy(dst_ptr=flags.unsafe_ptr(), src_buf=fl)
        if wit.ok(ctx, ob + 1, what):
            break
        tries += 1
        if tries >= WITNESS_TRIES:
            wit.fail()
    ctx.synchronize()
    if max_iter <= 0:
        return (0, 0)
    if flags[1] != Float32(0):
        return (0, 0)
    var c = 0
    var it = 0
    while it < max_iter:
        var tt = Float32(1)
        var accepted = False
        for _ in range(40):
            tries = 0
            while True:
                var nonce = wit.begin()
                var wf = wit.p()
                ctx.enqueue_function[lbd_trial_kernel](lp, Int32(p), Int32(c), tt, wf, Int32(0), nonce,
                                                       grid_dim=bt, block_dim=LBD_TPB)
                obj.enqueue(ctx, lp + lbd_th(p, 1 - c), lp + lbd_g(p, 1 - c), lp + lbd_sc(p, 1 - c), wf, bt, nonce)
                ctx.enqueue_function[lbd_step_kernel](lp, Int32(p), Int32(c), tt, tol, wf, Int32(bt + ob), nonce,
                                                      grid_dim=1, block_dim=LBD_TPB)
                ctx.enqueue_copy(dst_ptr=flags.unsafe_ptr(), src_buf=fl)
                if wit.ok(ctx, bt + ob + 1, what):
                    break
                tries += 1
                if tries >= WITNESS_TRIES:
                    wit.fail()
            ctx.synchronize()
            if flags[0] != Float32(0):
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        if not accepted:
            return (it, c)
        it += 1
        c = 1 - c
        if it >= max_iter:
            return (-it, c)
        if flags[1] != Float32(0):
            return (it, c)
    return (-it, c)
