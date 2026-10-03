# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A SEQUENTIAL L-BFGS for the linear lane (lane/algos-linear, 2026-09-27).

Nocedal & Wright, Numerical Optimization (2nd ed.), Algorithm 7.4 (the
two-loop recursion) inside Algorithm 7.5, with m = 10 pairs, the initial
scaling gamma = s'y / y'y (their eq. 7.20), and a backtracking Armijo line
search (c1 = 1e-4, step halved up to 40 times) in place of their Wolfe
search; a step inside float32's objective noise (1e-6 relative) is also
accepted when it meets the strong Wolfe curvature test (c2 = 0.9). A pair with s'y <= 1e-10 * |s|*|y| is skipped. Stops when
max|g| <= tol, when the line search cannot decrease f at float32 resolution,
or at max_iter. Every P-vector sum is x_linear/vfold.mojo's chunked
aligned-tree fold (lane cgr4-device-optim; it was one ascending chain);
the objective is a comptime function parameter, so each caller compiles
its own copy.

The three vector phases (`lb_direction`, `lb_accept`, `lb_pair`) are team
functions: the host and the one-block team path run them on the lead as a
team of one, the device driver (x_linear/lbfgs_device.mojo) on a block of
LBD_TPB threads, entries dealt across the team, every sum the vfold order.
The same words either way. A rejected pair (s'y too small) that overwrote
the oldest pair of a full history drops that pair (count M - 1): the ring
never mixes one pair's s and y with another's rho.

Objective contract (a team call, x_linear/team.mojo):
    f = obj(t, x, y, n, d, ip, fp, theta, toff, grad, goff, sc): every thread
    gets f, and grad is written and visible to every thread on return; sc is
    the caller's per-row scratch (the host objective's map, then fold; lane
    linear-cpu). The device objective runs the team schedule and ignores sc.
The P-vector algebra below runs on the lead thread; the other threads take
part in the objective calls and in the lead's broadcast decisions.
Work layout at fw[woff:]: tn P | g P | gn P | dir P | S m*P | Y m*P | rho m | al m.
"""
from x_linear.ops import FP, IP, fa, fs, fm, fd, fmad, fsqrt, fabs, fmax, ld, st, copy, fill
from x_linear.team import Team, team_at
from x_linear.vfold import vdot, vabsmax

comptime LBFGS_M = 10

comptime Objective = def(Team, FP, FP, Int, Int, IP, FP, FP, Int, FP, Int, FP) thin -> Float32


def lbfgs_work(p: Int) -> Int:
    return 4 * p + 2 * LBFGS_M * p + 2 * LBFGS_M


@always_inline
def solo_view(t: Team) -> Team:
    """The lead's team of one (its vector phases run alone)."""
    return Team(0, 1, t.slot_at, t.rows_at, t.n, t.own_at)


def lb_direction(
    v: Team, w: FP, g: Int, dr: Int, sS: Int, sY: Int, rho: Int, p: Int, count_in: Int, head_in: Int,
    tol: Float32, parts: FP,
) -> Tuple[Int, Float32, Int, Int]:
    """(stop, slope, count, head): stop when max|g| <= tol, else
    dr = -H g by the two-loop recursion (Nocedal & Wright Alg. 7.4); a
    direction that does not descend clears the history for -g, and stop
    when that does not descend either. Every thread returns the same."""
    var count = count_in
    var head = head_in
    var gmax = vabsmax(v, w, g, p, parts)
    if gmax <= tol:
        return (1, Float32(0), count, head)
    for j in range(v.tid, p, v.nt):
        st(w, dr + j, ld(w, g + j))
    v.sync()
    var al = InlineArray[Float32, LBFGS_M](fill=Float32(0))
    for kk in range(count):
        var k = (head - 1 - kk + 2 * LBFGS_M) % LBFGS_M
        var a = fm(ld(w, rho + k), vdot(v, w, sS + k * p, w, dr, p, parts))
        al[k] = a
        for j in range(v.tid, p, v.nt):
            st(w, dr + j, fs(ld(w, dr + j), fm(a, ld(w, sY + k * p + j))))
        v.sync()
    var gamma: Float32
    if count > 0:
        var k = (head - 1 + LBFGS_M) % LBFGS_M
        gamma = fd(vdot(v, w, sS + k * p, w, sY + k * p, p, parts), vdot(v, w, sY + k * p, w, sY + k * p, p, parts))
    else:
        gamma = fd(Float32(1), fmax(Float32(1), fsqrt(vdot(v, w, g, w, g, p, parts))))
    for j in range(v.tid, p, v.nt):
        st(w, dr + j, fm(gamma, ld(w, dr + j)))
    v.sync()
    for kk in range(count):
        var k = (head - count + kk + 2 * LBFGS_M) % LBFGS_M
        var b = fm(ld(w, rho + k), vdot(v, w, sY + k * p, w, dr, p, parts))
        var c = fs(al[k], b)
        for j in range(v.tid, p, v.nt):
            st(w, dr + j, fmad(c, ld(w, sS + k * p + j), ld(w, dr + j)))
        v.sync()
    for j in range(v.tid, p, v.nt):
        st(w, dr + j, -ld(w, dr + j))
    v.sync()
    var slope = vdot(v, w, g, w, dr, p, parts)
    if not (slope < 0):
        count = 0
        head = 0
        for j in range(v.tid, p, v.nt):
            st(w, dr + j, -ld(w, g + j))
        v.sync()
        slope = vdot(v, w, g, w, dr, p, parts)
        if not (slope < 0):
            return (1, slope, count, head)
    return (0, slope, count, head)


def lb_accept(v: Team, w: FP, gn: Int, dr: Int, p: Int, f: Float32, fnew: Float32, tt: Float32, slope: Float32,
              parts: FP) -> Int:
    """1 when the trial at step tt is accepted: Armijo (c1 = 1e-4), or,
    inside float32's objective noise (1e-6 relative), the strong Wolfe
    curvature test (c2 = 0.9). Every thread returns the same."""
    if fnew == fnew and fnew <= fa(f, fm(fm(Float32(1e-4), tt), slope)):
        return 1
    # float32's objective is noisy at about 1e-7 relative, so a decrease
    # below that cannot be seen: accept a step whose objective is within
    # that noise when the directional derivative has fallen to 0.9 of its
    # start (the strong Wolfe curvature test)
    if fnew == fnew and fnew <= fa(f, fa(fm(Float32(1e-6), fabs(f)), Float32(1e-30))):
        var dg = vdot(v, w, gn, w, dr, p, parts)
        if fabs(dg) <= fm(Float32(0.9), fabs(slope)):
            return 1
    return 0


def lb_pair(
    v: Team, th: FP, toff: Int, tn: FP, tnoff: Int, w: FP, g: Int, gn: Int, sS: Int, sY: Int, rho: Int,
    p: Int, count_in: Int, head_in: Int, parts: FP,
) -> Tuple[Int, Int]:
    """The new pair s = tn - th, y = gn - g into slot head; kept (rho, head
    and count advance) when s'y > 1e-10 |s||y| and s'y > 0. A rejected pair
    over the oldest of a full history drops that pair. (count, head)."""
    var count = count_in
    var head = head_in
    var k = head
    for j in range(v.tid, p, v.nt):
        st(w, sS + k * p + j, fs(ld(tn, tnoff + j), ld(th, toff + j)))
        st(w, sY + k * p + j, fs(ld(w, gn + j), ld(w, g + j)))
    v.sync()
    var sy = vdot(v, w, sS + k * p, w, sY + k * p, p, parts)
    var ss = vdot(v, w, sS + k * p, w, sS + k * p, p, parts)
    var yy = vdot(v, w, sY + k * p, w, sY + k * p, p, parts)
    if sy > fm(Float32(1e-10), fsqrt(fm(ss, yy))) and sy > 0:
        if v.lead():
            st(w, rho + k, fd(Float32(1), sy))
        head = (head + 1) % LBFGS_M
        if count < LBFGS_M:
            count += 1
    elif count == LBFGS_M:
        count = LBFGS_M - 1
    v.sync()
    return (count, head)


def lbfgs[obj: Objective](
    t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP,
    theta: FP, toff: Int, p: Int, max_iter: Int, tol: Float32, fw: FP, woff: Int, sc: FP,
) -> Int:
    """Minimizes obj over theta[toff:toff+p] in place. Returns iterations
    run (negative when it stopped on max_iter without meeting tol). The
    vector phases run on the lead alone (a team of one: no scratch)."""
    var tn = woff
    var g = tn + p
    var gn = g + p
    var dr = gn + p
    var sS = dr + p
    var sY = sS + LBFGS_M * p
    var rho = sY + LBFGS_M * p
    var f = obj(t, x, y, n, d, ip, fp, theta, toff, fw, g, sc)
    # the pair ring lives on the lead thread only (it is read nowhere else)
    var v = solo_view(t)
    var count = 0
    var head = 0
    var it = 0
    while it < max_iter:
        var flag = 0  # 0 search, 1 stop
        var slope = Float32(0)
        if t.lead():
            var r = lb_direction(v, fw, g, dr, sS, sY, rho, p, count, head, tol, fw)
            flag = r[0]
            slope = r[1]
            count = r[2]
            head = r[3]
        if t.bcast_int(flag, 1) == 1:
            return it
        var tt = Float32(1)
        var accepted = False
        var fnew = f
        for _ in range(40):
            if t.lead():
                for j in range(p):
                    st(fw, tn + j, fmad(tt, ld(fw, dr + j), ld(theta, toff + j)))
            t.sync()
            fnew = obj(t, x, y, n, d, ip, fp, fw, tn, fw, gn, sc)
            var ok = 0
            if t.lead():
                ok = lb_accept(v, fw, gn, dr, p, f, fnew, tt, slope, fw)
            if t.bcast_int(ok, 2) == 1:
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        if not accepted:
            return it
        it += 1
        if t.lead():
            var r = lb_pair(v, theta, toff, fw, tn, fw, g, gn, sS, sY, rho, p, count, head, fw)
            count = r[0]
            head = r[1]
            copy(theta, toff, fw, tn, p)
            copy(fw, g, fw, gn, p)
        t.sync()
        f = fnew
    return -it
