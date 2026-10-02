# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Nelder-Mead, statsforecast's (`include/statsforecast/nelder_mead.h`,
`nm::NelderMead`): box-clamped simplex, the initial simplex perturbed by
init_step (zero_pert at a zero coordinate), adaptive coefficients
gamma = 1 + 2/n, rho = 0.75 - 1/(2n), sigma = 1 - 1/n, stop when the
population standard deviation of the simplex values falls below tol_std,
reflection / expansion / outside and inside contraction / shrink exactly as
there, in float32. Runs inside one thread (one series).

The one difference: the reference orders the simplex with std::sort, whose
order among EQUAL values is unspecified; here ties keep the lower vertex
index first (a stable insertion sort), so the order is a function of the
values alone.

The objective is chosen at compile time: `Obj.eval(x)` of a struct
conforming to `Objective`."""
from std.memory import bitcast

from sequence.ops import FP, add, fma3, ld, mul, st, sub
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_sqrt


trait Objective:
    @always_inline
    def eval(mut self, x: FP) -> Float32:
        ...


@always_inline
def _clamp(v: Float32, lo: Float32, hi: Float32) -> Float32:
    var r = v if v > lo else lo
    return r if r < hi else hi


@always_inline
def _put(p: FP, i: Int, v: Float32) -> Bool:
    """st(p, i, v), and whether the word stored differs from the old one."""
    var old = bitcast[DType.uint32](p.unsafe_load(i))
    var w = ftz(v)
    p.unsafe_store(i, w)
    return old != bitcast[DType.uint32](w)


struct NMState(ImplicitlyCopyable, Movable):
    """The loop state of `nelder_mead` between iterations (lane neural-pass143):
    with the simplex and its values in scratch, everything the next iteration
    reads. `nm_steps` runs iterations from it and leaves it where it stopped,
    so a fit sliced over several device launches runs the same iterations,
    in the same order, as one call."""
    var changed: Bool
    var have_snap: Bool
    var snap_it: Int
    var power: Int
    var stop_at: Int
    var it: Int
    var best: Int
    var stall_ref: Float32
    var stall_at: Int
    var done: Bool

    @always_inline
    def __init__(out self):
        self.changed = True
        self.have_snap = False
        self.snap_it = 0
        self.power = 1
        self.stop_at = -1
        self.it = 0
        self.best = 0
        self.stall_ref = Float32(0.0)
        self.stall_at = 0
        self.done = False


@always_inline
def nm_start[O: Objective](
    mut obj: O, x0: FP, lower: FP, upper: FP, n: Int, scratch: FP,
    init_step: Float32, zero_pert: Float32,
) -> NMState:
    """`nelder_mead`'s initial simplex and its n + 1 values (scratch), and
    the loop state before the first iteration."""
    var simplex = scratch
    var fs = scratch + (n + 1) * n
    for i in range(n + 1):
        for j in range(n):
            st(simplex, i * n + j, _clamp(ld(x0, j), ld(lower, j), ld(upper, j)))
    for i in range(n):
        var v = ld(simplex, i * n + i)
        if v == Float32(0.0):
            v = zero_pert
        else:
            v = mul(v, add(Float32(1.0), init_step))
        st(simplex, i * n + i, _clamp(v, ld(lower, i), ld(upper, i)))
    for i in range(n + 1):
        st(fs, i, obj.eval(simplex + i * n))
    return NMState()


@always_inline
def nm_steps[O: Objective, CAP: Int = 9](
    mut obj: O, mut s: NMState, lower: FP, upper: FP, n: Int, scratch: FP,
    max_iter: Int, tol_std: Float32,
    snap: FP = FP(unsafe_from_address=64),
    stall_iters: Int = 0, stall_rel: Float32 = Float32(0.0),
    budget: Int = -1,
) -> Int:
    """`nelder_mead`'s iterations from state `s`: at most `budget` of them
    (all when budget < 0). `s.done` says whether the loop ended; the return
    is the number of iterations run here."""
    var nf = Float32(n)
    var gamma = add(Float32(1.0), ftz(identical_div(Float32(2.0), nf)))
    var rho = sub(Float32(0.75), ftz(identical_div(Float32(1.0), mul(Float32(2.0), nf))))
    var sigma = sub(Float32(1.0), ftz(identical_div(Float32(1.0), nf)))
    var simplex = scratch
    var fs = scratch + (n + 1) * n
    var xo = fs + (n + 1)
    var xr = xo + n
    var xe = xr + n
    var xt = xe + n
    var order = InlineArray[Int, CAP](fill=0)
    # FIXED POINT (Apple speed, 2026-09-28): the loop's whole state is the
    # simplex and its values. An iteration that leaves both bit for bit as
    # it found them (a stalled simplex whose shrink rounds back onto itself)
    # makes every later iteration the same, ending in the same state with
    # the same last evaluations, so the loop jumps to max_iter: the same
    # result and iteration count as running them all.
    # Every write to the simplex or its values inside the loop goes through
    # `_put`, which says whether the stored word differs from the old one.
    var changed = s.changed
    var use_snap = Int(snap) != 64
    var n_state = (n + 1) * n + (n + 1)
    var have_snap = s.have_snap
    var snap_it = s.snap_it
    var power = s.power
    var stop_at = s.stop_at
    var it = s.it
    var best = s.best
    var stall_ref = s.stall_ref
    var stall_at = s.stall_at
    var steps = 0
    var finished = True
    while it < max_iter:
        # the slice boundary (lane neural-pass143): the loop's state is
        # exactly what `s` and the scratch hold here, so a later call
        # resumes with the same iteration
        if budget >= 0 and steps >= budget:
            finished = False
            break
        steps += 1
        if it == stop_at:
            it = max_iter
            break
        # stable argsort of fs (ties: lower index first)
        for i in range(n + 1):
            order[i] = i
        for i in range(1, n + 1):
            var k = order[i]
            var j = i - 1
            while j >= 0 and ld(fs, order[j]) > ld(fs, k):
                order[j + 1] = order[j]
                j -= 1
            order[j + 1] = k
        best = order[0]
        var worst = order[n]
        var second = order[n - 1]
        # population standard deviation of fs
        var mean = Float32(0.0)
        for i in range(n + 1):
            mean = add(mean, ld(fs, i))
        mean = ftz(identical_div(mean, Float32(n + 1)))
        var ss = Float32(0.0)
        for i in range(n + 1):
            var d = sub(ld(fs, i), mean)
            ss = fma3(d, d, ss)
        if ftz(identical_sqrt(ftz(identical_div(ss, Float32(n + 1))))) < tol_std:
            break
        comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
            if stall_iters > 0:
                var fb = ld(fs, best)
                if it == 0 or stall_ref - fb > stall_rel * abs(stall_ref):
                    stall_ref = fb
                    stall_at = it
                elif it - stall_at >= stall_iters:
                    break
        if not changed:
            it = max_iter
            break
        changed = False
        if use_snap and stop_at < 0:
            if have_snap:
                var same = True
                for i in range(n_state):
                    if bitcast[DType.uint32](scratch.unsafe_load(i)) != bitcast[DType.uint32](snap.unsafe_load(i)):
                        same = False
                        break
                if same:
                    var period = it - snap_it
                    var rest = (max_iter - it) % period
                    stop_at = it + (rest if rest > 0 else period)
            if stop_at < 0 and (not have_snap or it - snap_it == power):
                for i in range(n_state):
                    snap.unsafe_store(i, scratch.unsafe_load(i))
                if have_snap:
                    power *= 2
                snap_it = it
                have_snap = True
        # centroid without the worst vertex
        for j in range(n):
            var s = Float32(0.0)
            for i in range(n + 1):
                s = add(s, ld(simplex, i * n + j))
            st(xo, j, ftz(identical_div(sub(s, ld(simplex, worst * n + j)), nf)))
        # reflection (alpha = 1)
        for j in range(n):
            var o = ld(xo, j)
            st(xr, j, _clamp(add(o, sub(o, ld(simplex, worst * n + j))), ld(lower, j), ld(upper, j)))
        var fr = obj.eval(xr)
        if ld(fs, best) <= fr and fr < ld(fs, second):
            for j in range(n):
                changed |= _put(simplex, worst * n + j, ld(xr, j))
            changed |= _put(fs, worst, fr)
            it += 1
            continue
        if fr < ld(fs, best):
            for j in range(n):
                var o = ld(xo, j)
                st(xe, j, _clamp(fma3(gamma, sub(ld(xr, j), o), o), ld(lower, j), ld(upper, j)))
            var fe = obj.eval(xe)
            if fe < fr:
                for j in range(n):
                    changed |= _put(simplex, worst * n + j, ld(xe, j))
                changed |= _put(fs, worst, fe)
            else:
                for j in range(n):
                    changed |= _put(simplex, worst * n + j, ld(xr, j))
                changed |= _put(fs, worst, fr)
            it += 1
            continue
        var accepted = False
        if ld(fs, second) <= fr and fr < ld(fs, worst):
            for j in range(n):
                var o = ld(xo, j)
                st(xt, j, _clamp(fma3(rho, sub(ld(xr, j), o), o), ld(lower, j), ld(upper, j)))
            var fc = obj.eval(xt)
            if fc <= fr:
                for j in range(n):
                    changed |= _put(simplex, worst * n + j, ld(xt, j))
                changed |= _put(fs, worst, fc)
                accepted = True
        else:
            for j in range(n):
                var o = ld(xo, j)
                st(xt, j, _clamp(sub(o, mul(rho, sub(ld(xr, j), o))), ld(lower, j), ld(upper, j)))
            var fc = obj.eval(xt)
            if fc < ld(fs, worst):
                for j in range(n):
                    changed |= _put(simplex, worst * n + j, ld(xt, j))
                changed |= _put(fs, worst, fc)
                accepted = True
        if not accepted:
            for i in range(n + 1):
                if i == best:
                    continue
                for j in range(n):
                    var b = ld(simplex, best * n + j)
                    changed |= _put(simplex, i * n + j,
                       _clamp(fma3(sigma, sub(ld(simplex, i * n + j), b), b), ld(lower, j), ld(upper, j)))
                changed |= _put(fs, i, obj.eval(simplex + i * n))
        it += 1
    s.changed = changed
    s.have_snap = have_snap
    s.snap_it = snap_it
    s.power = power
    s.stop_at = stop_at
    s.it = it
    s.best = best
    s.stall_ref = stall_ref
    s.stall_at = stall_at
    s.done = finished
    return steps


@always_inline
def nm_finish(x0: FP, n: Int, scratch: FP, s: NMState) -> Int:
    """The best vertex into x0; the iteration count `nelder_mead` returns."""
    var simplex = scratch
    for j in range(n):
        st(x0, j, ld(simplex, s.best * n + j))
    return s.it + 1


@always_inline
def nelder_mead[O: Objective, CAP: Int = 9](
    mut obj: O, x0: FP, lower: FP, upper: FP, n: Int, scratch: FP,
    init_step: Float32, zero_pert: Float32, max_iter: Int, tol_std: Float32,
    snap: FP = FP(unsafe_from_address=64),
    stall_iters: Int = 0, stall_rel: Float32 = Float32(0.0),
) -> Int:
    """Minimises obj over n <= CAP - 1 coordinates from x0; the best point is
    written back to x0. scratch holds (n + 1) n + (n + 1) + 4 n floats.
    Returns the iteration count.

    CYCLES (Apple speed, 2026-09-28): with `snap` ((n + 1) n + (n + 1)
    floats of the caller's) the loop also watches for its state (the simplex
    and its values, the first words of scratch) returning to an earlier one,
    Brent's way (a snapshot at iterations 0, 1, 2, 4, 8, ...). The loop is a
    function of that state alone, so from a repeat at period p it runs only
    the (max_iter - it) mod p iterations that remain of the last lap (a full
    lap when that is 0, so the last sort is the one the full run ends on)
    and stops: the same final state, best vertex, last evaluations and
    iteration count as running every iteration.

    FAST STALL STOP (apple2, 2026-09-28; FAST builds only, stall_iters > 0):
    stop once the best value has not dropped by more than
    stall_rel |best| for stall_iters iterations. Not the reference's rule
    (it changes where a capped run ends), so an IDENTICAL build compiles it
    out; tools/sequence_quality.py holds its paired quality check."""
    var s = nm_start(obj, x0, lower, upper, n, scratch, init_step, zero_pert)
    _ = nm_steps[O, CAP](obj, s, lower, upper, n, scratch, max_iter, tol_std, snap, stall_iters, stall_rel)
    return nm_finish(x0, n, scratch, s)
