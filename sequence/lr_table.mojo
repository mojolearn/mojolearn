# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The learning-rate schedules' values: the binary64 TABLE, its float32
decision, the exact route over a run of steps, and the one entry every
public schedule method calls (`lr_spec_fill`).

A schedule's value at step t is DEFINED as the float32 nearest an exact
rational (or, for the cosine forms, the float32 both ends of the exact
route's Taylor interval round to; sequence/lr_exact.mojo). The table
evaluates a run of steps in binary64, each value with a rigorous error
bound, and keeps a value only when both ends of its error interval round to
the same normal float32; every other step takes the exact route. So the
answer at every step is the exact route's float32, whatever run the step is
evaluated in.

MOVED WORD FOR WORD (lane neural-optimizers, 2026-10-21) from the shell
bindings/lr_table_helpers.mojo: `cos_sin_pi`, `cos_run`,
`schedule_block_values`, `onecycle_fill`, `decide_many`, `pow_values` and
`exact_block` are the bodies of its entries, the same binary64 operations
in the same order (every product that meets an addition spelled
`pinned_mul_f64`, never contracted into an fma). NEW here: `lr_spec_fill`,
which was Python before: `_LrTable.lr_at` / `lr_values` (the block walk,
the constant tail), `OneCycleLR.__init__` / `_fast_values` (the phases and
their step membership, in `fractions.Fraction`) and `_PowSched`'s blocks.

Host scalar code on every column: a schedule value is one number per
optimizer step (the CPU route by nature, DEVIATION 5540). No Python here."""
from std.math import ceil, floor
from std.memory import bitcast
from std.os import getenv

from checks.numerics import pinned_mul_f64
from sequence.bigrat import Nat, Rat, ncmp, nshl, nsub, rat_of_f64, rdiv
from sequence.lr_exact import LR_OK, LrOut, onecycle_exact, pow_exact, schedule_exact

comptime _PI: Float64 = 3.141592653589793
comptime _COS_F64_TERMS = 26
comptime _F64_EPS: Float64 = 2.220446049250313e-16
comptime _F32_MIN_NORMAL: Float64 = 1.1754943508222875e-38
comptime _F32_MAX: Float64 = 3.4028234663852886e38

comptime F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime U8P = MutPointer[UInt8, MutUntrackedOrigin]

#: steps of one table run (`_training_impl._LR_BLOCK`)
comptime LR_BLOCK = 256

#: schedule kinds of `LrSpec`
comptime LR_CONSTANT = 0
comptime LR_LINEAR = 1
comptime LR_COSINE = 2
comptime LR_ONECYCLE = 10
comptime LR_POW = 20

#: statuses beyond sequence/lr_exact.mojo's 0 ok, 1 overflow, 2 straddle,
#: 3 bad progress
comptime LR_ONE_BASED = 4      # a step below 1
comptime LR_BEYOND = 5         # OneCycleLR past total_steps + 1

#: added to an `LrSpec.kind`: every step on the exact route (a schedule's
#: `_lr_at_slow`, the table's reference)
comptime LR_EXACT_ONLY = 100


def cos_sin_pi(q_in: Float64) -> Tuple[Float64, Float64]:
    """(cos(pi q), sin(pi q)) by the 26-term Taylor series on q or 1 - q in
    [0, 1/2]."""
    var q = q_in
    var flip = q > 0.5
    if flip:
        q = 1.0 - q
    var x = pinned_mul_f64(_PI, q)
    var x2 = pinned_mul_f64(x, x)
    var c: Float64 = 0.0
    var sn: Float64 = 0.0
    var tc: Float64 = 1.0
    var ts = x
    for k in range(_COS_F64_TERMS):
        c += tc
        sn += ts
        tc = pinned_mul_f64(-tc, x2) / Float64((2 * k + 1) * (2 * k + 2))
        ts = pinned_mul_f64(-ts, x2) / Float64((2 * k + 2) * (2 * k + 3))
    return (-c if flip else c, sn)


def cos_run(n0: Float64, den: Float64, count: Int, mut cs: List[Float64], mut ce: List[Float64]) -> Bool:
    """cos(pi (n0 + k) / den) for k < count into cs and the error bound
    2^-43 + k 2^-45 into ce; False when the step angle pi / den exceeds 1/2."""
    var step = 1.0 / den
    if not (pinned_mul_f64(_PI, step) <= 0.5):
        return False
    var a = cos_sin_pi(n0 / den)
    var c = a[0]
    var sn = a[1]
    var r = cos_sin_pi(step)
    var cp = r[0]
    var sp = r[1]
    for k in range(count):
        cs[k] = c
        ce[k] = 1.1368683772161603e-13 + pinned_mul_f64(Float64(k), 2.842170943040401e-14)
        var nc = pinned_mul_f64(c, cp) - pinned_mul_f64(sn, sp)
        var ns = pinned_mul_f64(sn, cp) + pinned_mul_f64(c, sp)
        c = nc
        sn = ns
    return True


def schedule_block_values(
    kind: Int, peak: Float64, lo: Float64, w: Int, total: Int, t0: Int, t1: Int, vs: F64P, es: F64P,
) -> Int:
    """`_Schedule`'s binary64 values and bounds of steps t0 .. t1 - 1 into
    vs / es. kind 0 constant, 1 linear, 2 cosine; total -1: none. Returns 1
    when the run has no binary64 route (every step takes the exact route),
    else 0."""
    var m = 0
    var b = min(t1, w + 1)
    if t0 < b:
        if w >= (1 << 29):
            return 1
        # peak t is exact (a float32 times an integer under 2^29), the
        # division rounds once: bound 2^-50 of v
        for t in range(t0, b):
            var v = pinned_mul_f64(peak, Float64(t)) / Float64(w)
            vs[m] = v
            es[m] = pinned_mul_f64(v, 8.881784197001252e-16)
            m += 1
    var a = max(t0, w + 1)
    if a < t1:
        var k0 = a - w
        var k1 = t1 - w
        if kind == 1:
            var span = total - w
            if span >= (1 << 29):
                return 1
            var d = lo - peak
            var base = peak + abs(lo)
            for k in range(k0, k1):
                var v = peak + pinned_mul_f64(d, Float64(k)) / Float64(span)
                vs[m] = v
                es[m] = pinned_mul_f64(base + abs(v), 8.881784197001252e-16)
                m += 1
        elif kind == 2:
            var span_steps = total - w
            var span = peak - lo
            if not (span > 0.0) or span_steps >= (1 << 52):
                return 1
            var count = k1 - k0
            var cs = List[Float64](length=count, fill=0.0)
            var ce = List[Float64](length=count, fill=0.0)
            if not cos_run(Float64(k0), Float64(span_steps), count, cs, ce):
                return 1
            for j in range(count):
                var v = lo + pinned_mul_f64(span, 1.0 + cs[j]) / 2.0
                vs[m] = v
                es[m] = pinned_mul_f64(span, ce[j]) + pinned_mul_f64(8.0 * _F64_EPS, peak + abs(v))
                m += 1
        else:
            return 1
    return 0


def onecycle_fill(
    linear: Bool, at: Int, st0: Int, count: Int, start_f: Float64, end_f: Float64, af: Float64, bf: Float64,
    vs: F64P, es: F64P,
):
    """One OneCycleLR phase's binary64 values and bounds into vs / es from
    `at`: torch's linear form d ((st - start) / den) + a or its cosine form
    end + (start - end) / 2 (cos + 1). A cosine step angle over 1/2 leaves the
    entries untouched."""
    var den = end_f - start_f
    var scale = 16.0 * _F64_EPS
    if linear:
        var d = bf - af
        for k in range(count):
            var v = pinned_mul_f64(d, (Float64(st0 + k) - start_f) / den) + af
            vs[at + k] = v
            es[at + k] = pinned_mul_f64(scale, abs(af) + abs(bf) + abs(v))
        return
    var cs = List[Float64](length=count, fill=0.0)
    var ce = List[Float64](length=count, fill=0.0)
    if not cos_run(Float64(st0) - start_f, den, count, cs, ce):
        return
    var hd = abs(af - bf)
    for k in range(count):
        var v = bf + pinned_mul_f64((af - bf) / 2.0, cs[k] + 1.0)
        vs[at + k] = v
        es[at + k] = pinned_mul_f64(hd, ce[k]) + pinned_mul_f64(scale, abs(af) + abs(bf) + abs(v))


def decide_many(vs: F64P, es: F64P, count: Int, out: F32P, ok: U8P) -> Int:
    """Per value the float32 every real in [v - e, v + e] rounds to (ties to
    even), kept (ok = 1) when both ends round to the same normal positive
    float32. out gets that float32, ok 0/1. Returns how many steps were NOT
    decided."""
    var undecided = 0
    for i in range(count):
        var a = Float32(vs[i] - es[i])
        var b = Float32(vs[i] + es[i])
        out[i] = a
        var keep = a == b and Float64(a) >= _F32_MIN_NORMAL and Float64(a) <= _F32_MAX
        ok[i] = UInt8(1) if keep else UInt8(0)
        if not keep:
            undecided += 1
    return undecided


def pow_values(base: Float64, gamma: Float64, e0: Int, n: Int, step: Int, out: F32P) raises -> Tuple[Int, Int]:
    """StepLR / ExponentialLR: out[i] = the float32 nearest
    base gamma^((e0 + i) // step) for i < n. Returns (0, n), or (status, i)
    at the first failing step i."""
    var last_e = -1
    var last_v = Float32(0)
    for i in range(n):
        var e = (e0 + i) // step
        if e != last_e:
            var r = pow_exact(base, gamma, e)
            if r.status != 0:
                return (r.status, i)
            last_e = e
            last_v = r.value
        out[i] = last_v
    return (0, n)


def exact_block(
    kind: Int, ip: List[Int], fp: List[Float64], t0: Int, n: Int, out: F32P, ok: U8P, has_ok: Bool,
) raises -> Tuple[Int, Int]:
    """The exact route over a run: out[i] for every i < n whose ok[i] is 0
    (has_ok False: every i). kind 0: `_Schedule`, ip = [kind, warmup, total],
    fp = [peak, min_lr]; kind 1: OneCycleLR, ip = [linear, three_phase,
    total_steps], fp = [max_lr, div_factor, final_div_factor, phase-1 end,
    phase-2 end]. Returns (0, 0), or the first failing step's status and its
    index."""
    for i in range(n):
        if has_ok and ok[i] != 0:
            continue
        var r: LrOut
        if kind == 0:
            r = schedule_exact(ip[0], fp[0], fp[1], ip[1], ip[2], t0 + i)
        else:
            # OneCycleLR's torch step is t - 1
            r = onecycle_exact(ip[0] != 0, ip[1] != 0, fp[0], fp[1], fp[2], fp[3], fp[4], ip[2], t0 + i - 1)
        if r.status != 0:
            return (r.status, i)
        out[i] = r.value
    return (0, 0)


# ----------------------------------------------------------- the public entry


def nat_f64(num: Nat, den: Nat, neg: Bool) -> Float64:
    """Python's `float(Fraction(num, den))`: the nearest float64, ties to
    even, gradual underflow (den > 0; magnitudes past the float64 range give
    an infinity, where Python raises; no schedule value comes near it)."""
    if num.is_zero():
        return -0.0 if neg else 0.0
    # scaled(e) = num / (den 2^e) in [2^52, 2^53)
    var e = num.bits() - den.bits() - 53
    var A: Nat
    var B: Nat
    if e >= 0:
        A = num.copy()
        B = nshl(den, e)
    else:
        A = nshl(num, -e)
        B = den.copy()
    var hi53 = nshl(B, 53)
    while ncmp(A, hi53) >= 0:
        e += 1
        B = nshl(B, 1)
        hi53 = nshl(B, 53)
    var lo52 = nshl(B, 52)
    while ncmp(A, lo52) < 0:
        e -= 1
        A = nshl(A, 1)
        lo52 = nshl(B, 52)
    # below the smallest normal the quantum is 2^-1074: fewer bits survive
    var s = 0
    if e < -1074:
        s = -1074 - e
        B = nshl(B, s)
        e = -1074
    # m = floor(A / B) (under 2^(53 - s)), by restoring steps
    var m: UInt64 = 0
    var bit = 52 - s
    while bit >= 0:
        var t = nshl(B, bit)
        if ncmp(A, t) >= 0:
            A = nsub(A, t)
            m |= UInt64(1) << UInt64(bit)
        bit -= 1
    var c = ncmp(nshl(A, 1), B)
    if c > 0 or (c == 0 and (m & 1) == 1):
        m += 1
    var bits: UInt64
    if s > 0:
        # subnormal: m 2^-1074 (a carry into 2^52 is the smallest normal)
        bits = m
    else:
        if m == (UInt64(1) << 53):
            m = UInt64(1) << 52
            e += 1
        if e + 52 > 1023:
            bits = UInt64(0x7FF) << 52
        else:
            bits = (UInt64(e + 52 + 1023) << 52) | (m & UInt64(0xFFFFFFFFFFFFF))
    if neg:
        bits |= UInt64(1) << 63
    return bitcast[DType.float64](bits)


@fieldwise_init
struct LrSpec(Copyable, ImplicitlyCopyable, Movable):
    """One built-in schedule's configuration.

    kind 0 / 1 / 2 (ConstantLR, WarmupLinearLR, WarmupCosineLR): a = warmup
    steps, b = total steps (-1: none), f0 = peak_lr, f1 = min_lr.
    kind 10 (OneCycleLR): a = linear (1) or cosine (0), b = three_phase,
    c = total_steps, f0 = max_lr, f1 = div_factor, f2 = final_div_factor,
    f3 = pct_start.
    kind 20 (StepLR, ExponentialLR): a = step_size (1: exponential),
    f0 = base_lr, f1 = gamma."""
    var kind: Int
    var a: Int
    var b: Int
    var c: Int
    var f0: Float64
    var f1: Float64
    var f2: Float64
    var f3: Float64


def lr_table_on() -> Bool:
    """MOJOLEARN_LR_TABLE=0 or MOJOLEARN_LR_EXACT_ONLY=1 keeps every step on
    the exact route (the table's A/B); the answers are the same."""
    return String(getenv("MOJOLEARN_LR_TABLE")) != "0" and String(getenv("MOJOLEARN_LR_EXACT_ONLY")) != "1"


def _decide_run(vs: List[Float64], es: List[Float64], n: Int, out: F32P, mut ok: List[UInt8]) -> Int:
    var vp = vs.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var ep = es.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]()
    var kp = ok.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    return decide_many(vp, ep, n, out, kp)


def _onecycle_runs(spec: LrSpec, t0: Int, t1: Int, mut vs: List[Float64], mut es: List[Float64]):
    """`OneCycleLR._fast_values(t0, t1)`: the binary64 values of the steps
    strictly inside each phase (zeros where the exact route answers: a
    phase's two ends, its extra step, an empty phase, a step angle over 1/2).
    The phase ends and rates as `OneCycleLR.__init__` builds them: the ends
    in float64 as torch does, the rates the float64 nearest the exact
    rationals."""
    var linear = spec.a != 0
    var total = spec.c
    var e1 = pinned_mul_f64(spec.f3, Float64(total)) - 1.0
    var e2 = pinned_mul_f64(pinned_mul_f64(2.0, spec.f3), Float64(total)) - 2.0
    var mx = spec.f0
    var init = mx / spec.f1
    var low = Float64(0.0)
    try:
        var q = rdiv(rdiv(rat_of_f64(mx), rat_of_f64(spec.f1)), rat_of_f64(spec.f2))
        low = nat_f64(q.n.m, q.d, q.n.neg)
    except:
        # a zero divisor is refused when the schedule is built
        return
    var ends = List[Float64]()
    var aa = List[Float64]()
    var bb = List[Float64]()
    ends.append(e1)
    aa.append(init)
    bb.append(mx)
    if spec.b != 0:
        ends.append(e2)
        aa.append(mx)
        bb.append(init)
        ends.append(Float64(total - 1))
        aa.append(init)
        bb.append(low)
    else:
        ends.append(Float64(total - 1))
        aa.append(mx)
        bb.append(low)
    var n = t1 - t0
    var s0 = t0 - 1
    var s1 = s0 + n - 1
    var start = Float64(0.0)
    var last = len(ends) - 1
    var vp = vs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var ep = es.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for i in range(len(ends)):  # the at most three OneCycle phases
        var end = ends[i]
        if end != start:
            var fs = Int(floor(start))
            var lo = s0 if i == 0 else max(s0, fs + 1)
            var hi = s1 if i == last else min(s1, Int(floor(end)))
            lo = max(lo, fs + 1)
            hi = min(hi, Int(ceil(end)) - 1)
            if lo <= hi:
                onecycle_fill(linear, lo - s0, lo, hi - lo + 1, start, end, aa[i], bb[i], vp, ep)
        start = end


def _exact_words(spec: LrSpec) -> Tuple[Int, List[Int], List[Float64]]:
    """`exact_block`'s (kind, ip, fp) for a `_Schedule` or a OneCycleLR."""
    var ip = List[Int]()
    var fp = List[Float64]()
    if spec.kind == LR_ONECYCLE:
        ip.append(spec.a)
        ip.append(spec.b)
        ip.append(spec.c)
        fp.append(spec.f0)
        fp.append(spec.f1)
        fp.append(spec.f2)
        fp.append(pinned_mul_f64(spec.f3, Float64(spec.c)) - 1.0)
        fp.append(pinned_mul_f64(pinned_mul_f64(2.0, spec.f3), Float64(spec.c)) - 2.0)
        return (1, ip^, fp^)
    ip.append(spec.kind)
    ip.append(spec.a)
    ip.append(spec.b)
    fp.append(spec.f0)
    fp.append(spec.f1)
    return (0, ip^, fp^)


def lr_spec_fill(spec: LrSpec, t0: Int, n: Int, out: F32P) raises -> Tuple[Int, Int]:
    """The float32 learning rates of ONE-BASED steps t0 .. t0 + n - 1 into
    out. Returns (0, n), or (status, i): the first step that failed, at
    index i (LR_ONE_BASED below step 1, LR_BEYOND for a OneCycleLR step past
    total_steps + 1, and sequence/lr_exact.mojo's statuses). This is
    `lr_at(t)` (n = 1) and `lr_values(n)` (t0 = 1) of every built-in
    schedule, with the answers they always had."""
    if n <= 0:
        return (0, 0)
    if t0 < 1:
        return (LR_ONE_BASED, 0)
    var exact_only = spec.kind >= LR_EXACT_ONLY
    if exact_only:
        var plain = spec
        plain.kind = spec.kind - LR_EXACT_ONLY
        return _lr_spec_fill(plain, t0, n, out, False)
    return _lr_spec_fill(spec, t0, n, out, lr_table_on())


def _lr_spec_fill(spec: LrSpec, t0: Int, n: Int, out: F32P, table: Bool) raises -> Tuple[Int, Int]:
    if spec.kind == LR_POW:
        return pow_values(spec.f0, spec.f1, t0 - 1, n, spec.a, out)
    var tail_t: Int
    if spec.kind == LR_ONECYCLE:
        tail_t = spec.c + 2
    elif spec.b < 0:
        tail_t = spec.a + 1
    else:
        tail_t = max(spec.b, spec.a + 1)
    var words = _exact_words(spec)
    var ek = words[0]
    var ip = words[1].copy()
    var fp = words[2].copy()
    var t = t0
    var i = 0
    while i < n:
        if t >= tail_t:
            if spec.kind == LR_ONECYCLE:
                return (LR_BEYOND, i)
            # the constant tail: every later step has the tail step's value
            var tv = schedule_exact(spec.kind, spec.f0, spec.f1, spec.a, spec.b, tail_t)
            if tv.status != 0:
                return (tv.status, i)
            for k in range(i, n):
                out[k] = tv.value
            return (0, n)
        var t1 = min(min(t + LR_BLOCK, tail_t), t0 + n)
        var cnt = t1 - t
        var ok = List[UInt8](length=cnt, fill=UInt8(0))
        var decided = False
        if table:
            var vs = List[Float64](length=cnt, fill=0.0)
            var es = List[Float64](length=cnt, fill=0.0)
            var routed = True
            if spec.kind == LR_ONECYCLE:
                _onecycle_runs(spec, t, t1, vs, es)
            else:
                var vp = vs.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
                var ep = es.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
                routed = schedule_block_values(spec.kind, spec.f0, spec.f1, spec.a, spec.b, t, t1, vp, ep) == 0
            if routed:
                _ = _decide_run(vs, es, cnt, out + i, ok)
                decided = True
        var okp = ok.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
        var r = exact_block(ek, ip, fp, t, cnt, out + i, okp, decided)
        _ = ok^
        if r[0] != 0:
            return (r[0], i + r[1])
        t = t1
        i += cnt
    return (0, n)


def lr_spec_value(spec: LrSpec, t: Int) raises -> LrOut:
    """`lr_at(t)`: one step's float32 and its status."""
    var v = List[Float32](length=1, fill=Float32(0))
    var r = lr_spec_fill(spec, t, 1, v.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]())
    return LrOut(v[0], r[0])
