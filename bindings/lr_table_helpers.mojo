# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The learning-rate schedule TABLE in Mojo (lane py-runtime round 2): the
binary64 block route of `_training_impl._LrTable` and OneCycleLR's
`_fill`, which ran per optimizer step in Python.

A schedule's value at step t is DEFINED as the float32 nearest an exact
rational. The table evaluates a block of steps in binary64, each value with
a rigorous error bound, and keeps a value only when both ends of its error
interval round to the same normal float32 (`lr_decide`); every other step
takes the exact route, which stays Python's big-rational arithmetic (it
needs an arbitrary-precision rational Mojo's stdlib does not have; an owner
decision, docs/identical/py-runtime-classified.tsv). So the answer at every
step is the exact route's float32, as before.

SAME VALUES AS THE PYTHON THEY REPLACE: the same IEEE binary64 operations
in the same order, every product that meets an addition spelled
`pinned_mul_f64` so it is never contracted into an fma (Python never
fuses), and the same decision rule. The table only ever answers with a
value the exact route would give, so even a different binary64 value
would not change a float32; the bounds are copied with their margins.

Every loop here walks one block of at most `_LR_BLOCK` (256) steps or the
26 terms of a Taylor series: host scalar code on every column (no data
size, no device)."""
from std.math import isfinite
from std.python import PythonObject

from checks.numerics import pinned_mul_f64

comptime _PI: Float64 = 3.141592653589793
comptime _COS_F64_TERMS = 26
comptime _F64_EPS: Float64 = 2.220446049250313e-16
comptime _F32_MIN_NORMAL: Float64 = 1.1754943508222875e-38
comptime _F32_MAX: Float64 = 3.4028234663852886e38


def _f64p(addr: PythonObject) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    var a = Int(py=addr)
    if a == 0:
        raise Error("lr table: null float64 buffer")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=a)


def _cos_sin_pi(q_in: Float64) -> Tuple[Float64, Float64]:
    """`_training_impl._cos_sin_pi_f64`: (cos(pi q), sin(pi q)) by the
    26-term Taylor series on q or 1 - q in [0, 1/2]."""
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
    for k in range(_COS_F64_TERMS):  # small-loop(_COS_F64_TERMS: Taylor terms): the fixed 26-term series
        c += tc
        sn += ts
        tc = pinned_mul_f64(-tc, x2) / Float64((2 * k + 1) * (2 * k + 2))
        ts = pinned_mul_f64(-ts, x2) / Float64((2 * k + 2) * (2 * k + 3))
    return (-c if flip else c, sn)


def _cos_run(n0: Float64, den: Float64, count: Int, mut cs: List[Float64], mut ce: List[Float64]) -> Bool:
    """`_training_impl._cos_pi_run`: cos(pi (n0 + k) / den) for k < count
    into cs and the error bound 2^-43 + k 2^-45 into ce; False when the step
    angle pi / den exceeds 1/2."""
    var step = 1.0 / den
    if not (pinned_mul_f64(_PI, step) <= 0.5):
        return False
    var a = _cos_sin_pi(n0 / den)
    var c = a[0]
    var sn = a[1]
    var r = _cos_sin_pi(step)
    var cp = r[0]
    var sp = r[1]
    for k in range(count):  # small-loop(count: steps of one table block): the rotation recurrence over one block
        cs[k] = c
        ce[k] = 1.1368683772161603e-13 + pinned_mul_f64(Float64(k), 2.842170943040401e-14)
        var nc = pinned_mul_f64(c, cp) - pinned_mul_f64(sn, sp)
        var ns = pinned_mul_f64(sn, cp) + pinned_mul_f64(c, sp)
        c = nc
        sn = ns
    return True


def lr_schedule_values_binding(params: PythonObject, vs_addr: PythonObject, es_addr: PythonObject) raises -> PythonObject:
    """`_Schedule._fast_values(t0, t1)` into vs / es (t1 - t0 float64 each).
    params: [kind (0 constant, 1 linear, 2 cosine), peak, lo, warmup,
    total (-1: none), t0, t1]. Returns 1 when the block has no binary64
    route (Python's None: every step takes the exact route), else 0."""
    var kind = Int(py=params[0])
    var peak = Float64(py=params[1])
    var lo = Float64(py=params[2])
    var w = Int(py=params[3])
    var total = Int(py=params[4])
    var t0 = Int(py=params[5])
    var t1 = Int(py=params[6])
    var vs = _f64p(vs_addr)
    var es = _f64p(es_addr)
    var m = 0
    var b = min(t1, w + 1)
    if t0 < b:
        if w >= (1 << 29):
            return PythonObject(1)
        # peak t is exact (a float32 times an integer under 2^29), the
        # division rounds once: bound 2^-50 of v
        for t in range(t0, b):  # small-loop(b: steps of one table block): the warmup steps of the block
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
                return PythonObject(1)
            var d = lo - peak
            var base = peak + abs(lo)
            for k in range(k0, k1):  # small-loop(k1: steps of one table block): the linear decay steps of the block
                var v = peak + pinned_mul_f64(d, Float64(k)) / Float64(span)
                vs[m] = v
                es[m] = pinned_mul_f64(base + abs(v), 8.881784197001252e-16)
                m += 1
        elif kind == 2:
            var span_steps = total - w
            var span = peak - lo
            if not (span > 0.0) or span_steps >= (1 << 52):
                return PythonObject(1)
            var count = k1 - k0
            var cs = List[Float64](length=count, fill=0.0)
            var ce = List[Float64](length=count, fill=0.0)
            if not _cos_run(Float64(k0), Float64(span_steps), count, cs, ce):
                return PythonObject(1)
            for j in range(count):  # small-loop(count: steps of one table block): the cosine decay steps of the block
                var v = lo + pinned_mul_f64(span, 1.0 + cs[j]) / 2.0
                vs[m] = v
                es[m] = pinned_mul_f64(span, ce[j]) + pinned_mul_f64(8.0 * _F64_EPS, peak + abs(v))
                m += 1
        else:
            return PythonObject(1)
    return PythonObject(0)


def lr_onecycle_fill_binding(params: PythonObject, vs_addr: PythonObject, es_addr: PythonObject) raises -> PythonObject:
    """`OneCycleLR._fill` into vs / es from index `at`. params: [linear (1)
    or cos (0), at, st0, count, start_f, end_f, af, bf]. Returns 0 (a cos
    step angle over 1/2 leaves the entries untouched, as Python did)."""
    var linear = Int(py=params[0]) != 0
    var at = Int(py=params[1])
    var st0 = Int(py=params[2])
    var count = Int(py=params[3])
    var start_f = Float64(py=params[4])
    var end_f = Float64(py=params[5])
    var af = Float64(py=params[6])
    var bf = Float64(py=params[7])
    var vs = _f64p(vs_addr)
    var es = _f64p(es_addr)
    var den = end_f - start_f
    var scale = 16.0 * _F64_EPS
    if linear:
        var d = bf - af
        for k in range(count):  # small-loop(count: steps of one table block): the linear phase steps of the block
            var v = pinned_mul_f64(d, (Float64(st0 + k) - start_f) / den) + af
            vs[at + k] = v
            es[at + k] = pinned_mul_f64(scale, abs(af) + abs(bf) + abs(v))
        return PythonObject(0)
    var cs = List[Float64](length=count, fill=0.0)
    var ce = List[Float64](length=count, fill=0.0)
    if not _cos_run(Float64(st0) - start_f, den, count, cs, ce):
        return PythonObject(0)
    var hd = abs(af - bf)
    for k in range(count):  # small-loop(count: steps of one table block): the cosine phase steps of the block
        var v = bf + pinned_mul_f64((af - bf) / 2.0, cs[k] + 1.0)
        vs[at + k] = v
        es[at + k] = pinned_mul_f64(hd, ce[k]) + pinned_mul_f64(scale, abs(af) + abs(bf) + abs(v))
    return PythonObject(0)


def lr_decide_binding(vs_addr: PythonObject, es_addr: PythonObject, n: PythonObject,
                      out_addr: PythonObject, ok_addr: PythonObject) raises -> PythonObject:
    """`_training_impl._f32_decide_many`: per value the float32 every real in
    [v - e, v + e] rounds to (ties to even), kept (ok = 1) when both ends
    round to the same normal positive float32. out gets that float32 (as a
    float32 word), ok 0/1. Returns how many steps were NOT decided."""
    var count = Int(py=n)
    var vs = _f64p(vs_addr)
    var es = _f64p(es_addr)
    var oa = Int(py=out_addr)
    var ka = Int(py=ok_addr)
    if count > 0 and (oa == 0 or ka == 0):
        raise Error("lr table: null output buffer")
    var out = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=oa)
    var ok = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=ka)
    var undecided = 0
    for i in range(count):  # small-loop(count: steps of one table block): one decision per step
        var a = Float32(vs[i] - es[i])
        var b = Float32(vs[i] + es[i])
        out[i] = a
        var keep = a == b and Float64(a) >= _F32_MIN_NORMAL and Float64(a) <= _F32_MAX
        ok[i] = UInt8(1) if keep else UInt8(0)
        if not keep:
            undecided += 1
    return PythonObject(undecided)
