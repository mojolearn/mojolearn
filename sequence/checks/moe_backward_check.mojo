# SPDX-License-Identifier: Apache-2.0
"""The MoE block's tape backward (autograd/route_ops.mojo) checked three ways.

For each fixture case (several shapes, top-1 and top-k, with and without
renormalisation, experts with no pairs, more pairs than one grouping
block):
  1. HOST ORACLE, WORD FOR WORD: the HostExec tape run against a host walk
     that assembles the same products with plain loops (its own grouping,
     its own chaining, the same cells), every word of y and of the four
     gradients equal.
  2. FLOAT64 REFERENCE: an independent float64 chain rule over the same
     picks (every product spelled out, no shared code with the kernels);
     every tensor within a relative error of 1e-3 of its largest entry.
  3. DEVICE == HOST: on a machine with a GPU, the DeviceExec run against
     the HostExec run word for word. Each tensor's digest (FNV-1a over the
     words) is printed so NVIDIA, AMD and Apple columns can be compared.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . sequence/checks/moe_backward_check.mojo
"""
from std.math import exp
from std.memory import bitcast
from std.sys import has_accelerator

from sequence.exec import HostExec
from sequence.exec_device import DeviceExec
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, add, mul
from sequence.moe import op_moe_hidden, op_moe_out, op_moe_route
from autograd.kernels.moe import moe_bwd_cell, moe_bwd_segment_cell, op_moeb_dgu, op_moeb_route_vjp
from autograd.route_ops import moe_tape_train_step


def digest(values: List[Float32]) -> UInt64:
    var hash = UInt64(1469598103934665603)
    for i in range(len(values)):
        hash = (hash ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return hash


def fixture(n: Int, salt: Int, scale: Float32) -> List[Float32]:
    var v = List[Float32]()
    for i in range(n):
        var h = (i * 2654435761 + salt * 40503 + 12345) % 2003
        v.append((Float32(h) / Float32(2003.0) - Float32(0.5)) * scale)
    return v^


def _fp(xs: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(xs.unsafe_ptr()))


@fieldwise_init
struct MoeCase(Copyable, Movable):
    var T: Int
    var D: Int
    var F: Int
    var E: Int
    var k: Int
    var renorm: Int


@fieldwise_init
struct MoeResult(Copyable, Movable):
    var y: List[Float32]
    var logits: List[Float32]
    var sel: List[Float32]
    var w: List[Float32]
    var dx: List[Float32]
    var dwg: List[Float32]
    var dgu: List[Float32]
    var ddn: List[Float32]


@fieldwise_init
struct MoeInputs(Copyable, Movable):
    var x: List[Float32]
    var wg: List[Float32]
    var gu: List[Float32]
    var dn: List[Float32]
    var dy: List[Float32]


def inputs_of(c: MoeCase, salt: Int) -> MoeInputs:
    return MoeInputs(
        fixture(c.T * c.D, salt + 1, Float32(2.0)),
        fixture(c.E * c.D, salt + 2, Float32(1.5)),
        fixture(c.E * 2 * c.F * c.D, salt + 3, Float32(0.8)),
        fixture(c.E * c.D * c.F, salt + 4, Float32(0.8)),
        fixture(c.T * c.D, salt + 5, Float32(1.0)),
    )


def run_tape[E: Exec](mut ex: E, c: MoeCase, inp: MoeInputs) raises -> MoeResult:
    var T = c.T
    var D = c.D
    var F = c.F
    var En = c.E
    var k = c.k
    var X = ex.alloc(T * D)
    ex.upload(X, _fp(inp.x), T * D)
    var Wg = ex.alloc(En * D)
    ex.upload(Wg, _fp(inp.wg), En * D)
    var Gu = ex.alloc(En * 2 * F * D)
    ex.upload(Gu, _fp(inp.gu), En * 2 * F * D)
    var Dn = ex.alloc(En * D * F)
    ex.upload(Dn, _fp(inp.dn), En * D * F)
    var DY = ex.alloc(T * D)
    ex.upload(DY, _fp(inp.dy), T * D)
    var DX = ex.alloc(T * D)
    var DWg = ex.alloc(En * D)
    var DGu = ex.alloc(En * 2 * F * D)
    var DDn = ex.alloc(En * D * F)
    var Y = ex.alloc(T * D)
    var L = ex.alloc(T * En)
    var Sel = ex.alloc(T * k)
    var W = ex.alloc(T * k)
    moe_tape_train_step(ex, X, Wg, Gu, Dn, DY, DX, DWg, DGu, DDn, Y, L, Sel, W, T, D, F, En, k, c.renorm)
    var y = List[Float32](length=T * D, fill=Float32(0.0))
    ex.download(_fp(y), Y, T * D)
    var lg = List[Float32](length=T * En, fill=Float32(0.0))
    ex.download(_fp(lg), L, T * En)
    var sl = List[Float32](length=T * k, fill=Float32(0.0))
    ex.download(_fp(sl), Sel, T * k)
    var w = List[Float32](length=T * k, fill=Float32(0.0))
    ex.download(_fp(w), W, T * k)
    var dx = List[Float32](length=T * D, fill=Float32(0.0))
    ex.download(_fp(dx), DX, T * D)
    var dwg = List[Float32](length=En * D, fill=Float32(0.0))
    ex.download(_fp(dwg), DWg, En * D)
    var dgu = List[Float32](length=En * 2 * F * D, fill=Float32(0.0))
    ex.download(_fp(dgu), DGu, En * 2 * F * D)
    var ddn = List[Float32](length=En * D * F, fill=Float32(0.0))
    ex.download(_fp(ddn), DDn, En * D * F)
    ex.sync()
    return MoeResult(y^, lg^, sl^, w^, dx^, dwg^, dgu^, ddn^)


def oracle_walk(c: MoeCase, inp: MoeInputs) raises -> MoeResult:
    """The same products assembled by plain host loops: the forward's
    element bodies called in turn, its own stable grouping, its own
    chaining of the VJPs (the gather's scatter-add in pick order, x's two
    terms added experts first)."""
    var T = c.T
    var D = c.D
    var F = c.F
    var En = c.E
    var k = c.k
    var P = T * k
    var lg = List[Float32](length=T * En, fill=Float32(0.0))
    var pr = List[Float32](length=T * En, fill=Float32(0.0))
    var sl = List[Float32](length=P, fill=Float32(0.0))
    var w = List[Float32](length=P, fill=Float32(0.0))
    var a = Args()
    a.p0 = _fp(inp.x)
    a.p1 = _fp(inp.wg)
    a.p2 = _fp(lg)
    a.p3 = _fp(sl)
    a.p4 = _fp(w)
    a.p5 = _fp(pr)
    a.i0 = D
    a.i1 = En
    a.i2 = k
    a.i3 = c.renorm
    for t in range(T):
        op_moe_route(t, a)
    # KEEP-ALIVE (Mojo frees a local at its last named use): `pr` is named
    # only in `_fp(pr)` above, so without this use it is freed before the
    # loop writes the probabilities through a.p5 (a heap write after free).
    _ = len(pr)
    var h = List[Float32](length=P * F, fill=Float32(0.0))
    var b = Args()
    b.p0 = _fp(inp.x)
    b.p1 = _fp(inp.gu)
    b.p2 = _fp(sl)
    b.p3 = _fp(h)
    b.i0 = D
    b.i1 = F
    b.i2 = k
    for t in range(P * F):
        op_moe_hidden(t, b)
    var y = List[Float32](length=T * D, fill=Float32(0.0))
    var o = Args()
    o.p0 = _fp(h)
    o.p1 = _fp(inp.dn)
    o.p2 = _fp(sl)
    o.p3 = _fp(w)
    o.p4 = _fp(y)
    o.i0 = D
    o.i1 = F
    o.i2 = k
    for t in range(T * D):
        op_moe_out(t, o)
    # stable grouping: each expert's pairs ascending
    var order = List[Float32]()
    var poff = List[Float32]()
    for e in range(En):
        poff.append(Float32(len(order)))
        for p in range(P):
            if Int(sl[p]) == e:
                order.append(Float32(p))
    poff.append(Float32(len(order)))
    # out VJP
    var dyp = _fp(inp.dy)
    var q = List[Float32](length=P * F, fill=Float32(0.0))
    var dh = List[Float32](length=P * F, fill=Float32(0.0))
    for p in range(P):
        var tok = p // k
        var e = Int(sl[p])
        for f in range(F):
            var v = moe_bwd_cell(dyp + tok * D, 1, _fp(inp.dn) + e * D * F + f, F, D)
            q[p * F + f] = v
            dh[p * F + f] = mul(w[p], v)
    var dw = List[Float32](length=P, fill=Float32(0.0))
    for p in range(P):
        dw[p] = moe_bwd_cell(_fp(h) + p * F, 1, _fp(q) + p * F, 1, F)
    var ddn = List[Float32](length=En * D * F, fill=Float32(0.0))
    for e in range(En):
        var s0 = Int(poff[e])
        var n = Int(poff[e + 1]) - s0
        for d in range(D):
            for f in range(F):
                ddn[(e * D + d) * F + f] = moe_bwd_segment_cell[True](
                    _fp(order), s0, n, k, dyp + d, 0, D, _fp(w), _fp(h) + f, F, 0,
                )
    # hidden VJP
    var dg = List[Float32](length=P * 2 * F, fill=Float32(0.0))
    var g = Args()
    g.p0 = _fp(inp.x)
    g.p1 = _fp(inp.gu)
    g.p2 = _fp(sl)
    g.p4 = _fp(dh)
    g.p5 = _fp(dg)
    g.i0 = D
    g.i1 = F
    g.i2 = k
    for t in range(P * F):
        op_moeb_dgu(t, g)
    _ = len(dh)  # keep-alive: g.p4 points into dh (read by every element above)
    var dx = List[Float32](length=T * D, fill=Float32(0.0))
    for tok in range(T):
        for d in range(D):
            var acc = Float32(0.0)
            for j in range(k):
                var p = tok * k + j
                var e = Int(sl[p])
                acc = add(acc, moe_bwd_cell(_fp(dg) + p * 2 * F, 1, _fp(inp.gu) + e * 2 * F * D + d, D, 2 * F))
            dx[tok * D + d] = acc
    var dgu = List[Float32](length=En * 2 * F * D, fill=Float32(0.0))
    for e in range(En):
        var s0 = Int(poff[e])
        var n = Int(poff[e + 1]) - s0
        for r in range(2 * F):
            for d in range(D):
                dgu[(e * 2 * F + r) * D + d] = moe_bwd_segment_cell[False](
                    _fp(order), s0, n, k, _fp(dg) + r, 2 * F, 0, _fp(dg), _fp(inp.x) + d, 0, D,
                )
    # route VJP
    var dl = List[Float32](length=T * En, fill=Float32(0.0))
    var rv = Args()
    rv.p0 = _fp(lg)
    rv.p1 = _fp(sl)
    rv.p2 = _fp(w)
    rv.p3 = _fp(dw)
    rv.p4 = _fp(dl)
    rv.i1 = En
    rv.i2 = k
    rv.i3 = c.renorm
    for t in range(T * En):
        op_moeb_route_vjp(t, rv)
    _ = len(dw)  # keep-alive: rv.p3 points into dw (read by every element above)
    for tok in range(T):
        for d in range(D):
            var r = moe_bwd_cell(_fp(dl) + tok * En, 1, _fp(inp.wg) + d, D, En)
            # the tape's ACCUMULATE: the experts' term first, the router's added
            dx[tok * D + d] = add(dx[tok * D + d], r)
    var dwg = List[Float32](length=En * D, fill=Float32(0.0))
    for e in range(En):
        for d in range(D):
            dwg[e * D + d] = moe_bwd_cell(_fp(dl) + e, En, _fp(inp.x) + d, D, T)
    return MoeResult(y^, lg^, sl^, w^, dx^, dwg^, dgu^, ddn^)


def _sig64(v: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-v))


def reference64(c: MoeCase, inp: MoeInputs, sel: List[Float32]) -> List[List[Float64]]:
    """y, dx, drouter, dgate_up, ddown in float64 by the chain rule, over
    the run's picks: [y, dx, dwg, dgu, ddn]."""
    var T = c.T
    var D = c.D
    var F = c.F
    var En = c.E
    var k = c.k
    var y = List[Float64](length=T * D, fill=0.0)
    var dx = List[Float64](length=T * D, fill=0.0)
    var dwg = List[Float64](length=En * D, fill=0.0)
    var dgu = List[Float64](length=En * 2 * F * D, fill=0.0)
    var ddn = List[Float64](length=En * D * F, fill=0.0)
    for t in range(T):
        var lg = List[Float64](length=En, fill=0.0)
        var mx = Float64(0.0)
        for e in range(En):
            var s = Float64(0.0)
            for d in range(D):
                s += Float64(inp.x[t * D + d]) * Float64(inp.wg[e * D + d])
            lg[e] = s
            if e == 0 or s > mx:
                mx = s
        var z = Float64(0.0)
        var pr = List[Float64](length=En, fill=0.0)
        for e in range(En):
            pr[e] = exp(lg[e] - mx)
            z += pr[e]
        for e in range(En):
            pr[e] = pr[e] / z
        var tot = Float64(0.0)
        for j in range(k):
            tot += pr[Int(sel[t * k + j])]
        var wj = List[Float64](length=k, fill=0.0)
        for j in range(k):
            var pv = pr[Int(sel[t * k + j])]
            wj[j] = pv / tot if c.renorm != 0 else pv
        var dwj = List[Float64](length=k, fill=0.0)
        for j in range(k):
            var e = Int(sel[t * k + j])
            var hh = List[Float64](length=F, fill=0.0)
            var gg = List[Float64](length=F, fill=0.0)
            var uu = List[Float64](length=F, fill=0.0)
            for f in range(F):
                var g = Float64(0.0)
                var u = Float64(0.0)
                for d in range(D):
                    g += Float64(inp.x[t * D + d]) * Float64(inp.gu[(e * 2 * F + f) * D + d])
                    u += Float64(inp.x[t * D + d]) * Float64(inp.gu[(e * 2 * F + F + f) * D + d])
                gg[f] = g
                uu[f] = u
                hh[f] = g * _sig64(g) * u
            var dh = List[Float64](length=F, fill=0.0)
            for d in range(D):
                var s = Float64(0.0)
                for f in range(F):
                    s += hh[f] * Float64(inp.dn[(e * D + d) * F + f])
                y[t * D + d] += wj[j] * s
                var dyd = Float64(inp.dy[t * D + d])
                dwj[j] += dyd * s
                var ds = wj[j] * dyd
                for f in range(F):
                    ddn[(e * D + d) * F + f] += ds * hh[f]
                    dh[f] += ds * Float64(inp.dn[(e * D + d) * F + f])
            for f in range(F):
                var sg = _sig64(gg[f])
                var dg = dh[f] * uu[f] * sg * (1.0 + gg[f] * (1.0 - sg))
                var du = dh[f] * gg[f] * sg
                for d in range(D):
                    var xv = Float64(inp.x[t * D + d])
                    dgu[(e * 2 * F + f) * D + d] += dg * xv
                    dgu[(e * 2 * F + F + f) * D + d] += du * xv
                    dx[t * D + d] += dg * Float64(inp.gu[(e * 2 * F + f) * D + d])
                    dx[t * D + d] += du * Float64(inp.gu[(e * 2 * F + F + f) * D + d])
        var cdot = Float64(0.0)
        for j in range(k):
            cdot += dwj[j] * wj[j]
        var dp = List[Float64](length=En, fill=0.0)
        var dot = Float64(0.0)
        for j in range(k):
            var e = Int(sel[t * k + j])
            var dpv = (dwj[j] - cdot) / tot if c.renorm != 0 else dwj[j]
            dp[e] = dpv
            dot += dpv * pr[e]
        for e in range(En):
            var dle = pr[e] * (dp[e] - dot)
            for d in range(D):
                dwg[e * D + d] += dle * Float64(inp.x[t * D + d])
                dx[t * D + d] += dle * Float64(inp.wg[e * D + d])
    var out = List[List[Float64]]()
    out.append(y^)
    out.append(dx^)
    out.append(dwg^)
    out.append(dgu^)
    out.append(ddn^)
    return out^


def same_words(tag: String, name: String, a: List[Float32], b: List[Float32]) raises -> Int:
    if len(a) != len(b):
        raise Error(tag + " " + name + ": length " + String(len(a)) + " vs " + String(len(b)))
    var diff = 0
    var first = -1
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            diff += 1
            if first < 0:
                first = i
    print("MOE_BWD", tag, name, len(a), digest(a), digest(b),
          "SAME" if diff == 0 else "DIFFER words=" + String(diff) + " first=" + String(first))
    return 1 if diff > 0 else 0


def near(tag: String, name: String, got: List[Float32], want: List[Float64]) raises -> Int:
    var scale = Float64(1e-12)
    for i in range(len(want)):
        scale = max(scale, abs(want[i]))
    var err = Float64(0.0)
    for i in range(len(want)):
        err = max(err, abs(Float64(got[i]) - want[i]))
    var rel = err / scale
    var ok = rel <= 1e-3
    print("MOE_BWD_F64", tag, name, "max_rel", rel, "OK" if ok else "FAIL")
    return 0 if ok else 1


def compare_all(tag: String, a: MoeResult, b: MoeResult) raises -> Int:
    var bad = 0
    bad += same_words(tag, "y", a.y, b.y)
    bad += same_words(tag, "logits", a.logits, b.logits)
    bad += same_words(tag, "selected", a.sel, b.sel)
    bad += same_words(tag, "weights", a.w, b.w)
    bad += same_words(tag, "dx", a.dx, b.dx)
    bad += same_words(tag, "drouter", a.dwg, b.dwg)
    bad += same_words(tag, "dgate_up", a.dgu, b.dgu)
    bad += same_words(tag, "ddown", a.ddn, b.ddn)
    return bad


def main() raises:
    var cases = List[MoeCase]()
    cases.append(MoeCase(5, 8, 6, 4, 2, 1))
    cases.append(MoeCase(17, 16, 12, 8, 2, 1))
    cases.append(MoeCase(9, 8, 5, 3, 1, 0))
    cases.append(MoeCase(3, 8, 4, 8, 1, 1))      # most experts get no pair
    cases.append(MoeCase(70, 12, 10, 6, 3, 1))   # 210 pairs: four grouping blocks
    cases.append(MoeCase(33, 20, 7, 5, 2, 0))
    var bad = 0
    for ci in range(len(cases)):
        var c = cases[ci].copy()
        var tag = String("case") + String(ci)
        var inp = inputs_of(c, ci * 10)
        var host = HostExec()
        var got = run_tape(host, c, inp)
        var ora = oracle_walk(c, inp)
        bad += compare_all(tag + ".host_vs_oracle", got, ora)
        var rf = reference64(c, inp, got.sel)
        bad += near(tag, "y", got.y, rf[0])
        bad += near(tag, "dx", got.dx, rf[1])
        bad += near(tag, "drouter", got.dwg, rf[2])
        bad += near(tag, "dgate_up", got.dgu, rf[3])
        bad += near(tag, "ddown", got.ddn, rf[4])
        comptime if has_accelerator():
            var dev = DeviceExec()
            var dgot = run_tape(dev, c, inp)
            bad += compare_all(tag + ".device_vs_host", dgot, got)
            print("MOE_BWD_DIGEST", tag, "device dx", digest(dgot.dx), "drouter", digest(dgot.dwg),
                  "dgate_up", digest(dgot.dgu), "ddown", digest(dgot.ddn), "y", digest(dgot.y))
        print("MOE_BWD_DIGEST", tag, "host dx", digest(got.dx), "drouter", digest(got.dwg),
              "dgate_up", digest(got.dgu), "ddown", digest(got.ddn), "y", digest(got.y))
    if bad > 0:
        raise Error("MOE_BACKWARD_CHECK FAIL: " + String(bad) + " comparisons differ")
    print("MOE_BACKWARD_CHECK_PASS host tape == host walk word for word, float64 reference within 1e-3")
