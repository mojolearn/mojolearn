# SPDX-License-Identifier: Apache-2.0
# Lane ag-rankn-conv (docs/plans/AUTOGRAD_RANKN.md section 2.8): checks of
# the tape's rank-4 conv2d, max_pool2d and avg_pool2d on any executor
# (HostExec, Metal, NVIDIA, AMD).
#   conv2d: x [2, 3, 9, 11], weight [4, 3, 3, 3], bias [4]; stride 1 and 2,
#     padding 0 and 1, and one case without a bias. The forward against a
#     host oracle written elementwise in Mojo (float64); dx, dW and db
#     against central finite differences of that oracle's loss
#     sum(seed * conv(x)) (linear in each argument, so the differences are
#     the gradients up to float64 rounding); check_values.
#   pools: max and avg, 2x2 stride 2 and 3x3 stride 2, on distinct input
#     values (no tie within 2h, so the finite differences of the max are
#     the winner counts); forward against the oracle, dx by differences.
#   chain: conv2d (stride 1, padding 1, x needs no gradient) -> relu ->
#     max_pool2d 2x2 -> avg_pool2d 2x2 stride 1: digests only.
# Each case prints one `RANKN conv <case> digest <d>` line (the digest of
# every word it compared: the value, then each gradient) so the CPU,
# Metal, NVIDIA and AMD logs can be diffed, then one PASS line.
# Shapes are small, odd and off any board row: a non-square image, odd
# channels, a GEMM K of 27 and rows of 198 (stride 1) and 50 (stride 2).
from std.memory import bitcast
from autograd.tape import Tape, Tensor
from autograd.rankn import ShapeN, RN_MAX_RANK
from autograd.rankn_conv import rankn_conv2d, rankn_max_pool2d, rankn_avg_pool2d
from autograd.rules_conv2d import Conv2dRules
from autograd.checks.support import check_values
from sequence.exec_trait import Exec


def _data(n: Int, salt: Int) -> List[Float32]:
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 7919 + salt * 104729) % 1000) - 500) * Float32(0.00123)
    return v^


def _distinct(n: Int, salt: Int) raises -> List[Float32]:
    """n <= 1009 distinct values 0.00123 apart (7919 is a unit mod 1009)."""
    if n > 1009:
        raise Error("rankn_conv check: too many distinct values")
    var v = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(n: check data, host side)
        v[i] = Float32(((i * 7919 + salt * 31) % 1009) - 504) * Float32(0.00123)
    return v^


def _f64(v: List[Float32]) -> List[Float64]:
    var r = List[Float64](length=len(v), fill=Float64(0))
    for i in range(len(v)):  # small-loop(len(v): check oracle, host side)
        r[i] = Float64(v[i])
    return r^


def _f32(v: List[Float64]) -> List[Float32]:
    var r = List[Float32](length=len(v), fill=Float32(0))
    for i in range(len(v)):  # small-loop(len(v): check oracle, host side)
        r[i] = Float32(v[i])
    return r^


def _sh(dims: List[Int]) -> ShapeN:
    var d = InlineArray[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(len(dims): tensor rank, at most 4 here)
        d[k] = dims[k]
    return ShapeN(len(dims), d)


def _fold_digest(mut digest: UInt64, v: List[Float32]):
    for i in range(len(v)):  # small-loop(len(v): check output digest)
        digest = (digest ^ UInt64(bitcast[DType.uint32](v[i]))) * UInt64(1099511628211)


def _expect_shape(name: String, tape: Tape, t: Tensor, dims: List[Int]) raises:
    var s = tape.shape_of(t)
    if s.ndim != len(dims):
        raise Error(name + ": wrong rank")
    for k in range(len(dims)):  # small-loop(len(dims): tensor rank)
        if s.d[k] != dims[k]:
            print("FAIL", name, "axis", k, "got", s.d[k], "want", dims[k])
            raise Error(name + ": wrong shape")


def _dot(a: List[Float64], b: List[Float64]) -> Float64:
    var s = Float64(0)
    for i in range(len(a)):  # small-loop(len(a): check oracle, host side)
        s += a[i] * b[i]
    return s


# ---------------------------------------------------------------- oracles
# geometry g = [N, C, H, W, OC, KH, KW, S, P]
def _conv_ref(x: List[Float64], w: List[Float64], b: List[Float64], has_bias: Bool, g: List[Int]) -> List[Float64]:
    var N = g[0]; var C = g[1]; var H = g[2]; var W = g[3]
    var OC = g[4]; var KH = g[5]; var KW = g[6]; var S = g[7]; var P = g[8]
    var OH = (H + 2 * P - KH) // S + 1
    var OW = (W + 2 * P - KW) // S + 1
    var out = List[Float64](length=N * OC * OH * OW, fill=Float64(0))
    for n in range(N):  # small-loop(N: check oracle, host side)
        for oc in range(OC):  # small-loop(OC: check oracle)
            for oh in range(OH):  # small-loop(OH: check oracle)
                for ow in range(OW):  # small-loop(OW: check oracle)
                    var acc = b[oc] if has_bias else Float64(0)
                    for c in range(C):  # small-loop(C: check oracle)
                        for kh in range(KH):  # small-loop(KH: check oracle)
                            var h = oh * S - P + kh
                            if h < 0 or h >= H:
                                continue
                            for kw in range(KW):  # small-loop(KW: check oracle)
                                var ww = ow * S - P + kw
                                if ww < 0 or ww >= W:
                                    continue
                                acc += x[((n * C + c) * H + h) * W + ww] * w[((oc * C + c) * KH + kh) * KW + kw]
                    out[((n * OC + oc) * OH + oh) * OW + ow] = acc
    return out^


# pool geometry p = [N, C, H, W, K, S]
def _pool_ref(x: List[Float64], p: List[Int], is_max: Bool) -> List[Float64]:
    var N = p[0]; var C = p[1]; var H = p[2]; var W = p[3]; var K = p[4]; var S = p[5]
    var OH = (H - K) // S + 1
    var OW = (W - K) // S + 1
    var out = List[Float64](length=N * C * OH * OW, fill=Float64(0))
    for nc in range(N * C):  # small-loop(N C: check oracle, host side)
        for oh in range(OH):  # small-loop(OH: check oracle)
            for ow in range(OW):  # small-loop(OW: check oracle)
                var acc = Float64(0)
                var first = True
                for kh in range(K):  # small-loop(K: check oracle)
                    for kw in range(K):  # small-loop(K: check oracle)
                        var v = x[(nc * H + oh * S + kh) * W + ow * S + kw]
                        if is_max:
                            if first or v > acc:
                                acc = v
                            first = False
                        else:
                            acc += v
                if not is_max:
                    acc = acc / Float64(K * K)
                out[(nc * OH + oh) * OW + ow] = acc
    return out^


comptime _H = Float64(1.0) / Float64(4096)


def _fd_conv(which: Int, x: List[Float64], w: List[Float64], b: List[Float64], has_bias: Bool,
             g: List[Int], seed: List[Float64]) -> List[Float64]:
    """d sum(seed * conv) / d (x, w or b: which = 0, 1, 2), central differences."""
    var n = len(x) if which == 0 else (len(w) if which == 1 else len(b))
    var r = List[Float64](length=n, fill=Float64(0))
    for i in range(n):  # small-loop(n: check finite differences, host side)
        var xp = x.copy(); var wp = w.copy(); var bp = b.copy()
        var xm = x.copy(); var wm = w.copy(); var bm = b.copy()
        if which == 0:
            xp[i] += _H; xm[i] -= _H
        elif which == 1:
            wp[i] += _H; wm[i] -= _H
        else:
            bp[i] += _H; bm[i] -= _H
        var lp = _dot(seed, _conv_ref(xp, wp, bp, has_bias, g))
        var lm = _dot(seed, _conv_ref(xm, wm, bm, has_bias, g))
        r[i] = (lp - lm) / (2 * _H)
    return r^


def _fd_pool(x: List[Float64], p: List[Int], is_max: Bool, seed: List[Float64]) -> List[Float64]:
    var r = List[Float64](length=len(x), fill=Float64(0))
    for i in range(len(x)):  # small-loop(len(x): check finite differences, host side)
        var xp = x.copy()
        var xm = x.copy()
        xp[i] += _H
        xm[i] -= _H
        r[i] = (_dot(seed, _pool_ref(xp, p, is_max)) - _dot(seed, _pool_ref(xm, p, is_max))) / (2 * _H)
    return r^


# ---------------------------------------------------------------- cases
def _conv_case[E: Exec](mut ex: E, mut tape: Tape, stride: Int, padding: Int, has_bias: Bool) raises -> UInt64:
    tape.reset()
    comptime N = 2
    comptime C = 3
    comptime H = 9
    comptime W = 11
    comptime OC = 4
    comptime K = 3
    var name = String("conv2d/s") + String(stride) + "p" + String(padding)
    if not has_bias:
        name += "/nobias"
    var xs = _data(N * C * H * W, 1)
    var ws = _data(OC * C * K * K, 2)
    var bs = _data(OC, 3)
    var x = tape.tensor_nd(ex, xs, _sh([N, C, H, W]), True)
    var w = tape.tensor_nd(ex, ws, _sh([OC, C, K, K]), True)
    var bopt = Optional[Tensor]()
    if has_bias:
        bopt = Optional[Tensor](tape.tensor_nd(ex, bs, _sh([OC]), True))
    var out = rankn_conv2d(tape, ex, x, w, bopt, stride, padding)
    var OH = (H + 2 * padding - K) // stride + 1
    var OW = (W + 2 * padding - K) // stride + 1
    _expect_shape(name, tape, out, [N, OC, OH, OW])
    var gs = _data(N * OC * OH * OW, 4)
    var seed = tape.tensor(ex, gs, tape.rows(out), tape.cols(out))
    tape.backward_with_rules(ex, out, seed, Conv2dRules())
    var geo: List[Int] = [N, C, H, W, OC, K, K, stride, padding]
    var x64 = _f64(xs)
    var w64 = _f64(ws)
    var b64 = _f64(bs)
    var g64 = _f64(gs)
    var digest = UInt64(1469598103934665603)
    var v = tape.values(ex, out)
    check_values(name + "/forward", v, _f32(_conv_ref(x64, w64, b64, has_bias, geo)))
    _fold_digest(digest, v)
    var dx = tape.gradients(ex, x)
    check_values(name + "/dx", dx, _f32(_fd_conv(0, x64, w64, b64, has_bias, geo, g64)))
    _fold_digest(digest, dx)
    var dw = tape.gradients(ex, w)
    check_values(name + "/dw", dw, _f32(_fd_conv(1, x64, w64, b64, has_bias, geo, g64)))
    _fold_digest(digest, dw)
    if has_bias:
        var db = tape.gradients(ex, bopt.value())
        check_values(name + "/db", db, _f32(_fd_conv(2, x64, w64, b64, has_bias, geo, g64)))
        _fold_digest(digest, db)
    print("RANKN conv", name, "digest", digest)
    return digest


def _pool_case[E: Exec](mut ex: E, mut tape: Tape, is_max: Bool, k: Int, stride: Int) raises -> UInt64:
    tape.reset()
    comptime N = 2
    comptime C = 3
    comptime H = 9
    comptime W = 11
    var name = String("max_pool2d/k") if is_max else String("avg_pool2d/k")
    name += String(k) + "s" + String(stride)
    var xs = _distinct(N * C * H * W, k)
    var x = tape.tensor_nd(ex, xs, _sh([N, C, H, W]), True)
    var out: Tensor
    if is_max:
        out = rankn_max_pool2d(tape, ex, x, k, stride)
    else:
        out = rankn_avg_pool2d(tape, ex, x, k, stride)
    var OH = (H - k) // stride + 1
    var OW = (W - k) // stride + 1
    _expect_shape(name, tape, out, [N, C, OH, OW])
    var gs = _data(N * C * OH * OW, 5)
    var seed = tape.tensor(ex, gs, tape.rows(out), tape.cols(out))
    tape.backward_with_rules(ex, out, seed, Conv2dRules())
    var geo: List[Int] = [N, C, H, W, k, stride]
    var x64 = _f64(xs)
    var digest = UInt64(1469598103934665603)
    var v = tape.values(ex, out)
    check_values(name + "/forward", v, _f32(_pool_ref(x64, geo, is_max)))
    _fold_digest(digest, v)
    var dx = tape.gradients(ex, x)
    check_values(name + "/dx", dx, _f32(_fd_pool(x64, geo, is_max, _f64(gs))))
    _fold_digest(digest, dx)
    print("RANKN conv", name, "digest", digest)
    return digest


def _chain_case[E: Exec](mut ex: E, mut tape: Tape) raises -> UInt64:
    """conv2d -> relu -> max_pool2d -> avg_pool2d (stride 1), x a data input:
    the words of every gradient the chain produces."""
    tape.reset()
    var x = tape.tensor_nd(ex, _data(2 * 3 * 9 * 11, 6), _sh([2, 3, 9, 11]), False)
    var w = tape.tensor_nd(ex, _data(4 * 3 * 3 * 3, 7), _sh([4, 3, 3, 3]), True)
    var b = tape.tensor_nd(ex, _data(4, 8), _sh([4]), True)
    var c = rankn_conv2d(tape, ex, x, w, Optional[Tensor](b), 1, 1)
    var r = tape.relu(ex, c)
    var m = rankn_max_pool2d(tape, ex, r, 2)
    var a = rankn_avg_pool2d(tape, ex, m, 2, 1)
    _expect_shape("chain", tape, a, [2, 4, 3, 4])
    var seed = tape.tensor(ex, _data(tape.rows(a) * tape.cols(a), 9), tape.rows(a), tape.cols(a))
    tape.backward_with_rules(ex, a, seed, Conv2dRules())
    var digest = UInt64(1469598103934665603)
    _fold_digest(digest, tape.values(ex, a))
    _fold_digest(digest, tape.gradients(ex, w))
    _fold_digest(digest, tape.gradients(ex, b))
    print("RANKN conv chain digest", digest)
    return digest


def run_rankn_conv_checks[E: Exec](mut ex: E) raises:
    var tape = Tape()
    var summary = UInt64(1469598103934665603)
    var cases = 0
    for stride in range(1, 3):  # small-loop(2: strides)
        for padding in range(2):  # small-loop(2: paddings)
            summary = (summary ^ _conv_case(ex, tape, stride, padding, True)) * UInt64(1099511628211)
            cases += 1
    summary = (summary ^ _conv_case(ex, tape, 2, 1, False)) * UInt64(1099511628211)
    cases += 1
    for mi in range(2):  # small-loop(2: max, avg)
        for k in range(2, 4):  # small-loop(2: kernel sizes)
            summary = (summary ^ _pool_case(ex, tape, mi == 0, k, 2)) * UInt64(1099511628211)
            cases += 1
    summary = (summary ^ _chain_case(ex, tape)) * UInt64(1099511628211)
    cases += 1
    tape.release(ex)
    print("RANKN conv summary cases", cases, "digest", summary)
    print("PASS rankn_conv", cases, "cases")
