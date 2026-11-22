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
#   grouped conv2d (the groups argument): x [2, 4, 9, 11], groups 2 with
#     weight [6, 2, 3, 3] (stride 1, padding 1, bias), depthwise groups 4
#     with weight [8, 1, 3, 3] (a channel multiplier of 2; stride 2,
#     padding 1, bias) and depthwise weight [4, 1, 3, 3] (stride 1, no
#     padding, no bias): forward, dx, dW and db against the same float64
#     oracle and its finite differences. They print their own summary
#     line, so the ungrouped cases' summary digest keeps its value.
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
    var d = Array[Int, RN_MAX_RANK](fill=1)
    for k in range(len(dims)):  # small-loop(len(dims): tensor rank, at most 4 here)
        d[k] = dims[k]
    return ShapeN(len(dims), d)


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
# geometry g = [N, C, H, W, OC, KH, KW, S, P] and optionally groups G (1
# when absent): output channel oc sees input channels [gi Cg, (gi + 1) Cg),
# gi = oc // (OC / G), Cg = C / G; the weight is [OC, Cg, KH, KW]
def _conv_ref(x: List[Float64], w: List[Float64], b: List[Float64], has_bias: Bool, g: List[Int]) -> List[Float64]:
    var N = g[0]; var C = g[1]; var H = g[2]; var W = g[3]
    var OC = g[4]; var KH = g[5]; var KW = g[6]; var S = g[7]; var P = g[8]
    var G = g[9] if len(g) > 9 else 1
    var Cg = C // G
    var OCg = OC // G
    var OH = (H + 2 * P - KH) // S + 1
    var OW = (W + 2 * P - KW) // S + 1
    var out = List[Float64](length=N * OC * OH * OW, fill=Float64(0))
    for n in range(N):  # small-loop(N: check oracle, host side)
        for oc in range(OC):  # small-loop(OC: check oracle)
            for oh in range(OH):  # small-loop(OH: check oracle)
                for ow in range(OW):  # small-loop(OW: check oracle)
                    var acc = b[oc] if has_bias else Float64(0)
                    var c0 = (oc // OCg) * Cg
                    for cl in range(Cg):  # small-loop(C / G: check oracle)
                        var c = c0 + cl
                        for kh in range(KH):  # small-loop(KH: check oracle)
                            var h = oh * S - P + kh
                            if h < 0 or h >= H:
                                continue
                            for kw in range(KW):  # small-loop(KW: check oracle)
                                var ww = ow * S - P + kw
                                if ww < 0 or ww >= W:
                                    continue
                                acc += x[((n * C + c) * H + h) * W + ww] * w[((oc * Cg + cl) * KH + kh) * KW + kw]
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


# ---------------------------------------------------------------- cases

