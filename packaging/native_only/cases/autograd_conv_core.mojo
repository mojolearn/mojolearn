# SPDX-License-Identifier: Apache-2.0
# Complete tiny conv2d -> ReLU -> overlapping average-pool VJPs.
# Every input is dyadic and every intermediate/gradient fits exactly in
# binary32. The independent analytic chain oracle therefore needs no
# tolerance. It reuses only existing float64 test forward oracles, never
# product convolution/pooling/VJP bodies. Public Python class owners are not
# represented by the native Tape engine witness.
from std.memory import bitcast
from std.sys.compile import is_defined
from autograd import Tape, Tensor
from autograd.rankn_conv import rankn_conv2d, rankn_avg_pool2d
from autograd.rules_conv2d import Conv2dRules
from autograd.checks.rankn_conv_cases import _sh, _f64, _f32, _conv_ref, _pool_ref, _expect_shape
from sequence.exec_trait import Exec


def _assert_values(actual: List[Float32], expected: List[Float32], name: String, mut digest: UInt64) raises:
    if len(actual) != len(expected):
        raise Error("native conv chain output/gradient shape differs from oracle")
    for i in range(len(actual)):
        # Analytic equality includes either sign of exact zero. The actual
        # sign remains in the digest used for cross-vendor identity checks.
        if actual[i] != expected[i]:
            print("FAIL native_conv", name, "index", i, "actual", actual[i], "expected", expected[i])
            raise Error("native conv chain differs from exact analytic value/gradient")
        digest = (digest ^ UInt64(bitcast[DType.uint32](actual[i]))) * UInt64(1099511628211)


def _oracle(
    x: List[Float32], w: List[Float32], b: List[Float32], seed: List[Float32],
    geometry: List[Int], has_bias: Bool,
) raises -> List[List[Float32]]:
    var N = geometry[0]; var C = geometry[1]; var H = geometry[2]; var W = geometry[3]
    var OC = geometry[4]; var KH = geometry[5]; var KW = geometry[6]
    var stride = geometry[7]; var padding = geometry[8]; var groups = geometry[9]
    var OH = (H + 2 * padding - KH) // stride + 1
    var OW = (W + 2 * padding - KW) // stride + 1
    var PH = OH - 1; var PW = OW - 1
    var conv = _conv_ref(_f64(x), _f64(w), _f64(b), has_bias, geometry)
    var relu = conv.copy()
    var positive = 0; var negative = 0
    for i in range(len(relu)):
        if relu[i] > 0:
            positive += 1
        else:
            negative += 1
            relu[i] = 0
    if positive == 0 or negative == 0:
        raise Error("native conv fixture must exercise both ReLU branches")
    var pool_geometry: List[Int] = [N, OC, OH, OW, 2, 1]
    var pooled = _pool_ref(relu, pool_geometry, False)
    var dc = List[Float64](length=len(conv), fill=0.0)
    # First differentiate every overlapping 2x2 mean, then the ReLU gate.
    # At exactly zero use the public ReLU derivative convention, zero.
    for nc in range(N * OC):
        for ph in range(PH):
            for pw in range(PW):
                var contribution = Float64(seed[(nc * PH + ph) * PW + pw]) / 4.0
                for kh in range(2):
                    for kw in range(2):
                        var at = (nc * OH + ph + kh) * OW + pw + kw
                        if conv[at] > 0:
                            dc[at] += contribution
    var dx = List[Float64](length=len(x), fill=0.0)
    var dw = List[Float64](length=len(w), fill=0.0)
    var db = List[Float64](length=OC, fill=0.0)
    var per_group = C // groups
    for n in range(N):
        for oc in range(OC):
            var first_channel = (oc // (OC // groups)) * per_group
            for oh in range(OH):
                for ow in range(OW):
                    var grad = dc[((n * OC + oc) * OH + oh) * OW + ow]
                    if has_bias:
                        db[oc] += grad
                    for channel in range(per_group):
                        for kh in range(KH):
                            var ih = oh * stride - padding + kh
                            if ih < 0 or ih >= H:
                                continue
                            for kw in range(KW):
                                var iw = ow * stride - padding + kw
                                if iw < 0 or iw >= W:
                                    continue
                                var xi = ((n * C + first_channel + channel) * H + ih) * W + iw
                                var wi = ((oc * per_group + channel) * KH + kh) * KW + kw
                                dx[xi] += grad * Float64(w[wi])
                                dw[wi] += grad * Float64(x[xi])
    var result = List[List[Float32]]()
    result.append(_f32(conv))
    result.append(_f32(relu))
    result.append(_f32(pooled))
    result.append(_f32(dx))
    result.append(_f32(dw))
    result.append(_f32(db))
    return result^


def _case[E: Exec](mut ex: E, name: String, channels: Int, outputs: Int,
                   groups: Int, stride: Int, padding: Int, has_bias: Bool) raises:
    var tape = Tape()
    comptime N = 1
    comptime H = 4
    comptime W = 5
    comptime K = 2
    var xs = List[Float32](length=N * channels * H * W, fill=0.0)
    var ws = List[Float32](length=outputs * (channels // groups) * K * K, fill=0.0)
    var bs = List[Float32](length=outputs, fill=0.0)
    for i in range(len(xs)):
        xs[i] = Float32((i * 5) % 17 - 8) / Float32(16)
    for i in range(len(ws)):
        ws[i] = Float32((i * 7) % 13 - 6) / Float32(8)
    for i in range(outputs):
        # Products are multiples of 1/128; odd /256 offsets avoid an
        # accidental ReLU kink when bias is present, without hiding its gate.
        bs[i] = Float32(2 * i + 1) / Float32(256)
    var x = tape.tensor_nd(ex, xs, _sh([N, channels, H, W]), True)
    var w = tape.tensor_nd(ex, ws, _sh([outputs, channels // groups, K, K]), True)
    var bias = Optional[Tensor]()
    if has_bias:
        bias = Optional[Tensor](tape.tensor_nd(ex, bs, _sh([outputs]), True))
    var conv = rankn_conv2d(tape, ex, x, w, bias, stride, padding, groups)
    var relu = tape.relu(ex, conv)
    var pooled = rankn_avg_pool2d(tape, ex, relu, 2, 1)
    var OH = (H + 2 * padding - K) // stride + 1
    var OW = (W + 2 * padding - K) // stride + 1
    _expect_shape(name + "/conv", tape, conv, [N, outputs, OH, OW])
    _expect_shape(name + "/pool", tape, pooled, [N, outputs, OH - 1, OW - 1])
    var seeds = List[Float32](length=N * outputs * (OH - 1) * (OW - 1), fill=0.0)
    for i in range(len(seeds)):
        seeds[i] = Float32((i * 3) % 11 - 5) / Float32(16)
    var seed = tape.tensor(ex, seeds, tape.rows(pooled), tape.cols(pooled))
    tape.backward_with_rules(ex, pooled, seed, Conv2dRules())
    var geometry: List[Int] = [N, channels, H, W, outputs, K, K, stride, padding, groups]
    var expected = _oracle(xs, ws, bs, seeds, geometry, has_bias)
    var digest = UInt64(1469598103934665603)
    _assert_values(tape.values(ex, conv), expected[0], "conv forward", digest)
    _assert_values(tape.values(ex, relu), expected[1], "ReLU forward", digest)
    _assert_values(tape.values(ex, pooled), expected[2], "pool forward", digest)
    var dx = tape.gradients(ex, x)
    comptime if is_defined["MOJOLEARN_NATIVE_CONV_BAD_GRADIENT"]():
        dx[0] += Float32(1.0)
    _assert_values(dx, expected[3], "input gradient", digest)
    _assert_values(tape.gradients(ex, w), expected[4], "weight gradient", digest)
    if has_bias:
        _assert_values(tape.gradients(ex, bias.value()), expected[5], "bias gradient", digest)
    tape.release(ex)
    print("PASS native_conv_chain", name, "digest", digest)


def check_conv_chains[E: Exec](mut ex: E) raises:
    _case(ex, "ordinary_stride_padding", 2, 3, 1, 2, 1, True)
    _case(ex, "grouped_overlap", 4, 4, 2, 1, 0, True)
    _case(ex, "depthwise_multiplier", 3, 6, 3, 2, 1, True)
    _case(ex, "grouped_no_bias", 4, 4, 2, 1, 0, False)
