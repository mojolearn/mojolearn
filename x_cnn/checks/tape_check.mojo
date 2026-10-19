# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CNNClassifier's tape backward (autograd/rules_cnn.mojo) against the hand walk, word for word.

For each case (two conv blocks + head, N 8, 1x28x28, k 10; block 2 with and
without its max pool; the forward's saved cols and conv output kept or
recomputed; SGD with momentum and Adam), runs three trainer steps on two
fresh sets of resident arrays with the same parameters and data, one through
the hand walk (fit_epoch_d_binding's step, verbatim) and one through the tape
(cnn_backward_autograd, the binding's default arm), and requires every
step's loss, every gradient word of the last step, every updated parameter
word and every optimizer buffer word to be equal. Prints one FNV-1a digest
per tensor so columns can be compared across vendors.

    mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . x_cnn/checks/tape_check.mojo

Both arms are compiled into this binary (the AUTOGRAD_CNN define picks only
the binding's arm). A fixture whose gradients are all zero, or whose
parameters do not move, is refused as VACUOUS."""
from std.memory import bitcast
from x_cnn.ops import FP, IP, conv_params, pool_params, AH_ROW, adam_hyper_base, NN14_BOUNDED_IM2COL
from x_cnn.ops import CP_N, CP_C, CP_OC, CP_KH, CP_KW, CP_OH, CP_OW
from x_cnn.device import (
    res_alloc, res_free, res_upload, res_download, conv_block_forward_into, conv_block_backward_into,
    linear_forward_into, linear_backward_into, softmax_xent_res_loss, opt_many_resident_h, adam_hyper_resident,
)
from autograd.rules_cnn import cnn_backward_autograd, conv_block_out_count

comptime N = 8
comptime C0 = 1
comptime H0 = 28
comptime K = 10
comptime OC0 = 4
comptime OC1 = 8
comptime STEPS = 3


@always_inline
def _at(a: Int) -> FP:
    return FP(unsafe_from_address=a)


@always_inline
def _at_i(a: Int) -> IP:
    return IP(unsafe_from_address=a)


def digest(values: List[Float32]) -> UInt64:
    var hash = UInt64(1469598103934665603)
    for i in range(len(values)):
        hash = (hash ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return hash


def _down(addr: Int, n: Int) raises -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    res_download(addr, out.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), n)
    return out^


def _up(addr: Int, mut values: List[Float32]) raises:
    res_upload(addr, values.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), len(values))


def _fixture(n: Int, salt: Int, scale: Float32) -> List[Float32]:
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(Float32((i * 17 + salt * 13) % 101 - 50) / scale)
    return out^


def _conv_raw(n: Int, c: Int, h: Int, oc: Int) -> List[Int]:
    # Conv2d._params: 3x3, stride 1, pad 1, dilation 1, OH/OW filled, bias, no reversal
    return [n, c, h, h, oc, 3, 3, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]


def _pool_raw(n: Int, c: Int, h: Int) -> List[Int]:
    # MaxPool2d(2)._params: 2x2, stride 2, no pad, dilation 1, floor mode
    return [n, c, h, h, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]


struct Result(Movable):
    var names: List[String]
    var tensors: List[List[Float32]]
    var losses: List[Float32]

    def __init__(out self):
        self.names = List[String]()
        self.tensors = List[List[Float32]]()
        self.losses = List[Float32]()

    def add(mut self, name: String, var values: List[Float32]):
        self.names.append(name)
        self.tensors.append(values^)


def run[tape: Bool](pool2: Bool, save: Bool, adam: Bool) raises -> Result:
    """Three steps on fresh resident arrays: the hand walk or the tape."""
    # ---- the plan: block 0 conv 1->4 + pool (28 -> 14), block 1 conv 4->8 (+ pool 14 -> 7)
    var cs = List[List[Int32]]()
    var ps = List[List[Int32]]()
    var pools = List[Bool]()
    cs.append(conv_params(_conv_raw(N, C0, H0, OC0)))
    ps.append(pool_params(_pool_raw(N, OC0, H0)))
    pools.append(True)
    var h1 = H0 // 2
    cs.append(conv_params(_conv_raw(N, OC0, h1, OC1)))
    ps.append(pool_params(_pool_raw(N, OC1, h1)) if pool2 else List[Int32]())
    pools.append(pool2)
    var nb = 2
    var flat = conv_block_out_count(cs[1], ps[1], pools[1]) // N
    # ---- the arrays, as CNNClassifier._resident lays them out
    var nx = N * C0 * H0 * H0
    var a: List[Int] = [res_alloc(nx), res_alloc(N), res_alloc(N * K), res_alloc(N * K), res_alloc(N * K), 0]
    var bh = List[List[Int]]()
    var sizes = List[Int]()
    var pw = List[Int]()
    var pg = List[Int]()
    var pb = List[Int]()
    for j in range(nb):
        var oc = Int(cs[j][CP_OC])
        var ckk = Int(cs[j][CP_C]) * Int(cs[j][CP_KH]) * Int(cs[j][CP_KW])
        var rows = Int(cs[j][CP_N]) * Int(cs[j][CP_OH]) * Int(cs[j][CP_OW])
        var no = conv_block_out_count(cs[j], ps[j], pools[j])
        var w = res_alloc(oc * ckk)
        var b = res_alloc(oc)
        var gw = res_alloc(oc * ckk)
        var gb = res_alloc(oc)
        var cols = (res_alloc(1 if NN14_BOUNDED_IM2COL else rows * ckk)) if save else 0
        var yconv = res_alloc(rows * oc) if save else 0
        bh.append([w, b, gw, gb, res_alloc(no), res_alloc(no), res_alloc(no), cols, yconv])
        var wv = _fixture(oc * ckk, 2 * j, Float32(500.0))
        var bv = _fixture(oc, 2 * j + 1, Float32(5000.0))
        _up(w, wv)
        _up(b, bv)
        sizes.append(oc * ckk)
        sizes.append(oc)
        pw.append(w)
        pw.append(b)
        pg.append(gw)
        pg.append(gb)
    var head: List[Int] = [res_alloc(K * flat), res_alloc(K), res_alloc(K * flat), res_alloc(K)]
    var hwv = _fixture(K * flat, 7, Float32(5000.0))
    var hbv = _fixture(K, 8, Float32(5000.0))
    _up(head[0], hwv)
    _up(head[1], hbv)
    sizes.append(K * flat)
    sizes.append(K)
    pw.append(head[0])
    pw.append(head[1])
    pg.append(head[2])
    pg.append(head[3])
    for e in range(len(sizes)):
        pb.append(res_alloc(2 * sizes[e]))  # Adam's [m | v]; SGD reads the first half
    # ---- the data: the same rows every step (the gather is not under test)
    var xv = List[Float32](capacity=nx)
    for i in range(nx):
        xv.append(Float32((i * 37 + 11) % 97) / Float32(97.0))
    _up(a[0], xv)
    var lab = List[Int32](capacity=N)
    for i in range(N):
        lab.append(Int32((i * 3 + 1) % K))
    res_upload(a[1], FP(unsafe_from_address=Int(lab.unsafe_ptr())), N)
    # ---- the hyper rows, as fit_epoch_d_binding builds them
    var hbuf = res_alloc(STEPS * AH_ROW if adam else 12)
    if adam:
        adam_hyper_resident(adam_hyper_base(1e-3, 0.9, 0.999, 1e-8, 0.0, 0.0), 1, STEPS, hbuf)
    else:
        var sg: List[Float32] = [
            Float32(0.05), Float32(0.9), Float32(0), Float32(0), Float32(0), Float32(1),
            Float32(0.05), Float32(0.9), Float32(0), Float32(0), Float32(0), Float32(0),
        ]
        _up(hbuf, sg)
    var lbuf = res_alloc(STEPS)
    var tw = res_alloc(2)
    var one: List[Float32] = [Float32(1), Float32(0)]
    _up(tw, one)
    var m = N
    for st in range(STEPS):
        var src = a[0]
        for j in range(nb):
            conv_block_forward_into[True](
                _at(src), _at(bh[j][0]), _at(bh[j][1]), cs[j], ps[j], pools[j], _at(bh[j][4]),
                _at_i(bh[j][5]), bh[j][7], bh[j][8],
            )
            src = bh[j][4]
        linear_forward_into[True](_at(src), _at(head[0]), _at(head[1]), m, flat, K, _at(a[2]))
        softmax_xent_res_loss(_at(a[2]), _at_i(a[1]), m, K, _at(a[3]), _at(a[4]), lbuf, st)
        comptime if tape:
            cnn_backward_autograd(bh, head, a, cs, ps, pools, m, flat, K, tw, tw + 4)
        else:
            # fit_epoch_d_binding's hand step, verbatim
            var glast = bh[nb - 1][6] if nb > 0 else a[5]
            linear_backward_into[True](
                _at(src), _at(head[0]), _at(a[3]), m, flat, K, _at(glast), _at(head[2]), _at(head[3])
            )
            for jj in range(nb):
                var j = nb - 1 - jj
                var bsrc = bh[j - 1][4] if j > 0 else a[0]
                var dx = bh[j - 1][6] if j > 0 else 0
                var want = dx != 0
                conv_block_backward_into[True](
                    _at(bsrc), _at(bh[j][0]), _at(bh[j][1]), _at(bh[j][6]), _at_i(bh[j][5]), cs[j], ps[j],
                    pools[j], want, _at(dx) if want else _at(bh[j][3]), _at(bh[j][2]), _at(bh[j][3]), bh[j][7],
                    bh[j][8],
                )
        if adam:
            opt_many_resident_h[True](pw, pg, pb, sizes, hbuf + 4 * AH_ROW * st)
        else:
            opt_many_resident_h[False](pw, pg, pb, sizes, hbuf + (0 if st == 0 else 24))
    # ---- everything the step writes
    var r = Result()
    r.losses = _down(lbuf, STEPS)
    var pnames: List[String] = ["block0.w", "block0.b", "block1.w", "block1.b", "head.w", "head.b"]
    for e in range(len(sizes)):
        r.add(pnames[e] + " grad", _down(pg[e], sizes[e]))
        r.add(pnames[e] + " param", _down(pw[e], sizes[e]))
        r.add(pnames[e] + " opt", _down(pb[e], 2 * sizes[e]))
    r.add("block0.gout", _down(bh[0][6], conv_block_out_count(cs[0], ps[0], pools[0])))
    r.add("block1.gout", _down(bh[1][6], conv_block_out_count(cs[1], ps[1], pools[1])))
    r.add("glog", _down(a[3], N * K))
    # ---- vacuity: the step must move the parameters and reach every gradient
    for e in range(len(sizes)):
        var g = _down(pg[e], sizes[e])
        var nz = 0
        for i in range(len(g)):
            if (bitcast[DType.uint32](g[i]) & UInt32(0x7FFFFFFF)) != UInt32(0):
                nz += 1
        if nz == 0:
            raise Error("VACUOUS cnn tape check: gradient " + pnames[e] + " is all zero")
    var w0 = _fixture(sizes[0], 0, Float32(500.0))
    var w1 = _down(pw[0], sizes[0])
    var moved = 0
    for i in range(sizes[0]):
        if bitcast[DType.uint32](w0[i]) != bitcast[DType.uint32](w1[i]):
            moved += 1
    if moved == 0:
        raise Error("VACUOUS cnn tape check: the steps did not move block0.w")
    # ---- free
    for j in range(nb):
        for e in range(4, 9):  # out, idx, gout, cols, conv out (w, b, gw, gb below)
            if bh[j][e] != 0:
                res_free(bh[j][e])
    for e in range(len(sizes)):
        res_free(pw[e])
        res_free(pg[e])
        res_free(pb[e])
    for e in range(5):
        res_free(a[e])
    res_free(hbuf)
    res_free(lbuf)
    res_free(tw)
    _ = lab^
    _ = xv^
    _ = one^
    return r^


def compare(tag: String, hand: Result, tape: Result) -> Int:
    var bad = 0
    for s in range(len(hand.losses)):
        var same = bitcast[DType.uint32](hand.losses[s]) == bitcast[DType.uint32](tape.losses[s])
        print("CNN_TAPE", tag, "loss step", s, "hand", hand.losses[s], "tape", tape.losses[s], "SAME" if same else "DIFFER")
        if not same:
            bad += 1
    for t in range(len(hand.tensors)):
        ref h = hand.tensors[t]
        ref p = tape.tensors[t]
        var diff = abs(len(h) - len(p))
        for i in range(min(len(h), len(p))):
            if bitcast[DType.uint32](h[i]) != bitcast[DType.uint32](p[i]):
                diff += 1
        print("CNN_TAPE", tag, hand.names[t], len(h), "hand", digest(h), "tape", digest(p),
              String("SAME") if diff == 0 else String("DIFFER ") + String(diff))
        if diff != 0:
            bad += 1
    return bad


def run_case(tag: String, pool2: Bool, save: Bool, adam: Bool) raises -> Int:
    var hand = run[False](pool2, save, adam)
    var tape = run[True](pool2, save, adam)
    return compare(tag, hand, tape)


def main() raises:
    var bad = 0
    bad += run_case("sgd.pool.saved", True, True, False)
    bad += run_case("adam.pool.saved", True, True, True)
    bad += run_case("sgd.nopool.recompute", False, False, False)
    bad += run_case("adam.nopool.recompute", False, False, True)
    if bad != 0:
        raise Error("CNN_TAPE_CHECK_FAIL " + String(bad) + " tensors differ between the tape and the hand walk")
    print("CNN_TAPE_CHECK_PASS every loss, gradient, parameter and optimizer word equals the hand walk")
