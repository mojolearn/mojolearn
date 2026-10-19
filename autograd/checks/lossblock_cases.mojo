# SPDX-License-Identifier: Apache-2.0
# The MLP loss block (sequence/mlp.mojo MLP_LOSSBLOCK, lane ag-pass3-lossblock):
# the blocked ||W||^2, row-loss and epoch-loss folds against a float64 host
# sum, the seed against p - y bit for bit, and the previous kernels
# (MLP_BLOCKED_FOLDS: op_mlp_rowloss, op_mlp_l2part/l2fold, op_mlp_rowpart,
# op_mlp_bloss) beside them. A case whose layers and batch are at most 64
# terms is one block on both paths: its batch loss must keep the old bits.
# Every case prints its digests so the columns (host, Metal, NVIDIA, AMD) can
# be compared; the digests are the bit version mlp-lossblock-1.
from sequence.exec_trait import Exec
from sequence.mlp_fit import (
    MLPNet,
    _l2_blocked,
    mlp_epoch_loss_block,
    mlp_lb_l2_parts,
    mlp_lb_row_parts,
    mlp_loss_block,
)
from sequence.mlp import LOSS_SQUARED, LOSS_BINARY_LOG, LOSS_LOG, MLP_L2_BLOCK, MLP_ROW_BLOCK
from sequence.ops import FP, Args, OP_MLP_BLOSS, OP_MLP_ROWLOSS, OP_MLP_ROWPART, dummy_ptr, sub
from checks.numerics import ftz, identical_mul
from std.math import log
from std.memory import bitcast

#: relative tolerance of a float32 blocked fold against the float64 sum
comptime LB_TOL = 1.0e-4


def _ptr(mut values: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(values.unsafe_ptr()))


def _digest(values: List[Float32]) -> UInt64:
    var d = UInt64(1469598103934665603)
    for i in range(len(values)):
        d = (d ^ UInt64(bitcast[DType.uint32](values[i]))) * UInt64(1099511628211)
    return d


def _rel(got: Float64, want: Float64) -> Float64:
    var den = abs(want)
    if den < 1.0e-30:
        den = 1.0e-30
    return abs(got - want) / den


def _clip64(p: Float32) -> Float64:
    """sequence/mlp.mojo _clip, then widened."""
    comptime EPS: Float32 = 1.1920928955078125e-07
    var c = p
    if c < EPS:
        c = EPS
    if c > Float32(1.0) - EPS:
        c = Float32(1.0) - EPS
    return Float64(c)


def _batch_case[E: Exec](
    mut ex: E, name: String, var sizes: List[Int], B: Int, kind: Int, alpha: Float32, one_block: Bool,
) raises -> Bool:
    var out_act = 0 if kind == LOSS_SQUARED else 1
    var net = MLPNet(sizes^, 3, out_act)
    var L = net.n_layers()
    var O = net.sizes[L]
    var np_ = net.n_params()
    var params = List[Float32](length=np_, fill=0)
    for i in range(np_):
        params[i] = Float32((i * 37) % 101 - 50) / Float32(256)
    var pred = List[Float32](length=B * O, fill=0)
    var y = List[Float32](length=B * O, fill=0)
    for r in range(B):
        for o in range(O):
            var i = r * O + o
            if kind == LOSS_SQUARED:
                pred[i] = Float32((i * 13) % 41 - 20) / Float32(16)
                y[i] = Float32((i * 7) % 23 - 11) / Float32(8)
            else:
                pred[i] = Float32(0.05) + Float32(0.9) * Float32((i * 29) % 97) / Float32(96)
                if kind == LOSS_BINARY_LOG:
                    y[i] = Float32(1.0) if r % 3 == 0 else Float32(0.0)
                else:
                    y[i] = Float32(1.0) if o == r % O else Float32(0.0)
    var dP = ex.alloc(np_)
    ex.upload(dP, _ptr(params), np_)
    var dpred = ex.alloc(B * O)
    ex.upload(dpred, _ptr(pred), B * O)
    var dy = ex.alloc(B * O)
    ex.upload(dy, _ptr(y), B * O)
    var half = ftz(identical_mul(Float32(0.5), alpha))
    # the loss block (batch index 1 of a two-slot loss vector)
    var dseed = ex.alloc(B * O)
    var drow = ex.alloc(mlp_lb_row_parts(B))
    var dl2p = ex.alloc(mlp_lb_l2_parts(net))
    var dl2 = ex.alloc(L)
    var dloss = ex.alloc(2)
    mlp_loss_block(ex, net, dP, dpred, dy, dseed, drow, dl2p, dl2, dloss, 1, B, kind, half, dummy_ptr(), 0)
    # the previous kernels (MLP_BLOCKED_FOLDS), as mlp_fit issued them
    var oseed = ex.alloc(B * O)
    var orow = ex.alloc(B)
    var ol2 = ex.alloc(L)
    var l2_nb = 0
    for i in range(L):  # driver-loop: over layers
        l2_nb += (net.sizes[i] * net.sizes[i + 1] + MLP_L2_BLOCK - 1) // MLP_L2_BLOCK
    var ol2p = ex.alloc(l2_nb)
    var nrb = (B + MLP_ROW_BLOCK - 1) // MLP_ROW_BLOCK
    var orp = ex.alloc(nrb)
    var oloss = ex.alloc(2)
    var a = Args()
    a.p0 = dpred
    a.p1 = dy
    a.p2 = oseed
    a.p3 = orow
    a.i0 = kind
    a.i1 = O
    ex.launch[OP_MLP_ROWLOSS](a, B)
    _l2_blocked(ex, net, dP, ol2, ol2p)
    var rp = Args()
    rp.p0 = orow
    rp.p1 = orp
    rp.i0 = B
    rp.i1 = MLP_ROW_BLOCK
    ex.launch[OP_MLP_ROWPART](rp, nrb)
    var bl = Args()
    bl.p0 = orow
    bl.p1 = ol2
    bl.p2 = oloss
    bl.p3 = orp
    bl.i0 = 1
    bl.i1 = B
    bl.i2 = O
    bl.i3 = L
    bl.i4 = kind
    bl.i5 = nrb
    bl.f0 = half
    ex.launch[OP_MLP_BLOSS](bl, 1)
    ex.sync()
    var l2 = List[Float32](length=L, fill=0)
    var loss = List[Float32](length=2, fill=0)
    var seed = List[Float32](length=B * O, fill=0)
    var old_loss = List[Float32](length=2, fill=0)
    ex.download(_ptr(l2), dl2, L)
    ex.download(_ptr(loss), dloss, 2)
    ex.download(_ptr(seed), dseed, B * O)
    ex.download(_ptr(old_loss), oloss, 2)
    # the float64 host reference
    var ok = True
    var q64 = Float64(0.0)
    for l in range(L):  # driver-loop: over layers
        var off = net.w_off(l)
        var c = net.sizes[l] * net.sizes[l + 1]
        var s64 = Float64(0.0)
        for k in range(c):
            var w = Float64(params[off + k])
            s64 += w * w
        q64 += s64
        var e = _rel(Float64(l2[l]), s64)
        if e > LB_TOL:
            ok = False
            print("LOSSBLOCK", name, "l2 layer", l, "got", l2[l], "want", s64, "rel", e)
    var d64 = Float64(0.0)
    for i in range(B * O):
        var pv = pred[i]
        var yv = y[i]
        if kind == LOSS_SQUARED:
            var d = Float64(yv) - Float64(pv)
            d64 += d * d
        else:
            var pc = _clip64(pv)
            if yv != Float32(0.0):
                d64 -= Float64(yv) * log(pc)
            if kind == LOSS_BINARY_LOG:
                var ny = 1.0 - Float64(yv)
                if ny != 0.0:
                    d64 -= ny * log(1.0 - pc)
        if bitcast[DType.uint32](seed[i]) != bitcast[DType.uint32](sub(pv, yv)):
            ok = False
    var data64 = d64 / Float64(B * O) / 2.0 if kind == LOSS_SQUARED else d64 / Float64(B)
    var want = data64 + Float64(half) * q64 / Float64(B)
    var e = _rel(Float64(loss[1]), want)
    if e > LB_TOL:
        ok = False
    var same_old = bitcast[DType.uint32](loss[1]) == bitcast[DType.uint32](old_loss[1])
    if one_block and not same_old:
        ok = False
    print(
        "LOSSBLOCK", name, "layers", L, "rows", B, "loss", loss[1], "old", old_loss[1], "ref", want, "rel", e,
        "same_old", same_old, "l2_digest", _digest(l2), "loss_digest", bitcast[DType.uint32](loss[1]),
        "PASS" if ok else "FAIL",
    )
    _ = params^
    _ = pred^
    _ = y^
    return ok


def _epoch_case[E: Exec](mut ex: E, name: String, n_batches: Int, bs: Int, last: Int) raises -> Bool:
    """`last` (1 <= last <= bs) rows in the last batch, as mlp_fit's batches."""
    var N = (n_batches - 1) * bs + last
    var bl = List[Float32](length=n_batches, fill=0)
    for bi in range(n_batches):
        bl[bi] = Float32(0.4) + Float32((bi * 53) % 211) / Float32(512)
    var dl = ex.alloc(n_batches)
    ex.upload(dl, _ptr(bl), n_batches)
    var parts = ex.alloc(mlp_lb_row_parts(n_batches))
    var dc = ex.alloc(3)
    mlp_epoch_loss_block(ex, dl, parts, dc, n_batches, bs, N, 2)
    ex.sync()
    var c = List[Float32](length=3, fill=0)
    ex.download(_ptr(c), dc, 3)
    var acc = Float64(0.0)
    for bi in range(n_batches):
        var B = bs if bi < n_batches - 1 else last
        acc += Float64(bl[bi]) * Float64(B)
    var want = acc / Float64(N)
    var e = _rel(Float64(c[2]), want)
    var ok = e <= LB_TOL
    print("LOSSBLOCK", name, "batches", n_batches, "epoch", c[2], "ref", want, "rel", e,
          "digest", bitcast[DType.uint32](c[2]), "PASS" if ok else "FAIL")
    _ = bl^
    return ok


def run_lossblock_cases[E: Exec](mut ex: E) raises:
    var passed = 0
    var total = 0
    # one block everywhere (layers 15, 20, 8 weights; 11 rows): the old bits
    var s1: List[Int] = [3, 5, 4, 2]
    total += 1
    if _batch_case(ex, "one_block_squared", s1^, 11, LOSS_SQUARED, Float32(0.02), True):
        passed += 1
    var s2: List[Int] = [3, 5, 4, 1]
    total += 1
    if _batch_case(ex, "one_block_binary", s2^, 11, LOSS_BINARY_LOG, Float32(0.02), True):
        passed += 1
    # the HIGGS shape: 28 -> 256 -> 1, 512 rows, binary log loss
    var s3: List[Int] = [28, 256, 1]
    total += 1
    if _batch_case(ex, "higgs_shape", s3^, 512, LOSS_BINARY_LOG, Float32(1.0e-4), False):
        passed += 1
    # multiclass log loss, uneven block tails
    var s4: List[Int] = [7, 300, 129, 5]
    total += 1
    if _batch_case(ex, "softmax_tails", s4^, 1000, LOSS_LOG, Float32(0.02), False):
        passed += 1
    # six layers (two groups of four), a short last batch
    var s5: List[Int] = [3, 1100, 70, 90, 60, 40, 2]
    total += 1
    if _batch_case(ex, "six_layers", s5^, 37, LOSS_SQUARED, Float32(0.5), False):
        passed += 1
    total += 1
    if _epoch_case(ex, "epoch_short", 7, 64, 9):
        passed += 1
    total += 1
    if _epoch_case(ex, "epoch_higgs", 1954, 512, 64):
        passed += 1
    if passed == total:
        print("PASS lossblock_cases", passed)
    else:
        print("FAIL lossblock_cases", passed, "of", total)
        raise Error("lossblock_cases: a case failed")
