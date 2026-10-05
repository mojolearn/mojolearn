# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE SEQUENCE LANE'S SEAM CHECK (pass 2): every numeric seam of sequence/,
device and host against sequence/checks/oracle.mojo, bit for bit.

    tools/with_identical_mode.sh pixi run mojo run -I . sequence/checks/seams_check.mojo

Per seam: the fixture must first SEPARATE the pinned spelling from the
alternative (the oracle with alt=True), else VACUOUS and the check fails;
then the device column (`DeviceExec`, one GPU thread per element) and the
CPU column (`HostExec`) must each equal the oracle under IDENTICAL (FAST:
the counts are reported, no claim). Seams in host-only code (5509 shuffle,
5510 STL weights, 5513 Nelder-Mead order) have the host column only: the
device runs the same function inside its element body, and the lane check
compares the columns end to end. Each seam's result is recorded on the
identity card (MOJOLEARN_IDENTITY_TRACE) under the stage `sequence.<seam>`.
The sabotage arms, one per seam, are sequence/checks/sabotage/seam_55xx_*.patch
(tools/identity_lanes/sequence.checks)."""
from std.memory import bitcast

from checks.fixture_rng import binade_hashed_f32, hashed_signed_f32
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul, identical_sqrt
from checks.scaffold import bits
from core.identity_trace import IdentityTrace
from sequence.exec import Exec, HostExec
from sequence.exec_device import DeviceExec
from sequence.ops import (
    FP, Args, OP_GEMM, OP_COLSUM, OP_CELL_FWD, OP_CE, OP_OPT, OP_MLP_ROWLOSS, OP_COLSCALE, OP_CHOLSOLVE,
    OP_MOE_ROUTE, OP_LN_FWD, OP_CROSTON, OP_ETS_LIK, OP_ETS_INIT, CELL_LSTM, CELL_GRU, OPT_ADAM, OPT_ADAMAX,
    OP_AF_VEC, OP_LAMB_RATIO, OP_PROPHET_FEATURES, OPT_RMSPROP, OPT_ADAGRAD, OPT_LION, OPT_NADAM, OP_MLP_PERM,
)
from sequence.theta import theta_run
from sequence.garch import garch_sigma2
from sequence.mlp import LOSS_BINARY_LOG, SplitMix, fisher_yates, mlp_epoch_key, mlp_perm_args
from sequence.stl import stl_rwts
from sequence.nm import Objective, nelder_mead
from sequence.recurrent import Net, Work, forward, head, backward
from sequence.checks.oracle import (
    o_gemm, o_gemm_split, o_colsum, o_lstm, o_gru, o_bptt_dw, o_ce, o_adam, o_adamax, o_binary_logloss, o_shuffle, o_mlp_perm,
    o_stl_rwts, o_colscale, o_cholsolve, o_nm_quantized, quant_obj, o_moe_route, o_layer_norm, o_croston,
    o_ets_calc, o_ets_init, o_nadam, o_af_vec, o_rmsprop, o_adagrad, o_lion, o_lamb_ratio, o_theta_run,
    o_garch_sigma2, o_prophet_features,
)


# ------------------------------------------------------------ fixtures
def _mixed(n: Int, seed: UInt64) -> List[Float32]:
    """Values across eight binades (so a fold's order moves low bits), with
    a sign per value; exact, never zero or subnormal."""
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(binade_hashed_f32(seed, i, -2))
    return out^


def _signed(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    """Multiples of 0.001 in [-1, 1) times `scale` (ties are frequent)."""
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(hashed_signed_f32(seed, i) * scale)
    return out^


# ------------------------------------------------------------ comparison
def _diff(a: List[Float32], b: List[Float32]) -> Int:
    var c = abs(len(a) - len(b))
    for i in range(min(len(a), len(b))):
        if bits(a[i]) != bits(b[i]):
            c += 1
    return c


def _separates(seam: String, differing: Int) raises:
    if differing == 0:
        raise Error("VACUOUS " + seam + ": the fixture does not separate the pinned spelling from the alternative")
    print("  " + seam + ": fixture separates (" + String(differing) + " cells)")


def _same(seam: String, column: String, differing: Int) raises:
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if differing != 0:
            raise Error("FAIL " + seam + " " + column + ": " + String(differing) + " cells differ from the oracle")
        print("  " + seam + " " + column + ": == oracle")
    else:
        print("  " + seam + " " + column + ": FAST, " + String(differing) + " cells differ (no claim)")


def _check(seam: String, want: List[Float32], alt: List[Float32], dev: List[Float32], host: List[Float32], mut tr: IdentityTrace) raises:
    _separates(seam, _diff(want, alt))
    _same(seam, "device", _diff(dev, want))
    _same(seam, "host", _diff(host, want))
    tr.record_list_f32("sequence." + seam, dev)


def _check_host(seam: String, want: List[Float32], alt: List[Float32], host: List[Float32], mut tr: IdentityTrace) raises:
    _separates(seam, _diff(want, alt))
    _same(seam, "host", _diff(host, want))
    tr.record_list_f32("sequence." + seam, host)


def _cat(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = a.copy()
    for i in range(len(b)):
        out.append(b[i])
    return out^


# ------------------------------------------------------------ one launch
def _set_p(mut a: Args, i: Int, p: FP):
    if i == 0: a.p0 = p
    elif i == 1: a.p1 = p
    elif i == 2: a.p2 = p
    elif i == 3: a.p3 = p
    elif i == 4: a.p4 = p
    elif i == 5: a.p5 = p
    elif i == 6: a.p6 = p
    elif i == 7: a.p7 = p
    elif i == 8: a.p8 = p
    elif i == 9: a.p9 = p
    elif i == 10: a.p10 = p
    else: a.p11 = p


def _set_i(mut a: Args, i: Int, v: Int):
    if i == 0: a.i0 = v
    elif i == 1: a.i1 = v
    elif i == 2: a.i2 = v
    elif i == 3: a.i3 = v
    elif i == 4: a.i4 = v
    elif i == 5: a.i5 = v
    elif i == 6: a.i6 = v
    elif i == 7: a.i7 = v
    elif i == 8: a.i8 = v
    elif i == 9: a.i9 = v
    elif i == 10: a.i10 = v
    else: a.i11 = v


def _set_f(mut a: Args, i: Int, v: Float32):
    if i == 0: a.f0 = v
    elif i == 1: a.f1 = v
    elif i == 2: a.f2 = v
    elif i == 3: a.f3 = v
    elif i == 4: a.f4 = v
    elif i == 5: a.f5 = v
    elif i == 6: a.f6 = v
    else: a.f7 = v


def _run[OP: Int, E: Exec](
    mut ex: E, bufs: List[List[Float32]], ints: List[Int], flts: List[Float32], n: Int,
) raises -> List[List[Float32]]:
    """Buffer j of `bufs` becomes pointer slot j (uploaded), `ints` and
    `flts` the integer and float slots; launch OP over n elements, then
    every buffer comes back."""
    var a = Args()
    var ptrs = List[FP]()
    for j in range(len(bufs)):
        var nj = len(bufs[j])
        var p = ex.alloc(nj)
        var host = bufs[j].copy()
        ex.upload(p, _host_ptr(host), nj)
        _ = host^
        ptrs.append(p)
        _set_p(a, j, p)
    for j in range(len(ints)):
        _set_i(a, j, ints[j])
    for j in range(len(flts)):
        _set_f(a, j, flts[j])
    ex.launch[OP](a, n)
    ex.sync()
    var out = List[List[Float32]]()
    for j in range(len(bufs)):
        var nj = len(bufs[j])
        var back = List[Float32](length=nj, fill=Float32(0.0))
        ex.download(_host_ptr(back), ptrs[j], nj)
        out.append(back^)
    return out^


struct Cols(Movable):
    """Every buffer after one launch: `dev` from `DeviceExec`, `host` from
    `HostExec`."""
    var dev: List[List[Float32]]
    var host: List[List[Float32]]

    def __init__(out self, var dev: List[List[Float32]], var host: List[List[Float32]]):
        self.dev = dev^
        self.host = host^


def _both[OP: Int](
    mut hx: HostExec, mut dx: DeviceExec, bufs: List[List[Float32]], ints: List[Int], flts: List[Float32], n: Int,
) raises -> Cols:
    var d = _run[OP](dx, bufs, ints, flts, n)
    var h = _run[OP](hx, bufs, ints, flts, n)
    return Cols(d^, h^)


# ------------------------------------------------------------ 5545 MLP epoch order
def _perm_col[E: Exec](mut ex: E, n: Int, key: UInt64) raises -> List[Float32]:
    """`op_mlp_perm` over n positions on one executor, as mlp_fit launches it."""
    var p = ex.alloc(n)
    ex.launch[OP_MLP_PERM](mlp_perm_args(n, key, p), n)
    ex.sync()
    var back = List[Float32](length=n, fill=Float32(0.0))
    ex.download(_host_ptr(back), p, n)
    return back^


# ------------------------------------------------------------ 5504 BPTT
def _bptt[E: Exec](mut ex: E, net: Net, P: List[Float32], x: List[Float32], dy: List[Float32], T: Int, B: Int) raises -> Tuple[List[Float32], List[Float32]]:
    """(dW_ih of layer 0, dgx) after forward, head and backward."""
    var np_ = net.n_params()
    var Pd = ex.alloc(np_)
    var Gr = ex.alloc(np_)
    var xd = ex.alloc(len(x))
    var pc = P.copy()
    var xc = x.copy()
    var dyc = dy.copy()
    ex.upload(Pd, _host_ptr(pc), np_)
    ex.upload(xd, _host_ptr(xc), len(x))
    var w = Work(ex, net, T, B, True)
    var hT = forward(ex, net, Pd, xd, T, B, w)
    head(ex, net, Pd, hT, B, w.yhat)
    ex.upload(w.dy, _host_ptr(dyc), len(dy))
    backward(ex, net, Pd, Gr, xd, T, B, w)
    ex.sync()
    var GH = net.G() * net.H
    var dw = List[Float32](length=GH * net.D, fill=Float32(0.0))
    ex.download(_host_ptr(dw), Gr + net.w_ih(0), GH * net.D)
    var dgx = List[Float32](length=T * B * GH, fill=Float32(0.0))
    ex.download(_host_ptr(dgx), w.dgx, T * B * GH)
    _ = pc^
    _ = xc^
    _ = dyc^
    _ = w^
    return (dw^, dgx^)


# ------------------------------------------------------------ 5513 NM
struct QuantObj(Objective):
    var centre: List[Float32]
    var n: Int

    def __init__(out self, var centre: List[Float32]):
        self.n = len(centre)
        self.centre = centre^

    def eval(mut self, x: FP) -> Float32:
        var v = List[Float32]()
        for j in range(self.n):
            v.append(x.unsafe_load(j))
        return quant_obj(v, self.centre)


def _host_ptr(mut v: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(v.unsafe_ptr()))


def main() raises:
    var tr = IdentityTrace()
    var hx = HostExec()
    var dx = DeviceExec()

    # ---- 5500 GEMM, M 3, N 4, K 37, accumulate into C.
    var M = 3; var N = 4; var K = 37
    var A = _mixed(M * K, 11)
    var Bm = _mixed(K * N, 12)
    var C0 = _mixed(M * N, 13)
    var g = _both[OP_GEMM](hx, dx, [A.copy(), Bm.copy(), C0.copy()], [M, N, K, K, 1, N, 1, 1, N], List[Float32](), M * N)
    _check("5500_gemm_k_order", o_gemm(A, Bm, C0, M, N, K, False), o_gemm(A, Bm, C0, M, N, K, True),
           g.dev[2], g.host[2], tr)

    # ---- 5544 the host GEMM's vector cells (`sequence/host_gemm.mojo`):
    # M 7 (a 4-row block, then 3 single rows), N 37 (full vectors at every
    # host width, then a scalar tail), K 29, accumulate into C; B read along
    # n (stride 1), then B stored [N, K] (the packed panel). The device runs
    # op_gemm on both.
    var M2 = 7; var N2 = 37; var K2 = 29
    var A2 = _mixed(M2 * K2, 41)
    var B2 = _mixed(K2 * N2, 42)
    var C2 = _mixed(M2 * N2, 43)
    var B2t = List[Float32](length=K2 * N2, fill=Float32(0.0))
    for k in range(K2):
        for n in range(N2):
            B2t[n * K2 + k] = B2[k * N2 + n]
    var gv = _both[OP_GEMM](hx, dx, [A2.copy(), B2.copy(), C2.copy()], [M2, N2, K2, K2, 1, N2, 1, 1, N2], List[Float32](), M2 * N2)
    var gp = _both[OP_GEMM](hx, dx, [A2.copy(), B2t.copy(), C2.copy()], [M2, N2, K2, K2, 1, 1, K2, 1, N2], List[Float32](), M2 * N2)
    var want2 = o_gemm(A2, B2, C2, M2, N2, K2, False)
    var alt2 = o_gemm_split(A2, B2, C2, M2, N2, K2)
    _check("5544_host_gemm_vector", _cat(want2, want2), _cat(alt2, alt2),
           _cat(gv.dev[2], gp.dev[2]), _cat(gv.host[2], gp.host[2]), tr)

    # ---- 5501 column sums, R 41, C 3.
    var R = 41; var Cc = 3
    var X = _mixed(R * Cc, 21)
    var cs = _both[OP_COLSUM](hx, dx, [X.copy(), List[Float32](length=Cc, fill=Float32(0.0))], [R, Cc, Cc, 0], List[Float32](), Cc)
    _check("5501_colsum_order", o_colsum(X, R, Cc, False), o_colsum(X, R, Cc, True), cs.dev[1], cs.host[1], tr)

    # ---- 5502 LSTM step, B 4, H 8.
    var Bb = 4; var H = 8
    var gx = _signed(Bb * 4 * H, 31, 3.0)
    var gh = _signed(Bb * 4 * H, 32, 3.0)
    var cprev = _signed(Bb * H, 33, 5.0)
    var hprev = _signed(Bb * H, 34, 1.0)
    var z4 = List[Float32](length=Bb * 4 * H, fill=Float32(0.0))
    var zH = List[Float32](length=Bb * H, fill=Float32(0.0))
    var ls = _both[OP_CELL_FWD](hx, dx, [gx.copy(), gh.copy(), z4.copy(), hprev.copy(), cprev.copy(), zH.copy(), zH.copy()],
                                [CELL_LSTM, Bb, H], List[Float32](), Bb * H)
    _check("5502_lstm_cell", o_lstm(gx, gh, cprev, Bb, H, False), o_lstm(gx, gh, cprev, Bb, H, True),
           _cat(ls.dev[5], ls.dev[6]), _cat(ls.host[5], ls.host[6]), tr)

    # ---- 5503 GRU step, B 4, H 8.
    var gx3 = _signed(Bb * 3 * H, 41, 3.0)
    var gh3 = _signed(Bb * 3 * H, 42, 3.0)
    var hp3 = _signed(Bb * H, 43, 2.0)
    var z3 = List[Float32](length=Bb * 3 * H, fill=Float32(0.0))
    var gr = _both[OP_CELL_FWD](hx, dx, [gx3.copy(), gh3.copy(), z3.copy(), hp3.copy(), zH.copy(), zH.copy(), zH.copy()],
                                [CELL_GRU, Bb, H], List[Float32](), Bb * H)
    _check("5503_gru_update", o_gru(gx3, gh3, hp3, Bb, H, False), o_gru(gx3, gh3, hp3, Bb, H, True),
           gr.dev[5], gr.host[5], tr)

    # ---- 5504 BPTT weight gradient: an LSTM, D 3, H 4, one layer, O 2, T 6, B 3.
    var net = Net(CELL_LSTM, 3, 4, 1, 2)
    var T = 6; var B5 = 3
    var P = _signed(net.n_params(), 51, 0.5)
    var x5 = _mixed(T * B5 * 3, 52)
    var dy = _signed(B5 * 2, 53, 1.0)
    var bd = _bptt(dx, net, P, x5, dy, T, B5)
    var bh = _bptt(hx, net, P, x5, dy, T, B5)
    var GH = net.G() * net.H
    _check("5504_bptt_fold", o_bptt_dw(bh[1], x5, T, B5, GH, 3, False), o_bptt_dw(bh[1], x5, T, B5, GH, 3, True),
           bd[0], bh[0], tr)
    _same("5504_bptt_fold dgx", "device vs host", _diff(bd[1], bh[1]))
    # nr-small D3: K = T B = 600 > 512 rows, two blocks of the IDENTICAL
    # blocked weight-gradient order (one fold when it is off)
    var Tb = 200
    var xb5 = _mixed(Tb * B5 * 3, 54)
    var bdb = _bptt(dx, net, P, xb5, dy, Tb, B5)
    var bhb = _bptt(hx, net, P, xb5, dy, Tb, B5)
    _check("5504_bptt_fold_blocked", o_bptt_dw(bhb[1], xb5, Tb, B5, GH, 3, False),
           o_bptt_dw(bhb[1], xb5, Tb, B5, GH, 3, True), bdb[0], bhb[0], tr)
    _same("5504_bptt_fold_blocked dgx", "device vs host", _diff(bdb[1], bhb[1]))

    # ---- 5505 softmax cross entropy, B 3, C 40.
    var B6 = 3; var C6 = 40
    var logits = _signed(B6 * C6, 61, 6.0)
    var labels: List[Int] = [7, 0, 39]
    var lab = List[Float32]()
    for i in range(B6):
        lab.append(Float32(labels[i]))
    var scale6 = Float32(1.0) / Float32(3.0)
    var ce = _both[OP_CE](hx, dx, [logits.copy(), lab.copy(), List[Float32](length=B6 * C6, fill=Float32(0.0)),
                                   List[Float32](length=B6, fill=Float32(0.0))], [0, C6], [scale6], B6)
    _check("5505_ce_expsum", o_ce(logits, labels, B6, C6, scale6, False), o_ce(logits, labels, B6, C6, scale6, True),
           _cat(ce.dev[2], ce.dev[3]), _cat(ce.host[2], ce.host[3]), tr)

    # ---- 5506 Adam, 64 parameters, step 3, L2 weight decay.
    var n7 = 64
    var p7 = _signed(n7, 71, 1.0)
    var g7 = _mixed(n7, 72)
    var m7 = _signed(n7, 73, 0.1)
    var v7 = List[Float32]()
    var vs = _signed(n7, 74, 0.01)
    for i in range(n7):
        v7.append(abs(vs[i]) + Float32(1e-4))
    var b1 = Float32(0.9); var b2 = Float32(0.999); var lr = Float32(1e-3)
    # lr 1 and parameters near 1e-3: the step, where the two denominators
    # part, is not rounded away against p.
    var lrA = Float32(1.0)
    var pA = _signed(n7, 75, 0.001)
    var pw1 = ftz(identical_mul(ftz(identical_mul(b1, b1)), b1))
    var pw2 = ftz(identical_mul(ftz(identical_mul(b2, b2)), b2))
    var f5 = ftz(identical_div(lrA, Float32(1.0) - pw1))
    var f6 = ftz(identical_sqrt(Float32(1.0) - pw2))
    var z7 = List[Float32](length=n7, fill=Float32(0.0))
    var ad = _both[OP_OPT](hx, dx, [pA.copy(), g7.copy(), m7.copy(), v7.copy(), z7.copy()],
                           [OPT_ADAM, 3, 0], [lrA, b1, b2, Float32(1e-8), Float32(0.01), f5, f6, Float32(0.0)], n7)
    _check("5506_adam_denom", o_adam(pA, g7, m7, v7, b1, b2, Float32(1e-8), Float32(0.01), f5, f6, False),
           o_adam(pA, g7, m7, v7, b1, b2, Float32(1e-8), Float32(0.01), f5, f6, True),
           _cat(_cat(ad.dev[0], ad.dev[2]), ad.dev[3]), _cat(_cat(ad.host[0], ad.host[2]), ad.host[3]), tr)

    # ---- 5507 torch.lerp through Adamax, beta1 0.3 (w = 0.7, the upper branch).
    var b1x = Float32(0.3); var b2x = Float32(0.99)
    var clr = ftz(identical_div(lr, Float32(1.0) - b1x))
    var ax = _both[OP_OPT](hx, dx, [p7.copy(), g7.copy(), m7.copy(), v7.copy(), z7.copy()],
                           [OPT_ADAMAX, 1, 0], [lr, b1x, b2x, Float32(1e-8), Float32(0.0), clr, Float32(0.0), Float32(0.0)], n7)
    _check("5507_torch_lerp", o_adamax(p7, g7, m7, v7, b1x, b2x, Float32(1e-8), Float32(0.0), clr, False),
           o_adamax(p7, g7, m7, v7, b1x, b2x, Float32(1e-8), Float32(0.0), clr, True),
           _cat(_cat(ax.dev[0], ax.dev[2]), ax.dev[3]), _cat(_cat(ax.host[0], ax.host[2]), ax.host[3]), tr)

    # ---- 5507 torch.lerp through NAdam (w = 1 - beta1 = 0.7, the upper branch) and
    #      Adafactor's vector second moment (w 0.8): the same `lerp` body, their own ops.
    var bc2n = Float32(1.0) - ftz(identical_mul(b2x, b2x))
    var c1n = Float32(-4e-4); var c2n = Float32(-5e-4)
    var na = _both[OP_OPT](hx, dx, [p7.copy(), g7.copy(), m7.copy(), v7.copy(), z7.copy()],
                           [OPT_NADAM, 2, 0], [lr, b1x, b2x, Float32(1e-8), Float32(0.0), bc2n, c1n, c2n], n7)
    _check("5507_torch_lerp_nadam", o_nadam(p7, g7, m7, v7, b1x, b2x, Float32(1e-8), bc2n, c1n, c2n, False),
           o_nadam(p7, g7, m7, v7, b1x, b2x, Float32(1e-8), bc2n, c1n, c2n, True),
           _cat(_cat(na.dev[0], na.dev[2]), na.dev[3]), _cat(_cat(na.host[0], na.host[2]), na.host[3]), tr)
    var wv = Float32(0.8); var e1sq = Float32(1e-30)
    var av = _both[OP_AF_VEC](hx, dx, [g7.copy(), v7.copy(), z7.copy()], List[Int](), [wv, e1sq], n7)
    _check("5507_torch_lerp_adafactor", o_af_vec(g7, v7, wv, e1sq, False), o_af_vec(g7, v7, wv, e1sq, True),
           _cat(av.dev[1], av.dev[2]), _cat(av.host[1], av.host[2]), tr)

    # ---- 5508 MLP binary log loss, 10 rows, O 1: p at 0, 1 and near both ends.
    var pb: List[Float32] = [0.0, 1.0, 1e-9, 0.99999994, 0.3, 0.7, 5e-8, 0.5, 1.0, 0.0]
    var yb: List[Float32] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    var nb = len(pb)
    var ll = _both[OP_MLP_ROWLOSS](hx, dx, [pb.copy(), yb.copy(), List[Float32](length=nb, fill=Float32(0.0)),
                                             List[Float32](length=nb, fill=Float32(0.0))], [LOSS_BINARY_LOG, 1], List[Float32](), nb)
    _check("5508_logloss_clip", o_binary_logloss(pb, yb, False), o_binary_logloss(pb, yb, True),
           _cat(ll.dev[2], ll.dev[3]), _cat(ll.host[2], ll.host[3]), tr)

    # ---- 5509 the MLP shuffle (host), n 97, seed 20260927.
    var perm = List[Float32]()
    for i in range(97):
        perm.append(Float32(i))
    var rng = SplitMix(UInt64(20260927))
    fisher_yates(rng, perm)
    _check_host("5509_shuffle", o_shuffle(97, UInt64(20260927), False), o_shuffle(97, UInt64(20260927), True), perm, tr)

    # ---- 5510 STL robustness weights (host), 12 series of 24.
    var want10 = List[Float32](); var alt10 = List[Float32](); var got10 = List[Float32]()
    for s in range(12):
        var ys = _signed(24, UInt64(100 + s), 7.0)
        var fs = _signed(24, UInt64(200 + s), 3.0)
        var rw = List[Float32](length=24, fill=Float32(0.0))
        var sb = List[Float32](length=24, fill=Float32(0.0))
        stl_rwts(_host_ptr(ys), 24, _host_ptr(fs), _host_ptr(rw), _host_ptr(sb))
        want10 = _cat(want10, o_stl_rwts(ys, fs, False))
        alt10 = _cat(alt10, o_stl_rwts(ys, fs, True))
        got10 = _cat(got10, rw)
        _ = ys^
        _ = fs^
        _ = sb^
    _check_host("5510_stl_rwts", want10, alt10, got10, tr)

    # ---- 5511 VAR column scaling, R 9, 4 columns (one all zero).
    var R11 = 9; var M11 = 4
    var X11 = _mixed(R11 * M11, 111)
    for r in range(R11):
        X11[r * M11 + 2] = Float32(0.0)
        X11[r * M11 + 3] = X11[r * M11 + 3] * Float32(37.0)
    var sc = _both[OP_COLSCALE](hx, dx, [X11.copy(), List[Float32](length=M11, fill=Float32(0.0))], [R11, M11], List[Float32](), M11)
    _check("5511_var_pow2_scale", o_colscale(X11, R11, M11, False), o_colscale(X11, R11, M11, True),
           _cat(sc.dev[0], sc.dev[1]), _cat(sc.host[0], sc.host[1]), tr)

    # ---- 5512 VAR Cholesky solve: an SPD 7x7 with 2 right-hand sides, and an indefinite one.
    var m12 = 7; var K12 = 2
    var F = _mixed(m12 * m12, 121)
    var G = List[Float32](length=m12 * m12, fill=Float32(0.0))
    for i in range(m12):
        for j in range(m12):
            var acc = Float32(0.0)
            for k in range(m12):
                acc = acc + F[i * m12 + k] * F[j * m12 + k]
            G[i * m12 + j] = acc
    for i in range(m12):
        G[i * m12 + i] = G[i * m12 + i] + Float32(0.5)
    for i in range(m12):
        for j in range(i):
            G[j * m12 + i] = G[i * m12 + j]
    var B12 = _mixed(m12 * K12, 122)
    var ch = _both[OP_CHOLSOLVE](hx, dx, [G.copy(), B12.copy(), [Float32(-1.0)]], [m12, K12], List[Float32](), 1)
    var Gbad = G.copy()
    Gbad[4 * m12 + 4] = Float32(-3.0)
    var chb = _both[OP_CHOLSOLVE](hx, dx, [Gbad.copy(), B12.copy(), [Float32(-1.0)]], [m12, K12], List[Float32](), 1)
    _check("5512_var_cholesky",
           _cat(o_cholsolve(G, B12, m12, K12, False), o_cholsolve(Gbad, B12, m12, K12, False)),
           _cat(o_cholsolve(G, B12, m12, K12, True), o_cholsolve(Gbad, B12, m12, K12, True)),
           _cat(_cat(_cat(ch.dev[0], ch.dev[1]), ch.dev[2]), _cat(_cat(chb.dev[0], chb.dev[1]), chb.dev[2])),
           _cat(_cat(_cat(ch.host[0], ch.host[1]), ch.host[2]), _cat(_cat(chb.host[0], chb.host[1]), chb.host[2])), tr)

    # ---- 5513 Nelder-Mead simplex order (host): a staircase objective, 3 coordinates, 4 starts.
    var want13 = List[Float32](); var alt13 = List[Float32](); var got13 = List[Float32]()
    for s in range(4):
        var c13 = _signed(3, UInt64(130 + s), 2.0)
        var x13 = _signed(3, UInt64(140 + s), 1.0)
        var lo: List[Float32] = [-5.0, -5.0, -5.0]
        var hi: List[Float32] = [5.0, 5.0, 5.0]
        want13 = _cat(want13, o_nm_quantized(x13, lo, hi, c13, Float32(0.5), Float32(0.25), 60, Float32(1e-4), False))
        alt13 = _cat(alt13, o_nm_quantized(x13, lo, hi, c13, Float32(0.5), Float32(0.25), 60, Float32(1e-4), True))
        var obj = QuantObj(c13.copy())
        var xw = x13.copy()
        var scratch = List[Float32](length=(3 + 1) * 3 + (3 + 1) + 4 * 3, fill=Float32(0.0))
        var its = nelder_mead(obj, _host_ptr(xw), _host_ptr(lo), _host_ptr(hi), 3, _host_ptr(scratch),
                              Float32(0.5), Float32(0.25), 60, Float32(1e-4))
        xw.append(Float32(its))
        got13 = _cat(got13, xw)
        _ = scratch^
        _ = lo^
        _ = hi^
    _check_host("5513_nm_tie_order", want13, alt13, got13, tr)

    # ---- 5514 MoE routing, T 4, D 3, E 6, k 2: experts 1 and 4 share a router row (exact ties).
    var T14 = 4; var D14 = 3; var E14 = 6; var k14 = 2
    var x14 = _signed(T14 * D14, 141, 1.0)
    var W14 = _signed(E14 * D14, 142, 1.0)
    for d in range(D14):
        var v14 = W14[D14 + d] + (Float32(4.0) if d == 0 else Float32(0.0))
        W14[D14 + d] = v14
        W14[4 * D14 + d] = v14
    var mo = _both[OP_MOE_ROUTE](hx, dx, [x14.copy(), W14.copy(), List[Float32](length=T14 * E14, fill=Float32(0.0)),
                                          List[Float32](length=T14 * k14, fill=Float32(0.0)),
                                          List[Float32](length=T14 * k14, fill=Float32(0.0)),
                                          List[Float32](length=T14 * E14, fill=Float32(0.0))],
                                 [D14, E14, k14, 1], List[Float32](), T14)
    _check("5514_moe_route_tie", o_moe_route(x14, W14, T14, D14, E14, k14, True, False),
           o_moe_route(x14, W14, T14, D14, E14, k14, True, True),
           _cat(_cat(mo.dev[2], mo.dev[3]), mo.dev[4]), _cat(_cat(mo.host[2], mo.host[3]), mo.host[4]), tr)

    # ---- 5515 LayerNorm forward, M 3, D 33, weight and bias.
    var M15 = 3; var D15 = 33
    var x15 = _mixed(M15 * D15, 151)
    var w15 = _signed(D15, 152, 2.0)
    var b15 = _signed(D15, 153, 1.0)
    var eps15 = Float32(1e-5)
    var ln = _both[OP_LN_FWD](hx, dx, [x15.copy(), w15.copy(), b15.copy(), List[Float32](length=M15 * D15, fill=Float32(0.0)),
                                       List[Float32](length=M15, fill=Float32(0.0)), List[Float32](length=M15, fill=Float32(0.0))],
                              [D15, 1, 1], [eps15], M15)
    _check("5515_layernorm_stats", o_layer_norm(x15, w15, b15, M15, D15, eps15, False),
           o_layer_norm(x15, w15, b15, M15, D15, eps15, True),
           _cat(_cat(ln.dev[3], ln.dev[4]), ln.dev[5]), _cat(_cat(ln.host[3], ln.host[4]), ln.host[5]), tr)

    # ---- 5516 SES recursion through Croston: 5 intermittent series of 40, classic and optimized.
    var B16 = 5; var n16 = 40
    var y16 = List[Float32]()
    var raw16 = _signed(B16 * n16, 161, 9.0)
    for i in range(B16 * n16):
        var v = raw16[i]
        y16.append(Float32(Int(v)) if v > Float32(4.0) else Float32(0.0))
    var want16 = List[Float32](); var alt16 = List[Float32](); var dev16 = List[Float32](); var host16 = List[Float32]()
    for variant in range(2):
        var cr = _both[OP_CROSTON](hx, dx, [y16.copy(), List[Float32](length=B16, fill=Float32(0.0)),
                                            List[Float32](length=B16 * 2 * n16, fill=Float32(0.0))],
                                   [n16, variant], List[Float32](), B16)
        want16 = _cat(want16, o_croston(y16, B16, n16, variant, False))
        alt16 = _cat(alt16, o_croston(y16, B16, n16, variant, True))
        dev16 = _cat(dev16, cr.dev[1])
        host16 = _cat(host16, cr.host[1])
    _check("5516_ses_recursion", want16, alt16, dev16, host16, tr)

    # ---- 5517 ETS seasonal update through Calc (OP_ETS_LIK): 4 series of 36, m 5,
    #      (A error, A season, trend), (M, M, trend), (M, A, no trend); lik, final l, b, s.
    var B17 = 4; var n17 = 36; var m17 = 5
    var raw17 = _signed(B17 * n17, 171, 3.0)
    var wav17 = _signed(m17, 172, 4.0)
    var y17 = List[Float32]()
    for i in range(B17 * n17):
        y17.append(Float32(30.0) + wav17[(i % n17) % m17] + raw17[i])
    var ps17 = _signed(B17 * 8, 173, 1.0)
    var want17 = List[Float32](); var alt17 = List[Float32](); var dev17 = List[Float32](); var host17 = List[Float32]()
    for cfg in range(3):
        var err17 = 0 if cfg == 0 else 1
        var seas17 = 2 if cfg == 1 else 1
        var tr17 = cfg != 2
        var par17 = List[Float32]()
        for s in range(B17):
            par17.append(Float32(0.3) + Float32(0.2) * ps17[8 * s])                 # alpha
            par17.append(Float32(0.05) + Float32(0.04) * ps17[8 * s + 1])           # beta
            par17.append(Float32(0.2) + Float32(0.15) * ps17[8 * s + 2])            # gamma
            par17.append(Float32(0.9) + Float32(0.05) * ps17[8 * s + 3])            # phi
            par17.append(Float32(30.0) + ps17[8 * s + 4])                           # l0
            par17.append(Float32(0.1) * ps17[8 * s + 5])                            # b0
            for j in range(m17):
                var w = wav17[(m17 - 1 - j) % m17]
                par17.append(Float32(1.0) + Float32(0.02) * w if seas17 == 2 else w)
        var ec = _both[OP_ETS_LIK](hx, dx, [y17.copy(), par17.copy(), List[Float32](length=B17 * (3 + m17), fill=Float32(0.0)),
                                            List[Float32](length=B17 * m17, fill=Float32(0.0))],
                                   [n17, 0, err17, 1 if tr17 else 0, 0, 0, seas17, m17], List[Float32](), B17)
        want17 = _cat(want17, o_ets_calc(y17, B17, n17, err17, tr17, seas17, m17, par17, False))
        alt17 = _cat(alt17, o_ets_calc(y17, B17, n17, err17, tr17, seas17, m17, par17, True))
        dev17 = _cat(dev17, ec.dev[2])
        host17 = _cat(host17, ec.host[2])
    _check("5517_ets_seasonal_update", want17, alt17, dev17, host17, tr)

    # ---- 5518 ETS initstate by decomposition (OP_ETS_INIT, n >= 3m): 3 series of 40,
    #      m 6 (even, half-weight ends) and m 5, additive and multiplicative, with a trend.
    var B18 = 3; var n18 = 40
    var raw18 = _signed(B18 * n18, 181, 5.0)
    var want18 = List[Float32](); var alt18 = List[Float32](); var dev18 = List[Float32](); var host18 = List[Float32]()
    for cfg in range(4):
        var m18 = 6 if cfg < 2 else 5
        var seas18 = 1 if cfg % 2 == 0 else 2
        var wav18 = _signed(m18, 182 + UInt64(cfg), 6.0)
        var y18 = List[Float32]()
        for i in range(B18 * n18):
            var t = i % n18
            y18.append(Float32(40.0) + Float32(0.25) * Float32(t) + wav18[t % m18] + raw18[i])
        var st18 = 6 * n18 + 8
        var ic = _both[OP_ETS_INIT](hx, dx, [y18.copy(), List[Float32](length=B18 * (1 + m18), fill=Float32(0.0)),
                                             List[Float32](length=1, fill=Float32(0.0)),
                                             List[Float32](length=B18 * st18, fill=Float32(0.0))],
                                    [n18, 0, 0, 1, 0, 0, seas18, m18, st18], List[Float32](), B18)
        want18 = _cat(want18, o_ets_init(y18, B18, n18, True, seas18, m18, False))
        alt18 = _cat(alt18, o_ets_init(y18, B18, n18, True, seas18, m18, True))
        dev18 = _cat(dev18, ic.dev[1])
        host18 = _cat(host18, ic.host[1])
    _check("5518_ets_decompose_ma", want18, alt18, dev18, host18, tr)

    # ---- 5536 RMSprop, centered, momentum 0.9, L2 weight decay: 64 parameters whose
    #      second moment sits just above gavg^2, so v - gavg^2 cancels and its rounding shows.
    var n36 = 64
    var p36 = _signed(n36, 361, 0.001)
    var g36 = _mixed(n36, 362)
    var ga36 = _signed(n36, 363, 0.5)
    var v36 = List[Float32]()
    for i in range(n36):
        v36.append(ga36[i] * ga36[i] + Float32(1e-4))
    var b36 = _signed(n36, 364, 0.1)
    var al36 = Float32(0.99); var mu36 = Float32(0.9); var wd36 = Float32(0.01)
    var rp = _both[OP_OPT](hx, dx, [p36.copy(), g36.copy(), b36.copy(), v36.copy(), ga36.copy()],
                           [OPT_RMSPROP, 3, 1], [Float32(1.0), Float32(0.0), al36, Float32(1e-8), wd36, Float32(0.0),
                                                 Float32(0.0), mu36], n36)
    _check("5536_rmsprop_centered_var",
           o_rmsprop(p36, g36, b36, v36, ga36, Float32(1.0), al36, Float32(1e-8), wd36, mu36, False),
           o_rmsprop(p36, g36, b36, v36, ga36, Float32(1.0), al36, Float32(1e-8), wd36, mu36, True),
           _cat(_cat(_cat(rp.dev[0], rp.dev[2]), rp.dev[3]), rp.dev[4]),
           _cat(_cat(_cat(rp.host[0], rp.host[2]), rp.host[3]), rp.host[4]), tr)

    # ---- 5537 Adagrad, 64 parameters, step 3, weight decay 0.01.
    var s37 = List[Float32]()
    var r37 = _signed(n36, 371, 1.0)
    for i in range(n36):
        s37.append(abs(r37[i]) + Float32(0.01))
    var clr37 = Float32(0.05)
    var zz = List[Float32](length=n36, fill=Float32(0.0))
    var ag = _both[OP_OPT](hx, dx, [p36.copy(), g36.copy(), zz.copy(), s37.copy(), zz.copy()],
                           [OPT_ADAGRAD, 3, 0], [Float32(0.1), Float32(0.0), Float32(0.0), Float32(1e-10), wd36, clr37,
                                                 Float32(0.0), Float32(0.0)], n36)
    _check("5537_adagrad_sum", o_adagrad(p36, g36, s37, clr37, Float32(1e-10), wd36, False),
           o_adagrad(p36, g36, s37, clr37, Float32(1e-10), wd36, True),
           _cat(ag.dev[0], ag.dev[3]), _cat(ag.host[0], ag.host[3]), tr)

    # ---- 5538 Lion, 64 parameters, betas (0.9, 0.99), weight decay 0.1.
    var m38 = _signed(n36, 381, 0.5)
    var lr38 = Float32(1e-3)
    var li = _both[OP_OPT](hx, dx, [p36.copy(), g36.copy(), m38.copy(), zz.copy(), zz.copy()],
                           [OPT_LION, 1, 0], [lr38, Float32(0.9), Float32(0.99), Float32(0.0), Float32(0.1), Float32(0.0),
                                              Float32(0.0), Float32(0.0)], n36)
    _check("5538_lion_momentum", o_lion(p36, g36, m38, lr38, Float32(0.9), Float32(0.99), Float32(0.1), False),
           o_lion(p36, g36, m38, lr38, Float32(0.9), Float32(0.99), Float32(0.1), True),
           _cat(li.dev[0], li.dev[2]), _cat(li.host[0], li.host[2]), tr)

    # ---- 5539 LAMB trust ratio: 5 tensors (one all-zero parameter), trust_clip off and on.
    var offs39: List[Int] = [0, 7, 20, 33, 40, 64]
    var offf39 = List[Float32]()
    for i in range(len(offs39)):
        offf39.append(Float32(offs39[i]))
    var p39 = _mixed(n36, 391)
    for i in range(33, 40):
        p39[i] = Float32(0.0)
    var u39 = _mixed(n36, 392)
    var nseg = len(offs39) - 1
    var want39 = List[Float32](); var alt39 = List[Float32](); var dev39 = List[Float32](); var host39 = List[Float32]()
    for clip in range(2):
        var lr39 = _both[OP_LAMB_RATIO](hx, dx, [p39.copy(), u39.copy(), offf39.copy(), List[Float32](length=nseg, fill=Float32(0.0))],
                                        [clip], List[Float32](), nseg)
        want39 = _cat(want39, o_lamb_ratio(p39, u39, offs39, clip == 1, False))
        alt39 = _cat(alt39, o_lamb_ratio(p39, u39, offs39, clip == 1, True))
        dev39 = _cat(dev39, lr39.dev[3])
        host39 = _cat(host39, lr39.host[3])
    _check("5539_lamb_trust_ratio", want39, alt39, dev39, host39, tr)

    # ---- 5541 Theta's level recursion (host; theta_run is op_theta's body): a trending
    #      series of 30, STM, OTM (theta 2.7) and DSTM, alpha 0.37, level0 y0 / 2.
    var n41 = 30
    var r41 = _signed(n41, 411, 3.0)
    var y41 = List[Float32]()
    for i in range(n41):
        y41.append(Float32(20.0) + Float32(0.3) * Float32(i) + r41[i])
    var want41 = List[Float32](); var alt41 = List[Float32](); var got41 = List[Float32]()
    for model in range(3):
        var th41 = Float32(2.7) if model == 1 else Float32(2.0)
        var l041 = y41[0] * Float32(0.5)
        want41 = _cat(want41, o_theta_run(y41, model, l041, Float32(0.37), th41, False))
        alt41 = _cat(alt41, o_theta_run(y41, model, l041, Float32(0.37), th41, True))
        var yw = y41.copy()
        var st41 = List[Float32](length=5 * n41, fill=Float32(0.0))
        var e41 = List[Float32](length=n41, fill=Float32(0.0))
        var mse = theta_run(_host_ptr(yw), n41, model, l041, Float32(0.37), th41, _host_ptr(st41), _host_ptr(e41))
        got41 = _cat(_cat(got41, st41), e41)
        got41.append(mse)
        _ = yw^
    _check_host("5541_theta_level", want41, alt41, got41, tr)

    # ---- 5542 GARCH variance recursion (host; garch_sigma2 is op_garch's body): 40 returns,
    #      GJR(1, 1, 1) and GARCH(2, 0, 2); one step under its floor, one over its cap.
    var n42 = 40
    var r42 = _signed(n42, 421, 2.0)
    var vb42 = List[Float32]()
    for t in range(n42):
        vb42.append(Float32(50.0) if t == 10 else Float32(1e-6))
        vb42.append(Float32(0.2) if t == 20 else Float32(1e3))
    var bc42 = Float32(1.3)
    var want42 = List[Float32](); var alt42 = List[Float32](); var got42 = List[Float32]()
    for cfg in range(2):
        var p42 = 1 if cfg == 0 else 2
        var o42 = 1 if cfg == 0 else 0
        var q42 = 1 if cfg == 0 else 2
        var par42: List[Float32]
        if cfg == 0:
            par42 = [Float32(0.05), Float32(0.08), Float32(0.1), Float32(0.85)]
        else:
            par42 = [Float32(0.03), Float32(0.07), Float32(0.02), Float32(0.5), Float32(0.37)]
        want42 = _cat(want42, o_garch_sigma2(par42, r42, p42, o42, q42, bc42, vb42, False))
        alt42 = _cat(alt42, o_garch_sigma2(par42, r42, p42, o42, q42, bc42, vb42, True))
        var pw = par42.copy(); var rw = r42.copy(); var vw = vb42.copy()
        var s42 = List[Float32](length=n42, fill=Float32(0.0))
        garch_sigma2(_host_ptr(pw), _host_ptr(rw), n42, p42, o42, q42, bc42, _host_ptr(vw), _host_ptr(s42))
        got42 = _cat(got42, s42)
        _ = pw^
        _ = rw^
        _ = vw^
    _check_host("5542_garch_recursion", want42, alt42, got42, tr)

    # ---- 5543 Prophet Fourier features: 16 rows, seasonalities of order 3 and 10, 2 holidays.
    var N43 = 16; var nh43 = 2
    var ord43: List[Int] = [3, 10]
    var ordf43: List[Float32] = [3.0, 10.0]
    var K43 = 2 * (3 + 10) + nh43
    var fr43 = List[Float32]()
    var raw43 = _signed(N43 * 2, 431, 1.0)
    for i in range(N43 * 2):
        fr43.append(abs(raw43[i]))
    var hol43 = _signed(N43 * nh43, 432, 1.0)
    var pf = _both[OP_PROPHET_FEATURES](hx, dx, [fr43.copy(), ordf43.copy(), hol43.copy(),
                                                 List[Float32](length=N43 * K43, fill=Float32(0.0))],
                                        [2, nh43, K43], List[Float32](), N43)
    _check("5543_prophet_fourier", o_prophet_features(fr43, ord43, hol43, N43, nh43, False),
           o_prophet_features(fr43, ord43, hol43, N43, nh43, True), pf.dev[3], pf.host[3], tr)

    # ---- 5545 the MLP device epoch order (`op_mlp_perm`, roadmap D13), n 97
    # (a 256-value domain: the cycle walk runs), seed 20260927, epoch 3.
    var key45 = mlp_epoch_key(UInt64(20260927), 3)
    _check("5545_mlp_epoch_perm", o_mlp_perm(97, UInt64(20260927), 3, False), o_mlp_perm(97, UInt64(20260927), 3, True),
           _perm_col(dx, 97, key45), _perm_col(hx, 97, key45), tr)

    _ = dx^
    _ = hx^
    print("PASS sequence seams (28 seams: 5500-5518, 5536-5539, 5541-5545; 5540 is sched_check.py)")
