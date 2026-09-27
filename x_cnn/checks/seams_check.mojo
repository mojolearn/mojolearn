# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S SEAM CHECK (pass 2): every numeric seam of x_cnn, device
and host against x_cnn/checks/oracle.mojo, bit for bit.

    tools/with_identical_mode.sh pixi run mojo run -I . x_cnn/checks/seams_check.mojo

Per seam: the fixture must first SEPARATE the pinned spelling from the
alternative (the oracle with alt=True), else VACUOUS and the check fails;
then the device column (x_cnn/device.mojo) and the CPU column
(x_cnn/host/ops_host.mojo) must each equal the oracle under IDENTICAL
(FAST: the counts are reported, no claim). Each seam's device result is
recorded on the identity card (MOJOLEARN_IDENTITY_TRACE) under the stage
`x_cnn.<seam>`. The sabotage arms, one per seam, are
x_cnn/checks/sabotage/seam_57xx_*.patch (tools/identity_lanes/cnn.checks)."""
from std.memory import bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from core.identity_trace import IdentityTrace
from x_cnn.ops import conv_params, pool_params, CP_OH, CP_OW, PP_OH, PP_OW
from x_cnn.checks.oracle import (
    o_col2im, o_conv_dw, o_bn_stats, o_dropout_mask, o_spmm, o_softmax, o_maxpool, o_avgpool, o_sgd, o_gcn_norm,
)
import x_cnn.device as D
import x_cnn.host.ops_host as Hh


def _fixture(n: Int, seed: UInt64, ties: Bool = False) -> List[Float32]:
    """Mixed-scale values (so a fold's order moves low bits), a -0.0, a
    subnormal, and with `ties` small integers (exact ties)."""
    var s = seed * 6364136223846793005 + 1442695040888963407
    var out = List[Float32](capacity=n)
    for i in range(n):
        s = s * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int((s >> 40) & 0xFFFFFF)) / Float32(16777216)
        if ties:
            out.append(Float32(Int(u * Float32(4))))
        else:
            var scale = Float32(1000) if i % 3 == 1 else (Float32(0.001) if i % 3 == 2 else Float32(1))
            out.append((u * Float32(2) - Float32(1)) * scale)
    if n > 2 and not ties:
        out[0] = Float32(-0.0)
        out[2] = bitcast[DType.float32](UInt32(0x00000005))
    return out^


def _diff(a: List[Float32], b: List[Float32]) -> Int:
    var c = abs(len(a) - len(b))
    for i in range(min(len(a), len(b))):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
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
    tr.record_list_f32("x_cnn." + seam, dev)


def _slice(v: List[Float32], lo: Int, hi: Int) -> List[Float32]:
    var out = List[Float32](capacity=hi - lo)
    for i in range(lo, hi):
        out.append(v[i])
    return out^


def main() raises:
    var tr = IdentityTrace()
    # ---- conv: 5700 col2im, 5701 dW. N2 C3 H7 W6, OC4, k3 s1 p1 (overlapping windows).
    var N = 2; var C = 3; var H = 7; var W = 6; var OC = 4; var K = 3
    var raw: List[Int] = [N, C, H, W, OC, K, K, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    var prm = conv_params(raw)
    var OH = Int(prm[CP_OH]); var OW = Int(prm[CP_OW])
    var x = _fixture(N * C * H * W, 1)
    var w = _fixture(OC * C * K * K, 2)
    var dout = _fixture(N * OC * OH * OW, 3)
    var bd = D.conv2d_backward_device(x, w, dout, prm)
    var bh = Hh.conv2d_backward_host(x, w, dout, prm)
    var nx = N * C * H * W
    var nw = OC * C * K * K
    # the oracle's col2im input: dcols = G W (the NN GEMM is the pinned contract, not this seam)
    var rows = N * OH * OW
    var g = List[Float32]()
    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                for oc in range(OC):
                    g.append(ftz(dout[((n * OC + oc) * OH + oh) * OW + ow]))
    var dcols = Hh.gemm_host(g, w, rows, C * K * K, OC, 0)
    _check("5700_col2im", o_col2im(dcols, N, C, H, W, K, K, 1, 1, 1, 1, 1, 1, OH, OW, False),
           o_col2im(dcols, N, C, H, W, K, K, 1, 1, 1, 1, 1, 1, OH, OW, True),
           _slice(bd, 0, nx), _slice(bh, 0, nx), tr)
    # 5701 needs k (= N*OH*OW rows) above the contract's leaf so the pinned tree and one serial fold differ
    var N2 = 16; var H2 = 16; var W2 = 16
    var raw2: List[Int] = [N2, 2, H2, W2, 3, K, K, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
    var prm2 = conv_params(raw2)
    var OH2 = Int(prm2[CP_OH]); var OW2 = Int(prm2[CP_OW])
    var x2 = _fixture(N2 * 2 * H2 * W2, 4)
    var w2 = _fixture(3 * 2 * K * K, 5)
    var d2 = _fixture(N2 * 3 * OH2 * OW2, 6)
    var nx2 = N2 * 2 * H2 * W2
    var nw2 = 3 * 2 * K * K
    var bd2 = D.conv2d_backward_device(x2, w2, d2, prm2)
    var bh2 = Hh.conv2d_backward_host(x2, w2, d2, prm2)
    _check("5701_conv_dw", o_conv_dw(x2, d2, N2, 2, H2, W2, 3, K, K, 1, 1, 1, 1, OH2, OW2, False),
           o_conv_dw(x2, d2, N2, 2, H2, W2, 3, K, K, 1, 1, 1, 1, OH2, OW2, True),
           _slice(bd2, nx2, nx2 + nw2), _slice(bh2, nx2, nx2 + nw2), tr)
    _ = nw

    # ---- 5702 BatchNorm statistics. N 9, C 3, HW 11.
    var bn_n = 9; var bn_c = 3; var bn_hw = 11
    var bx = _fixture(bn_n * bn_c * bn_hw, 7)
    var bnp: List[Int32] = [Int32(bn_n), Int32(bn_c), Int32(bn_hw)]
    var aux = List[Float32](length=2 + 7 * bn_c, fill=Float32(0))
    aux[0] = Float32(1e-5)
    aux[1] = Float32(0.1)
    for c in range(bn_c):
        aux[2 + 5 * bn_c + c] = Float32(1)
    var running = List[Float32](length=2 * bn_c, fill=Float32(0))
    var nbx = bn_n * bn_c * bn_hw
    var fd = D.batchnorm_forward_device(bx, running, aux, bnp, True)
    var fh = Hh.batchnorm_forward_host(bx, running, aux, bnp, True)
    var off = nbx + 2 * bn_c + 2
    _check("5702_bn_stats", o_bn_stats(bx, bn_n, bn_c, bn_hw, Float32(1e-5), False),
           o_bn_stats(bx, bn_n, bn_c, bn_hw, Float32(1e-5), True),
           _slice(fd, off, off + 3 * bn_c), _slice(fh, off, off + 3 * bn_c), tr)

    # ---- 5703 Dropout2d mask. N 12, C 5, HW 2, p 0.4.
    var dn = 12; var dc = 5; var dhw = 2
    var thresh = UInt64(1717986918)  # round(0.4 * 2^32)
    var dprm: List[Int32] = [Int32(dn), Int32(dc), Int32(dhw), Int32(12345), Int32(7), Int32(Int(thresh >> 16)), Int32(Int(thresh & 0xFFFF))]
    var dxin = List[Float32](length=dn * dc * dhw, fill=Float32(1))
    var dh: List[Float32] = [Float32(0.4)]
    var ddev = D.dropout2d_device(dxin, dprm, dh)
    var dhost = Hh.dropout2d_host(dxin, dprm, dh)
    var nd = dn * dc * dhw
    _check("5703_dropout_mask", o_dropout_mask(dn, dc, dhw, UInt32(12345), UInt32(7), thresh, Float32(0.4), False),
           o_dropout_mask(dn, dc, dhw, UInt32(12345), UInt32(7), thresh, Float32(0.4), True),
           _slice(ddev, nd, 2 * nd), _slice(dhost, nd, 2 * nd), tr)

    # ---- 5704 SpMM, 5710 GCN norm. 30 nodes, row r has entries (r*7+j) % 30 for j < 1 + r % 5, sorted.
    var gn = 30
    var F = 4
    var rowptr = List[Int]()
    var col = List[Int]()
    var rowof = List[Int]()
    rowptr.append(0)
    for r in range(gn):
        var cs = List[Int]()
        for j in range(1 + r % 5):
            var c = (r * 7 + j * 11) % gn
            var seen = False
            for t in cs:
                if t == c:
                    seen = True
            if not seen:
                cs.append(c)
        # ascending column order
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                if cs[b] < cs[a]:
                    var t = cs[a]
                    cs[a] = cs[b]
                    cs[b] = t
        for c in cs:
            col.append(c)
            rowof.append(r)
        rowptr.append(len(col))
    var nnz = len(col)
    var csr = List[Int32]()
    for v in rowptr:
        csr.append(Int32(v))
    for v in col:
        csr.append(Int32(v))
    for v in rowof:
        csr.append(Int32(v))
    var vals = _fixture(nnz, 8)
    var hfeat = _fixture(gn * F, 9)
    var sp: List[Int32] = [Int32(gn), Int32(F), Int32(nnz), 0]
    _check("5704_spmm", o_spmm(vals, hfeat, rowptr, col, gn, F, 0, False), o_spmm(vals, hfeat, rowptr, col, gn, F, 0, True),
           D.spmm_device(vals, hfeat, csr, sp), Hh.spmm_host(vals, hfeat, csr, sp), tr)
    var ew = List[Float32]()
    for e in range(nnz):
        ew.append(Float32(0.25) + Float32(e % 7) * Float32(0.37))
    var gp: List[Int32] = [Int32(gn), 1, Int32(nnz), 0]
    _check("5710_gcn_norm", o_gcn_norm(ew, rowptr, col, gn, False), o_gcn_norm(ew, rowptr, col, gn, True),
           D.gcn_norm_device(ew, csr, gp), Hh.gcn_norm_host(ew, csr, gp), tr)

    # ---- 5705 NaN canon / +inf limit, 5708 softmax fold order. 6 rows x 7 classes.
    var sn = 6; var sk = 7
    var logits = _fixture(sn * sk, 10)
    for j in range(sk):
        logits[j] = logits[j] * Float32(20)
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    logits[1 * sk + 2] = inf
    logits[1 * sk + 5] = inf
    logits[2 * sk + 0] = inf
    var labels = List[Int]()
    var labels32 = List[Int32]()
    for i in range(sn):
        labels.append((i * 3) % sk)
        labels32.append(Int32((i * 3) % sk))
    var smd = D.softmax_xent_device(logits, labels32, sn, sk)
    var smh = Hh.softmax_xent_host(logits, labels32, sn, sk)
    # the devices return the loss MEAN in the last slot; compare grad and proba
    var ng = 2 * sn * sk
    _check("5705_nan_canon", _slice(o_softmax(logits, labels, sn, sk, False, False), 0, ng),
           _slice(o_softmax(logits, labels, sn, sk, True, False), 0, ng), _slice(smd, 0, ng), _slice(smh, 0, ng), tr)
    var finite = _fixture(sn * sk, 11)
    for i in range(sn * sk):
        finite[i] = finite[i] * Float32(0.01)
    var sfd = D.softmax_xent_device(finite, labels32, sn, sk)
    var sfh = Hh.softmax_xent_host(finite, labels32, sn, sk)
    _check("5708_softmax_fold", _slice(o_softmax(finite, labels, sn, sk, False, False), 0, ng),
           _slice(o_softmax(finite, labels, sn, sk, False, True), 0, ng), _slice(sfd, 0, ng), _slice(sfh, 0, ng), tr)

    # ---- 5706 max pool ties, 5707 avg pool divisor. NC 6, 7x7, k3 s2 p1.
    var pnc = 6; var ph = 7; var pw = 7
    var praw: List[Int] = [1, pnc, ph, pw, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0, 1, 0]
    var pprm = pool_params(praw)
    var POH = Int(pprm[PP_OH]); var POW = Int(pprm[PP_OW])
    var tx = _fixture(pnc * ph * pw, 12, True)
    var idd = List[Int32]()
    var idh = List[Int32]()
    var mdev = D.maxpool2d_forward_device(tx, pprm, idd)
    var mhost = Hh.maxpool2d_forward_host(tx, pprm, idh)
    for v in idd:
        mdev.append(Float32(Int(v)))
    for v in idh:
        mhost.append(Float32(Int(v)))
    _check("5706_maxpool_tie", o_maxpool(tx, pnc, ph, pw, 3, 2, 1, POH, POW, False),
           o_maxpool(tx, pnc, ph, pw, 3, 2, 1, POH, POW, True), mdev, mhost, tr)
    var ax = _fixture(pnc * ph * pw, 13)
    _check("5707_avgpool_div", o_avgpool(ax, pnc, ph, pw, 3, 2, 1, POH, POW, False),
           o_avgpool(ax, pnc, ph, pw, 3, 2, 1, POH, POW, True),
           D.avgpool2d_forward_device(ax, pprm), Hh.avgpool2d_forward_host(ax, pprm), tr)

    # ---- 5709 SGD contraction.
    var sw = _fixture(257, 14)
    var sg = _fixture(257, 15)
    var sv = _fixture(257, 16)
    var hy: List[Float32] = [Float32(0.05), Float32(0.9), Float32(1e-3)]
    _check("5709_sgd_fma", o_sgd(sw, sg, sv, hy[0], hy[1], hy[2], False), o_sgd(sw, sg, sv, hy[0], hy[1], hy[2], True),
           D.sgd_device(sw, sg, sv, hy), Hh.sgd_host(sw, sg, sv, hy), tr)
    print("PASS x_cnn seams_check")
