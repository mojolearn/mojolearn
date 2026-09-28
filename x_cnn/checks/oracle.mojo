# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CNN LANE'S HOST ORACLES, one per numeric seam (pass 2).

Each restates its seam as plain nested host loops over the SAME numeric
primitives (ftz, identical_mul/div/exp/log/rsqrt, gemm_oracle), written
from the reference semantics and NOT from x_cnn/ops.mojo: a bug or a
sabotage in the shared element functions moves the device and host columns
together, and only an independent restatement catches it. Every oracle takes
an `alt` flag that spells the seam the OTHER legal way; x_cnn/checks/
seams_check.mojo first requires pinned != alt on its fixture (VACUOUS
otherwise), then device == oracle and host == oracle bit for bit.

Seams (DEVIATION numbers, lane cnn 5700-5799):
  5700 col2im: per input pixel, a gather over (kh, kw) ascending    alt: descending
  5701 conv weight gradient: GEMM TN, the pinned leaves + fold      alt: one serial fold
  5702 BatchNorm statistics: one fold per channel over (n, hw)      alt: n descending
  5703 Dropout2d: Philox counter = n*C + c, integer threshold       alt: counter c*N + n
  5704 SpMM (GCN/SAGE): row entries folded in column order          alt: reversed
  5705 NaN canon + softmax +inf limit                               alt: raw arithmetic
  5706 max pool: the FIRST maximum wins (strict >)                  alt: the last (>=)
  5707 avg pool: window sum, then / divisor                         alt: sum * (1 / divisor)
  5708 softmax: the exp-sum folded in column order                  alt: reversed
  5709 SGD: momentum * v + d, the product pinned (no FMA)           alt: fused multiply-add
  5710 GCN norm: dis[src] * w * dis[dst] left to right              alt: w * (dis[src] * dis[dst])
  5711 pad backward (reflect/replicate/circular): (hp, wp) ascending alt: descending
  5712 adaptive average pooling backward: (oh, ow) ascending        alt: descending
  5713 SAGE max backward: a source's targets ascending, ties split    alt: descending
  5714 row L2 normalize: the squares folded in column order          alt: reversed
  5715 Adam/AdamW: every product pinned (no FMA)                      alt: v's update fused
"""
from std.math import fma
from std.memory import bitcast
from checks.numerics import ftz, identical_div, identical_mul, identical_exp, identical_log, identical_rsqrt, identical_sqrt
from core.philox import philox4x32_10
from gemm.host.identical_gemm import gemm_oracle, gemm_oracle_serial, OP_TN


def _nan() -> Float32:
    return bitcast[DType.float32](UInt32(0x7FC00000))


def _inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


# ---------------------------------------------------------------- 5700
def o_col2im(
    dcols: List[Float32], N: Int, C: Int, H: Int, W: Int, KH: Int, KW: Int, SH: Int, SW: Int,
    PH: Int, PW: Int, DH: Int, DW: Int, OH: Int, OW: Int, alt: Bool,
) -> List[Float32]:
    var out = List[Float32](capacity=N * C * H * W)
    var ckk = C * KH * KW
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    var acc = Float32(0)
                    for a in range(KH):
                        var kh = KH - 1 - a if alt else a
                        for b in range(KW):
                            var kw = KW - 1 - b if alt else b
                            var th = h + PH - kh * DH
                            var tw = w + PW - kw * DW
                            if th < 0 or tw < 0 or th % SH != 0 or tw % SW != 0:
                                continue
                            var oh = th // SH
                            var ow = tw // SW
                            if oh >= OH or ow >= OW:
                                continue
                            var r = (n * OH + oh) * OW + ow
                            acc = ftz(acc + ftz(dcols[r * ckk + (c * KH + kh) * KW + kw]))
                    out.append(acc)
    return out^


# ---------------------------------------------------------------- 5701
def o_conv_dw(
    x: List[Float32], dout: List[Float32], N: Int, C: Int, H: Int, W: Int, OC: Int, KH: Int, KW: Int,
    SH: Int, SW: Int, PH: Int, PW: Int, OH: Int, OW: Int, alt: Bool,
) -> List[Float32]:
    """dW (OC x C*KH*KW) = G^T cols over the N*OH*OW rows (dilation 1)."""
    var rows = N * OH * OW
    var ckk = C * KH * KW
    var cols = List[Float32](capacity=rows * ckk)
    var g = List[Float32](capacity=rows * OC)
    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                for c in range(C):
                    for kh in range(KH):
                        for kw in range(KW):
                            var h = oh * SH - PH + kh
                            var w = ow * SW - PW + kw
                            var v = Float32(0)
                            if h >= 0 and h < H and w >= 0 and w < W:
                                v = ftz(x[((n * C + c) * H + h) * W + w])
                            cols.append(v)
                for oc in range(OC):
                    g.append(ftz(dout[((n * OC + oc) * OH + oh) * OW + ow]))
    if alt:
        return gemm_oracle_serial(g, cols, OP_TN, OC, ckk, rows)
    return gemm_oracle(g, cols, OP_TN, OC, ckk, rows)


# ---------------------------------------------------------------- 5702
def o_bn_stats(x: List[Float32], N: Int, C: Int, HW: Int, eps: Float32, alt: Bool) -> List[Float32]:
    """[mean C | biased var C | invstd C]."""
    var mean = List[Float32]()
    var var_b = List[Float32]()
    var inv = List[Float32]()
    var count = Float32(N * HW)
    for c in range(C):
        var acc = Float32(0)
        for a in range(N):
            var n = N - 1 - a if alt else a
            for k in range(HW):
                acc = ftz(acc + ftz(x[(n * C + c) * HW + k]))
        var m = ftz(identical_div(acc, count))
        var sq = Float32(0)
        for a in range(N):
            var n = N - 1 - a if alt else a
            for k in range(HW):
                var d = ftz(ftz(x[(n * C + c) * HW + k]) - m)
                sq = ftz(sq + ftz(identical_mul(d, d)))
        var v = ftz(identical_div(sq, count))
        mean.append(m)
        var_b.append(v)
        inv.append(ftz(identical_rsqrt(ftz(v + eps))))
    mean.extend(var_b^)
    mean.extend(inv^)
    return mean^


# ---------------------------------------------------------------- 5703
def o_dropout_mask(
    N: Int, C: Int, HW: Int, seed_lo: UInt32, seed_hi: UInt32, thresh: UInt64, p: Float32, alt: Bool
) -> List[Float32]:
    var out = List[Float32](capacity=N * C * HW)
    var scale = ftz(identical_div(Float32(1), ftz(Float32(1) - p)))
    for n in range(N):
        for c in range(C):
            var ctr_word = UInt32(c * N + n) if alt else UInt32(n * C + c)
            var u = philox4x32_10(SIMD[DType.uint32, 4](ctr_word, 0, 0, 0), SIMD[DType.uint32, 2](seed_lo, seed_hi))[0]
            var m = scale if UInt64(u) >= thresh else Float32(0)
            for _ in range(HW):
                out.append(m)
    return out^


# ---------------------------------------------------------------- 5704
def o_spmm(
    vals: List[Float32], h: List[Float32], rowptr: List[Int], col: List[Int], n: Int, F: Int, mode: Int, alt: Bool
) -> List[Float32]:
    var out = List[Float32](capacity=n * F)
    for r in range(n):
        var lo = rowptr[r]
        var hi = rowptr[r + 1]
        for f in range(F):
            var acc = Float32(0)
            for t in range(hi - lo):
                var e = hi - 1 - t if alt else lo + t
                var v = ftz(h[col[e] * F + f])
                if mode == 0:
                    v = ftz(identical_mul(ftz(vals[e]), v))
                elif mode == 2:
                    v = ftz(identical_div(v, ftz(vals[e])))
                acc = ftz(acc + v)
            if mode == 1 and hi > lo:
                acc = ftz(identical_div(acc, Float32(hi - lo)))
            out.append(acc)
    return out^


# ---------------------------------------------------------------- 5705, 5708
def o_softmax(logits: List[Float32], labels: List[Int], n: Int, k: Int, raw: Bool, rev: Bool) -> List[Float32]:
    """[grad n*k | proba n*k | row loss n] of mean softmax cross entropy.
    raw: plain arithmetic, no +inf limit and no NaN canon (5705's alt);
    rev: the exp-sum folded from the last column (5708's alt)."""
    var grad = List[Float32](length=n * k, fill=Float32(0))
    var proba = List[Float32](length=n * k, fill=Float32(0))
    var loss = List[Float32](length=n, fill=Float32(0))
    for i in range(n):
        var mx = ftz(logits[i * k])
        for j in range(1, k):
            if ftz(logits[i * k + j]) > mx:
                mx = ftz(logits[i * k + j])
        var y = labels[i]
        if mx == _inf() and not raw:
            var cnt = 0
            for j in range(k):
                if ftz(logits[i * k + j]) == _inf():
                    cnt += 1
            var share = ftz(identical_div(Float32(1), Float32(cnt)))
            for j in range(k):
                var pr = share if ftz(logits[i * k + j]) == _inf() else Float32(0)
                proba[i * k + j] = pr
                var t = ftz(pr - Float32(1)) if j == y else pr
                grad[i * k + j] = ftz(identical_div(t, Float32(n)))
            loss[i] = Float32(0) if ftz(logits[i * k + y]) == _inf() else _inf()
            continue
        var s = Float32(0)
        for a in range(k):
            var j = k - 1 - a if rev else a
            s = ftz(s + ftz(identical_exp(ftz(ftz(logits[i * k + j]) - mx))))
        for j in range(k):
            var pr = ftz(identical_div(ftz(identical_exp(ftz(ftz(logits[i * k + j]) - mx))), s))
            var t = ftz(pr - Float32(1)) if j == y else pr
            var gv = ftz(identical_div(t, Float32(n)))
            if not raw:
                pr = _nan() if pr != pr else pr
                gv = _nan() if gv != gv else gv
            proba[i * k + j] = pr
            grad[i * k + j] = gv
        var lv = ftz(ftz(identical_log(s)) - ftz(ftz(logits[i * k + y]) - mx))
        loss[i] = _nan() if (lv != lv and not raw) else lv
    grad.extend(proba^)
    grad.extend(loss^)
    return grad^


# ---------------------------------------------------------------- 5706
def o_maxpool(x: List[Float32], NC: Int, H: Int, W: Int, K: Int, S: Int, P: Int, OH: Int, OW: Int, alt: Bool) -> List[Float32]:
    """[out | winner index as float] (square window, dilation 1)."""
    var out = List[Float32]()
    var idx = List[Float32]()
    for nc in range(NC):
        for oh in range(OH):
            for ow in range(OW):
                var best = Float32(0)
                var bi = -1
                for kh in range(K):
                    for kw in range(K):
                        var h = oh * S - P + kh
                        var w = ow * S - P + kw
                        if h < 0 or h >= H or w < 0 or w >= W:
                            continue
                        var v = ftz(x[(nc * H + h) * W + w])
                        var take = (v >= best) if alt else (v > best)
                        if bi < 0 or take or v != v:
                            best = v
                            bi = h * W + w
                out.append(best)
                idx.append(Float32(bi))
    out.extend(idx^)
    return out^


# ---------------------------------------------------------------- 5707
def o_avgpool(x: List[Float32], NC: Int, H: Int, W: Int, K: Int, S: Int, P: Int, OH: Int, OW: Int, alt: Bool) -> List[Float32]:
    """count_include_pad=True, floor mode."""
    var out = List[Float32]()
    for nc in range(NC):
        for oh in range(OH):
            for ow in range(OW):
                var hs = oh * S - P
                var ws = ow * S - P
                var div = (min(hs + K, H + P) - hs) * (min(ws + K, W + P) - ws)
                var acc = Float32(0)
                for kh in range(K):
                    for kw in range(K):
                        var h = hs + kh
                        var w = ws + kw
                        if h < 0 or h >= H or w < 0 or w >= W:
                            continue
                        acc = ftz(acc + ftz(x[(nc * H + h) * W + w]))
                if alt:
                    out.append(ftz(identical_mul(acc, ftz(identical_div(Float32(1), Float32(div))))))
                else:
                    out.append(ftz(identical_div(acc, Float32(div))))
    return out^


# ---------------------------------------------------------------- 5709
def o_sgd(w: List[Float32], g: List[Float32], v: List[Float32], lr: Float32, mom: Float32, wd: Float32, alt: Bool) -> List[Float32]:
    """[w' | v']."""
    var nw = List[Float32]()
    var nv = List[Float32]()
    for i in range(len(w)):
        var wi = ftz(w[i])
        var d = ftz(g[i])
        if wd != Float32(0):
            d = ftz(d + ftz(identical_mul(wd, wi)))
        var vi = d
        if mom != Float32(0):
            if alt:
                vi = ftz(fma(mom, ftz(v[i]), d))
            else:
                vi = ftz(ftz(identical_mul(mom, ftz(v[i]))) + d)
        nv.append(vi)
        nw.append(ftz(wi - ftz(identical_mul(lr, vi))))
    nw.extend(nv^)
    return nw^


# ---------------------------------------------------------------- 5710
def o_gcn_norm(w: List[Float32], rowptr: List[Int], col: List[Int], n: Int, alt: Bool) -> List[Float32]:
    var dis = List[Float32]()
    for r in range(n):
        var acc = Float32(0)
        for e in range(rowptr[r], rowptr[r + 1]):
            acc = ftz(acc + ftz(w[e]))
        dis.append(ftz(identical_rsqrt(acc)) if acc > Float32(0) else Float32(0))
    var out = List[Float32]()
    for r in range(n):
        for e in range(rowptr[r], rowptr[r + 1]):
            var s = ftz(dis[col[e]])
            var t = ftz(dis[r])
            if alt:
                out.append(ftz(identical_mul(ftz(w[e]), ftz(identical_mul(s, t)))))
            else:
                out.append(ftz(identical_mul(ftz(identical_mul(s, ftz(w[e]))), t)))
    return out^


# ---------------------------------------------------------------- 5711
def _o_src(hp: Int, before: Int, size: Int, mode: Int) -> Int:
    var h = hp - before
    if h >= 0 and h < size:
        return h
    if mode == 0:
        return -1
    if mode == 1:
        return -h if h < 0 else 2 * (size - 1) - h
    if mode == 2:
        return 0 if h < 0 else size - 1
    return (h % size + size) % size


def o_pad_bwd(g: List[Float32], NC: Int, H: Int, W: Int, t: Int, b: Int, l: Int, r: Int, mode: Int, alt: Bool) -> List[Float32]:
    """dx of a 2-D pad: every padded position's gradient summed onto its
    source pixel in (hp, wp) ascending order; alt: descending."""
    var Hp = H + t + b
    var Wp = W + l + r
    var out = List[Float32](capacity=NC * H * W)
    for nc in range(NC):
        for h in range(H):
            for w in range(W):
                var acc = Float32(0)
                for a in range(Hp):
                    var hp = Hp - 1 - a if alt else a
                    if _o_src(hp, t, H, mode) != h:
                        continue
                    for c in range(Wp):
                        var wp = Wp - 1 - c if alt else c
                        if _o_src(wp, l, W, mode) != w:
                            continue
                        acc = ftz(acc + ftz(g[(nc * Hp + hp) * Wp + wp]))
                out.append(acc)
    return out^


# ---------------------------------------------------------------- 5712
def o_adapt_avg_bwd(g: List[Float32], NC: Int, H: Int, W: Int, OH: Int, OW: Int, alt: Bool) -> List[Float32]:
    """dx of adaptive average pooling: g / window size summed over the
    windows holding each pixel, (oh, ow) ascending; alt: descending."""
    var out = List[Float32](capacity=NC * H * W)
    for nc in range(NC):
        for h in range(H):
            for w in range(W):
                var acc = Float32(0)
                for a in range(OH):
                    var oh = OH - 1 - a if alt else a
                    var hs = (oh * H) // OH
                    var he = ((oh + 1) * H + OH - 1) // OH
                    if h < hs or h >= he:
                        continue
                    for b in range(OW):
                        var ow = OW - 1 - b if alt else b
                        var ws = (ow * W) // OW
                        var we = ((ow + 1) * W + OW - 1) // OW
                        if w < ws or w >= we:
                            continue
                        var gv = ftz(g[(nc * OH + oh) * OW + ow])
                        acc = ftz(acc + ftz(identical_div(gv, Float32((he - hs) * (we - ws)))))
                out.append(acc)
    return out^


# ---------------------------------------------------------------- 5713, 5714
def o_adapt_max_fwd(x: List[Float32], NC: Int, H: Int, W: Int, OH: Int, OW: Int, alt: Bool) -> List[Float32]:
    """[out | winner index as float] of adaptive max pooling (torch's
    windows [floor(o*H/OH), ceil((o+1)*H/OH))), (h, w) ascending, the first
    maximum wins, a NaN wins (DEVIATION 5706's tie); alt: the last maximum."""
    var out = List[Float32]()
    var idx = List[Float32]()
    for nc in range(NC):
        for oh in range(OH):
            var hs = (oh * H) // OH
            var he = ((oh + 1) * H + OH - 1) // OH
            for ow in range(OW):
                var ws = (ow * W) // OW
                var we = ((ow + 1) * W + OW - 1) // OW
                var best = Float32(0)
                var bi = -1
                for h in range(hs, he):
                    for w in range(ws, we):
                        var v = ftz(x[(nc * H + h) * W + w])
                        var take = (v >= best) if alt else (v > best)
                        if bi < 0 or take or v != v:
                            best = v
                            bi = h * W + w
                out.append(best)
                idx.append(Float32(bi))
    out.extend(idx^)
    return out^


def o_sage_max_bwd(
    h: List[Float32], g: List[Float32], rowptr: List[Int], col: List[Int], n: Int, F: Int, alt: Bool
) -> List[Float32]:
    """The SAGE max aggregation's gradient: rowptr/col are the FORWARD
    (target) CSR; the max, its tie count, then for every source the sum over
    its outgoing entries (targets ascending) of g / count where it is a
    maximum. alt: the targets descending."""
    var mx = List[Float32](length=n * F, fill=Float32(0))
    var cnt = List[Float32](length=n * F, fill=Float32(0))
    for t in range(n):
        for f in range(F):
            var first = True
            for e in range(rowptr[t], rowptr[t + 1]):
                var v = ftz(h[col[e] * F + f])
                if first or v > mx[t * F + f]:
                    mx[t * F + f] = v
                    cnt[t * F + f] = Float32(1)
                    first = False
                elif v == mx[t * F + f]:
                    cnt[t * F + f] = cnt[t * F + f] + Float32(1)
    # every (source, target) entry, grouped by source with targets ascending
    var out = List[Float32](capacity=n * F)
    for s in range(n):
        var targets = List[Int]()
        for t in range(n):
            for e in range(rowptr[t], rowptr[t + 1]):
                if col[e] == s:
                    targets.append(t)
        for f in range(F):
            var acc = Float32(0)
            var v = ftz(h[s * F + f])
            for a in range(len(targets)):
                var t = targets[len(targets) - 1 - a] if alt else targets[a]
                var o = t * F + f
                if v == mx[o] and cnt[o] > Float32(0):
                    acc = ftz(acc + ftz(identical_div(ftz(g[o]), cnt[o])))
            out.append(acc)
    return out^


def o_l2norm(x: List[Float32], n: Int, F: Int, alt: Bool) -> List[Float32]:
    """Row L2 normalization, the squares folded in column order (alt: reversed)."""
    var out = List[Float32](capacity=n * F)
    for r in range(n):
        var sq = Float32(0)
        for a in range(F):
            var f = F - 1 - a if alt else a
            var v = ftz(x[r * F + f])
            sq = ftz(sq + ftz(identical_mul(v, v)))
        var norm = ftz(identical_sqrt(sq))
        var den = norm if norm > Float32(1e-12) else Float32(1e-12)
        for f in range(F):
            out.append(ftz(identical_div(ftz(x[r * F + f]), den)))
    return out^


# ---------------------------------------------------------------- 5715
def o_adam(w: List[Float32], g: List[Float32], mv: List[Float32], h: List[Float32], alt: Bool) -> List[Float32]:
    """torch.optim.Adam/AdamW's element step (hyper as adam_at's); alt: the
    second-moment update fused (fma(b2, v, (1-b2) g g))."""
    var n = len(w)
    var nw = List[Float32]()
    var nm = List[Float32]()
    var nv = List[Float32]()
    for i in range(n):
        var wi = ftz(w[i])
        var gi = ftz(g[i])
        if h[6] != Float32(0):
            if h[7] != Float32(0):
                wi = ftz(identical_mul(wi, h[8]))
            else:
                gi = ftz(gi + ftz(identical_mul(h[6], wi)))
        var m = ftz(mv[i])
        var v = ftz(mv[n + i])
        m = ftz(m + ftz(identical_mul(h[1], ftz(gi - m))))
        var sq = ftz(identical_mul(h[3], ftz(identical_mul(gi, gi))))
        if alt:
            v = ftz(fma(h[2], v, sq))
        else:
            v = ftz(ftz(identical_mul(h[2], v)) + sq)
        var denom = ftz(ftz(identical_div(ftz(identical_sqrt(v)), h[5])) + h[4])
        nm.append(m)
        nv.append(v)
        nw.append(ftz(wi - ftz(identical_mul(h[0], ftz(identical_div(m, denom))))))
    nw.extend(nm^)
    nw.extend(nv^)
    return nw^
