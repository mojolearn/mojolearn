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
  5702 BatchNorm statistics: blocked fold per channel over (n, hw)  alt: n descending
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
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz, identical_div, identical_mul, identical_exp, identical_log, identical_rsqrt, identical_sqrt
from core.philox import philox4x32_10
from gemm.contract import OP_TN
from gemm.host.identical_gemm import gemm_oracle, gemm_oracle_serial


def _nan() -> Float32:
    return bitcast[DType.float32](UInt32(0x7FC00000))


def _inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


# ---------------------------------------------------------------- 5700


# ---------------------------------------------------------------- 5701


# ---------------------------------------------------------------- 5702
#: The oracle's own reading of the blocked-fold switch (x_cnn/ops.mojo
#: BN_FOLD_BLOCK; lane idn-loss-norm-folds).
# TOMBSTONE: MOJOLEARN_BN_FOLD_BLOCK_OFF (DELETE-OLD-PATH: no A/B recorded for the old path) deleted 2026-11-15 by lane/expdel2-33-neural; code recoverable at c3d1c7865f.
# Restore: git apply experiments/removed/MOJOLEARN_BN_FOLD_BLOCK_OFF.patch
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (umbrella old-path switch, EXP:2438 DELETED; non-owner sites) deleted 2026-11-15 by lane/expdel2-33-neural; code recoverable at c3d1c7865f.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-33-neural.patch; record in docs/TOMBSTONES.md.
comptime O_BN_FOLD_BLOCK = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def _o_blocked_sum(v: List[Float32]) -> Float32:
    """The channel fold's order (x_cnn/ops.mojo BN_FOLD_BLOCK): consecutive
    blocks of B values (the smallest power of two >= 64 with B * B >= the
    count), each an ascending chain from +0.0, the block sums then added
    ascending from +0.0. One chain when the blocked fold is compiled out."""
    var count = len(v)
    var B = count if count > 0 else 1
    comptime if O_BN_FOLD_BLOCK:
        B = 64
        while B * B < count:
            B *= 2
    var total = Float32(0)
    var parts = List[Float32]()
    var lo = 0
    while lo < count:
        var hi = min(lo + B, count)
        var acc = Float32(0)
        for j in range(lo, hi):
            acc = ftz(acc + v[j])
        parts.append(acc)
        lo = hi
    for b in range(len(parts)):
        total = ftz(total + parts[b])
    return total


# ---------------------------------------------------------------- 5703


# ---------------------------------------------------------------- 5704


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


# ---------------------------------------------------------------- 5707


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


# ---------------------------------------------------------------- 5711


# ---------------------------------------------------------------- 5712


# ---------------------------------------------------------------- 5713, 5714


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
