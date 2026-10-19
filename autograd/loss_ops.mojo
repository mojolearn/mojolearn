# SPDX-License-Identifier: Apache-2.0
# Element bodies of the loss and metric family, shared by HostExec and
# DeviceExec (sequence/dispatch.mojo routes OP_LS_FIRST..OP_LS_LAST here).
#
# Every reduction in this family is the tape's blocked fold
# (autograd/fold.mojo, OP_AG_REDUCE_PART / OP_AG_REDUCE_FOLD); these bodies
# only compute per-element or per-row terms, their VJPs, the AUC sort and
# rank counts, and one-word epilogues. Each body reads and writes only the
# slots listed in its docstring, never reads back a slot it writes, and
# loads through branches, never through a pointer select (Metal rule,
# autograd/ops.mojo _ld_right).
#
# Class codes and binary labels travel as float values (Float32(code)),
# exact below 2^24. The AUC path keeps its counts as float values too, so
# it refuses n >= 2^24 on the host (autograd/metrics.mojo).
from std.math import nan
from sequence.ops import Args, FP, ld, st, add, sub, mul
from checks.numerics import (
    ftz, identical_div, identical_exp, identical_log, identical_log1p, identical_sigmoid,
    identical_sqrt,
)

# Codes 1000..1010 are the autograd ops (autograd/ops.mojo); this family is
# the contiguous range below, asserted free in sequence/dispatch.mojo.
comptime OP_LS_ELEM_FWD = 1200
comptime OP_LS_ELEM_VJP = 1201
comptime OP_LS_CE_FWD = 1202
comptime OP_LS_CE_VJP = 1203
comptime OP_LS_METRIC_TERM = 1204
comptime OP_LS_PACK = 1205
comptime OP_LS_BITONIC = 1206
comptime OP_LS_SCAN_PART = 1207
comptime OP_LS_SCAN_OFFS = 1208
comptime OP_LS_SCAN_FILL = 1209
comptime OP_LS_AUC_TERM = 1210
comptime OP_LS_FINISH = 1211
comptime OP_LS_FIRST = OP_LS_ELEM_FWD
comptime OP_LS_LAST = OP_LS_FINISH

# Elementwise loss kinds (a.i0 of OP_LS_ELEM_FWD / OP_LS_ELEM_VJP).
comptime LS_MSE = 1
comptime LS_MAE = 2
comptime LS_HUBER = 3
comptime LS_BCE_LOGITS = 4
# Forward only: the clipped-probability log-loss terms of the MLP trainer
# (sequence/mlp.mojo op_mlp_rowloss, one term per element instead of one
# chain per row). No VJP: the trainer seeds its own combined derivative.
comptime LS_LOG_PROB = 5
comptime LS_BINARY_LOG_PROB = 6
# Row loss from logits and integer codes (OP_LS_CE_FWD / OP_LS_CE_VJP).
comptime LS_CROSS_ENTROPY = 7

# Metric term kinds (a.i0 of OP_LS_METRIC_TERM).
comptime MT_EQUAL = 1
comptime MT_ARGMAX_EQUAL = 2
comptime MT_LOG_LOSS = 3
comptime MT_SQUARED = 4
comptime MT_ABSOLUTE = 5
comptime MT_CENTERED = 6

# One-word epilogues (a.i0 of OP_LS_FINISH).
comptime FIN_SQRT = 1
comptime FIN_R2 = 2
comptime FIN_AUC = 3

# Label value of a padding slot in the AUC sort (real labels are 0 or 1).
comptime AUC_PAD = Float32(2)


@always_inline
def _div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def _abs(x: Float32) -> Float32:
    return x if x >= 0 else -x


@always_inline
def _sign(x: Float32) -> Float32:
    if x > 0:
        return Float32(1)
    if x < 0:
        return Float32(-1)
    return Float32(0)


@always_inline
def clip_probability(p: Float32) -> Float32:
    """Float32 epsilon clipping, the bounds of metrics/impl/log_loss.mojo and
    sequence/mlp.mojo _clip: [2^-23, 1 - 2^-23]."""
    comptime EPS: Float32 = 1.1920928955078125e-07
    if p < EPS:
        return EPS
    if p > Float32(1.0) - EPS:
        return Float32(1.0) - EPS
    return p


@always_inline
def loss_term(kind: Int, p: Float32, y: Float32, delta: Float32) -> Float32:
    """The per-element loss of prediction p against target y."""
    if kind == LS_MSE:
        # sub then mul: the bits of the tape's subtract + multiply (mse_loss)
        var d = sub(p, y)
        return mul(d, d)
    if kind == LS_MAE:
        return _abs(sub(p, y))
    if kind == LS_HUBER:
        # torch.nn.HuberLoss: 0.5 d^2 if |d| < delta, else delta (|d| - delta / 2)
        var d = sub(p, y)
        var ad = _abs(d)
        if ad < delta:
            return mul(Float32(0.5), mul(d, d))
        return mul(delta, sub(ad, mul(Float32(0.5), delta)))
    if kind == LS_BCE_LOGITS:
        # p is the logit: max(p, 0) - p y + log1p(exp(-|p|)), torch's stable form
        var x = p
        var pos = x if x > 0 else Float32(0)
        var soft = ftz(identical_log1p(ftz(identical_exp(-_abs(x)))))
        return add(sub(pos, mul(x, y)), soft)
    if kind == LS_LOG_PROB:
        if y == Float32(0):
            return Float32(0)
        return -mul(y, ftz(identical_log(clip_probability(p))))
    # LS_BINARY_LOG_PROB
    var pc = clip_probability(p)
    var acc = Float32(0)
    if y != Float32(0):
        acc = mul(y, ftz(identical_log(pc)))
    var ny = sub(Float32(1), y)
    if ny != Float32(0):
        acc = add(acc, mul(ny, ftz(identical_log(sub(Float32(1), pc)))))
    return -acc


@always_inline
def loss_term_dp(kind: Int, p: Float32, y: Float32, delta: Float32, g: Float32) -> Float32:
    """g * d loss_term / d p (LS_MSE .. LS_BCE_LOGITS)."""
    if kind == LS_MSE:
        # 2 (g d): the bits of the tape's multiply VJP added to itself
        return mul(Float32(2), mul(g, sub(p, y)))
    if kind == LS_MAE:
        return mul(g, _sign(sub(p, y)))
    if kind == LS_HUBER:
        var d = sub(p, y)
        if _abs(d) < delta:
            return mul(g, d)
        return mul(g, mul(delta, _sign(d)))
    # LS_BCE_LOGITS
    return mul(g, sub(ftz(identical_sigmoid(p)), y))


@always_inline
def loss_term_dy(kind: Int, p: Float32, y: Float32, delta: Float32, g: Float32) -> Float32:
    """g * d loss_term / d y (LS_MSE .. LS_BCE_LOGITS)."""
    if kind == LS_BCE_LOGITS:
        return mul(g, -p)
    return -loss_term_dp(kind, p, y, delta, g)


@always_inline
def op_ls_elem_fwd(t: Int, a: Args):
    """p0 prediction, p1 target, p2 terms. i0 kind, f0 delta (Huber).
    Thread t = element t."""
    st(a.p2, t, loss_term(a.i0, ld(a.p0, t), ld(a.p1, t), a.f0))


@always_inline
def op_ls_elem_vjp(t: Int, a: Args):
    """p0 prediction, p1 target, p2 upstream (the terms' gradient), p3
    destination. i0 kind, i8 operand (0 prediction, 1 target), f0 delta.
    Thread t = element t."""
    var p = ld(a.p0, t)
    var y = ld(a.p1, t)
    var g = ld(a.p2, t)
    if a.i8 == 0:
        st(a.p3, t, loss_term_dp(a.i0, p, y, a.f0, g))
    else:
        st(a.p3, t, loss_term_dy(a.i0, p, y, a.f0, g))


@always_inline
def _row_max(p: FP, base: Int, cols: Int) -> Float32:
    var m = ld(p, base)
    for c in range(1, cols):
        var v = ld(p, base + c)
        if v > m:
            m = v
    return m


@always_inline
def _row_expsum(p: FP, base: Int, cols: Int, m: Float32) -> Float32:
    var s = Float32(0)
    for c in range(cols):
        s = add(s, ftz(identical_exp(sub(ld(p, base + c), m))))
    return s


@always_inline
def op_ls_ce_fwd(t: Int, a: Args):
    """Softmax cross-entropy of row t from logits: p0 logits (rows x i1),
    p1 codes (rows, float values), p2 row loss. log(sum exp(z - m)) -
    (z_y - m), the max subtracted first: the row loss of sequence/ops.mojo
    op_ce, term for term. A code outside [0, i1) gives NaN."""
    var cols = a.i1
    var base = t * cols
    var y = Int(ld(a.p1, t))
    if y < 0 or y >= cols:
        a.p2.unsafe_store(t, nan[DType.float32]())
        return
    var m = _row_max(a.p0, base, cols)
    var ls = ftz(identical_log(_row_expsum(a.p0, base, cols, m)))
    st(a.p2, t, sub(ls, sub(ld(a.p0, base + y), m)))


@always_inline
def op_ls_ce_vjp(t: Int, a: Args):
    """Row t of the cross-entropy VJP: p0 logits (rows x i1), p1 codes, p2
    upstream (rows), p3 destination (rows x i1) = (softmax - onehot) * g:
    op_ce's gradient expression with its scale replaced by the upstream."""
    var cols = a.i1
    var base = t * cols
    var y = Int(ld(a.p1, t))
    var g = ld(a.p2, t)
    var m = _row_max(a.p0, base, cols)
    var s = _row_expsum(a.p0, base, cols, m)
    for c in range(cols):
        var p = _div(ftz(identical_exp(sub(ld(a.p0, base + c), m))), s)
        if c == y:
            p = sub(p, Float32(1))
        st(a.p3, base + c, mul(p, g))


@always_inline
def op_ls_metric_term(t: Int, a: Args):
    """Per-element or per-row metric terms into p2[t]. i0 kind:
      MT_EQUAL        element t: 1 if code p0[t] equals code p1[t], else 0.
      MT_ARGMAX_EQUAL row t: 1 if p0[t] is the first maximum of row t of
                      p1 (rows x i1 scores), else 0.
      MT_LOG_LOSS     row t: -log(clip(probability of class p0[t])), p1 the
                      rows x i1 probabilities; i1 = 1 holds the positive
                      class only (1 - p for a negative row, then clipped).
      MT_SQUARED      element t: (p0[t] - p1[t])^2.
      MT_ABSOLUTE     element t: |p0[t] - p1[t]|.
      MT_CENTERED     element t: (p0[t] - p1[0])^2, p1 one word (the mean)."""
    var kind = a.i0
    if kind == MT_EQUAL:
        var same = Int(ld(a.p0, t)) == Int(ld(a.p1, t))
        st(a.p2, t, Float32(1) if same else Float32(0))
    elif kind == MT_ARGMAX_EQUAL:
        var cols = a.i1
        var base = t * cols
        var best = 0
        var m = ld(a.p1, base)
        for c in range(1, cols):
            var v = ld(a.p1, base + c)
            if v > m:
                m = v
                best = c
        st(a.p2, t, Float32(1) if best == Int(ld(a.p0, t)) else Float32(0))
    elif kind == MT_LOG_LOSS:
        var cols = a.i1
        var y = Int(ld(a.p0, t))
        var q: Float32
        if cols == 1:
            var p = ld(a.p1, t)
            q = p if y == 1 else sub(Float32(1), p)
        else:
            if y < 0 or y >= cols:
                a.p2.unsafe_store(t, nan[DType.float32]())
                return
            q = ld(a.p1, t * cols + y)
        st(a.p2, t, -ftz(identical_log(clip_probability(q))))
    elif kind == MT_SQUARED:
        var d = sub(ld(a.p0, t), ld(a.p1, t))
        st(a.p2, t, mul(d, d))
    elif kind == MT_ABSOLUTE:
        st(a.p2, t, _abs(sub(ld(a.p0, t), ld(a.p1, t))))
    else:
        var d = sub(ld(a.p0, t), ld(a.p1, 0))
        st(a.p2, t, mul(d, d))


@always_inline
def op_ls_pack(t: Int, a: Args):
    """AUC sort input, thread t < padded length: p0 scores, p1 labels (n,
    float 0/1), p2 keys, p3 sort labels; i0 n. Slots t >= n are padding
    (label AUC_PAD), ordered after every real row by the sort."""
    if t < a.i0:
        st(a.p2, t, ld(a.p0, t))
        st(a.p3, t, Float32(1) if ld(a.p1, t) == Float32(1) else Float32(0))
    else:
        st(a.p2, t, Float32(0))
        st(a.p3, t, AUC_PAD)


@always_inline
def _after(ka: Float32, la: Float32, kb: Float32, lb: Float32) -> Bool:
    """(ka, la) sorts after (kb, lb): padding last, then by score."""
    if la == AUC_PAD:
        return lb != AUC_PAD
    if lb == AUC_PAD:
        return False
    return ka > kb


@always_inline
def op_ls_bitonic(t: Int, a: Args):
    """One compare-exchange stage of a bitonic sort network over the padded
    power-of-two length: p0 keys, p1 labels (exchanged together), i0 the
    merge size k, i1 the partner distance j. Thread t owns the pair
    (i, i + j), i = 2 j (t // j) + t % j; ascending when i & k == 0. The
    pairs of a stage are disjoint, so the network is the same on every
    executor and the result is independent of thread order."""
    var j = a.i1
    var i = 2 * j * (t // j) + t % j
    var l = i + j
    var ka = ld(a.p0, i)
    var la = ld(a.p1, i)
    var kb = ld(a.p0, l)
    var lb = ld(a.p1, l)
    var swap: Bool
    if (i & a.i0) == 0:
        swap = _after(ka, la, kb, lb)
    else:
        swap = _after(kb, lb, ka, la)
    if swap:
        st(a.p0, i, kb)
        st(a.p1, i, lb)
        st(a.p0, l, ka)
        st(a.p1, l, la)


@always_inline
def op_ls_scan_part(t: Int, a: Args):
    """Block t of i1 rows of the n = i0 sorted labels p0: p1[t] = the count
    of positive labels in the block (an exact integer)."""
    var lo = t * a.i1
    var hi = min(lo + a.i1, a.i0)
    var c = Float32(0)
    for k in range(lo, hi):
        if ld(a.p0, k) == Float32(1):
            c = add(c, Float32(1))
    st(a.p1, t, c)


@always_inline
def op_ls_scan_offs(t: Int, a: Args):
    """p1[t] = the block counts p0[0:t] added ascending (exclusive offset of
    block t). Thread t = block t; i0 blocks (unused by the body)."""
    var s = Float32(0)
    for b in range(t):
        s = add(s, ld(a.p0, b))
    st(a.p1, t, s)


@always_inline
def op_ls_scan_fill(t: Int, a: Args):
    """Exclusive positive counts of the sorted labels: p2[k] = positives in
    p0[0:k] for k in block t (i1 rows a block, n = i0, i2 blocks), from the
    block's offset p1[t]; the last block also writes p2[n] (all positives)."""
    var lo = t * a.i1
    var hi = min(lo + a.i1, a.i0)
    var run = ld(a.p1, t)
    for k in range(lo, hi):
        st(a.p2, k, run)
        if ld(a.p0, k) == Float32(1):
            run = add(run, Float32(1))
    if t == a.i2 - 1:
        st(a.p2, a.i0, run)


@always_inline
def _lower_bound(keys: FP, n: Int, v: Float32) -> Int:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if ld(keys, mid) < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@always_inline
def _upper_bound(keys: FP, n: Int, v: Float32) -> Int:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = (lo + hi) // 2
        if ld(keys, mid) <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@always_inline
def op_ls_auc_term(t: Int, a: Args):
    """Row t of the sorted (p0 keys, p1 labels), n = i0, p2 the exclusive
    positive counts (n + 1 words), p3 terms. A positive row in the tie
    group [i, j) of its score counts 2 x (negatives below i) + (negatives
    in the group), the tie-aware pair count of metrics/impl/binary_ranking
    .mojo; the term is that count over 2 x negatives. Negative rows give 0.
    The group bounds are binary searches, so ties cost no serial walk."""
    var n = a.i0
    if ld(a.p1, t) != Float32(1):
        st(a.p3, t, Float32(0))
        return
    var v = ld(a.p0, t)
    var i = _lower_bound(a.p0, n, v)
    var j = _upper_bound(a.p0, n, v)
    var pi = Int(ld(a.p2, i))
    var pj = Int(ld(a.p2, j))
    var negatives = n - Int(ld(a.p2, n))
    if negatives == 0:
        st(a.p3, t, Float32(0))
        return
    var c = 2 * (i - pi) + (j - i) - (pj - pi)
    st(a.p3, t, _div(Float32(c), Float32(2 * negatives)))


@always_inline
def op_ls_finish(t: Int, a: Args):
    """One-word epilogue into p2[0]. i0 mode:
      FIN_SQRT  sqrt(p0[0]).
      FIN_R2    1 - p0[0] / p1[0] (residual over total sum of squares); a
                constant target gives 1 for a perfect fit, else 0.
      FIN_AUC   p0[0] (the folded terms) / positives, positives = p1[i1]
                of the n = i1 rows; -1 when a class is missing."""
    var mode = a.i0
    if mode == FIN_SQRT:
        st(a.p2, 0, ftz(identical_sqrt(ld(a.p0, 0))))
    elif mode == FIN_R2:
        var ssr = ld(a.p0, 0)
        var sst = ld(a.p1, 0)
        if sst == Float32(0):
            st(a.p2, 0, Float32(1) if ssr == Float32(0) else Float32(0))
        else:
            st(a.p2, 0, sub(Float32(1), _div(ssr, sst)))
    else:
        var n = a.i1
        var positives = Int(ld(a.p1, n))
        if positives == 0 or positives == n:
            st(a.p2, 0, Float32(-1))
        else:
            st(a.p2, 0, _div(ld(a.p0, 0), Float32(positives)))


@always_inline
def apply_losses[OP: Int](t: Int, a: Args):
    comptime if OP == OP_LS_ELEM_FWD:
        op_ls_elem_fwd(t, a)
    elif OP == OP_LS_ELEM_VJP:
        op_ls_elem_vjp(t, a)
    elif OP == OP_LS_CE_FWD:
        op_ls_ce_fwd(t, a)
    elif OP == OP_LS_CE_VJP:
        op_ls_ce_vjp(t, a)
    elif OP == OP_LS_METRIC_TERM:
        op_ls_metric_term(t, a)
    elif OP == OP_LS_PACK:
        op_ls_pack(t, a)
    elif OP == OP_LS_BITONIC:
        op_ls_bitonic(t, a)
    elif OP == OP_LS_SCAN_PART:
        op_ls_scan_part(t, a)
    elif OP == OP_LS_SCAN_OFFS:
        op_ls_scan_offs(t, a)
    elif OP == OP_LS_SCAN_FILL:
        op_ls_scan_fill(t, a)
    elif OP == OP_LS_AUC_TERM:
        op_ls_auc_term(t, a)
    elif OP == OP_LS_FINISH:
        op_ls_finish(t, a)
