# SPDX-License-Identifier: Apache-2.0
# THE KERNEL BODIES OF THE CTC LOSS OF SEQUENTIAL (lane nn-losses-2, 2026-11-23).
#
# torch.nn.CTCLoss on log-probabilities (B, T, C) (one sample's (T, C) rows contiguous, the row-major (B, T, C)
# layout of a Sequential output) against padded targets. One `seq_kernel` body per code, shared by HostExec and
# DeviceExec through sequence/dispatch.mojo, so the host column and every GPU run the same words. Codes
# OP_CTC_FIRST..OP_CTC_LAST (5640..5649).
#
# THE TARGET ROWS. A batch's target is (B, 2 + S): [input length, target length, label 0 .. label S-1] per row
# (the fit packs them once per fit with OP_CTC_PACK; autograd/nn_ctc.mojo ctc_loss reads them). Labels past the
# target length are never read. The extended label sequence of a target of length L is blank, l0, blank, l1, ..,
# l(L-1), blank: 2 L + 1 states.
#
# ONE SCAN PER ROW. OP_CTC_FWD and OP_CTC_VJP run one thread per batch row (rows in parallel); each thread scans the
# row's time steps in order, all 2 L + 1 states of a step before the next step, in log space. There is no launch per
# time step: forward is ONE launch (the alpha recursion, the row's negative log-likelihood, the reduction factor),
# backward ONE launch (the beta recursion, two rows of scratch, and the gradient of each step as soon as its beta
# row is known). Order: every log-sum-exp adds its terms in the fixed order (stay, advance one, skip one), with
# max + log1p(exp(-|a - b|)) pairwise, so the words are one order on the host and on every vendor.
#
# THE GRADIENT is the derivative in the log-probabilities as independent inputs:
#   d nll / d lp[t, c] = -(1 / p) sum_{s : l'_s = c} alpha_t(s) beta_t(s) / y_t(c) = -sum_s exp(a + b - lp + nll)
# (alpha and beta both carry step t's emission, Graves 2006). It is computed in the NORMALIZED form (2026-11-25,
# ctc_row_fwd / ctc_row_vjp): the scans read lp - M_t (the step's largest log-probability) and normalize each step's
# states by their log-sum, and the step's gradient is the posterior exp(q_s - log sum q) of its states, so neither
# the likelihood nor a common offset of the log-probabilities enters a word. PyTorch's ctc_loss backward returns this plus
# exp(lp[t, c]) (the gradient it would have through a log_softmax); passed back through log_softmax both give the
# same logits gradient (softmax - q), because q sums to 1 over the classes of each valid step. Steps at or after
# the input length take gradient 0. A row whose alignment is impossible (nll = +inf) takes gradient 0.
from std.memory import bitcast
from std.sys.compile import is_defined
from sequence.ops import Args, FP, ld, st, add, sub, mul
from checks.numerics import ftz, identical_div, identical_exp, identical_log1p

comptime OP_CTC_FWD = 5640
comptime OP_CTC_VJP = 5641
comptime OP_CTC_PACK = 5642
comptime OP_CTC_ADMIT = 5643
comptime OP_CTC_FIRST = 5640
comptime OP_CTC_LAST = 5649

# Fault injection (verify's sabotage arm of the CTC family, never in a product build): the skip transition is
# allowed between two equal labels, so "a a" collapses as "a". checks/native_tests/nn_losses_2.mojo's
# brute-force oracle must FAIL under it (its targets hold a repeated label).
comptime NN_CTC_SABOTAGE_REPEAT = is_defined["MOJOLEARN_NN_LOSSES_SABOTAGE_CTC_REPEAT"]()

# The target row's words (ctc_loss's (B, 2 + S) target).
comptime CTC_IN_LEN = 0
comptime CTC_TGT_LEN = 1
comptime CTC_LABELS = 2


@always_inline
def ctc_neg_inf() -> Float32:
    """-inf, log(0): the log-probability of an unreachable state."""
    return bitcast[DType.float32](UInt32(0xFF800000))


@always_inline
def ctc_lse2(a: Float32, b: Float32) -> Float32:
    """log(exp(a) + exp(b)) = max + log1p(exp(-|a - b|)); -inf terms drop out (exact)."""
    var ni = ctc_neg_inf()
    if a == ni:
        return b
    if b == ni:
        return a
    var m = a if a > b else b
    var d = sub(a, b)
    var ad = d if d >= Float32(0) else -d
    return add(m, ftz(identical_log1p(ftz(identical_exp(-ad)))))


@always_inline
def ctc_label(tg: FP, tb: Int, s: Int, blank: Int) -> Int:
    """The class of extended state s of the target row at tb: blank at even s, label (s - 1) / 2 at odd s."""
    if s % 2 == 0:
        return blank
    return Int(ld(tg, tb + CTC_LABELS + (s - 1) // 2))


@always_inline
def ctc_skip(tg: FP, tb: Int, s: Int, blank: Int) -> Bool:
    """Does state s (s >= 2) take the skip transition from s - 2? Only a label different from the label two
    states back (the blank between two equal labels cannot be skipped)."""
    if s < 2 or s % 2 == 0:
        return False
    comptime if NN_CTC_SABOTAGE_REPEAT:
        return True
    return ctc_label(tg, tb, s, blank) != ctc_label(tg, tb, s - 2, blank)


@always_inline
def ctc_factor(L: Int, reduction: Int) -> Float32:
    """PyTorch's per-row factor: 'mean' divides each row's loss by its target length (clamped to 1) before the
    batch mean; 'sum' adds the rows as they are (reduction 1)."""
    if reduction == 1:
        return Float32(1)
    var n = L if L > 0 else 1
    return ftz(identical_div(Float32(1), Float32(n)))


@always_inline
def ctc_step_shift(lp: FP, base: Int, C: Int, raw: Int) -> Float32:
    """The step's shift M_t: the largest log-probability of the step's C classes (ascending fold), +0.0 when every
    class is -inf (the step then makes every state unreachable on its own) or in the raw form (i8 = 1). Every
    recursion reads lp - M_t: for log-probabilities that all carry a common offset k the subtraction is exact
    (both words share an exponent range), so the scans see the same words with and without the offset."""
    if raw != 0:
        return Float32(0)
    var ni = ctc_neg_inf()
    var m = ld(lp, base)
    for c in range(1, C):
        var v = ld(lp, base + c)
        if v > m:
            m = v
    if m == ni:
        return Float32(0)
    return m


@always_inline
def ctc_states_lse(p: FP, base: Int, n: Int) -> Float32:
    """log(sum_s exp(p[base + s])) over the n states, folded pairwise in ascending state order (-inf: none
    reachable)."""
    var acc = ctc_neg_inf()
    for s in range(n):
        acc = ctc_lse2(acc, ld(p, base + s))
    return acc


@always_inline
def ctc_states_shift(p: FP, base: Int, n: Int, z: Float32):
    """Subtract the step's normalizer z from each reachable state (-inf states stay -inf)."""
    var ni = ctc_neg_inf()
    for s in range(n):
        var v = ld(p, base + s)
        if v != ni:
            st(p, base + s, sub(v, z))


@always_inline
def ctc_row_fwd(b: Int, a: Args):
    """OP_CTC_FWD, thread b = batch row b. p0 log-probabilities (B, T, C), p1 target rows (B, i2), p2 the row's
    loss value (B), p3 alpha scratch (B, T, 2 S + 1), p4 the row's raw negative log-likelihood (B); i0 T, i1 C,
    i2 the target row width 2 + S, i3 blank, i4 reduction (0 mean, 1 sum), i5 zero_infinity, i8 the raw form (the
    sabotage arm's unnormalized recursion; 0 in the product).

    THE NORMALIZED FORM (lane stream-fits, 2026-11-25). Step t reads the shifted log-probabilities lp - M_t
    (ctc_step_shift) and stores its alphas normalized: each step's reachable states minus their log-sum A_t, so
    every stored alpha is at most 0 and of the magnitude of one step's terms, never of the whole row's likelihood.
    The row's log-likelihood is the running sum of (M_t + A_t) in ascending t plus the log-sum of the last two
    states. The backward (ctc_row_vjp) reads these alphas; its occupations never meet the likelihood (before
    2026-11-25 it took exp(alpha + beta - lp + nll) with alpha, beta and nll at the row's magnitude, so a long
    sequence or a large common offset moved its words by ulp(nll))."""
    var T = a.i0
    var C = a.i1
    var TW = a.i2
    var blank = a.i3
    var raw = a.i8
    var SP = 2 * (TW - CTC_LABELS) + 1
    var tb = b * TW
    var tin = Int(ld(a.p1, tb + CTC_IN_LEN))
    var L = Int(ld(a.p1, tb + CTC_TGT_LEN))
    var n = 2 * L + 1
    var lb = b * T * C
    var ab = b * T * SP
    var ni = ctc_neg_inf()
    if tin < 1 or tin > T or L < 0 or n > SP:
        # no alignment (the fit's admission refuses these; a tape caller's malformed row gives +inf)
        st(a.p4, b, -ni)
        st(a.p2, b, Float32(0) if a.i5 != 0 else -ni)
        return
    for s in range(n):
        st(a.p3, ab + s, ni)
    var m = ctc_step_shift(a.p0, lb, C, raw)
    st(a.p3, ab, sub(ld(a.p0, lb + blank), m))
    if L > 0:
        st(a.p3, ab + 1, sub(ld(a.p0, lb + ctc_label(a.p1, tb, 1, blank)), m))
    var logp = Float32(0)
    var reachable = True
    var z = Float32(0)
    if raw == 0:
        z = ctc_states_lse(a.p3, ab, n)
        if z == ni:
            reachable = False
        else:
            ctc_states_shift(a.p3, ab, n, z)
    logp = add(m, z)
    var t = 1
    while reachable and t < tin:
        var prev = ab + (t - 1) * SP
        var row = ab + t * SP
        m = ctc_step_shift(a.p0, lb + t * C, C, raw)
        for s in range(n):
            var v = ld(a.p3, prev + s)
            if s >= 1:
                v = ctc_lse2(v, ld(a.p3, prev + s - 1))
            if ctc_skip(a.p1, tb, s, blank):
                v = ctc_lse2(v, ld(a.p3, prev + s - 2))
            if v == ni:
                st(a.p3, row + s, ni)
            else:
                st(a.p3, row + s, add(sub(ld(a.p0, lb + t * C + ctc_label(a.p1, tb, s, blank)), m), v))
        z = Float32(0)
        if raw == 0:
            z = ctc_states_lse(a.p3, row, n)
            if z == ni:
                reachable = False
            else:
                ctc_states_shift(a.p3, row, n, z)
        logp = add(logp, add(m, z))
        t += 1
    var nll = -ni
    if reachable:
        var last = ab + (tin - 1) * SP
        var ll = ld(a.p3, last + n - 1)
        if L > 0:
            ll = ctc_lse2(ll, ld(a.p3, last + n - 2))
        if ll != ni:
            nll = -add(logp, ll)
    st(a.p4, b, nll)
    var value = nll
    if a.i5 != 0 and nll == -ni:
        value = Float32(0)
    st(a.p2, b, mul(value, ctc_factor(L, a.i4)))


@always_inline
def ctc_row_vjp(b: Int, a: Args):
    """OP_CTC_VJP, thread b = batch row b: the gradient of the row's loss value in the log-probabilities into p6
    (B, T, C), every word of the row written (0 where no state of a step carries the class, and at every step from
    the input length on). p0 .. p4 and i0 .. i5, i8 as OP_CTC_FWD (p3, p4 its outputs), p5 the upstream gradient
    of the row values (B), p7 beta scratch (B, 2, 2 S + 1).

    Normalized form: the betas read lp - M_t and are normalized per step as the alphas are; a step's occupations
    are q_s = a_s + b_s - (lp_t(c_s) - M_t) over its states with both finite, and the posterior of state s is
    exp(q_s - Q_t), Q_t their log-sum (ascending pairwise fold). The posteriors of a step add up to one, as the
    exact ones do (alpha_t(s) beta_t(s) / y_t(c_s) summed over s is the likelihood at every t), so nothing of the
    row's magnitude enters a word. The raw form (i8 = 1, the sabotage arm) is the pre-2026-11-25 formula
    exp(alpha + beta - lp + nll) on unshifted, unnormalized scans."""
    var T = a.i0
    var C = a.i1
    var TW = a.i2
    var blank = a.i3
    var raw = a.i8
    var SP = 2 * (TW - CTC_LABELS) + 1
    var tb = b * TW
    var tin = Int(ld(a.p1, tb + CTC_IN_LEN))
    var L = Int(ld(a.p1, tb + CTC_TGT_LEN))
    var n = 2 * L + 1
    var lb = b * T * C
    var ab = b * T * SP
    var bb = b * 2 * SP
    var ni = ctc_neg_inf()
    for k in range(T * C):
        st(a.p6, lb + k, Float32(0))
    var nll = ld(a.p4, b)
    if nll == -ni:
        return
    var coef = -mul(ld(a.p5, b), ctc_factor(L, a.i4))
    var t = tin - 1
    while t >= 0:
        var cur = bb + (t % 2) * SP
        var nxt = bb + ((t + 1) % 2) * SP
        var row = lb + t * C
        var m = ctc_step_shift(a.p0, row, C, raw)
        for s in range(n):
            var v = ni
            if t == tin - 1:
                if s == n - 1 or (L > 0 and s == n - 2):
                    v = sub(ld(a.p0, row + ctc_label(a.p1, tb, s, blank)), m)
            else:
                var nx = ld(a.p7, nxt + s)
                if s + 1 < n:
                    nx = ctc_lse2(nx, ld(a.p7, nxt + s + 1))
                if s + 2 < n and ctc_skip(a.p1, tb, s + 2, blank):
                    nx = ctc_lse2(nx, ld(a.p7, nxt + s + 2))
                if nx != ni:
                    v = add(sub(ld(a.p0, row + ctc_label(a.p1, tb, s, blank)), m), nx)
            st(a.p7, cur + s, v)
        # the step's normalizers: the betas' log-sum, then the occupations' log-sum Q (raw form: Q = -nll)
        var q = -nll
        if raw == 0:
            var zb = ctc_states_lse(a.p7, cur, n)
            if zb != ni:
                ctc_states_shift(a.p7, cur, n, zb)
            q = ni
            for s in range(n):
                var al = ld(a.p3, ab + t * SP + s)
                var be = ld(a.p7, cur + s)
                if al == ni or be == ni:
                    continue
                var c = ctc_label(a.p1, tb, s, blank)
                q = ctc_lse2(q, sub(add(al, be), sub(ld(a.p0, row + c), m)))
        # the step's gradient: the posterior of each state, added to its class in ascending state order
        if q != ni:
            for s in range(n):
                var al = ld(a.p3, ab + t * SP + s)
                var be = ld(a.p7, cur + s)
                if al == ni or be == ni:
                    continue
                var c = ctc_label(a.p1, tb, s, blank)
                var e = ftz(identical_exp(sub(sub(add(al, be), sub(ld(a.p0, row + c), m)), q)))
                st(a.p6, row + c, add(ld(a.p6, row + c), e))
        for c in range(C):
            st(a.p6, row + c, mul(coef, ld(a.p6, row + c)))
        t -= 1


@always_inline
def ctc_pack(w: Int, a: Args):
    """OP_CTC_PACK, thread w = word w of the (N, 2 + S) target rows in p3: column 0 the input length (p2[r] when
    i2 = 1, else the constant i1), column 1 the target length p1[r], columns 2.. the labels p0 (N, S); i0 = 2 + S."""
    var TW = a.i0
    var r = w // TW
    var k = w % TW
    if k == CTC_IN_LEN:
        st(a.p3, w, ld(a.p2, r) if a.i2 == 1 else Float32(a.i1))
    elif k == CTC_TGT_LEN:
        st(a.p3, w, ld(a.p1, r))
    else:
        st(a.p3, w, ld(a.p0, r * (TW - CTC_LABELS) + k - CTC_LABELS))


@always_inline
def ctc_admit_word(w: Int, a: Args):
    """OP_CTC_ADMIT, thread w = word w of the (N, 2 + S) target rows in p0: p1[w] holds int32 0 when the word may
    be read by the row scans, else 1. i0 = 2 + S, i1 = T, i2 = C, i3 = blank. Column 0 (the input length) is a
    whole number in [0, T]; column 1 (the target length) a whole number in [0, S]; a label before its row's
    target length is a whole class index in [0, C) other than blank; a label at or past it is never read by the
    scans, so it is admitted whatever it holds. Each range test comes before the float-to-integer conversion, so
    no NaN or out-of-range value is converted. The words are in the host admission's order (row, input length,
    target length, labels), so the first flagged word is the word the host loop refused."""
    var TW = a.i0
    var S = TW - CTC_LABELS
    var k = w % TW
    var v = ld(a.p0, w)
    var bad = 0
    if k == CTC_IN_LEN:
        bad = 1
        if v >= Float32(0) and v <= Float32(a.i1):
            if Float32(Int(v)) == v:
                bad = 0
    elif k == CTC_TGT_LEN:
        bad = 1
        if v >= Float32(0) and v <= Float32(S):
            if Float32(Int(v)) == v:
                bad = 0
    else:
        var tl = ld(a.p0, w - k + CTC_TGT_LEN)
        if tl >= Float32(0) and tl <= Float32(S) and Float32(k - CTC_LABELS) < tl:
            bad = 1
            if v >= Float32(0) and v < Float32(a.i2):
                if Float32(Int(v)) == v and Int(v) != a.i3:
                    bad = 0
    # The flag is an int32 bit pattern: a raw store. `st` flushes subnormals, and int32 1 is a float32
    # subnormal, so every flag read back 0 and nothing was ever refused (w3-integrate-10).
    a.p1.unsafe_store(w, bitcast[DType.float32](Int32(bad)))


@always_inline
def apply_nn_ctc[OP: Int](t: Int, a: Args):
    comptime if OP == OP_CTC_FWD:
        ctc_row_fwd(t, a)
    elif OP == OP_CTC_VJP:
        ctc_row_vjp(t, a)
    elif OP == OP_CTC_PACK:
        ctc_pack(t, a)
    elif OP == OP_CTC_ADMIT:
        ctc_admit_word(t, a)
