# SPDX-License-Identifier: Apache-2.0
# Lane nn-losses (2026-11-23): the element bodies of the losses added for
# `Sequential.compile(loss=...)`: smooth L1 (beta), negative log-likelihood
# on log-probabilities, KL divergence on log-probabilities, and the
# positive-class weight of binary cross-entropy with logits. Each is ONE
# `@always_inline` body called by autograd/loss_ops.mojo's loss_term /
# loss_term_dp / loss_term_dy, which every column runs: the host loop, the
# device kernel (OP_LS_ELEM_FWD / OP_LS_ELEM_VJP), the loss fold's leaf
# chain (autograd/fold_ops.mojo OP_AGS_LOSS_PART / OP_AGS_LOSS_VJP) and the
# trainers' loss_value_into. No launch, allocation or Python in a body.
#
# Launches (Tape.loss_reduced, the form the Sequential fit records): one
# fold launch forward when the terms fit one leaf block, else the leaf launch
# plus the fold finish (autograd/fold.mojo loss_fold); one launch backward per
# operand that takes a gradient (OP_AGS_LOSS_VJP, the model's output only in a
# fit). The reduction factor (1 / elements, 1 / rows or 1) is a host scalar
# computed once per call.
from sequence.ops import add, sub, mul
from checks.numerics import ftz, identical_div, identical_exp, identical_log, identical_log1p, identical_sigmoid

# Elementwise loss kinds of this lane (a.i0 of OP_LS_ELEM_FWD / VJP; codes
# 1..4 and 7 are autograd/loss_ops.mojo's, 5 and 6 stay retired).
comptime LS_SMOOTH_L1 = 8
comptime LS_NLL = 9
comptime LS_KL_DIV = 10


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


# ---------------------------------------------------------------- smooth L1
@always_inline
def smooth_l1_term(p: Float32, y: Float32, beta: Float32) -> Float32:
    """torch.nn.SmoothL1Loss: 0.5 d^2 / beta where |d| < beta, else |d| - 0.5 beta (d = p - y). beta 0 is the
    absolute error (no element is inside the threshold)."""
    var d = sub(p, y)
    var ad = _abs(d)
    if ad < beta:
        return mul(Float32(0.5), ftz(identical_div(mul(d, d), beta)))
    return sub(ad, mul(Float32(0.5), beta))


@always_inline
def smooth_l1_dp(p: Float32, y: Float32, beta: Float32, g: Float32) -> Float32:
    """g * d smooth_l1_term / d p: g d / beta inside the threshold, g sign(d) outside."""
    var d = sub(p, y)
    if _abs(d) < beta:
        return mul(g, ftz(identical_div(d, beta)))
    return mul(g, _sign(d))


# ---------------------------------------------------------------- negative log-likelihood
@always_inline
def nll_term(p: Float32, y: Float32) -> Float32:
    """torch.nn.NLLLoss on a one-hot (or class-weight) target: -y p, where p is a log-probability. A zero target
    gives 0 whatever p is, so a log-probability of -inf in another class costs nothing (NLLLoss reads only the
    target class)."""
    if y == Float32(0):
        return Float32(0)
    return -mul(y, p)


@always_inline
def nll_dp(p: Float32, y: Float32, g: Float32) -> Float32:
    if y == Float32(0):
        return Float32(0)
    return -mul(g, y)


@always_inline
def nll_dy(p: Float32, y: Float32, g: Float32) -> Float32:
    return -mul(g, p)


# ---------------------------------------------------------------- KL divergence
@always_inline
def kl_div_term(p: Float32, y: Float32) -> Float32:
    """torch.nn.KLDivLoss(log_target=False): y (log y - p), where p is the log-probability of the model and y the
    target probability; 0 where y <= 0 (torch's xlogy convention at y = 0)."""
    if not (y > Float32(0)):
        return Float32(0)
    return mul(y, sub(ftz(identical_log(y)), p))


@always_inline
def kl_div_dp(p: Float32, y: Float32, g: Float32) -> Float32:
    if not (y > Float32(0)):
        return Float32(0)
    return -mul(g, y)


@always_inline
def kl_div_dy(p: Float32, y: Float32, g: Float32) -> Float32:
    if not (y > Float32(0)):
        return Float32(0)
    return mul(g, add(sub(ftz(identical_log(y)), p), Float32(1)))


# ---------------------------------------------------------------- BCE with logits and pos_weight
@always_inline
def bce_softplus_neg(x: Float32) -> Float32:
    """softplus(-x) = max(-x, 0) + log1p(exp(-|x|)), without overflow."""
    var neg = -x if x < 0 else Float32(0)
    return add(neg, ftz(identical_log1p(ftz(identical_exp(-_abs(x))))))


@always_inline
def bce_logits_term(x: Float32, y: Float32, w: Float32) -> Float32:
    """torch.nn.BCEWithLogitsLoss(pos_weight=w): -[w y log sigmoid(x) + (1 - y) log(1 - sigmoid(x))]
    = max(x, 0) - x y + log1p(exp(-|x|)) + (w - 1) y softplus(-x). With w = 1 the last term is skipped, so the
    words are the unweighted loss's."""
    var pos = x if x > 0 else Float32(0)
    var soft = ftz(identical_log1p(ftz(identical_exp(-_abs(x)))))
    var base = add(sub(pos, mul(x, y)), soft)
    var wm1 = sub(w, Float32(1))
    if wm1 == Float32(0):
        return base
    var neg = -x if x < 0 else Float32(0)
    return add(base, mul(mul(wm1, y), add(neg, soft)))


@always_inline
def bce_logits_dp(x: Float32, y: Float32, w: Float32, g: Float32) -> Float32:
    """g ((1 + (w - 1) y) sigmoid(x) - w y); g (sigmoid(x) - y) when w = 1."""
    var s = ftz(identical_sigmoid(x))
    var wm1 = sub(w, Float32(1))
    if wm1 == Float32(0):
        return mul(g, sub(s, y))
    var lw = add(Float32(1), mul(wm1, y))
    return mul(g, sub(mul(lw, s), mul(w, y)))


@always_inline
def bce_logits_dy(x: Float32, y: Float32, w: Float32, g: Float32) -> Float32:
    """g (-x + (w - 1) softplus(-x)); g (-x) when w = 1."""
    var wm1 = sub(w, Float32(1))
    if wm1 == Float32(0):
        return mul(g, -x)
    return mul(g, add(-x, mul(wm1, bce_softplus_neg(x))))
