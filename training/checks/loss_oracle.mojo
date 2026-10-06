# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host FP32 oracle of softmax cross-entropy under profile `mojolearn.identical.loss.ce.fp32.v1`, and its Float64 tolerance reference. The shared contract (constants, configuration, refusals) is in `training/checks/loss_contract.mojo`; `refuse_nonfinite` there is a THIRD COPY (DEVIATION 1164)."""

from training.neural_ab_profile_contract import NN54_LOSS_PROFILE, nn_reduce_host_admitted
from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_log,
    identical_mul,
)
from gemm.contract import OP_NN
from gemm.host.neural_gemm import gemm_oracle
from gemm.host.neural_gemm import gemm_host_rows
from training.checks.loss_contract import (
    CE_MAX_EXACT_COUNT,
    CE_MAX_ROWS,
    CE_NEG_INF_BITS,
    CE_POS_INF_BITS,
    CE_SIGN_BIT,
    IGNORE_INDEX_DEFAULT,
    REDUCTION_MEAN,
    REDUCTION_NONE,
    REDUCTION_SUM,
    CeConfig,
    ce_count,
    ce_divisor,
    ce_nonfinite_message,
    ce_one_minus_eps,
    ce_refuse_inputs,
    ce_refuse_shape,
    ce_refuse_targets,
    ce_smoothing_targets,
    ce_ones,
    neg_by_bits,
    reduction_name,
    refuse_nonfinite,
)



from training.neural_identical_experiments import IDN_LOSS_TOKEN_TREE_V2
from training.loss_reduction_v2 import loss_token_tree_v2_host


def ce_fold(
    values: List[Float32], base: Int, count: Int, ones: List[Float32]
) -> Float32:
    """One fold of `values[base .. The two must agree bit for bit and that agreement is contract clause (a)."""
    var row = List[Float32]()
    for t in range(count):
        row.append(values[base + t])
    var out = gemm_host_rows(row, ones, OP_NN, 1, 1, count)
    return out[0]


def ce_total_fold(values: List[Float32], count: Int, ones: List[Float32]) -> Float32:
    """NN54 v2 changes only row-total order; normalization stays unchanged."""
    comptime if NN54_LOSS_PROFILE:
        return nn_reduce_host_admitted[128, False](rebind[MutPointer[Float32, MutAnyOrigin]](values.unsafe_ptr()), count)
    comptime if IDN_LOSS_TOKEN_TREE_V2:
        return loss_token_tree_v2_host(values, 0, count)
    return ce_fold(values, 0, count, ones)


def ce_fold_serial_diagnostic(
    values: List[Float32], base: Int, count: Int
) -> Float32:
    """The WHOLE-AXIS ASCENDING CHAIN."""
    var acc = Float32(0.0)
    for t in range(count):
        acc = ftz(ftz(acc) + ftz(values[base + t]))
    return ftz(acc)




struct CeStages(Movable):
    """Every recorded stage of one loss call, in the card's order. A check that hashes an empty list records a zero-length stage, which the differ must treat as "absent" rather than as "agreeing with an absent one" -- `IdentityTrace.record_device`'s own length hazard, pointed the other way."""

    var max_v: List[Float32]        # [N]      L1
    var shift: List[Float32]        # [N, V]   L2
    var expo: List[Float32]         # [N, V]   L3
    var denom: List[Float32]        # [N]      L4
    var logdenom: List[Float32]     # [N]      L5
    var logp_target: List[Float32]  # [N]      L6
    var nll: List[Float32]          # [N]      L7
    var logp: List[Float32]         # [N, V]   L8   smoothing only
    var logp_sum: List[Float32]     # [N]      L9   smoothing only
    var smooth: List[Float32]       # [N]      L10  smoothing only
    var row: List[Float32]          # [N]      L11
    var count: Int                  # exact
    var divisor: List[Float32]      # [1]
    var total: List[Float32]        # [1]      L12
    var loss: List[Float32]         # [1]      L13
    var target_vec: List[Float32]   # [2]      L15  (T_TARGET, T_OTHER)
    var weights: List[Float32]      # [N, V]   L14
    var dlogits: List[Float32]      # [N, V]   L16

    def __init__(out self):
        self.max_v = List[Float32]()
        self.shift = List[Float32]()
        self.expo = List[Float32]()
        self.denom = List[Float32]()
        self.logdenom = List[Float32]()
        self.logp_target = List[Float32]()
        self.nll = List[Float32]()
        self.logp = List[Float32]()
        self.logp_sum = List[Float32]()
        self.smooth = List[Float32]()
        self.row = List[Float32]()
        self.count = 0
        self.divisor = List[Float32]()
        self.total = List[Float32]()
        self.loss = List[Float32]()
        self.target_vec = List[Float32]()
        self.weights = List[Float32]()
        self.dlogits = List[Float32]()







def _row_max(logits: List[Float32], base: Int, vocab: Int) -> Float32:
    """Seam L1, contract 5.1. **THE FOLD SHAPE IS FREE AND THIS FUNCTION'S ASCENDING LOOP IS NOT PART OF THE CONTRACT.** `portable_fmaxf` canonicalizes NaN first, flushes both operands and selects on `_total_order_key` (under which `+0.0` keys at `0x80000000` and `-0.0` at `0x7FFFFFFF`), so the result is commutative and associative over all of Float32 including both zeros and NaN."""
    from std.memory import bitcast

    var m = bitcast[DType.float32](CE_NEG_INF_BITS)
    for v in range(vocab):
        m = identical_fmax(m, logits[base + v])
    return m


def _row_combine(
    nll: Float32, smooth: Float32, one_minus_eps: Float32, eps: Float32
) -> Float32:
    """Seam L11, the label-smoothing combine, contract 6.2(b)."""
    var a = ftz(identical_mul(one_minus_eps, nll))
    var b = ftz(identical_mul(eps, smooth))
    return ftz(ftz(a) + ftz(b))


def ce_forward_oracle(
    logits: List[Float32], targets: List[Int32], cfg: CeConfig
) raises -> CeStages:
    """Native host forward under CE_NUMERICAL_PROFILE.

    The default retains v1; NI35 selects its separate L12 token-total tree.
    NI35's source has not been compiled or checked for cross-column identity.
    """
    var n = ce_refuse_inputs(logits, targets, cfg)
    var v = cfg.vocab
    var smoothing = cfg.smoothing_is_spelled()
    var one_minus = ce_one_minus_eps(cfg.eps)
    var tv = ce_smoothing_targets(cfg.eps, v)

    var wide = v
    if n > wide:
        wide = n
    var ones = ce_ones(wide)

    var st = CeStages()
    st.target_vec.append(tv[0])
    st.target_vec.append(tv[1])

    for i in range(n):
        var base = i * v
        var y = Int(targets[i])
        var ignored = y == cfg.ignore_index

        var m = _row_max(logits, base, v)
        st.max_v.append(m)

        for vv in range(v):
            var s = ftz(ftz(logits[base + vv]) - ftz(m))
            st.shift.append(s)
            st.expo.append(identical_exp(s))

        var denom = ce_fold(st.expo, base, v, ones)
        st.denom.append(denom)

        var logdenom = ftz(identical_log(ftz(denom)))
        st.logdenom.append(logdenom)

        var ty = y
        if ignored:
            ty = 0  # a placeholder index; `row` below discards the result
        var lp_y = ftz(ftz(st.shift[base + ty]) - ftz(logdenom))
        st.logp_target.append(lp_y)
        var nll = neg_by_bits(lp_y)
        st.nll.append(nll)

        var row_loss = nll
        if smoothing:
            for vv in range(v):
                st.logp.append(
                    ftz(ftz(st.shift[base + vv]) - ftz(logdenom))
                )
            var lpsum = ce_fold(st.logp, base, v, ones)
            st.logp_sum.append(lpsum)
            var sm = neg_by_bits(
                ftz(identical_div(ftz(lpsum), Float32(v)))
            )
            st.smooth.append(sm)
            row_loss = _row_combine(nll, sm, one_minus, ftz(cfg.eps))
        if ignored:
            row_loss = Float32(0.0)
        st.row.append(row_loss)

    st.count = ce_count(targets, cfg.ignore_index)

    if cfg.reduction == REDUCTION_NONE:
        return st^

    var total = ce_total_fold(st.row, n, ones)
    st.total.append(total)

    var divisor = ce_divisor(cfg.reduction, st.count, cfg.num_items)
    st.divisor.append(divisor)
    st.loss.append(ftz(identical_div(ftz(total), divisor)))
    return st^




def ce_backward_oracle(
    mut st: CeStages, targets: List[Int32], cfg: CeConfig
) raises:
    """**THE NORMATIVE BACKWARD ANSWER**, `dLoss/dlogits`, filled into `st.weights` and `st.dlogits`. `st` must be the output of `ce_forward_oracle` on the same inputs and the same config."""
    if cfg.reduction == REDUCTION_NONE:
        raise Error(
            String("ce: REDUCTION_NONE has no backward (contract section 11)")
        )
    if len(st.divisor) != 1:
        raise Error(
            String("ce: backward called on stages with no divisor; run")
            + " ce_forward_oracle with a SUM or MEAN reduction first"
        )
    if len(st.target_vec) != 2:
        raise Error(String("ce: backward called on empty stages"))

    var v = cfg.vocab
    var n = len(targets)
    var divisor = st.divisor[0]
    var t_target = st.target_vec[0]
    var t_other = st.target_vec[1]

    for i in range(n):
        var base = i * v
        var y = Int(targets[i])
        var ignored = y == cfg.ignore_index
        var denom = st.denom[i]
        for vv in range(v):
            var w = ftz(
                identical_div(ftz(st.expo[base + vv]), ftz(denom))
            )
            st.weights.append(w)
            if ignored:
                st.dlogits.append(Float32(0.0))
                continue
            var t = t_other
            if vv == y:
                t = t_target
            st.dlogits.append(
                ftz(identical_div(ftz(ftz(w) - ftz(t)), divisor))
            )




def ce_exact_uniform_gradient(
    vocab: Int, target: Int, is_target: Bool, divisor: Float32
) raises -> Float32:
    """Contract 12.1's closed form, ONE cell. REFUSES a non-power-of-two `vocab` or `divisor`, because the exactness argument depends on both and an unchecked exactness argument is how a gate comes to assert what the code does rather than what it should do (contract 12.4 guard 2)."""
    if vocab < 1 or (vocab & (vocab - 1)) != 0:
        raise Error(
            String("ce exact fixture: vocab ") + String(vocab)
            + " is not a power of two; 1/vocab would not be exact"
        )
    if not _is_exact_power_of_two(divisor):
        raise Error(
            String("ce exact fixture: divisor is not a power of two")
        )
    if target < 0 or target >= vocab:
        raise Error(String("ce exact fixture: target out of range"))
    var w = Float32(1.0) / Float32(vocab)
    var d = w
    if is_target:
        d = w - Float32(1.0)
    return d / divisor


def ce_exact_saturating_gradient(
    high_count: Int, is_high: Bool, is_target: Bool, divisor: Float32
) raises -> Float32:
    """Contract 12.2's closed form, ONE cell. This family adds four things the uniform one lacks -- a genuine argmax structure, an exercised underflow edge, exact `+0.0` weights whose SIGN must be `+`, and a case where the TARGET is a low cell, where `dl[y]` is exactly `(-1) / divisor`."""
    if high_count < 1 or (high_count & (high_count - 1)) != 0:
        raise Error(
            String("ce exact fixture: high_count ") + String(high_count)
            + " is not a power of two"
        )
    if not _is_exact_power_of_two(divisor):
        raise Error(
            String("ce exact fixture: divisor is not a power of two")
        )
    var w = Float32(0.0)
    if is_high:
        w = Float32(1.0) / Float32(high_count)
    var d = w
    if is_target:
        d = w - Float32(1.0)
    return d / divisor


def _is_exact_power_of_two(x: Float32) -> Bool:
    """A positive normal Float32 whose mantissa field is zero."""
    var b = rebind[UInt32](x.to_bits())
    if (b & CE_SIGN_BIT) != UInt32(0):
        return False
    var e = (b >> 23) & UInt32(0xFF)
    if e == UInt32(0) or e == UInt32(0xFF):
        return False
    return (b & UInt32(0x007FFFFF)) == UInt32(0)




def ce_forward_f64(
    logits: List[Float32], targets: List[Int32], cfg: CeConfig
) raises -> List[Float64]:
    """The per-row loss in double precision, plain `Float64` arithmetic, no pins and no partition."""
    from std.math import exp, log

    var n = len(targets)
    var v = cfg.vocab
    var out = List[Float64]()
    for i in range(n):
        var base = i * v
        var y = Int(targets[i])
        if y == cfg.ignore_index:
            out.append(Float64(0.0))
            continue
        var m = Float64(logits[base])
        for vv in range(1, v):
            var xv = Float64(logits[base + vv])
            if xv > m:
                m = xv
        var s = Float64(0.0)
        for vv in range(v):
            s += exp(Float64(logits[base + vv]) - m)
        var logdenom = log(s)
        var nll = logdenom - (Float64(logits[base + y]) - m)
        if cfg.eps == Float32(0.0):
            out.append(nll)
            continue
        var lpsum = Float64(0.0)
        for vv in range(v):
            lpsum += (Float64(logits[base + vv]) - m) - logdenom
        var smooth = -(lpsum / Float64(v))
        var e64 = Float64(cfg.eps)
        out.append((Float64(1.0) - e64) * nll + e64 * smooth)
    return out^
