# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host FP32 oracle of softmax cross-entropy under profile `mojolearn.identical.loss.ce.fp32.v1`, and its Float64 tolerance reference. - `refuse_nonfinite` here is a THIRD COPY (DEVIATION 1164)."""

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_log,
    identical_mul,
)
from gemm.checks.gemm_oracle import OP_NN, gemm_oracle



comptime REDUCTION_NONE = 0
comptime REDUCTION_SUM = 1
comptime REDUCTION_MEAN = 2

comptime IGNORE_INDEX_DEFAULT = -100

comptime CE_MAX_EXACT_COUNT = 16777216

comptime CE_MAX_ROWS = 4000000

comptime CE_POS_INF_BITS = UInt32(0x7F800000)
comptime CE_NEG_INF_BITS = UInt32(0xFF800000)
comptime CE_SIGN_BIT = UInt32(0x80000000)




def neg_by_bits(x: Float32) -> Float32:
    """DEVIATION 1154: IEEE negation spelled as an XOR of the sign bit. It is refused only because it presents a floating-point operation where an exact bit operation will do, and because an XOR cannot flush, cannot round and cannot be contracted (IDENTITY_PATHS row 9)."""
    from std.memory import bitcast

    var b = rebind[UInt32](x.to_bits())
    return bitcast[DType.float32](b ^ CE_SIGN_BIT)


def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """IDENTITY_PATHS row 39: a NaN or an infinity in an input is REFUSED BY NAME before any recorded stage. DEVIATION 1164.** The first is `mamba/checks/mamba_oracle.mojo:57` and the second is `training/checks/optimizer_oracle.mojo:162`, landed by the concurrent optimizer lane on 2026-08-25; all three must stay the same shape."""
    for i in range(len(values)):
        var au = rebind[UInt32](values[i].to_bits()) & UInt32(0x7FFFFFFF)
        if au > CE_POS_INF_BITS:
            raise Error(
                String("ce: NaN in ") + name + " at flat index " + String(i)
                + " REFUSED (row 39: NaN payloads are vendor-shaped; no"
                + " stage may record one)"
            )
        if au == CE_POS_INF_BITS:
            raise Error(
                String("ce: infinity in ") + name + " at flat index "
                + String(i) + " REFUSED (row 39)"
            )




@fieldwise_init
struct CeConfig(Copyable, Movable):
    """One loss call's configuration."""

    var vocab: Int
    var ignore_index: Int
    var reduction: Int
    var eps: Float32
    var num_items: Int

    @staticmethod
    def causal_lm(vocab: Int) -> Self:
        """`ForCausalLMLoss`'s own defaults with no `num_items_in_batch`, which is the MEAN arm (`fixed_cross_entropy` :39)."""
        return Self(vocab, IGNORE_INDEX_DEFAULT, REDUCTION_MEAN, 0.0, 0)

    def smoothing_is_spelled(self) -> Bool:
        """Contract 6.2(c)."""
        return self.eps != Float32(0.0)


def ce_one_minus_eps(eps: Float32) -> Float32:
    """`ONE_MINUS_EPS`, contract section 3."""
    return ftz(Float32(1.0) - ftz(eps))


def ce_smoothing_targets(eps: Float32, vocab: Int) -> Tuple[Float32, Float32]:
    """`(T_TARGET, T_OTHER)`, seam L15, contract 6.3."""
    var one_minus = ce_one_minus_eps(eps)
    var other = ftz(identical_div(ftz(eps), Float32(vocab)))
    var target = ftz(one_minus + other)
    return (target, other)


def ce_count(targets: List[Int32], ignore_index: Int) -> Int:
    """`count`, the number of rows whose target is not `ignore_index`. **AN INTEGER, and contract 5.5 turns that into a design constraint.** It is exact, order-free and vendor-free, and it is the reason contract section 11 refuses a per-class `weight` vector -- a weighted mean's denominator is a SUM OF FLOATS and would need a fold, a clause, a fixture and a sabotage of its own."""
    var c = 0
    for i in range(len(targets)):
        if Int(targets[i]) != ignore_index:
            c += 1
    return c


def ce_divisor(reduction: Int, count: Int, num_items: Int) raises -> Float32:
    """**THE ONE PRODUCER OF `divisor`**, seam L13's and seam L16's, contract 5.5. NONE -> refused; there is no reduced loss and no backward SUM -> Float32(num_items) when supplied, else exactly +1.0 MEAN -> Float32(count) THE SUM's DIVIDE BY EXACTLY `1.0` IS SPELLED ANYWAY."""
    if reduction == REDUCTION_NONE:
        raise Error(
            String("ce: REDUCTION_NONE has no divisor and no backward")
            + " (contract section 11)"
        )
    if reduction == REDUCTION_SUM:
        if num_items > 0:
            if num_items > CE_MAX_EXACT_COUNT:
                raise Error(
                    String("ce: num_items ") + String(num_items)
                    + " exceeds CE_MAX_EXACT_COUNT; Float32(num_items) would"
                    + " round (contract section 3)"
                )
            return Float32(num_items)
        if num_items < 0:
            raise Error(
                String("ce: num_items_in_batch ") + String(num_items)
                + " is negative; pass 0 for 'not supplied' (contract 5.5)"
            )
        return Float32(1.0)
    if reduction == REDUCTION_MEAN:
        if count <= 0:
            raise Error(
                String("ce: MEAN over zero unignored rows REFUSED. Torch")
                + " returns NaN here and a NaN payload is vendor-shaped"
                + " (row 39: 0x7fc00000 Apple, 0x7fffffff NVIDIA,"
                + " 0xffc00000 AMD). Contract section 11."
            )
        if count > CE_MAX_EXACT_COUNT:
            raise Error(
                String("ce: count ") + String(count)
                + " exceeds CE_MAX_EXACT_COUNT; Float32(count) would round"
                + " (contract section 3)"
            )
        return Float32(count)
    raise Error(String("ce: unknown reduction ") + String(reduction))


def reduction_name(reduction: Int) -> String:
    if reduction == REDUCTION_NONE:
        return String("NONE")
    if reduction == REDUCTION_SUM:
        return String("SUM")
    if reduction == REDUCTION_MEAN:
        return String("MEAN")
    return String("REDUCTION?")




def ce_ones(n: Int) -> List[Float32]:
    """`n` entries of exactly `Float32(1.0)`, the right operand of every fold in this file."""
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


def ce_fold(
    values: List[Float32], base: Int, count: Int, ones: List[Float32]
) -> Float32:
    """One fold of `values[base .. The two must agree bit for bit and that agreement is contract clause (a)."""
    var row = List[Float32]()
    for t in range(count):
        row.append(values[base + t])
    var out = gemm_oracle(row, ones, OP_NN, 1, 1, count)
    return out[0]


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




def ce_refuse_inputs(
    logits: List[Float32], targets: List[Int32], cfg: CeConfig
) raises -> Int:
    """Every refusal of contract section 8 and section 3, in one place, BEFORE any recorded stage. **A target equal to `ignore_index` is ignored even when `ignore_index` happens to be a valid class index.** That is torch's behavior and it is admitted rather than refused, so a caller who sets `ignore_index = 0` on a real vocabulary loses class 0 and this profile does not stop them."""
    if cfg.vocab < 1:
        raise Error(String("ce: vocab ") + String(cfg.vocab) + " < 1 REFUSED")
    if cfg.vocab > CE_MAX_EXACT_COUNT:
        raise Error(
            String("ce: vocab ") + String(cfg.vocab)
            + " exceeds CE_MAX_EXACT_COUNT; Float32(vocab) would round and"
            + " seam L10's divide would stop being checkable by hand"
        )
    var n = len(targets)
    if n < 1:
        raise Error(String("ce: N < 1 REFUSED"))
    if n > CE_MAX_ROWS:
        raise Error(
            String("ce: N ") + String(n) + " exceeds CE_MAX_ROWS; the batch"
            + " fold's k would leave the range gemm v1's own sweep has"
            + " exercised (contract section 3)"
        )
    if len(logits) != n * cfg.vocab:
        raise Error(
            String("ce: logits hold ") + String(len(logits))
            + " floats, expected N*V = " + String(n * cfg.vocab)
        )
    var eb = rebind[UInt32](cfg.eps.to_bits()) & UInt32(0x7FFFFFFF)
    if eb >= CE_POS_INF_BITS:
        raise Error(String("ce: label_smoothing is not finite REFUSED"))
    if cfg.eps < Float32(0.0) or cfg.eps >= Float32(1.0):
        raise Error(
            String("ce: label_smoothing must be in [0, 1) (contract"
                   " section 3)")
        )
    refuse_nonfinite("logits", logits)
    for i in range(n):
        var t = Int(targets[i])
        if t == cfg.ignore_index:
            continue
        if t < 0 or t >= cfg.vocab:
            raise Error(
                String("ce: target ") + String(t) + " at row " + String(i)
                + " is neither ignore_index nor in [0, vocab) REFUSED"
            )
    return n




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
    """**THE NORMATIVE FORWARD ANSWER of `mojolearn.identical.loss.ce.fp32.v1`.** Scalar, single threaded, host. That is deterministic, the same on every vendor, and admitted."""
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

    var total = ce_fold(st.row, 0, n, ones)
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
