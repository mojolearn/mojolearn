# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The contract half of profile `mojolearn.identical.loss.ce.fp32.v1` that the
device loss and the host oracle share: the reductions, the ignore index, the
bounds, the bit constants, `neg_by_bits`, the configuration, the count and
divisor, and the refusals. Moved out of `training/checks/loss_oracle.mojo`
(which imports them back) so the device training modules import the contract
without linking the CPU oracle (cpu-gpu-cleanup n-train-mamba, 2026-10-02).
`refuse_nonfinite` here is the plain ascending scan; the host-lane fast path
it had in the oracle is gone (a scan, same refusals, same messages)."""

from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_log,
    identical_mul,
)

# TOMBSTONE: MOJOLEARN_IDN_CE_TOKEN_FOLD=2 (noise) deleted 2026-10-10 by DELSHA_CE; code recoverable at ca25d9321.
# (it set the token-tree256 v2 loss profile here.) Restore: git apply experiments/removed/MOJOLEARN_IDN_CE_TOKEN_FOLD-arm2.patch
comptime CE_NUMERICAL_PROFILE = "mojolearn.identical.loss.ce.fp32.v1"


def ce_numerical_profile() -> String:
    return String(CE_NUMERICAL_PROFILE)


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


def ce_nonfinite_message(name: String, index: Int, is_nan: Bool) -> String:
    """THE ONE SPELLING of this profile's non-finite refusal (DEVIATION
    2514 step 2). `refuse_nonfinite` below raises through it over a host
    List, and `training/checks/loss.mojo::ce_refuse_device_inputs` raises
    through it with the index `device_first_nonfinite` returned, so the
    host and device paths cannot produce two messages for one defect. The
    two strings are byte for byte the ones `refuse_nonfinite` built inline
    before this function existed; loss_check clause (f) asserts the
    device message EQUALS the host message on the same planted List."""
    if is_nan:
        return (
            String("ce: NaN in ") + name + " at flat index " + String(index)
            + " REFUSED (row 39: NaN payloads are vendor-shaped; no"
            + " stage may record one)"
        )
    return (
        String("ce: infinity in ") + name + " at flat index "
        + String(index) + " REFUSED (row 39)"
    )


def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """IDENTITY_PATHS row 39: a NaN or an infinity in an input is REFUSED BY NAME before any recorded stage. DEVIATION 1164.** The first is `mamba/checks/mamba_oracle.mojo:57` and the second is `training/checks/optimizer_oracle.mojo:162`, landed by the concurrent optimizer lane on 2026-08-25; all three must stay the same shape. The message is `ce_nonfinite_message`'s (DEVIATION 2514)."""
    for i in range(len(values)):
        var au = rebind[UInt32](values[i].to_bits()) & UInt32(0x7FFFFFFF)
        if au > CE_POS_INF_BITS:
            raise Error(ce_nonfinite_message(name, i, True))
        if au == CE_POS_INF_BITS:
            raise Error(ce_nonfinite_message(name, i, False))




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



def ce_refuse_shape(n: Int, n_logits: Int, cfg: CeConfig) raises:
    """The scalar refusals of `ce_refuse_inputs`, first third: vocab, N,
    the logits length and the smoothing constant, in that order. `n` is
    the row count (`len(targets)` on the host, `n_rows` on the device) and
    `n_logits` the logits length. Split out at DEVIATION 2514 step 2 so the
    device entry can run the same checks in the same order without a List;
    `ce_refuse_inputs` calls this, then `refuse_nonfinite`, then
    `ce_refuse_targets`, and its walk is unchanged."""
    if cfg.vocab < 1:
        raise Error(String("ce: vocab ") + String(cfg.vocab) + " < 1 REFUSED")
    if cfg.vocab > CE_MAX_EXACT_COUNT:
        raise Error(
            String("ce: vocab ") + String(cfg.vocab)
            + " exceeds CE_MAX_EXACT_COUNT; Float32(vocab) would round and"
            + " seam L10's divide would stop being checkable by hand"
        )
    if n < 1:
        raise Error(String("ce: N < 1 REFUSED"))
    if n > CE_MAX_ROWS:
        raise Error(
            String("ce: N ") + String(n) + " exceeds CE_MAX_ROWS; the batch"
            + " fold's k would leave the range gemm v1's own sweep has"
            + " exercised (contract section 3)"
        )
    if n_logits != n * cfg.vocab:
        raise Error(
            String("ce: logits hold ") + String(n_logits)
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


def ce_refuse_targets(targets: List[Int32], cfg: CeConfig) raises:
    """The last third of `ce_refuse_inputs`: every target in `[0, vocab)`
    or equal to `ignore_index`, walked in row order. An INTEGER refusal;
    the same loop serves the host and the device entry (which downloads
    the M int32 targets, see `ce_refuse_device_inputs`)."""
    for i in range(len(targets)):
        var t = Int(targets[i])
        if t == cfg.ignore_index:
            continue
        if t < 0 or t >= cfg.vocab:
            raise Error(
                String("ce: target ") + String(t) + " at row " + String(i)
                + " is neither ignore_index nor in [0, vocab) REFUSED"
            )


def ce_refuse_inputs(
    logits: List[Float32], targets: List[Int32], cfg: CeConfig
) raises -> Int:
    """Every refusal of contract section 8 and section 3, in one place, BEFORE any recorded stage. **A target equal to `ignore_index` is ignored even when `ignore_index` happens to be a valid class index.** That is torch's behavior and it is admitted rather than refused, so a caller who sets `ignore_index = 0` on a real vocabulary loses class 0 and this profile does not stop them. Since DEVIATION 2514 step 2 it is the three parts in the order they always ran: shape, the non-finite scan of `logits`, the targets walk."""
    var n = len(targets)
    ce_refuse_shape(n, len(logits), cfg)
    refuse_nonfinite("logits", logits)
    ce_refuse_targets(targets, cfg)
    return n



def ce_ones(n: Int) -> List[Float32]:
    """`n` entries of exactly `Float32(1.0)`, the right operand of every fold of the profile."""
    return List[Float32](length=n, fill=Float32(1.0))
