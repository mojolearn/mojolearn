"""The host FP32 oracle of softmax cross-entropy under profile
`mojolearn.identical.loss.ce.fp32.v1`, and its Float64 tolerance reference.

**THIS BANNER WAS FALSE AND IS CORRECTED. COMPILED AND RUN ON ONE DEVICE.**
Until 2026-08-31 this header read "THIS FILE HAS NEVER BEEN COMPILED AND HAS
NEVER BEEN EXECUTED", and added that no device had run a single stage of it and
no gate had ever failed against it. **Commit `ecd1a436` is the first execution
of this file, with `loss.mojo`, `loss_check.mojo` and `loss_fixture.mojo`**,
and clause (a) matched this oracle against the device BITWISE over 24 cases and
61,925 cells. Clause (f) then failed against a real defect in `loss.mojo`,
which is a gate failing against this file's contract exactly as intended. The
numbers in the docstrings below were derived on paper on 2026-08-25 and the run
did not contradict them; they are still paper arithmetic where they are not
marked measured. Written by the training lane, DEVIATIONS 1150-1169. **One
device, not three.**

WHAT IS OWED, and none of it is in this file
---------------------------------------------
  - `training/mojo_only/loss_fixture.mojo` and
    `training/mojo_only/loss_check.mojo` DO NOT EXIST. **Not one clause of
    `training/IDENTICAL_LOSS_CONTRACT.md` has been falsified by a sabotage.**
    Section 10.1 of that document is a specification for gates, not a report
    of any.
  - `training/corpus/` does not exist, so `ce_forward_f64` below has nothing
    independent to be checked against.
  - `refuse_nonfinite` here is a THIRD COPY (DEVIATION 1164). The first is
    `mamba/mojo_only/mamba_oracle.mojo:57`; the second landed in
    `training/mojo_only/optimizer_oracle.mojo:162` on 2026-08-25 while this
    file was being written, and that lane's own header already counts three
    copies and owes the same lift. It belongs in `mojo_only/numerics.mojo`;
    that file is under concurrent edit by the numerics lane and the lift is
    theirs. **Two lanes in one directory now want the same edit**, which is
    the argument for making it once. Contract section OWED item 1.
  - `neg_by_bits` (DEVIATION 1154) belongs beside `identical_mul` in
    `mojo_only/numerics.mojo` for the reason DEVIATION 826 gives about
    `pinned_mul`. Contract OWED item 2.
  - The three decisions marked UNVERIFIED below need a PyTorch checkout that
    `/Users/andrewhendel/CascadeProjects/upstream/` does not have. Contract
    section 1 and OWED item 8.

NOT A PORT. `torch.nn.functional.cross_entropy` dispatches into ATen, which
could not be read. What IS pinned from a readable reference is the WRAPPER --
`transformers/src/transformers/loss/loss_utils.py::fixed_cross_entropy`
(:32-46) and `::ForCausalLMLoss` (:49-71) at
`d56c55bf564ddb176759eb6ec199442682564916` -- the FP32 upcast, the flatten to
`[-1, vocab_size]`, the `ignore_index`, and the `loss / num_items_in_batch`
divide. The ARITHMETIC ORDER below is this repository's own and is stated in
`training/IDENTICAL_LOSS_CONTRACT.md`, clause by clause.

WHAT THIS FILE IS FOR
---------------------
1. **Be the definition.** The NORMATIVE answer of the profile is
   `ce_forward_oracle` and `ce_backward_oracle`. Not "close to"; the same
   bits, on Apple, NVIDIA and AMD, at every launch geometry and every row
   chunking.
2. **Be built from the DECLARED helpers.** Every seam is
   `mojo_only.numerics`'s own `ftz`, `identical_mul`, `identical_exp`,
   `identical_log`, `identical_div` and `identical_fmax`, imported rather
   than copied, so this file cannot drift into an independent opinion about
   what IDENTICAL means. The consequence is that under `NUMERIC_FAST` both
   pins compile away and **THIS FILE IS NOT THE CONTRACT** -- it is the FAST
   spelling of the same loops. Every check must print the mode it compiled
   in.
3. **CONTAIN NO FOLD OF ITS OWN.** All three reductions (the vocabulary
   denominator, the vocabulary log-probability sum, and the batch sum) are
   routed to `gemm/mojo_only/gemm_oracle.mojo::gemm_oracle` against a vector
   of exact ones. DEVIATION 1152. That is deliberately the shape
   `gemm/mojo_only/gemm_backward.mojo` has -- a routing layer whose "no
   arithmetic of its own" claim is falsifiable rather than decorative -- and
   the claim here is narrower and equally checkable: **the only `+` between
   two Float32 values in this file is in `_row_combine` (seam L11, the
   label-smoothing combine), in `ce_smoothing_targets` (a HOST constant
   computed once per configuration) and in `ce_fold_serial_diagnostic`
   (which is not on any normative path).** If a fourth one appears, a fold
   has grown by hand and the contract has to say so.

`[[mojo-buffer-freed-at-last-use]]`: nothing here touches a device.
`[[mojo-string-float-roundtrip]]`: nothing here prints; the check will, and
it must print hex bits beside every decimal.
"""

from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_fmax,
    identical_log,
    identical_mul,
)
from gemm.mojo_only.gemm_oracle import OP_NN, gemm_oracle


# ===========================================================================
# PROFILE CONSTANTS AND ENUMERATIONS (contract section 3)
# ===========================================================================

#: No reduction: the output is `ce.row`, `[N]`. There is no `ce.loss` stage
#: and there is NO BACKWARD (contract section 11 -- a per-row upstream vector
#: adds a second product per cell whose placement is a real decision and this
#: lane has no caller for it).
comptime REDUCTION_NONE = 0
#: `fixed_cross_entropy` :39's `"sum"`. The divisor is `num_items_in_batch`
#: when the caller supplies one (:45) and exactly `+1.0` otherwise.
comptime REDUCTION_SUM = 1
#: `fixed_cross_entropy` :39's `"mean"`. The divisor is `Float32(count)`,
#: `count` being the number of rows whose target is not `ignore_index`.
comptime REDUCTION_MEAN = 2

#: `fixed_cross_entropy` :36. Any Int outside `[0, V)` works; this is torch's
#: default and is the only value the transformers wrapper ever passes.
comptime IGNORE_INDEX_DEFAULT = -100

#: Contract section 3's refusal. Above this `Float32(count)` is no longer
#: exact and the MEAN's divisor would round, which would make the divisor a
#: function of nothing a reader can check by hand.
comptime CE_MAX_EXACT_COUNT = 16777216

#: Contract section 3's other refusal. `gemm_device_check.mojo`'s forward
#: sweep reaches `k = 4,000,000` (case `k4M.P1024.cap`), so a batch fold at
#: `k = N` up to four million is inside the range the GEMM partition has
#: actually been exercised at. Past that it is not, and a refusal says so
#: rather than a docstring.
comptime CE_MAX_ROWS = 4000000

#: `+inf` and `-inf` by bits, because a compare cannot be trusted to see them
#: on a column that flushes compare operands (IDENTITY_PATHS row 49).
comptime CE_POS_INF_BITS = UInt32(0x7F800000)
comptime CE_NEG_INF_BITS = UInt32(0xFF800000)
#: The sign bit, for `neg_by_bits`.
comptime CE_SIGN_BIT = UInt32(0x80000000)


# ===========================================================================
# THE TWO PRIMITIVES THIS LANE ADDS (contract 4.3 and section 8)
# ===========================================================================


def neg_by_bits(x: Float32) -> Float32:
    """DEVIATION 1154: IEEE negation spelled as an XOR of the sign bit.

    Seams L7 (`nll = -lp_y`) and L10 (`smooth = -(lpsum / V)`).

    THREE SPELLINGS AND ONLY THIS ONE IS RIGHT AT EVERY INPUT.

      - `0.0 - x` is **WRONG at `x = +0.0`**, where round-to-nearest gives
        `+0.0` and IEEE negation gives `-0.0`. Contract 8.1 shows
        `lp_y == +0.0` is reachable -- at `V == 1`, and at any `V` where
        every non-target exponential underflows to exactly `+0.0` -- so this
        is not a hypothetical. Sabotage `L_NEG_VIA_ZERO_SUB`.
      - `identical_mul(x, -1.0)` is CORRECT at every input including both
        zeros (`fma(+0, -1, -0)` is `(-0) + (-0)` = `-0`, and
        `fma(-0, -1, -0)` is `(+0) + (-0)` = `+0`). It is refused only
        because it presents a floating-point operation where an exact bit
        operation will do, and because an XOR cannot flush, cannot round and
        cannot be contracted (IDENTITY_PATHS row 9).
      - a source-level `-x` is the compiler's choice between the two above.
        It is very probably a sign-bit XOR on all three backends and this
        profile does not rest on "very probably".

    NO FLUSH HERE, AND NONE IS NEEDED. The operand arrives already flushed
    from the seam that produced it, so a flush would be bitwise redundant;
    and **integer operations do not flush on Metal** (`_ftz_always`'s
    docstring, DEVIATION 746 (i)), which is the second reason the bit
    spelling is the safe one at a seam a Metal kernel will also spell.

    Exact on NaN too (it flips the payload's sign bit and nothing else), but
    contract section 8 refuses a NaN long before this is reached.
    """
    from std.memory import bitcast

    var b = rebind[UInt32](x.to_bits())
    return bitcast[DType.float32](b ^ CE_SIGN_BIT)


def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """IDENTITY_PATHS row 39: a NaN or an infinity in an input is REFUSED BY
    NAME before any recorded stage.

    **A THIRD COPY. DEVIATION 1164.** The first is
    `mamba/mojo_only/mamba_oracle.mojo:57` and the second is
    `training/mojo_only/optimizer_oracle.mojo:162`, landed by the concurrent
    optimizer lane on 2026-08-25; all three must stay the same shape. It is copied rather than imported because importing it would drag
    `mamba/mojo_only/mamba_fixture.mojo` (and therefore a Mamba block's whole
    configuration) into every training build, which is the same dependency-
    direction argument `mojo_only/numerics.mojo::_total_order_key` makes
    about `extratrees`. **The right home is `numerics.mojo` and the lift is
    the numerics lane's**; contract OWED item 1.

    Why refuse rather than propagate. A computed NaN's PAYLOAD is
    vendor-shaped -- row 39 measured `0x7fc00000` on Apple, `0x7fffffff` on
    NVIDIA and `0xffc00000` on AMD for one IEEE answer -- so a certified
    stage may not contain one, and torch's behavior here (propagate) is a
    KNOWING DEPARTURE recorded in contract section 11.

    **TESTED BY BITS, NOT BY COMPARES.** Metal flushes COMPARE operands
    (row 49's measurement), so a compare-written test has one meaning on one
    column and another elsewhere. `(bits & 0x7FFFFFFF) > 0x7F800000` is a
    NaN of either sign and any payload; `== 0x7F800000` is an infinity of
    either sign. Integer operations do not flush anywhere.

    THE ONE DIFFERENCE FROM THE MAMBA COPY, stated so the two can be diffed:
    that one reaches the bits through `bitcast[DType.uint32](values[i])` and
    this one through `rebind[UInt32](values[i].to_bits())`, which is the
    spelling `mojo_only/numerics.mojo::_total_order_key` and
    `::portable_logf` both use. The two are the same reinterpretation and
    neither rounds; when the lift of OWED item 1 happens, one of them
    survives.
    """
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


# ===========================================================================
# THE CONFIGURATION, AND THE ONE PRODUCER OF THE DIVISOR (contract 5.5)
# ===========================================================================


@fieldwise_init
struct CeConfig(Copyable, Movable):
    """One loss call's configuration. Every field is a HOST value known
    before any launch; none of them is data dependent.

    `eps` is the label-smoothing coefficient and **`eps == 0.0` selects a
    DIFFERENT CODE PATH** rather than a bit-inert arm -- contract 6.2(c),
    DEVIATION 1155. `num_items` is `fixed_cross_entropy` :35's
    `num_items_in_batch`, and a value below 1 means "not supplied".
    """

    var vocab: Int
    var ignore_index: Int
    var reduction: Int
    var eps: Float32
    var num_items: Int

    @staticmethod
    def causal_lm(vocab: Int) -> Self:
        """`ForCausalLMLoss`'s own defaults with no `num_items_in_batch`,
        which is the MEAN arm (`fixed_cross_entropy` :39)."""
        return Self(vocab, IGNORE_INDEX_DEFAULT, REDUCTION_MEAN, 0.0, 0)

    def smoothing_is_spelled(self) -> Bool:
        """Contract 6.2(c). The comparison is against a HOST constant, so
        this is a configuration branch and never a data-dependent one.

        At `eps == 0` the smoothing arm is ALMOST bit-inert and the almost is
        the whole clause: `identical_mul(0.0, smooth)` is `+0.0` for a
        non-negative `smooth`, and `ftz(nll + (+0.0))` is `nll` for every
        value except `nll = -0.0`, where it LAUNDERS to `+0.0`. Contract 8.1
        shows `nll == -0.0` is reachable. So spelling the arm unconditionally
        would make turning smoothing off move a bit on exactly one row shape.
        Sabotage `L_SMOOTH_ALWAYS_SPELLED`.
        """
        return self.eps != Float32(0.0)


def ce_one_minus_eps(eps: Float32) -> Float32:
    """`ONE_MINUS_EPS`, contract section 3. One rounding, on the host, ONCE
    per configuration -- never recomputed per row, so a row's bits cannot
    depend on where the subtraction happened."""
    return ftz(Float32(1.0) - ftz(eps))


def ce_smoothing_targets(eps: Float32, vocab: Int) -> Tuple[Float32, Float32]:
    """`(T_TARGET, T_OTHER)`, seam L15, contract 6.3. HOST, once.

        T_OTHER  = ftz(identical_div(eps, Float32(vocab)))
        T_TARGET = ftz(ONE_MINUS_EPS + T_OTHER)

    This is the analytic gradient of contract 6.1 with respect to `x_v`,
    worked out --
    `d/dx_v [(1-e) nll + e smooth] = w_v - (1-e) onehot_v - e/V` -- and it is
    spelled as a target VECTOR rather than as three terms so the gradient
    kernel performs one subtraction and one division per cell and nothing
    else.

    **AT `eps == 0` THESE ARE EXACTLY `1.0` AND `+0.0`**, so unlike the
    forward, the backward's smoothing arm is bitwise inert at eps zero and
    needs no branch. That asymmetry is `[[reached-but-inert]]` in the other
    direction and contract 6.3 names it: a sabotage that deletes a branch the
    gradient does not have moves nothing, and that is not evidence about the
    gradient.

    `Float32(vocab)` is exact for every `vocab` under `CE_MAX_EXACT_COUNT`.
    """
    var one_minus = ce_one_minus_eps(eps)
    var other = ftz(identical_div(ftz(eps), Float32(vocab)))
    var target = ftz(one_minus + other)
    return (target, other)


def ce_count(targets: List[Int32], ignore_index: Int) -> Int:
    """`count`, the number of rows whose target is not `ignore_index`.

    **AN INTEGER, and contract 5.5 turns that into a design constraint.** It
    is exact, order-free and vendor-free, and it is the reason contract
    section 11 refuses a per-class `weight` vector -- a weighted mean's
    denominator is a SUM OF FLOATS and would need a fold, a clause, a fixture
    and a sabotage of its own.
    """
    var c = 0
    for i in range(len(targets)):
        if Int(targets[i]) != ignore_index:
            c += 1
    return c


def ce_divisor(reduction: Int, count: Int, num_items: Int) raises -> Float32:
    """**THE ONE PRODUCER OF `divisor`**, seam L13's and seam L16's, contract
    5.5. DEVIATION 1156.

    Pure and host-side: it reads `reduction`, `count` and `num_items` and
    nothing else -- no buffer, no device, no launch, no `N`. That is
    `gemm/mojo_only/gemm_backward.mojo`'s "the backward shape has exactly two
    producers" discipline applied to the one quantity in this lane that two
    different passes have to agree about. **To make the forward and the
    backward disagree about the scale you would have to edit this one
    function**, and it is testable with no GPU present.

        NONE  ->  refused; there is no reduced loss and no backward
        SUM   ->  Float32(num_items) when supplied, else exactly +1.0
        MEAN  ->  Float32(count)

    THE SUM's DIVIDE BY EXACTLY `1.0` IS SPELLED ANYWAY. It is bitwise inert
    at every value the accumulator can hold (`portable_divf(x, 1.0)` returns
    `x` for every finite `x` and both zero signs), and spelling it keeps ONE
    code path so there is no branch whose two arms have to be shown to agree.

    **`count == 0` IS REFUSED BY NAME.** Torch computes `0.0 / 0.0` there and
    returns a NaN; a computed NaN's payload is vendor-shaped (row 39), so a
    certified stage may not contain one. Contract section 11 records the
    departure. Sabotage `L_GRAD_DIVISOR_IS_N` substitutes `N` for `count` in
    the backward and **is bit-inert whenever no row is ignored** -- the gate
    must PREDICT that inertness and assert it as a mask, or the sabotage will
    look broken on a fixture with no ignored rows.
    """
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


# ===========================================================================
# THE FOLDS, ALL THREE, ROUTED TO gemm v1 (contract 5.3, 5.4; DEVIATION 1152)
# ===========================================================================
# THERE IS NO FOLD IN THIS FILE. What follows is addressing.
#
# `identical_gemm`'s leaf loop computes `acc = ftz(fma(ftz(a), ftz(b), acc))`,
# and with `b` exactly `1.0` the product `a * 1.0` is exact, so
# `fma(a, 1, acc)` is ONE rounding of `a + acc`. The ones vector therefore
# turns a sum into the contract's own ascending flushed chain inside a leaf
# and the contract's own balanced tree across leaves, with no second fold
# shape anywhere and no second thing to certify.
#
# **THAT SENTENCE IS NOT THIS LANE'S INVENTION.** It is
# `gemm/mojo_only/gemm_backward.mojo`'s module docstring, DEVIATION 851,
# where the bias gradient's row sum was routed the same way for the same
# reason. This lane applies the finding to three more reductions.
#
# The departure from `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.3 --
# which pinned a SERIAL ASCENDING CHAIN for softmax's denominator -- is
# DEVIATION 1152 and rests on one fact: that contract's reason for the serial
# chain was that its fold LENGTH (the kv length) varies between prefill and
# decode, so `P = f(k)` would give a row two different trees. **The
# vocabulary never varies.** Contract 3.1 is the argument and contract 3.1's
# last paragraph is its price -- a per-row or masked vocabulary is REFUSED.


def ce_ones(n: Int) -> List[Float32]:
    """`n` entries of exactly `Float32(1.0)`, the right operand of every fold
    in this file.

    **A WRONG VALUE HERE IS A WRONG ANSWER WITH NO SYMPTOM**, because any
    vector produces a plausible weighted sum. That is
    `identical_gemm_backward_bias_ones_floats`'s warning verbatim, and the
    gate for it is not "the buffer was allocated" -- it is the
    EXACT-ANALYTIC arm of contract 12.3, which fails if the ones are anything
    else.

    One vector serves both axes: the caller allocates `max(V, N)` entries and
    passes the same list to every fold, because `gemm_oracle` at `n == 1`
    reads only `b[0 .. k)`.
    """
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


def ce_fold(
    values: List[Float32], base: Int, count: Int, ones: List[Float32]
) -> Float32:
    """One fold of `values[base .. base + count)`, as gemm v1 `OP_NN` at
    `(1, 1, count)` against `ones`. Seams L4, L9 and L12, all three.

    `_a_at(a, OP_NN, 0, p, 1, count)` is `a[p]` and
    `_b_at(b, OP_NN, p, 0, 1, count)` is `b[p]`, so this is exactly the
    contract's leaf-and-tree over the slice -- `L = contract_leaf_size(count)`
    and `P = ceil(count / L)`, from `count` alone.

    THE SLICE IS COPIED INTO A FRESH `List` and that is a real cost. It is
    `O(count)` per fold on top of an oracle that is already `O(N * V)`
    scalar host Mojo, so it changes the constant and not the shape.
    `[[mojo-list-float32-not-implicitly-copyable]]` is the reason it is an
    explicit `append` loop rather than an assignment: `var row = values` does
    not compile for `List[Float32]`, and `.copy()` would copy the whole
    buffer rather than the slice.

    A device implementation does NOT copy: it passes the `[N, V]` buffer
    whole as an `m x k` operand and lets `identical_gemm` address it, which
    is what `training/mojo_only/loss.mojo` does. The two must agree bit for
    bit and that agreement is contract clause (a).
    """
    var row = List[Float32]()
    for t in range(count):
        row.append(values[base + t])
    var out = gemm_oracle(row, ones, OP_NN, 1, 1, count)
    return out[0]


def ce_fold_serial_diagnostic(
    values: List[Float32], base: Int, count: Int
) -> Float32:
    """The WHOLE-AXIS ASCENDING CHAIN. **DIAGNOSTIC, NOT NORMATIVE.**

    The twin of `gemm_oracle_serial_cell`, and it exists for the same three
    reasons that one does. It is the simplest thing a person can check by
    hand; it is what `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md` 5.3
    pins for ITS softmax, so the difference between the two lanes is a value
    that can be printed rather than an argument; and it is the `L_DENOM_
    SERIAL_CHAIN` sabotage's expected answer, which means the check can
    assert the sabotage's output instead of merely observing that it moved.

    **It equals `ce_fold` when and only when `count <= 128`** (there
    `P == 1` and the tree performs no addition -- gemm contract 7.3). Above
    that the two are DIFFERENT ANSWERS and the contract's is `ce_fold`. Do
    not describe this function as "the right answer" at the shipped
    vocabulary.

    This function DOES contain a `+`, and it is the one exception to the
    module docstring's claim. It is not on any normative path.
    """
    var acc = Float32(0.0)
    for t in range(count):
        acc = ftz(ftz(acc) + ftz(values[base + t]))
    return ftz(acc)


# ===========================================================================
# THE STAGES (contract section 9)
# ===========================================================================


struct CeStages(Movable):
    """Every recorded stage of one loss call, in the card's order.

    `logp` and `logp_sum` and `smooth` are EMPTY unless
    `CeConfig.smoothing_is_spelled()`; `loss` is empty under
    `REDUCTION_NONE`; `weights` and `dlogits` are empty until
    `ce_backward_oracle` runs. A check that hashes an empty list records a
    zero-length stage, which the differ must treat as "absent" rather than
    as "agreeing with an absent one" -- `IdentityTrace.record_device`'s own
    length hazard, pointed the other way.

    `count` and `divisor` and `target_vec` are HOST constants and are on the
    card anyway, for the reason the transformer contract put `rope.inv_freq`
    there: a constant computed with the wrong spelling is a silent
    divergence that no activation stage localizes, and `divisor` in
    particular is read by two different passes.
    """

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


# ===========================================================================
# THE REFUSALS (contract section 8)
# ===========================================================================


def ce_refuse_inputs(
    logits: List[Float32], targets: List[Int32], cfg: CeConfig
) raises -> Int:
    """Every refusal of contract section 8 and section 3, in one place,
    BEFORE any recorded stage. Returns `N`.

    THE ORDER MATTERS. The shape refusals come first because a wrong `N`
    makes every later message wrong; the nonfinite scan comes before any
    arithmetic because that is what "before any recorded stage" means; and
    the target-range scan is an INTEGER compare, which needs no flush
    reasoning at all.

    **A target equal to `ignore_index` is ignored even when `ignore_index`
    happens to be a valid class index.** That is torch's behavior and it is
    admitted rather than refused, so a caller who sets `ignore_index = 0` on
    a real vocabulary loses class 0 and this profile does not stop them.
    """
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


# ===========================================================================
# THE FORWARD (contract section 2, seams L1 through L13)
# ===========================================================================


def _row_max(logits: List[Float32], base: Int, vocab: Int) -> Float32:
    """Seam L1, contract 5.1. The fold of `identical_fmax` over EVERY element
    of the row, seeded `-inf`.

    **THE FOLD SHAPE IS FREE AND THIS FUNCTION'S ASCENDING LOOP IS NOT PART
    OF THE CONTRACT.** `portable_fmaxf` canonicalizes NaN first, flushes both
    operands and selects on `_total_order_key` (under which `+0.0` keys at
    `0x80000000` and `-0.0` at `0x7FFFFFFF`), so the result is commutative
    and associative over all of Float32 including both zeros and NaN. This is
    the only place in the profile where an execution plan may choose its own
    tree, and it may because the operation is exactly associative rather than
    because the difference is thought to be small.

    THE SEED IS `-inf` AND NOT `+0.0` AND NOT `-FLT_MAX`. DEVIATION 1151.

      - `+0.0` is WRONG: a row whose logits are all negative -- which is most
        rows of a trained head -- would have its maximum clamped to `+0.0`,
        every shift would be too negative, and **the loss would be quietly
        wrong on every vendor identically**, which bit-identity cannot see.
        Sabotage `L_MAX_SEED_ZERO`.
      - `-FLT_MAX` (which MAX's own `softmax_kernel:958` seeds with) is
        CORRECT under this profile only because contract section 8 refuses an
        infinite logit somewhere else. `key(-inf)` is `0x007FFFFF`, strictly
        below the key of every finite value, so `-inf` is a true identity and
        needs no distant refusal to hold it up. MAX is inconsistent with
        itself here anyway -- `_softmax_warp_kernel:1063` in the same file
        seeds with `min_or_neg_inf`.

    `core/pinned_reduce.mojo::pinned_block_max` may NOT be used for this. Its
    fold is a plain `other > red[tid]` compare (:159-190), the exact spelling
    IDENTITY_PATHS row 13 closed everywhere else, and its own block comment
    (:145-155) requires a caller whose inputs can carry `+-0.0` to say why
    first. This caller cannot: IDENTITY_PATHS row 39 MEASURED
    `max(+0.0, -0.0)` as `-0.0` on Apple and `+0.0` on NVIDIA and AMD, and a
    row of 128,256 logits reaches both zero signs easily.
    """
    from std.memory import bitcast

    var m = bitcast[DType.float32](CE_NEG_INF_BITS)
    for v in range(vocab):
        m = identical_fmax(m, logits[base + v])
    return m


def _row_combine(
    nll: Float32, smooth: Float32, one_minus_eps: Float32, eps: Float32
) -> Float32:
    """Seam L11, the label-smoothing combine, contract 6.2(b).

        ftz( ftz(identical_mul(ONE_MINUS_EPS, nll))
           + ftz(identical_mul(EPS, smooth)) )

    **UNFUSED.** Two rounded products, then one add of two already-rounded
    values. An `identical_mul_add(eps, smooth, product1)` is ONE rounding
    where this is three, and it is the natural thing for a kernel to write.
    That is `transformer/IDENTICAL_TRANSFORMER_CONTRACT.md`'s S10 clause at a
    different seam and it has the same sabotage shape,
    `L_SMOOTH_FUSED_COMBINE`.

    **`identical_mul` and not `*`.** DEVIATION 826. `identical_mul(a, b)` is
    `identical_mul_add(a, b, -0.0)`, which presents no syntactic multiply for
    a codegen to contract into the following add -- the gemm README's F3 scar
    records `var p = a * b; p + c` being contracted ACROSS STATEMENTS on this
    host, so a separate statement is not a barrier -- and whose `-0.0` addend
    preserves a negative-zero product where a `+0.0` addend would launder it
    (the gemm lane's F6a lesson).

    **THIS IS THE ONLY `+` BETWEEN TWO Float32 VALUES ON A NORMATIVE PATH IN
    THIS FILE.** The module docstring makes that a checkable claim; if a
    second one appears, a fold has grown by hand and the contract has to say
    so. (`ce_fold_serial_diagnostic` has one and is not normative;
    `ce_smoothing_targets` has one and is a HOST CONSTANT computed once.)
    """
    var a = ftz(identical_mul(one_minus_eps, nll))
    var b = ftz(identical_mul(eps, smooth))
    return ftz(ftz(a) + ftz(b))


def ce_forward_oracle(
    logits: List[Float32], targets: List[Int32], cfg: CeConfig
) raises -> CeStages:
    """**THE NORMATIVE FORWARD ANSWER of
    `mojolearn.identical.loss.ce.fp32.v1`.**

    Scalar, single threaded, host. Performance is irrelevant here and it is
    meant to stay that way: it is `O(N * V)` with a `List` copy per fold, and
    at the shipped Llama-3 shape it would be absurd. Its job is to be the
    definition, and the device is what has to be fast.

    THE ORDER, and every step names its seam and its contract clause.

      L1  m        = fold of identical_fmax, any order, seed -inf   (5.1)
      L2  shift[v] = ftz(ftz(x_v) - ftz(m))                         (4)
      L3  e[v]     = identical_exp(shift[v])                        (4)
      L4  denom    = gemm v1 OP_NN (1,1,V) against ones             (5.3)
      L5  logdenom = identical_log(denom)                           (4)
      L6  lp_y     = ftz(ftz(shift[y]) - ftz(logdenom))             (4.2)
      L7  nll      = neg_by_bits(lp_y)                              (4.3)
      L8  lp[v]    = ftz(ftz(shift[v]) - ftz(logdenom))       smoothing only
      L9  lpsum    = gemm v1 OP_NN (1,1,V) against ones       smoothing only
      L10 smooth   = neg_by_bits(ftz(identical_div(lpsum, V))) smoothing only
      L11 row      = the combine, or nll when eps == 0              (6.2)
      L12 total    = gemm v1 OP_NN (1,1,N) against ones             (5.4)
      L13 loss     = identical_div(total, divisor)                  (5.5)

    THREE THEOREMS THAT REMOVE A WHOLE CLASS OF HAZARDS, contract 8.2, and
    they are what let this function have no special cases.

      (a) `m` is the total-order maximum of the flushed row, so
          `ftz(x_v) - m <= +0.0` for every `v`. **The shift is provably
          non-positive, so `identical_exp` never takes its overflow branch.**
      (b) The argmax contributes `identical_exp(+0.0)`, which is exactly
          `1.0`, and every other term is non-negative, and the leaf chain
          seeded `+0.0` cannot lose it. **`denom` is provably in
          `[1.0, Float32(V)]`**, so `identical_log`'s argument is never zero,
          never negative, never subnormal and never infinite, and every one
          of `portable_logf`'s special-value branches is unreachable from
          here.
      (c) `nll = logdenom - shift[y]` with `logdenom >= 0` and
          `shift[y] <= 0`, so **the row loss is provably non-negative** and
          `ce.total` can never form `inf - inf` and can never produce a NaN.

    THE ONE STATED GAP. `shift[y]` CAN be `-inf` from FINITE logits, when
    `x_y` is near `-FLT_MAX` and `m` near `+FLT_MAX` and the subtraction
    overflows. Then `nll` is `+inf` and the loss is `+inf`. That is
    deterministic, the same on every vendor, and admitted. What is not
    reachable is a NaN, by (c). DEVIATION 1161.

    IGNORED ROWS, contract 7.3. `row[i] = +0.0`, STORED and not skipped, and
    `+0.0` rather than `-0.0` because L12's leaf accumulator is seeded `+0.0`
    and `acc + (+0.0) == acc` for every `acc` a `+0.0`-seeded chain can hold.
    So an ignored row is BITWISE INERT in the batch fold, which is the
    transformer contract's 7.1 theorem with a row where it had a masked key.
    It is NOT inert in `P` (the fold still has `N` terms) and it is NOT inert
    on the card. Sabotage `L_IGNORED_ROW_NEG_ZERO` must move `ce.row` and
    must NOT move `ce.total`.

    ROW INDEPENDENCE, contract 7.1. Nothing in the per-row block below reads
    `n`, the row chunk, a launch geometry or a vendor. A caller may split the
    rows into chunks of any size and concatenate. **Only L12 reads `n`**, and
    its fold length IS the batch, which is the same finding
    `gemm_backward_b_call` recorded about the weight gradient.
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

        # L1. The fold shape is free; this ascending loop is one legal
        # schedule of it and NOT part of the contract (5.1).
        var m = _row_max(logits, base, v)
        st.max_v.append(m)

        # L2 and L3, elementwise. Every operand flushed as loaded, every
        # result flushed as written (IDENTITY_PATHS row 10's checklist unit).
        for vv in range(v):
            var s = ftz(ftz(logits[base + vv]) - ftz(m))
            st.shift.append(s)
            st.expo.append(identical_exp(s))

        # L4. Routed. No fold here.
        var denom = ce_fold(st.expo, base, v, ones)
        st.denom.append(denom)

        # L5. By theorem (b) the argument is in [1, V].
        var logdenom = ftz(identical_log(ftz(denom)))
        st.logdenom.append(logdenom)

        # L6 and L7. An ignored row still records a log-probability and an
        # nll -- they are real numbers and they are on the card -- and only
        # `row` is forced to +0.0. Recording them costs nothing and it is
        # what makes `L_IGNORED_ROW_NEG_ZERO` localizable to `ce.row`.
        var ty = y
        if ignored:
            ty = 0  # a placeholder index; `row` below discards the result
        var lp_y = ftz(ftz(st.shift[base + ty]) - ftz(logdenom))
        st.logp_target.append(lp_y)
        var nll = neg_by_bits(lp_y)
        st.nll.append(nll)

        var row_loss = nll
        if smoothing:
            # L8. The same expression as L6 at every v, so a divergence at
            # `ce.logp` and not at `ce.logp_target` is a vocabulary-indexing
            # defect and nothing else.
            for vv in range(v):
                st.logp.append(
                    ftz(ftz(st.shift[base + vv]) - ftz(logdenom))
                )
            # L9. Routed. No fold here.
            var lpsum = ce_fold(st.logp, base, v, ones)
            st.logp_sum.append(lpsum)
            # L10. ONE division by V FIRST, then L11's product by eps. A
            # host-folded `eps/V` constant is a different answer and is
            # sabotage `L_SMOOTH_FOLDED_CONSTANT` (contract 6.2(a)).
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

    # L12. Routed. The one fold in this file whose `k` is a launch quantity.
    var total = ce_fold(st.row, 0, n, ones)
    st.total.append(total)

    # L13. One division, never a reciprocal multiplied in (contract 5.5,
    # transformer 5.4's paragraph at a second site). The divisor comes from
    # the ONE producer, which the backward calls too.
    var divisor = ce_divisor(cfg.reduction, st.count, cfg.num_items)
    st.divisor.append(divisor)
    st.loss.append(ftz(identical_div(ftz(total), divisor)))
    return st^


# ===========================================================================
# THE BACKWARD (contract section 6.4, seams L14 through L16)
# ===========================================================================


def ce_backward_oracle(
    mut st: CeStages, targets: List[Int32], cfg: CeConfig
) raises:
    """**THE NORMATIVE BACKWARD ANSWER**, `dLoss/dlogits`, filled into
    `st.weights` and `st.dlogits`.

    `st` must be the output of `ce_forward_oracle` on the same inputs and the
    same config. The backward reads `st.expo`, `st.denom` and `st.divisor`
    and recomputes nothing, which is deliberate -- **a second spelling of the
    softmax is a second thing that can be wrong**, and the gemm oracle's own
    header makes the same argument about its serial diagnostic.

      L14  w[v] = ftz(identical_div(e[v], denom))
      L15  t[v] = T_TARGET if v == y else T_OTHER      (host constants)
      L16  dl[v] = ftz(identical_div(ftz(ftz(w[v]) - t[v]), divisor))

    **L14 IS TRANSFORMER DEVIATION 806 AT A SECOND SITE.** ONE division per
    weight, never `e * (1/denom)`, which rounds twice where a division rounds
    once and differs in the last bit on ordinary inputs. The transformer
    contract pinned this for `attn.weights` and this lane's brief required
    consistency with it; the two must agree and sabotage
    `L_GRAD_RECIPROCAL_MUL` is the falsifier here.

    **`w = exp(logp)` IS REFUSED.** One more transcendental where a division
    suffices, and it is a different answer. Sabotage `L_W_VIA_EXP_LOGP`.

    **L16's DIVISION LIVES PER CELL, NOT PER ROW.** `d / divisor` rounds once
    where `d * (1/divisor)` rounds twice, and the same paragraph that pins
    L13 pins this. At `divisor == +1.0` (the plain SUM) it is bitwise inert
    at every value `d` can hold, and it is spelled anyway so there is one
    code path -- exactly the argument `ce_divisor` makes about returning
    `+1.0` instead of a flag.

    **THE GRADIENT IS THE ANALYTIC GRADIENT OF THE REAL-VALUED LOSS**, not
    the derivative of the floating-point expression the forward evaluated.
    Nobody computes the latter and this profile does not pretend to. Contract
    section 11.

    IGNORED ROWS write `V` copies of `+0.0`, STORED and not skipped (contract
    7.3). `+0.0` and not `-0.0` for the same reason the row loss is, and
    because a `-0.0` gradient cell would flow into whatever the optimizer
    does next.

    **REDUCTION_NONE HAS NO BACKWARD** and raises. A per-row upstream vector
    adds a second product per cell whose placement -- before or after L16's
    division -- is a real decision with two different answers, and this lane
    has no caller for it. An unused wrapper is an ungated one. Contract
    section 11.

    HOW THIS IS GATED BITWISE RATHER THAN APPROXIMATELY -- contract section
    12, and the summary is that FINITE DIFFERENCES ARE DEMOTED (DEVIATION
    1163). On a UNIFORM row at a power-of-two `V` with a power-of-two
    divisor, every quantity here is exactly representable: the shift is
    `+0.0`, every exponential is exactly `1.0`, the denominator is exactly
    `Float32(V)` in every fold order, `w` is exactly `1/V`, and the two
    gradient values are the dyadic rationals `(1/V - 1)/divisor` and
    `(1/V)/divisor`. **A person can write those down**, so the check asserts
    the BIT PATTERN and there is no epsilon anywhere in it. Contract 12.1
    also says what that family cannot do -- being exact, it separates NO
    spelling from any other, so run alone it would pass every sabotage in
    contract 10.1 and gate nothing about the arithmetic. It is the
    CORRECTNESS half and needs clause (a)'s per-stage bitwise comparison
    beside it.
    """
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
            # L14, one division per weight. Recorded even for an ignored row:
            # the softmax of an ignored row is a real number, the card
            # records it, and only the GRADIENT is forced to +0.0. That keeps
            # `L_IGNORED_ROW_NEG_ZERO`'s reach localized to one stage.
            var w = ftz(
                identical_div(ftz(st.expo[base + vv]), ftz(denom))
            )
            st.weights.append(w)
            if ignored:
                st.dlogits.append(Float32(0.0))
                continue
            # L15, an INTEGER compare. Integers do not flush anywhere.
            var t = t_other
            if vv == y:
                t = t_target
            # L16, one subtraction and one division.
            st.dlogits.append(
                ftz(identical_div(ftz(ftz(w) - ftz(t)), divisor))
            )


# ===========================================================================
# THE EXACT-ANALYTIC REFERENCES (contract section 12)
# ===========================================================================
# These two are NOT the oracle and they are NOT derived from it. They are the
# closed forms of contract 12.1 and 12.2, written down independently, and the
# EXACT-ANALYTIC arm asserts the oracle and the device both equal them BY
# BITS. `IDENTICAL_BACKWARD_PLAN.md`'s G1 makes the same move for its routing
# table -- "the expected table is written out a SECOND time by hand rather
# than derived from the code under test" -- and it is the only thing that
# makes an expected value evidence rather than a restatement.


def ce_exact_uniform_gradient(
    vocab: Int, target: Int, is_target: Bool, divisor: Float32
) raises -> Float32:
    """Contract 12.1's closed form, ONE cell.

    On a row whose `vocab` logits are all the SAME value, with `vocab` a
    power of two and `divisor` a power of two,

        dl[target]   = (1/vocab - 1) / divisor      EXACT
        dl[v != y]   = (1/vocab)     / divisor      EXACT

    both dyadic rationals. At `vocab = 4` and `divisor = 2` they are
    `-0.375` and `+0.125`.

    **THE ARITHMETIC BELOW IS PLAIN Mojo AND NOT THE PROFILE'S SEAMS, ON
    PURPOSE.** A reference written through `identical_div` and `ftz` would
    agree with the oracle because both went through the same helpers, which
    is a tautology rather than a check. Every value here is exact in Float32,
    so a plain `/` and a plain `-` are correctly rounded to the same bits on
    every column measured, and the comparison means something.

    REFUSES a non-power-of-two `vocab` or `divisor`, because the exactness
    argument depends on both and an unchecked exactness argument is how a
    gate comes to assert what the code does rather than what it should do
    (contract 12.4 guard 2).
    """
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
    """Contract 12.2's closed form, ONE cell.

    On a row with `high_count` copies of a value `a` and the rest at
    `a - 200.0`, with `high_count` a power of two, `portable_expf` returns
    exactly `+0.0` below `-87.33655`, so

        w = 1/high_count  at a high cell,  EXACTLY +0.0 at a low cell
        dl = (w - onehot) / divisor

    and every cell is again a dyadic rational. This family adds four things
    the uniform one lacks -- a genuine argmax structure, an exercised
    underflow edge, exact `+0.0` weights whose SIGN must be `+`, and a case
    where the TARGET is a low cell, where `dl[y]` is exactly
    `(-1) / divisor`.

    Plain arithmetic, for `ce_exact_uniform_gradient`'s reason.
    """
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
    """A positive normal Float32 whose mantissa field is zero. By BITS,
    because a compare-written test of a small value has one meaning on a
    column that flushes compare operands and another elsewhere (row 49)."""
    var b = rebind[UInt32](x.to_bits())
    if (b & CE_SIGN_BIT) != UInt32(0):
        return False
    var e = (b >> 23) & UInt32(0xFF)
    if e == UInt32(0) or e == UInt32(0xFF):
        return False
    return (b & UInt32(0x007FFFFF)) == UInt32(0)


# ===========================================================================
# THE Float64 TOLERANCE REFERENCE (contract 12.3 arm A3)
# ===========================================================================


def ce_forward_f64(
    logits: List[Float32], targets: List[Int32], cfg: CeConfig
) raises -> List[Float64]:
    """The per-row loss in double precision, plain `Float64` arithmetic, no
    pins and no partition. Returns `[N]`.

    **A TOLERANCE INSTRUMENT, NEVER A BITWISE ONE.** That sentence is
    `mamba/mojo_only/mamba_oracle.mojo`'s about its own Float64 arm and it
    means the same thing here. It exists so that a systematic error which
    `ce_exact_uniform_gradient`'s algebra and `ce_forward_oracle`'s seams
    SHARE would be visible -- neither of the other two arms can see a
    mistake they both make.

    It uses `std.math`'s `exp` and `log` deliberately, NOT
    `identical_exp64` / `identical_log64`, because the point is an
    INDEPENDENT number. A reference routed through this repository's own
    portable polynomials would be checking the polynomials against
    themselves.

    Label smoothing is included; the reduction is not, because a Float64
    reduction of Float32 row losses answers a question nobody asked. Compare
    ROWS.
    """
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
