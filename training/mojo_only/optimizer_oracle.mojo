"""The IDENTICAL FP32 optimizer step, written out, on the host.

**NOTHING IN THIS FILE HAS EVER BEEN COMPILED OR EXECUTED.** It was
written against `training/IDENTICAL_OPTIMIZER_CONTRACT.md` and against the
repository's existing pinned primitives, and no compiler has seen it. What
is owed is listed at the bottom of that contract, section 16; the largest
single item is `training/mojo_only/optimizer_check.mojo`, which does not
exist, so **not one clause below has been falsified by a sabotage.**

**NOT A PORT, and it replaces no upstream call.** cuML, cuVS and RAFT
contain no optimizer -- they are inference and classical-ML libraries --
so there is no upstream in this repository's mirror set to check against
and this file is what stands in for one. PyTorch is the DESIGN reference
(`torch.optim.SGD`, `torch.optim.Adam`, `torch.optim.AdamW`,
`torch.nn.utils.clip_grad_norm_`) and it is **NOT a bit reference**, for
two reasons the contract measures rather than asserts -- section 5.2 (the
bias correction is not computed in float64 here) and section 5.3
(`1 - beta2` in Float32 differs from PyTorch's float64 route by about
1.3e-5 relative, which is the third significant decimal of the coefficient
that drives `v`).

**AND THE CITATIONS ARE FROM MEMORY.** There is no PyTorch checkout under
`/Users/andrewhendel/CascadeProjects/upstream/`, so every "the reference
spells it" sentence below violates `read-their-source-against-ours` and is
owed a source read (contract 16.5). Where the source turns out to disagree
with a clause, the CLAUSE stands and the citation gets corrected -- the
profile is normative here -- but the departure must then be renumbered as
deliberate rather than accidental.

WHAT THIS FILE IS FOR
---------------------
`training/IDENTICAL_OPTIMIZER_CONTRACT.md` is the contract in prose. This
is the same contract in code and the two must be read together; every
function cites its clause. Performance is irrelevant here -- it is scalar
Mojo on the host, single threaded, and it is meant to stay that way. Its
jobs are three.

1. **Be the definition.** The NORMATIVE answer of profile
   `mojolearn.identical.optimizer.fp32.v1` is `optimizer_step_oracle`.
   Not "close to"; the same bits.
2. **Be the instrument that proves a fixture separates.** Every contested
   decision in the contract has an adversarial spelling, and "does this
   fixture distinguish them" is a question this file answers rather than
   one the kernel is trusted about.
3. **Be the host half of the alignment predicate.**
   `microbatch_split_is_identical` computes clause 9.2 from `T` and `A`,
   so a trainer asks instead of guessing.

BUILT FROM THE DECLARED HELPERS, ON PURPOSE
--------------------------------------------
Every multiply is `mojo_only.numerics.identical_mul`, every multiply-add
is `identical_mul_add`, every division is `identical_div`, every square
root is `identical_sqrt` and every seam is `ftz`. Not local copies of
them; the actual helpers, so this file cannot drift into an independent
opinion about what IDENTICAL means. The consequence is the one
`gemm_oracle.mojo`'s header names -- under `NUMERIC_FAST` the helpers
compile away and THIS FILE IS NOT THE CONTRACT, it is the FAST spelling of
the same loops, and any check must print the mode it compiled in.

**AND EVERY REDUCTION IS DELEGATED.** The gradient-norm clip's sums of
squares are `gemm/mojo_only/gemm_oracle.mojo::gemm_oracle` calls at
`m = n = 1`, `k = N`, `OP_NT` (contract clause 3.2). Not a fold written
here. That inherits the v1 leaf partition, the v1 balanced tree, the
odd-tail carry, the no-padding clause, and a MEASUREMENT -- the v1 device
card was bit-identical on Apple M4, NVIDIA H100 and AMD MI325X at leg 11,
commit 144aa5b. A fold written fresh in this lane would inherit none of
that and would owe its own three-vendor leg.

NO DEVICE KERNEL LIVES HERE, DELIBERATELY. The device spelling is
`training/mojo_only/optimizer.mojo`. A host-only oracle also runs with no
GPU present, which is a real virtue in a reference.

OWED, AND NONE OF IT IS IN THIS LANE'S WRITE SET
------------------------------------------------
The full list is contract section 16. The four that matter most here.

1. `training/mojo_only/optimizer_check.mojo`, the gate. It does not
   exist, so **not one clause below has been falsified by a sabotage**
   and every "this separates" sentence is a prediction.
2. A `pixi.toml` task (`check-optimizer`) and the sabotage build arms.
3. A PyTorch source read. Every "the reference spells it" citation here
   is from memory (see above) and violates
   `read-their-source-against-ours`.
4. `IDENTITY_PATHS.md` rows for the optimizer step and the global-norm
   clip, and the correction of `IDENTICAL_BACKWARD_PLAN.md` 4.2, which
   still says "the composition and the operation ORDER are unwritten".

Also owed and smaller. A shared home for `refuse_nonfinite` (three copies
now exist in this tree), `PORTED_MAP.tsv` rows for `training/`, and the
adversarial corpus the fixture properties below require --
`adv_subnormal_square`, `adv_dead_unit_v`, `adv_dampening_first_step`,
`adv_param_order_five` and `adv_pow_step_1000`.

`[[mojo-string-float-roundtrip]]`: nothing here prints. A check that does
must print hex bits beside every decimal.
`[[mojo-list-not-implicitly-copyable]]`: every `List` return below
transfers with `^` and `_slice` copies element by element on purpose.
"""

from gemm.mojo_only.gemm_oracle import OP_NT, contract_leaf_size, gemm_oracle
from mojo_only.numerics import (
    ftz,
    identical_div,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


# ===========================================================================
# THE PROFILE CONSTANTS (contract sections 0.1 and 1)
# ===========================================================================

#: SGD, with optional momentum, dampening, Nesterov and COUPLED L2 decay.
comptime OPT_SGD = 0
#: Adam with COUPLED decay -- the decay is folded into the GRADIENT, so it
#: passes through `m` and `v` and is itself smoothed and normalized.
comptime OPT_ADAM = 1
#: AdamW with DECOUPLED decay -- the decay multiplies the PARAMETER and the
#: gradient is untouched. Contract 7.4; this is the ONLY difference between
#: the two algorithms and it is an ORDER, not a coefficient.
comptime OPT_ADAMW = 2

#: `torch.nn.utils.clip_grad_norm_`'s hardcoded `1e-6`, as Float32 bits.
#: A PROFILE constant and NOT a caller-supplied number. A caller who
#: could change it could change the bits of every parameter in the model
#: through one scalar. Contract 1 and 3.4a.
comptime CLIP_EPS_BITS: UInt32 = 0x358637BD

#: `1.0`, written as bits so the file has one spelling of it.
comptime ONE_BITS: UInt32 = 0x3F800000

#: The canonical quiet NaN pattern, recorded here and DELIBERATELY UNUSED.
#: No seam in this profile ever produces or returns a NaN -- contract 8a
#: refuses a non-finite input by name instead of propagating one, because
#: NaN payloads are vendor shaped (row 39 measured three payloads for one
#: IEEE answer) and a certified stage may not contain one. It is written
#: down so that a later reader who needs the pattern does not invent a
#: second spelling of it, and if a NaN-producing seam is ever added this
#: constant is where its canonicalization belongs.
comptime QNAN_BITS: UInt32 = 0x7FC00000

#: `+inf` bits, for the by-BITS finiteness test. Row 49: Metal flushes
#: COMPARE operands, so a compare-written finiteness test has two meanings
#: and a bit test has one.
comptime INF_BITS: UInt32 = 0x7F800000


def clip_eps() -> Float32:
    """The profile's `CLIP_EPS`, from its bit pattern rather than from a
    decimal literal (`[[mojo-string-float-roundtrip]]`)."""
    from std.memory import bitcast

    return bitcast[DType.float32](CLIP_EPS_BITS)


# ===========================================================================
# CLAUSE 8a: THE REFUSAL (contract section 8)
# ===========================================================================


def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """Row 39. A NaN or infinity in a gradient, a parameter or a state is
    REFUSED BY NAME before any recorded stage.

    NaN PAYLOADS are vendor shaped -- row 39 measured three payloads for
    one IEEE answer -- so a certified card may never contain a computed
    NaN, and a cross-vendor gate would otherwise have to compare NaN cells
    as "is NaN" rather than by bits, which is a hole in a bitwise claim.

    Tested BY BITS, not by compares. Metal flushes COMPARE operands (row
    49, measured 2026-08-23), so a compare-written test means one thing on
    Metal and another on CUDA. `bitcast` and an integer mask mean one
    thing everywhere.

    **THE COST IS REAL AND THE CONTRACT STATES IT.** Production training
    loops SKIP a step whose gradient norm is non-finite and that behavior
    is useful. Under this profile it is not arithmetic, it is a CONTROL
    decision, and it belongs outside the pinned region as an explicit
    recorded branch (a `skipped` flag in the card, with the step index) so
    two runs can be compared on whether they skipped the same steps.
    Folding it in as a `select` would put a NaN inside a certified stage.

    THIS IS THE THIRD COPY OF THIS PREDICATE IN THE TREE. The others are
    `mamba/mojo_only/mamba_oracle.mojo:57` and its own callers. Three
    copies of one predicate have three chances to drift, which is
    `identical_mul`'s complaint about `pinned_mul` verbatim. The canonical
    home is `mojo_only/numerics.mojo`, which is outside this lane's write
    set -- contract 16.7."""
    from std.memory import bitcast

    for i in range(len(values)):
        var au = bitcast[DType.uint32](values[i]) & UInt32(0x7FFFFFFF)
        if au > INF_BITS:
            raise Error(
                String("optimizer: NaN in ")
                + name
                + String(" at flat index ")
                + String(i)
                + String(" REFUSED (row 39: NaN payloads are vendor-shaped")
                + String("; no stage may record one)")
            )
        if au == INF_BITS:
            raise Error(
                String("optimizer: infinity in ")
                + name
                + String(" at flat index ")
                + String(i)
                + String(" REFUSED (row 39)")
            )


def refuse_nonfinite_scalar(name: String, v: Float32) raises:
    """`refuse_nonfinite` for one value. The total gradient norm goes
    through this before it reaches the clamp, so contract 8c's claim that
    the profile's single compare-select sees only finite non-negative
    input is established rather than assumed."""
    var one = List[Float32]()
    one.append(v)
    refuse_nonfinite(name, one)


# ===========================================================================
# CLAUSE 5.1: `beta^t` AS A PURE FUNCTION OF THE INTEGER t
# ===========================================================================


def pow_int_f32(base: Float32, t: Int) -> Float32:
    """DEVIATION 1171. `base^t` by LSB-first binary exponentiation over
    `identical_mul`, with `ftz` at every step. Contract clause 5.1.

    **THIS IS THE CLAUSE THAT MAKES STEP-COUNT INVARIANCE STRUCTURAL**,
    and that is why it is not a running product.

    Three properties.

    1. It uses only the profile's own arithmetic, so the same function
       evaluated on a DEVICE returns the same bits. No float64, so
       `mojolearn-hardware-limits` is satisfied and so is the host FTZ
       hazard the contract's 5.2 names (x86 and arm honor denormals by
       default, but FTZ and DAZ are mode bits a linked library can set,
       and `beta2^t` spends thousands of steps in the subnormal band).
    2. It is NOT a general `pow`. `portable_powf` is `exp(p * log(x))`
       through two Cephes polynomials and is not exact even at `t = 1`.
       `t` is an integer and integer powers need no transcendental.
    3. **It is a pure function of `t`.** There is no running `beta_pow`
       state to checkpoint, nothing to reconstruct on resume, and no way
       for a resumed run to disagree with a continuous one about what
       `beta1^t` is. The gate then verifies what the construction
       promises rather than being the only thing standing between the
       profile and a silent drift.

    What it refuses is `b_t = b_{t-1} * beta`, which is what an
    implementation reaches for because it is one multiply per step. That
    is `t - 1` sequential roundings and it is a different number.

    PREDICTED, DERIVED OFF-REPOSITORY IN HOST DOUBLE PRECISION, NOT
    MEASURED, and the gate must re-derive all of it. With the profile's
    flush, this function and the running product FIRST DIFFER AT `t = 7`,
    for `beta = 0.9` and `beta = 0.999` alike. **`t = 1` through `t = 6`
    agree exactly, so a sabotage arm that runs to `t = 4` is VACUOUS**,
    and that is precisely where a hand-written fixture stops.

    Also predicted, `0.9^t` reaches exactly `+0.0` at `t = 829` and
    `0.999^t` at `t = 87,295`, after which `bc` is exactly `1.0` and stays
    there. That is the correct Float32 answer -- `0.9^829` is below the
    smallest normal -- and it is a pure function of `t` on every column,
    so it is admitted rather than special-cased. The monotonicity argument
    is that `b` in the loop below is `base^(2^j)`, and once that flushes
    to zero every `t` with that bit set has `base^t <= base^(2^j)`, which
    is also below the flush threshold, so there is no `t` at which the
    early zero is wrong.

    `[[mojo-amp-plus-is-bitwise-and]]`: the bit test below is `e & 1` and
    the shift is `e >> 1`. `&+` is bitwise AND in Mojo with no compile
    error and appears nowhere in this file.

    `t <= 0` returns `1.0`, which is the empty product. The profile's `t`
    is one-based and a caller passing 0 is a bug the caller should see as
    `bc == 0.0` and a division by zero at `step_size`, not as a silently
    plausible number.
    """
    from std.memory import bitcast

    var acc = bitcast[DType.float32](ONE_BITS)
    var b = base
    var e = t
    while e > 0:
        if (e & 1) != 0:
            acc = ftz(identical_mul(acc, b))
        b = ftz(identical_mul(b, b))
        e = e >> 1
    return acc


# ===========================================================================
# THE CONFIGURATION AND THE PER-STEP HOST SCALARS (contract 1 and 7.1)
# ===========================================================================


@fieldwise_init
struct OptimizerConfig(Copyable, Movable):
    """One parameter group's configuration. Every field is `Float32` and
    **a run's card records each one as an eight-hex-digit bit pattern**,
    not as a decimal string (contract section 1). Two runs are comparable
    only when those patterns are equal.

    `max_norm <= 0.0` means the gradient-norm clip is OFF, and that is the
    configuration contract 11(c)'s parameter-count-invariance gate must
    run under -- with clipping ON, one parameter's update depends on every
    other parameter in the model, by the reference's own semantics and not
    by a defect (contract 3.5)."""

    var kind: Int
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var momentum: Float32
    var dampening: Float32
    var nesterov: Bool
    var max_norm: Float32


@fieldwise_init
struct StepScalars(Copyable, Movable):
    """The host scalars of contract 7.1, computed ONCE per step per
    parameter group.

    **A KERNEL MAY NOT RECOMPUTE THESE.** They WOULD agree if it did,
    because every step of `step_scalars` is a pinned primitive -- but a
    per-element recomputation is where a `powf` creeps in, it is
    `O(log t)` work per element instead of per step, and having exactly
    one producer is what makes "the same scalars reached every element" a
    structural fact rather than a check. `OPT_SAB_SCALARS_PER_ELEMENT` is
    the REACH probe for that ban and its bit result is EXPECTED to be
    inert; contract section 12's last row says so in advance, because an
    arm whose predicted answer is "no bits move" is only worth having when
    it is labelled that way before it runs."""

    var b1t: Float32
    var b2t: Float32
    var bc1: Float32
    var bc2: Float32
    var step_size: Float32
    var rt_bc2: Float32
    var c1: Float32
    var c2: Float32
    var decay_mul: Float32
    var neg_lr: Float32
    var c_damp: Float32


def step_scalars(cfg: OptimizerConfig, t: Int) -> StepScalars:
    """Contract 7.1, line for line. Seams H1 through H6.

    `c1 = ftz(1.0 - beta1)` and `c2 = ftz(1.0 - beta2)` are **Float32
    subtractions from the Float32 hyperparameter**, contract clause 5.3,
    and that is a deliberate departure from PyTorch with a measured cost.
    PREDICTED, DERIVED OFF-REPOSITORY, NOT MEASURED:

        1 - 0.9    here 0x3DCCCCD0   PyTorch's float64 route 0x3DCCCCCD
        1 - 0.999  here 0x3A831200   PyTorch's float64 route 0x3A83126F
        1 - 0.99   here 0x3C23D700   PyTorch's float64 route 0x3C23D70A

    At `beta2 = 0.999` those differ by about **1.3e-5 relative**, which is
    not a last-bit difference. It is the third significant decimal of the
    coefficient that drives `v`. It is written here rather than left in
    the contract alone because a reader who sees `1.0 - beta2` below will
    assume it is the obvious thing, and the obvious thing is a choice.

    `step_size` carries `1 / bc1` and `rt_bc2` carries `sqrt(bc2)`, so the
    bias correction lands on the STEP SIZE and on the DENOMINATOR rather
    than on `m` and `v` -- contract clause 7.2c, the implementation shape
    rather than the documented-pseudocode shape, separated by
    `OPT_SAB_MHAT_FORM`.

    `decay_mul` is AdamW's `1 - lr*wd` and is computed even for SGD and
    Adam, where it is unused. Computing it unconditionally costs two host
    flops per step and removes a branch from the one function every mode
    goes through."""
    from std.memory import bitcast

    var one = bitcast[DType.float32](ONE_BITS)

    var b1t = pow_int_f32(cfg.beta1, t)
    var b2t = pow_int_f32(cfg.beta2, t)
    var bc1 = ftz(one - b1t)
    var bc2 = ftz(one - b2t)
    var step_size = ftz(identical_div(cfg.lr, bc1))
    var rt_bc2 = ftz(identical_sqrt(bc2))
    var c1 = ftz(one - cfg.beta1)
    var c2 = ftz(one - cfg.beta2)
    var decay_mul = ftz(one - ftz(identical_mul(cfg.lr, cfg.weight_decay)))
    # Negation is EXACT in IEEE-754 -- a sign-bit flip, no rounding, no
    # seam. It is written as its own scalar so that the per-element `fma`
    # at O14 and S5 has a ready operand and so that no kernel spells a
    # subtraction where the contract spells a fused multiply-add.
    var neg_lr = -cfg.lr
    var c_damp = ftz(one - cfg.dampening)

    return StepScalars(
        b1t,
        b2t,
        bc1,
        bc2,
        step_size,
        rt_bc2,
        c1,
        c2,
        decay_mul,
        neg_lr,
        c_damp,
    )


# ===========================================================================
# CLAUSE 9.2: THE MICROBATCH ALIGNMENT PREDICATE
# ===========================================================================


def microbatch_split_is_identical(t_tokens: Int, a: Int) -> Bool:
    """Contract clause 9.2. **True when accumulating `a` microbatches over
    `t_tokens` tokens reproduces the unsplit gradient BIT FOR BIT.**

    The optimizer step imposes no alignment requirement on its own
    arithmetic -- it is elementwise. It imposes one on whoever produced
    the gradient, and this predicate is that requirement in a form a
    trainer can ask rather than guess.

    The four structural conditions, and the fifth is on the CALLER.

    1. `contract_leaf_size(t_tokens / a) == contract_leaf_size(t_tokens)`.
       The v1 leaf rule holds `L` at 128 for every `k` in `(128, 131072]`,
       so this says the full token count and the microbatch size land in
       the same band of the rule.
    2. `t_tokens mod L == 0`, so there is no ragged last leaf and
       `P = t_tokens / L` exactly.
    3. `a` divides `P`.
    4. `a` is a power of two.
    5. NOT CHECKABLE HERE, and it is the one a caller gets wrong. The
       cross-microbatch combination must be the v1 BALANCED TREE over the
       `a` pieces in ascending microbatch index, `ftz(ftz(x) + ftz(y))` at
       every node -- **not a running serial sum.**

    Conditions 3 and 4 together are what make each microbatch's leaf range
    a COMPLETE SUBTREE. With `a` a power of two and `P` divisible by `a`,
    the `a` piece roots are exactly the nodes of level `log2(a)` of the
    unsplit tree, and the tree over them is the remainder of the unsplit
    tree.

    THE MEASURED EVIDENCE THIS GENERALIZES, and what it cannot supply.
    `IDENTICAL_BACKWARD_PLAN.md` 5.2, gate G5, 2026-08-25: `T = 512` split
    256/256 and `T = 384` split 256/128 each moved 0 of 35 gradient cells,
    host and device agreeing, while 150/150 of 300, 200/312 of 512 and
    192/192 of 384 moved 31, 30 and 31. **Every aligned case measured is
    `a = 2`, and over TWO pieces a serial running sum and a balanced tree
    are the same operation**, so condition 5 is NOT established by that
    measurement and a reader who concludes "the accumulator just has to be
    the flushed add" is over-reading it. At `a = 4` they separate --- the
    tree computes `ftz(ftz(p0+p1) + ftz(p2+p3))` and a running sum
    computes `ftz(ftz(ftz(p0+p1) + p2) + p3)`. `T = 512, a = 4` is the
    fixture that closes it and it is inside the range
    `gemm_device_check.mojo` already sweeps. Sabotage
    `OPT_SAB_MICROBATCH_SERIAL`.

    A caller for whom this returns False is not blocked -- the run is
    still deterministic on one machine. What it loses is the freedom to
    change `a`. **The microbatch count is then part of the run's numerical
    specification and a reproducibility claim must name it.**
    """
    if a <= 0 or t_tokens <= 0:
        return False
    if t_tokens % a != 0:
        return False
    var leaf_full = contract_leaf_size(t_tokens)
    var leaf_piece = contract_leaf_size(t_tokens // a)
    if leaf_full != leaf_piece:
        return False
    if leaf_full <= 0:
        return False
    if t_tokens % leaf_full != 0:
        return False
    var p_count = t_tokens // leaf_full
    if p_count % a != 0:
        return False
    # `a` a power of two. Written as a loop rather than as `a & (a - 1)`
    # because the mask spelling is one character from `&+`, which is
    # bitwise AND in Mojo with no compile error
    # (`[[mojo-amp-plus-is-bitwise-and]]`), and because this runs once.
    var q = a
    while q > 1:
        if q % 2 != 0:
            return False
        q = q // 2
    return True


# ===========================================================================
# CLAUSE 3: THE GRADIENT-NORM CLIP
# ===========================================================================


def _slice(xs: List[Float32], begin: Int, count: Int) -> List[Float32]:
    """One tensor's elements as their own `List`, because `gemm_oracle`
    takes a whole operand. `O(N)` per tensor per step and this is a host
    oracle where that is free.

    `[[mojo-list-not-implicitly-copyable]]`: `var a = b` on a
    `List[Float32]` fails with "value of type 'List[Float32]' cannot be
    implicitly copied", so this builds element by element and transfers
    the result. The device spelling in `optimizer.mojo` needs no such
    copy -- it hands the kernel an OFFSET into the flat buffer -- and that
    asymmetry is the price of the oracle being scalar and simple."""
    var out = List[Float32]()
    for i in range(count):
        out.append(xs[begin + i])
    return out^


def clip_tensor_sumsq_oracle(
    grads: List[Float32], begin: Int, count: Int
) -> Float32:
    """DEVIATION 1178, contract clause 3.2. **The sum of squares of one
    parameter tensor IS a v1 GEMM call**, at `m = n = 1`, `k = N`,
    `OP_NT`, with the gradient as both operands.

    Not a fold written here. Not `pinned_block_sum`. Not a `sum` helper.

    At `m = n = 1` and `OP_NT` the GEMM's own accessors reduce to `a[p]`
    and `b[p]` -- `_a_at(a, OP_NT, 0, p, 1, N)` is `a[0*N + p]` and
    `_b_at(b, OP_NT, p, 0, 1, N)` is `b[0*N + p]` -- so this really is the
    sum of squares and not a shape that happens to be near one.

    Three things are inherited entire and they are the whole reason for
    the clause.

    1. The partition is `contract_leaf_size(N)` and NOTHING else -- not
       the block count, not the vendor, not the occupancy, not how many
       other tensors are in the launch. gemm contract section 6.
    2. The fold is v1's fixed balanced tree over ADJACENT leaves with the
       odd tail CARRIED, including the no-padding and no-stride-pairing
       clauses and their fixtures F7, F8 and F9. gemm contract 7.2.
    3. **A MEASUREMENT.** The v1 device card was bit-identical Apple M4
       against NVIDIA H100 against AMD MI325X at leg 11, commit 144aa5b,
       60 stages each. A fold written fresh here would inherit nothing.

    What would make a sabotage of this clause pass while gating nothing --
    any fixture whose tensors all have `N <= 128`. There `P == 1`, the
    tree has no arithmetic node, and the v1 answer IS the serial ascending
    chain, so a hand-written serial fold passes. **One tensor must have
    `N > 128`, and one should have `N = 300`** -- `P = 3` with a
    44-element ragged last leaf, the gemm lane's own ragged fixture.
    Sabotage `OPT_SAB_CLIP_SERIAL_FOLD`.

    `N == 0` returns `+0.0`; `gemm_oracle` at `k == 0` writes the `+0.0`
    its section 8 requires rather than skipping the cell."""
    var g = _slice(grads, begin, count)
    var out = gemm_oracle(g, g, OP_NT, 1, 1, count)
    if len(out) == 0:
        return Float32(0.0)
    return out[0]


def clip_coefficient(total_norm: Float32, max_norm: Float32) raises -> Float32:
    """Contract clauses 3.4a and 3.4b. Seams C6 and C7.

        denom = ftz(total_norm + CLIP_EPS)
        coef  = ftz(identical_div(max_norm, denom))
        coef  = coef if coef < 1.0 else 1.0

    The division is a TRUE divide through `identical_div`, never a
    reciprocal followed by a multiply (contract 4c). A reciprocal-multiply
    is EXACT when the denominator is a power of two, so a fixture of
    power-of-two denominators cannot see that sabotage -- plant
    denominators that are not. Sabotage `OPT_SAB_RECIP_MUL`.

    **This is the ONLY compare-select in the whole profile** (contract
    8c), and it is on a value `refuse_nonfinite_scalar` has already
    established is finite. `total_norm` is a square root and is therefore
    non-negative, `CLIP_EPS` is positive, so `denom` is positive and
    `coef` is non-negative. Row 13's selection hazard -- `-0.0` and `+0.0`
    comparing equal, order deciding which survives -- has no reachable
    site here, and that statement is what makes the `amsgrad` exclusion
    load bearing, since `max(v_max, v)` would create the first one."""
    from std.memory import bitcast

    refuse_nonfinite_scalar(String("clip.total_norm"), total_norm)
    var one = bitcast[DType.float32](ONE_BITS)
    var denom = ftz(total_norm + clip_eps())
    var coef = ftz(identical_div(max_norm, denom))
    if coef < one:
        return coef
    return one


def clip_grad_norm_oracle(
    mut grads: List[Float32],
    offsets: List[Int],
    max_norm: Float32,
    mut sumsq_out: List[Float32],
    mut norm_out: List[Float32],
    mut total_out: List[Float32],
) raises -> Float32:
    """DEVIATIONS 1178, 1179, 1180. The whole clip pass, contract section 3.

    `offsets` has length `J + 1` and `offsets[j] .. offsets[j+1]` is
    tensor `j`'s slice of the flat `grads` buffer. **`j` IS the
    `param_id`**, and the ascending order of `j` is the cross-tensor
    summation order.

    THE TWO-LEVEL SHAPE IS THE REFERENCE'S AND IT IS LOAD BEARING.
    `torch.nn.utils.clip_grad_norm_` does not take one flat sum of squares
    over the model. It takes a per-tensor L2 norm, stacks those into a
    vector, and takes the L2 norm of THAT. In exact arithmetic the two are
    the same number; **in Float32 they are not**, because each per-tensor
    `sqrt` rounds and each result is then squared again inside the outer
    norm. The profile takes the reference's shape -- `COPY, DO NOT
    IMPROVE` at a place where the improvement is one line and tempting.
    Sabotage `OPT_SAB_CLIP_FLAT_NORM`; a fixture with `J == 1` cannot see
    it, because squaring a rounded square root usually lands back in the
    same binade, so **at least three tensors of different lengths with
    norms several binades apart.**

    **`param_id` IS PART OF THE PROFILE** (contract 3.3). It is assigned
    once when the model is registered, it is written into the checkpoint,
    and it is not a dictionary iteration order, not the order the
    optimizer happened to receive the tensors in, not a device pointer
    order and not a name sort that changes when a layer is renamed. This
    is the clause most likely to be waved past, and the reason is that it
    looks like bookkeeping. It is not. It is the cross-tensor summation
    order. Sabotage `OPT_SAB_CLIP_PARAM_ORDER`; **a fixture with `J == 2`
    cannot see it at all**, because reversing two elements swaps the two
    children of one tree node and `a + b` equals `b + a` bitwise. `J >= 3`
    is required and `J = 5` is better -- gemm contract 7.2.2 notes `P = 5`
    is the smallest `P` that carries an odd tail twice.

    THE RESCALE IS UNCONDITIONAL (contract 3.4c) and a "skip when no
    clipping is needed" optimization is FORBIDDEN. `identical_mul(1.0, x)`
    is `fma(1.0, x, -0.0)`, which returns `x` exactly for every finite `x`
    including both signed zeros -- so for a NORMAL gradient the multiply
    and the skip agree. They do NOT agree for a SUBNORMAL gradient,
    because the multiply's operand and result flushes turn it into a
    signed zero and the skip leaves the original bit pattern in the
    buffer.

    And the honest half, which belongs at the clause and not in a
    footnote. **The difference is CARD VISIBLE and DOWNSTREAM INERT.** The
    `clip.grad` stage hash differs between the two spellings; the
    optimizer's own first act is `ftz` on the gradient load, so by the
    time the value reaches `m` and `v` the two have converged. This clause
    protects the card and any OTHER consumer of the gradient buffer -- a
    logger, a second optimizer, a gradient-statistics pass -- and it does
    not protect the parameter update. Sabotage
    `OPT_SAB_CLIP_SKIP_AT_ONE`, and it must compare `clip.grad`, not
    `param.out`, on a fixture with a SUBNORMAL gradient cell.

    Returns the clamped coefficient. `sumsq_out`, `norm_out` and
    `total_out` are filled with the card's `clip.sumsq`, `clip.norm` and
    `[clip.total_sumsq, clip.total_norm, clip.coef]` stages.
    """
    var j_count = len(offsets) - 1
    if j_count <= 0:
        return Float32(0.0)

    # C1 through C3, per tensor, in ASCENDING param_id.
    for j in range(j_count):
        var begin = offsets[j]
        var count = offsets[j + 1] - begin
        var s = ftz(clip_tensor_sumsq_oracle(grads, begin, count))
        sumsq_out.append(s)
        norm_out.append(ftz(identical_sqrt(s)))

    # C4 and C5, the OUTER norm, over the vector of per-tensor norms.
    # The same delegation as C2: one v1 GEMM at m = n = 1, k = J.
    var norms_copy = _slice(norm_out, 0, len(norm_out))
    var tot = gemm_oracle(norms_copy, norms_copy, OP_NT, 1, 1, j_count)
    var total_sumsq = Float32(0.0)
    if len(tot) > 0:
        total_sumsq = ftz(tot[0])
    var total_norm = ftz(identical_sqrt(total_sumsq))

    var coef = clip_coefficient(total_norm, max_norm)

    total_out.append(total_sumsq)
    total_out.append(total_norm)
    total_out.append(coef)

    # C8. UNCONDITIONAL, every element of every tensor.
    for i in range(len(grads)):
        grads[i] = ftz(identical_mul(coef, ftz(grads[i])))

    return coef


# ===========================================================================
# CLAUSE 7.2: THE ADAM AND ADAMW ELEMENT
# ===========================================================================


@fieldwise_init
struct AdamElement(Copyable, Movable):
    """One element's results, including the two intermediates the card
    records (`adam.denom` and `adam.q`). They are in the card because they
    are the cheapest early-divergence addresses in the update -- a run
    that agrees at `adam.v` and disagrees at `adam.denom` has a section-4
    problem, and one that disagrees at `adam.q` has a section-4c
    problem."""

    var p: Float32
    var m: Float32
    var v: Float32
    var denom: Float32
    var q: Float32


def adam_element_oracle(
    p_in: Float32,
    g_in: Float32,
    m_in: Float32,
    v_in: Float32,
    cfg: OptimizerConfig,
    sc: StepScalars,
) -> AdamElement:
    """Contract 7.2, seams O1 through O14, in order. DEVIATIONS 1172,
    1174, 1175, 1176.

    `g_in` is ALREADY CLIPPED. Clipping is a separate pass over the whole
    model (section 3) and cannot be an elementwise decision, because its
    coefficient is a function of every gradient there is.

    THE FOUR CONTESTED DECISIONS INSIDE THIS SEQUENCE, each named so
    nobody has to derive which line is a choice.

    **7.2a, the moment recurrences are a PRODUCT then an FMA**, mirroring
    `exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)` -- two roundings,
    in that order. The alternative is `lerp`, `m + c1*(g - m)`, which
    recent PyTorch uses in some paths and which is a different number.
    Sabotage `OPT_SAB_MOMENT_LERP`.

    **7.2b, the `v` term associates as `c2 * (g * g)`**, forming the
    square first. `addcmul_(grad, grad, value = 1 - beta2)` does not say
    which way it associates and `(c2 * g) * g` is the other reading.
    Sabotage `OPT_SAB_SQ_ASSOC`; a fixture of round `g` values will not
    separate them, so plant hashed `g`.

    **7.2c, the bias correction lands on the DENOMINATOR and the STEP
    SIZE**, not on `m` and `v`. O11 divides by `rt_bc2` and `step_size`
    already carries `1/bc1`. The documented-pseudocode alternative forms
    `m_hat` and `v_hat` explicitly and then
    `p -= lr * m_hat / (sqrt(v_hat) + eps)`, which is one more rounding
    and a different number. Sabotage `OPT_SAB_MHAT_FORM`.

    **7.2d, the update at O14 is ONE FUSED ROUNDING**,
    `fma(-step_size, q, p)`, not a rounded product then a rounded
    subtract. Only fusion has a portable spelling (gemm contract section
    4). *What makes a fusion sabotage inert is the most important
    non-vacuity note in this lane*: `check-ieee-arith` scored Metal as
    UNFUSED over 2^20 HASHED patterns and the verdict was WRONG, because
    **zero of those patterns separate a fused `a*b + c` from an unfused
    one** -- random exponents put the product and the addend so far apart
    that both spellings round identically. The fixture must be BUILT to
    separate, which means `step_size * q` and `p` must be within a few
    binades of each other with the product's tail nonzero. A random
    fixture reports INERT and the report is false.

    AND ONE DECISION THAT LOOKS LIKE A NO-OP AND IS NOT. The weight-decay
    branch is CONDITIONAL on `wd != 0.0`, matching the reference, and
    `fma(0.0, p, g)` is **not** bitwise `g` in every case -- at `g` equal
    to `-0.0` it gives `0*p + (-0.0)`, which is `+0.0` for positive `p`.
    So the branch is not a free optimization and the arms differ on a
    negative-zero gradient. It is written as a branch because the
    reference has one; the alternative would be a v2.

    Contract 4d, the `eps` position. It is added OUTSIDE the square root,
    at O12, to the bias-corrected denominator. Sabotage
    `OPT_SAB_EPS_INSIDE_SQRT`, and it is INERT unless the fixture plants
    `v` in the 1e-20 to 1e-12 band -- with `eps = 1e-8`, `eps^2` is 1e-16,
    and an ordinary gradient gives `v` around 1e-4 where the two spellings
    agree to the last bit.

    Contract 6, seam O7: `g` at 1e-25 is a perfectly ordinary normal
    `Float32` and `g*g` at 1e-50 is not representable as a normal at all.
    Flush it and `v` picks up exactly `c2 * 0` on every column; carry it
    and Metal disagrees with CUDA from that step onward FOREVER, because
    `v` is a running state. Sabotage `OPT_SAB_FTZ_LATE`, inert on any
    fixture whose gradients are within a few binades of 1.0.
    """
    var g = ftz(g_in)  # O1
    var p = ftz(p_in)  # O2
    var mp = ftz(m_in)  # O3
    var vp = ftz(v_in)  # O3

    if cfg.weight_decay != Float32(0.0):
        if cfg.kind == OPT_ADAMW:
            # O4b. DECOUPLED: the PARAMETER is multiplied and the gradient
            # is untouched. A PRODUCT (`p * (1 - lr*wd)`), matching
            # `param.mul_(1 - lr * weight_decay)`. The additive reading
            # `p - lr*wd*p` is the other spelling in circulation and is a
            # different number; sabotage `OPT_SAB_DECAY_ADD_FORM`.
            p = ftz(identical_mul(sc.decay_mul, p))
        else:
            # O4a. COUPLED: the decay enters the GRADIENT, so it passes
            # through `m` and `v` and is itself smoothed and normalized.
            # ONE fused rounding.
            g = ftz(identical_mul_add(cfg.weight_decay, p, g))

    var ms = ftz(identical_mul(cfg.beta1, mp))  # O5, PRODUCT
    var m = ftz(identical_mul_add(sc.c1, g, ms))  # O6, FUSED
    var g2 = ftz(identical_mul(g, g))  # O7, PRODUCT
    var vs = ftz(identical_mul(cfg.beta2, vp))  # O8, PRODUCT
    var v = ftz(identical_mul_add(sc.c2, g2, vs))  # O9, FUSED

    var s = ftz(identical_sqrt(v))  # O10
    var sd = ftz(identical_div(s, sc.rt_bc2))  # O11
    var dn = ftz(sd + cfg.eps)  # O12, eps OUTSIDE the sqrt
    var q = ftz(identical_div(m, dn))  # O13, a TRUE divide
    var p_out = ftz(identical_mul_add(-sc.step_size, q, p))  # O14, FUSED

    return AdamElement(p_out, m, v, dn, q)


# ===========================================================================
# CLAUSE 7.3: THE SGD ELEMENT
# ===========================================================================


@fieldwise_init
struct SgdElement(Copyable, Movable):
    """One element's results. `direction` is the value actually stepped
    along and is in the card as `sgd.dir`, because with Nesterov on it is
    neither the gradient nor the buffer and a run that diverges there has
    a clause-7.3c problem rather than a clause-7.3a one."""

    var p: Float32
    var buf: Float32
    var direction: Float32


def sgd_element_oracle(
    p_in: Float32,
    g_in: Float32,
    buf_in: Float32,
    buf_initialized: Bool,
    cfg: OptimizerConfig,
    sc: StepScalars,
) -> SgdElement:
    """Contract 7.3, seams S1 through S5. DEVIATION 1177.

    **7.3a, THE FIRST MOMENTUM STEP IS A COPY.** `b_1 = g`, not
    `c_damp * g`. That is `buf = grad.clone()` and it means the dampening
    factor is not applied on the step that CREATES the buffer.

    *What makes a sabotage of this inert*: at `dampening = 0.0`, `c_damp`
    is exactly `1.0` and `identical_mul(1.0, g)` returns `g` for every
    finite `g`, **so the default configuration cannot see this clause at
    all.** The fixture must set `dampening != 0` and must compare at
    `t = 1`, then at `t = 2` and beyond to show the divergence persists
    rather than washing out. Sabotage `OPT_SAB_MOMENTUM_FIRST_STEP`.

    **7.3b, `buf_initialized` IS CHECKPOINT STATE.** A resume that
    reinitializes it recomputes `b` as a copy of the current gradient
    instead of continuing the recurrence, and the two runs diverge at that
    step and never reconverge. This is the SGD half of contract 11(d) and
    it is the half a checkpoint format forgets, because the flag is a
    `Bool` and everything around it is a tensor. Sabotage
    `OPT_SAB_RESUME_REINIT`.

    **7.3c, Nesterov is `g + momentum * b`, in that operand order**,
    matching `grad.add(buf, alpha = momentum)`. The transposed reading
    `b + momentum * g` is a different number. Sabotage
    `OPT_SAB_NESTEROV_ORDER`, and note it is ALSO inert at `t = 1`, where
    `b == g` makes the two readings the same expression.

    The coupled weight decay at S3 is the same seam as Adam's O4a and
    carries the same negative-zero caveat -- `fma(0.0, p, g)` is not
    bitwise `g` when `g` is `-0.0`, so the `wd != 0` branch is not a free
    optimization.
    """
    var g = ftz(g_in)  # S1
    var p = ftz(p_in)  # S2

    if cfg.weight_decay != Float32(0.0):
        g = ftz(identical_mul_add(cfg.weight_decay, p, g))  # S3, FUSED

    var b = ftz(buf_in)
    if cfg.momentum != Float32(0.0):
        if not buf_initialized:
            # 7.3a. A COPY. No arithmetic, no seam of its own -- it
            # inherits the flush `g` already carries.
            b = g
        else:
            var bs = ftz(identical_mul(cfg.momentum, b))  # PRODUCT
            b = ftz(identical_mul_add(sc.c_damp, g, bs))  # FUSED
        if cfg.nesterov:
            g = ftz(identical_mul_add(cfg.momentum, b, g))  # 7.3c, FUSED
        else:
            # A COPY, no arithmetic.
            g = b

    var p_out = ftz(identical_mul_add(sc.neg_lr, g, p))  # S5, FUSED
    return SgdElement(p_out, b, g)


# ===========================================================================
# THE CARD, AND THE NORMATIVE STEP
# ===========================================================================


struct OptimizerStages(Movable):
    """Every recorded stage of one step, in the card's order (contract
    section 10). Flat buffers indexed the way `param`, `grad`, `m` and `v`
    are, with `offsets` separating the tensors.

    The one-element `sched_*` stages are in the card ON PURPOSE. They are
    the cheapest possible early-divergence address -- if a cross-vendor
    run differs at `sched_pow2` the problem is contract section 5 and not
    a single one of the millions of elementwise updates downstream of it.
    Hashing a one-element buffer costs nothing and skipping it costs a
    debugging session."""

    var clip_sumsq: List[Float32]  # [J]
    var clip_norm: List[Float32]  # [J]
    var clip_total: List[Float32]  # [3] total_sumsq, total_norm, coef
    var clip_grad: List[Float32]  # [sum N_j], the rescaled gradient
    var sched: List[Float32]  # [7] see `sched_field_name`
    var adam_m: List[Float32]  # [sum N_j]
    var adam_v: List[Float32]
    var adam_denom: List[Float32]
    var adam_q: List[Float32]
    var sgd_buf: List[Float32]
    var sgd_dir: List[Float32]
    var param_out: List[Float32]

    def __init__(out self):
        self.clip_sumsq = List[Float32]()
        self.clip_norm = List[Float32]()
        self.clip_total = List[Float32]()
        self.clip_grad = List[Float32]()
        self.sched = List[Float32]()
        self.adam_m = List[Float32]()
        self.adam_v = List[Float32]()
        self.adam_denom = List[Float32]()
        self.adam_q = List[Float32]()
        self.sgd_buf = List[Float32]()
        self.sgd_dir = List[Float32]()
        self.param_out = List[Float32]()


def sched_field_name(i: Int) -> String:
    """The `sched` stage's field order, so a card reader does not have to
    count. Matches the order `optimizer_step_oracle` appends in."""
    if i == 0:
        return String("sched.pow1")
    if i == 1:
        return String("sched.pow2")
    if i == 2:
        return String("sched.bc1")
    if i == 3:
        return String("sched.bc2")
    if i == 4:
        return String("sched.step_size")
    if i == 5:
        return String("sched.rt_bc2")
    if i == 6:
        return String("sched.decay_mul")
    return String("sched.?")


def optimizer_step_oracle(
    mut param: List[Float32],
    mut grad: List[Float32],
    mut m_state: List[Float32],
    mut v_state: List[Float32],
    mut buf_initialized: List[Bool],
    offsets: List[Int],
    cfg: OptimizerConfig,
    t: Int,
) raises -> OptimizerStages:
    """**THE NORMATIVE ANSWER of `mojolearn.identical.optimizer.fp32.v1`.**

    One step, in order --- refuse, clip, host scalars, then the elementwise
    update. `param`, `grad`, `m_state` and `v_state` are flat buffers with
    `offsets[j] .. offsets[j+1]` being tensor `j`; `j` IS the `param_id`
    and its ascending order is the cross-tensor summation order of clause
    3.3. `buf_initialized` has length `J` and is SGD's per-tensor flag
    from clause 7.3b.

    For SGD, `m_state` carries the momentum buffer and `v_state` is
    unused. One state pair serves both algorithms so that a checkpoint has
    one shape.

    `t` is ONE-BASED. The first step of a run is `t = 1`.

    THE ORDER OF THE FOUR PHASES IS PART OF THE CONTRACT.

    1. **Refuse** (8a). Non-finite input is refused BY NAME and BY BITS
       before any recorded stage.
    2. **Clip** (section 3), over the WHOLE model, before anything
       elementwise. Its coefficient is a function of every gradient there
       is, which is why it cannot be an elementwise decision and why
       contract 3.5 says that with clipping ON a parameter's update is
       NOT independent of the rest of the model. That is the reference's
       semantics, not a defect, and 11(c)'s parameter-count-invariance
       gate must therefore run with `max_norm <= 0`.
    3. **Host scalars** (7.1), ONCE. Not per element.
    4. **The elementwise update** (7.2 or 7.3).

    Weight decay lands in phase 4 and WHERE it lands is the entire
    difference between Adam and AdamW (7.4). **At `weight_decay = 0.0` the
    two algorithms are the same arithmetic**, so a fixture at the
    reference's own default cannot distinguish them. That is the single
    most likely vacuous gate in this lane. Sabotage
    `OPT_SAB_ADAMW_AS_ADAM`.

    This function has never been compiled. What it is least confident
    about is listed in the contract's section 16.
    """
    var stages = OptimizerStages()
    var j_count = len(offsets) - 1
    if j_count <= 0:
        return stages^

    # ---- PHASE 1: refuse (contract 8a) --------------------------------
    refuse_nonfinite(String("input.param"), param)
    refuse_nonfinite(String("input.grad"), grad)
    refuse_nonfinite(String("state.m"), m_state)
    refuse_nonfinite(String("state.v"), v_state)

    # ---- PHASE 2: the global norm clip (contract section 3) -----------
    # `max_norm <= 0.0` means OFF. The gradient buffer is then untouched
    # and the `clip.*` stages are EMPTY rather than filled with a coef of
    # 1.0, because an empty stage and a stage that says "no clipping
    # happened" are different claims and a card should make the
    # difference visible.
    if cfg.max_norm > Float32(0.0):
        # Local lists, then transferred into the card. Passing a STRUCT
        # FIELD as a `mut` argument would be the shorter spelling and is
        # the kind of place-expression borrow this file cannot check
        # without a compiler, so it takes the spelling that needs no
        # judgement call. `[[mojo-list-not-implicitly-copyable]]`: the
        # transfers below are `^` and not assignments.
        var sq = List[Float32]()
        var nm = List[Float32]()
        var tt = List[Float32]()
        _ = clip_grad_norm_oracle(grad, offsets, cfg.max_norm, sq, nm, tt)
        for i in range(len(grad)):
            stages.clip_grad.append(grad[i])
        stages.clip_sumsq = sq^
        stages.clip_norm = nm^
        stages.clip_total = tt^

    # ---- PHASE 3: the host scalars, ONCE (contract 7.1) ---------------
    var sc = step_scalars(cfg, t)
    stages.sched.append(sc.b1t)
    stages.sched.append(sc.b2t)
    stages.sched.append(sc.bc1)
    stages.sched.append(sc.bc2)
    stages.sched.append(sc.step_size)
    stages.sched.append(sc.rt_bc2)
    stages.sched.append(sc.decay_mul)

    # ---- PHASE 4: the elementwise update ------------------------------
    var n_total = len(param)
    if cfg.kind == OPT_SGD:
        for j in range(j_count):
            var begin = offsets[j]
            var end = offsets[j + 1]
            var was_init = buf_initialized[j]
            for i in range(begin, end):
                var e = sgd_element_oracle(
                    param[i], grad[i], m_state[i], was_init, cfg, sc
                )
                param[i] = e.p
                m_state[i] = e.buf
                stages.sgd_buf.append(e.buf)
                stages.sgd_dir.append(e.direction)
                stages.param_out.append(e.p)
            # 7.3b. The flag flips ONCE, AFTER the whole tensor, and it is
            # per TENSOR rather than per element -- a per-element flag
            # would let the first element of a tensor take the copy arm
            # and the second take the recurrence arm on the SAME step.
            if cfg.momentum != Float32(0.0):
                buf_initialized[j] = True
    else:
        for i in range(n_total):
            var e = adam_element_oracle(
                param[i], grad[i], m_state[i], v_state[i], cfg, sc
            )
            param[i] = e.p
            m_state[i] = e.m
            v_state[i] = e.v
            stages.adam_m.append(e.m)
            stages.adam_v.append(e.v)
            stages.adam_denom.append(e.denom)
            stages.adam_q.append(e.q)
            stages.param_out.append(e.p)

    return stages^
