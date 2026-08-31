# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Blocked right-looking FP32 Cholesky, and the profile it belongs to.

Profile `mojolearn.identical.cholesky.fp32.v1`. `A = L L^T` in place, lower
triangular, row-major, contiguous. cuSOLVER's `potrf` with `uplo = LOWER`,
except that there is no cuSOLVER source to transliterate (DEVIATION 1631) so
nothing here is a port and this file says so rather than citing a line
number it cannot have.

**NOT A PORT.** cuML and cuVS do not implement Cholesky. Every Cholesky in
either library is a cuSOLVER call -- `cuvs/src/neighbors/scann/detail/
scann_avq.cuh:179-200` (`potrf` then `potrs`) is the only factorization from
scratch in the two trees, and `cuml/src/solver/lars_impl.cuh:315-320` reaches
RAFT's rank-one UPDATE, which is itself three cuBLAS calls around a host
`std::sqrt`. cuSOLVER and cuBLAS are CLOSED; `VENDOR_LIBS.md`'s surviving
exception says call the platform equivalent because there is nothing to port,
and `IDENTITY_PATHS.md`'s opening rule says a mode has three moves. There is
no MAX `potrf` to call, so the move here is not REPLACE-with-a-vendor-call
and it is not REFUSE. It is: write the factorization with every numeric
decision named, which is what this file is.

The one thing in the RAPIDS trees that IS portable source and IS mirrored is
`raft/linalg/detail/cholesky_r1_update.cuh`; it lives under
`cholesky/derived/` and `cholesky/DERIVATION_MAP.tsv` records what it does and
does not cover.

# =========================================================================
# DEVIATION 1630: `NB` IS A NUMERIC PARAMETER, NOT A TUNING KNOB, AND UNDER
# `NUMERIC_IDENTICAL` IT IS PINNED FOR EVERY SHAPE AND A CALLER HINT IS
# REFUSED.
#
# This is the finding this lane exists to record, and it is the one a
# reviewer should look for first.
#
# THE ARGUMENT. A blocked right-looking Cholesky at block size `NB` computes
# cell `A[i][j]` of the trailing submatrix as
#
#     A[i][j] - sum over panels p of ( sum over k in panel p of L[i][k] L[j][k] )
#
# The INNER sums are folded per panel and the OUTER sum is a running
# subtraction, one per panel, applied in panel order. Change `NB` and the
# partition of `k` into panels changes, so the bracketing of that sum
# changes, so -- float addition not being associative -- the answer's last
# bits change. At `NB = 32` a cell of the last trailing block accumulates
# ceil(j/32) separate roundings of grouped partials; at `NB = 64` it
# accumulates ceil(j/64) of them, over different groups.
#
# The same argument in one sentence: **a block size in a blocked
# factorization is a summation order, and `original/numerics.mojo`'s
# classifying question -- does it change the SEQUENCE or the PRECISION of
# the arithmetic -- answers YES.** It is exactly IDENTITY_PATHS' "some
# scheduling decisions ARE numeric decisions" case, and it looks like a
# tuning knob for the same reason the histogram replication factor did.
#
# WHAT MAKES IT DIFFERENT FROM THE GEMM LANE'S LEAF SIZE. `gemm/`'s
# `contract_leaf_size(k)` is a pure function of `k`, so an identical GEMM at
# a given `k` has ONE partition on every vendor and at every shape. `NB` is
# NOT derived from `n`: it is a constant of the profile, so that a 48 x 48
# matrix and the leading 48 x 48 block of a 4096 x 4096 matrix factor
# through the same panel boundaries. Deriving `NB` from `n` would make the
# factor of a submatrix differ from the corresponding part of the factor of
# the whole, which is a property Gaussian-process and LARS-style incremental
# callers depend on.
#
# WHAT IS PINNED AND WHAT IS NOT:
#
#   NUMERIC, pinned under IDENTICAL   `CHOL_NB_PINNED`; the panel column
#                                     order (serial, ascending); the inner
#                                     sum order (k ascending); the fold that
#                                     computes the trailing update (the
#                                     gemm profile's balanced tree, and at
#                                     k = 32 <= CONTRACT_K_LEAF_MIN that is
#                                     one leaf, i.e. an ascending chain);
#                                     the accumulator width (float32, one
#                                     `fma` per term); the jitter.
#   SCHEDULING, free in BOTH modes    `panel_tpb`, `elem_tpb`, `solve_tpb`,
#                                     every grid shape, the allocation's
#                                     padding, the batch a matrix is
#                                     factored inside.
#
# `solve_tpb` is a PARAMETER and not a constant precisely so that the panel
# solve runs at more than one block width in the gates: at one block, an
# order that reads `block_idx.x` is indistinguishable from a pinned one, and
# `CHOL_SAB_PANEL_ROTATE` is the arm that would hide there.
#
# UNDER `NUMERIC_FAST` `NB` IS FREE, and `check_block_size_is_pinned` shows
# two values of it producing DIFFERENT BITS on the ill-conditioned fixture.
# That negative result is what makes the pin load-bearing rather than
# decorative: a pin nobody can show a difference for is a pin nobody has
# tested.
# =========================================================================

# =========================================================================
# DEVIATION 1634: LAPACK'S `info` CONTRACT, AND THE PIVOT DECISION ITSELF IS
# PINNED.
#
# `potrf_lower` returns `info`: 0 on success, or `k > 0` when the leading
# minor of order `k` is not positive definite. That is `dpotrf`'s contract
# and `gbdt/lapack/linear_system.mojo` already mirrors it for the host 6x6.
#
# WHAT IS NEW HERE AND IS THE WHOLE POINT: **failure is DATA-DEPENDENT, so
# a factorization that succeeds on one vendor and fails on another is the
# worst outcome this lane can produce** -- worse than differing bits,
# because one column returns a factor and the other returns an error, and
# no bitwise gate downstream of it ever runs. Two things therefore have to
# be identical and both are:
#
#   (a) THE VALUE COMPARED. `s = A[j][j] - sum_{k<j} L[j][k]^2`, folded
#       ascending through `identical_mul_add`, flushed through `ftz` at
#       every step. It is bit-identical on every column by the same
#       construction as every other pinned expression here.
#   (b) THE COMPARISON. `not (s > 0.0)` fails the pivot. Spelled that way
#       and not as `s <= 0.0` so that a NaN `s` FAILS rather than passing:
#       `NaN > 0` is false on every column, `NaN <= 0` is false on every
#       column. It is a compare, so no library and no rounding mode reaches
#       it, and both zeros fail it (`+0.0 > 0.0` and `-0.0 > 0.0` are both
#       false), which is what LAPACK's own `s <= 0` does too.
#
# THE `ftz` IN (a) IS LOAD-BEARING AND IS NOT DECORATION. A pivot that
# lands in the subnormal range is flushed to a signed zero, so it FAILS the
# test -- on every column, including the ones whose hardware keeps
# subnormals. Without it, Metal (which flushes in hardware) would refuse a
# matrix that CUDA (which does not) would factor, and the two columns would
# disagree about whether the input is positive definite at all. That is
# `FIX_DENORMAL_PIVOT` and `CHOL_SAB_NO_FTZ_PIVOT`; the sabotage is
# expected INERT on Apple and expected to FLIP the outcome on NVIDIA and
# AMD, and it is recorded that way rather than claimed as a passing gate on
# one box.
#
# THE HOST ROUND TRIP PER PANEL IS DELIBERATE. `info` is read back after
# every panel, and the driver stops. That is a control-plane decision of
# exactly the kind `HOST_AND_DEVICE.md` governs, it costs one drain per
# panel, and it is what both LAPACK and RAFT do -- `cholesky_r1_update.cuh:
# 105-108` copies two scalars to the host, takes the square root THERE, and
# copies one back, once per rank. Continuing past a failed pivot to save
# the drain would divide by a zero diagonal and fill the trailing block
# with infinities and NaNs, and a NaN in a recorded stage carries the
# VENDOR'S payload (IDENTITY_PATHS row 39, FACT 2), which would put a
# per-vendor bit pattern into a card that claims to be vendor-independent.
# =========================================================================

# =========================================================================
# DEVIATION 1636: THE TRAILING UPDATE IS `identical_gemm_into`, AND
# `linalg.matmul` IS REFUSED.
#
# `A22 -= L21 L21^T` is the only place in the factorization where a matrix
# product appears. It is computed by `gemm/original/gemm_identical.mojo::
# identical_gemm_into` at `OP_NT` -- profile `mojolearn.identical.gemm.
# fp32.v1` -- into a caller-owned workspace, followed by a pinned
# elementwise subtract over the LOWER triangle only.
#
# Not by a hand-written contraction here, because one already exists and is
# gated at 62 shapes across eight execution plans with six sabotages
# (`gemm/README.md`); a second contraction in this repository would be a
# second thing to get wrong, and this lane's README says so under WHAT THIS
# LANE REUSES RATHER THAN REWRITES.
#
# Not by `linalg.matmul` either, in either mode of this path, because its
# k-split is a per-vendor summation order and nothing in this repository
# can pin it, read it or check it (`core/gemm.mojo::gemm_tn`'s refusal
# carries the same argument at length). `CHOL_SAB_VENDOR_MATMUL` is the arm
# that swaps it in so the gate can be shown to see the difference.
#
# THE SYMMETRIC HALF IS COMPUTED AND THROWN AWAY. `L21 L21^T` is symmetric,
# so a `syrk` would compute half of it. This computes the whole `n_trail x
# n_trail` product and subtracts only the lower triangle. That is a
# measured-nothing speed cost and a real correctness gain: the alternative
# is a triangular GEMM shape that the gemm profile does not have, which
# would mean either a new profile arm or a hand-written contraction, and
# both are worse than doing twice the FLOPs in a lane that has never
# published a number.
# =========================================================================

# =========================================================================
# DEVIATION 1640: THE STRICT UPPER TRIANGLE OF THE FACTOR IS ZEROED.
#
# LAPACK leaves it untouched (`dpotrf` documents the other triangle as "not
# referenced"), and so does every cuSOLVER caller. This zeroes it, once, at
# the end, with `+0.0`.
#
# The reason is the card. `core/identity_trace.mojo` hashes a BUFFER, and a
# buffer whose upper triangle still holds the input's upper triangle hashes
# the INPUT there -- which is fine -- while a buffer whose upper triangle
# was never written at all hashes uninitialized memory, which differs run to
# run on ONE machine and would make the instrument report divergence
# everywhere (that file's own `record_device` docstring names the failure).
# Zeroing costs one elementwise pass and makes `chol.factor` a hash of the
# factor and of nothing else. It also makes `L L^T` checkable with a plain
# GEMM instead of a triangular one, which is what
# `check_potrf_reconstructs` does.
# =========================================================================

# =========================================================================
# DEVIATION 1637: THE JITTER IS PART OF THE PROFILE.
#
# `add_jitter` adds a constant to the diagonal. Under IDENTICAL the constant
# must be one of exactly two pinned values -- `CHOL_JITTER_NONE` (+0.0, the
# no-ridge case, which must stay expressible) and `CHOL_JITTER_PINNED`
# (2^-20) -- and any other value RAISES BY NAME.
#
# WHY A CALLER MUST NOT CHOOSE IT. Three lanes are going to call this
# (Gaussian processes, kernel ridge, GMM) and all three need a ridge,
# because an RBF Gram matrix in FP32 is numerically singular well before it
# is mathematically singular. If each picks its own, then "the same fit
# gives the same model" becomes "the same fit gives the same model provided
# every caller agrees about a number none of them writes down", and the
# claim stops being checkable. The jitter is an input to the arithmetic in
# exactly the way the block size is.
#
# WHY 2^-20 AND WHY A POWER OF TWO. Its float32 bits are `0x35800000` and
# they are stated here by hand -- no decimal string is parsed anywhere on
# the path, so `[[mojo-string-float-roundtrip]]` cannot reach it. A decimal
# ridge like `1e-6` would have to survive `String(Float32)` in a fixture, a
# log line or a Python binding, and that round trip is known broken in this
# toolchain. 2^-20 is about 9.54e-07, the same order every GP library
# defaults to, and adding it to a diagonal entry near 1.0 is exact.
#
# WHAT IS NOT PINNED: whether to jitter at all, and how many times. A caller
# that needs a bigger ridge applies the pinned one twice and says so; the
# ladder is 2^-20, 2^-19, 2^-18 ... and each rung is a fresh, statable
# input. A relative ridge is refused outright and
# `CHOL_SAB_JITTER_RELATIVE` is the arm that shows what it would cost.
# =========================================================================

# =========================================================================
# DEVIATION 1638: NON-FINITE AND NON-SYMMETRIC INPUT ARE REFUSED BY NAME,
# ON THE HOST, BEFORE ANY LAUNCH.
#
# LAPACK and cuSOLVER read one triangle and say nothing about the other, so
# a caller who builds a Gram matrix with a bug in its upper half gets a
# clean answer to a question it did not ask. `chol_validate_matrix` refuses
# instead, naming the cell and both values, and it refuses NaN and infinity
# the same way `kde/`'s DEVIATION 604 does and for the same reason: every
# stage here is a recorded card stage, and a computed NaN carries the
# VENDOR'S payload (Apple 0x7fc00000, NVIDIA 0x7fffffff, AMD 0xffc00000 --
# IDENTITY_PATHS row 39), so a NaN in a certified stage is three answers
# wearing one name.
#
# THE SYMMETRY TOLERANCE IS PINNED and is RELATIVE: cell `(i, j)` and cell
# `(j, i)` must differ by at most `CHOL_SYM_REL_TOL` (2^-20) times the
# larger of their magnitudes. A relative test, because a Gram matrix's
# entries span many orders of magnitude and an absolute tolerance is a
# statement about the data's scale rather than about its symmetry. Both
# tolerance and comparison are host-side and exact, so the refusal is the
# same refusal on every host.
#
# ONE THING THE SYMMETRY TEST CANNOT SEE, stated because it would otherwise
# look like a hole: `+0.0` at `(i, j)` against `-0.0` at `(j, i)` passes,
# since their difference is `+0.0`. It does not matter -- only the lower
# triangle is ever read, so the upper cell's zero sign reaches nothing.
# `FIX_SIGNED_ZERO` plants exactly that case and
# `check_signed_zero_and_denormal` asserts the factor's zero signs come out
# of the LOWER triangle's values.
# =========================================================================

# =========================================================================
# DEVIATION 1635: FLOAT32 ON THE DEVICE, FLOAT64 ONLY IN THE HOST ORACLE.
#
# cuSOLVER instantiates `potrf` for double and every RAPIDS caller offers
# both. Metal has no float64 (`mojolearn hardware limits`), so the device
# path here is float32 end to end and `cholesky/original/cholesky_oracle.
# mojo::reference_potrf_f64` is the only float64 in the lane. A float64
# request is refused by name at the host entry rather than silently
# narrowed.
#
# The consequence a Gaussian-process caller must know: FP32 Cholesky loses
# roughly half the digits of an FP64 one at the same conditioning, which is
# why DEVIATION 1637's ridge is not optional in practice. The oracle
# measures the gap per fixture and `check_potrf_vs_oracle` prints it.
# =========================================================================

# =========================================================================
# DEVIATION 1641: THE PANEL IS ONE BLOCK WITH A PINNED SERIAL COLUMN ORDER,
# AND NO FLOAT CROSSES A THREAD BOUNDARY IN IT.
#
# `panel_factor_kernel` runs on ONE block. Its outer loop over the panel's
# columns is SERIAL and ASCENDING; within a column, thread 0 computes the
# diagonal and every other element of the column is computed entirely by one
# thread, from its own ascending `fma` chain, out of values written before
# the preceding `barrier()`.
#
# So there is no cross-thread combination anywhere for a block size to
# reorder, exactly as `gemm/original/gemm_identical.mojo`'s structural fact
# 2 says of its tile plans, and launch invariance across `panel_tpb` is a
# property of the kernel's SHAPE rather than of a check that happens to
# pass. `panel_tpb` decides only WHICH thread computes which row of the
# column.
#
# It is deliberately the simplest correct shape and not a fast one. A real
# panel factorization stages the panel in threadgroup memory; that would be
# a page count to pin, and a second thing to pin is a second thing to get
# wrong. The price is stated in `cholesky/README.md` under WHAT IS OWED and
# no number is claimed, because nothing has been run.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.gemm import gemm_nt
from core.identity_trace import IdentityTrace
from cholesky.original.chol_sabotage import (
    CHOL_SAB_JITTER_RELATIVE,
    CHOL_SAB_NB_FROM_LAUNCH,
    CHOL_SAB_NONE,
    CHOL_SAB_VENDOR_MATMUL,
    chol_sabotage_is_kernel_arm,
    sabotage_jitter_diag_kernel,
    sabotage_logdet_kernel,
    sabotage_panel_factor_kernel,
    sabotage_trsm_panel_kernel,
)
from cholesky.original.trsm import CHOL_SOLVE_TPB, trsm_panel_kernel
from cholesky.derived.matrix.detail.matrix import (
    copy_vector_from_matrix_diagonal_kernel,
)
from gemm.original.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.original.gemm_oracle import OP_NT
from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_div,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)


#: The profile's name. Changing `CHOL_NB_PINNED`, the jitter, the panel
#: column order, the inner sum order or the trailing update's fold creates a
#: v2; it does not amend v1. Same discipline as
#: `mojolearn.identical.gemm.fp32.v1`.
comptime CHOL_PROFILE = "mojolearn.identical.cholesky.fp32.v1"

#: **NUMERIC. DEVIATION 1630.** The panel width, pinned for every shape under
#: IDENTICAL. 32 rather than 64 or 128 because at 32 the trailing update's
#: `k` is 32, which is at or below `gemm`'s `CONTRACT_K_LEAF_MIN` (128), so
#: `contract_leaf_size(32)` is 32 and the product folds as ONE leaf -- a
#: plain ascending chain with no tree at all. That makes the trailing
#: update's arithmetic statable in one sentence and makes the oracle's
#: replay of it a serial loop. A wider panel would put a fold tree inside the
#: update, which is legal under the gemm profile and correct, and is a v2
#: decision rather than a free one.
comptime CHOL_NB_PINNED = 32

#: SCHEDULING. Threads in the single panel block. Free in both modes.
comptime CHOL_PANEL_TPB = 128

#: SCHEDULING. Threads per block for the elementwise kernels. Free.
comptime CHOL_ELEM_TPB = 256

#: **NUMERIC. DEVIATION 1637.** The pinned ridge, `2^-20`. Written as its
#: float32 BITS and bitcast, never as a decimal string: `[[mojo-string-float
#: -roundtrip]]` says `String(Float32)` does not round trip in this
#: toolchain, and a profile constant that a log line cannot reproduce is a
#: profile constant nobody can check.
comptime CHOL_JITTER_BITS: UInt32 = 0x35800000

#: The no-ridge case, which has to stay expressible or a caller who does not
#: want a ridge is forced to pass an unpinned zero and get refused.
comptime CHOL_JITTER_NONE = Float32(0.0)

#: **NUMERIC. DEVIATION 1638.** The symmetry tolerance, `2^-20`, relative to
#: the larger magnitude of the two cells compared.
comptime CHOL_SYM_REL_TOL_BITS: UInt32 = 0x35800000


def chol_jitter_pinned() -> Float32:
    """`CHOL_JITTER_PINNED` as a value. A function rather than a `comptime`
    binding because the constant is defined by its BITS and the bitcast is
    the definition."""
    return bitcast[DType.float32](CHOL_JITTER_BITS)


def chol_sym_rel_tol() -> Float32:
    """`CHOL_SYM_REL_TOL` as a value. Same reason."""
    return bitcast[DType.float32](CHOL_SYM_REL_TOL_BITS)


@fieldwise_init
struct CholRun(Copyable, Movable):
    """What one `potrf_lower` actually did, READ BACK FROM THE RUN.

    `check_block_size_is_pinned` reads `nb` from here rather than assuming
    it, which is the whole difference between a gate and a comment: a driver
    that silently ignored the pin, or a sabotage that derived `nb` from the
    launch, changes this field and the check sees it.
    """

    var info: Int
    """LAPACK's `info`. 0 on success; `k > 0` means the leading minor of
    order `k` is not positive definite and the factorization stopped at
    column `k - 1`. DEVIATION 1634."""

    var nb: Int
    """The panel width that ACTUALLY ran."""

    var n_panels: Int
    """How many panels the driver walked before finishing or stopping.

    NOT a diagnostic either: at a given `n` it is a function of `nb` alone,
    so a run that reports the pinned `nb` and the wrong panel count is a run
    whose driver loop disagrees with its own partition.

    THE RIDGE IS NOT A FIELD HERE, on purpose. `potrf_lower` does not apply
    it -- `add_jitter` does, before the factorization -- so a `jitter` field
    on this struct could only ever restate a constant, which is a field that
    LOOKS like a read-back and is not one. `cholesky/estimator.mojo`'s
    `CholeskyFactor` carries the value that was actually added, because that
    is the level at which it was actually chosen."""


def chol_nb_for(n: Int, nb_hint: Int) raises -> Int:
    """**THE PIN. DEVIATION 1630.** The panel width this run will use.

    Under `NUMERIC_IDENTICAL` the answer is `CHOL_NB_PINNED` and a hint that
    asks for anything else RAISES BY NAME rather than being quietly ignored,
    because a silently ignored hint is how a caller comes to believe it
    tuned something. Under `NUMERIC_FAST` the hint is honored and the answer
    is whatever it asked for, clamped to `[1, n]`.

    A hint EQUAL to the pinned value is accepted in both modes, so a caller
    that wants to be explicit about running the profile can be, and so the
    default argument does not itself have to be mode-dependent.
    """
    if n <= 0:
        raise Error(
            "chol_nb_for: n must be positive, got " + String(n)
        )
    var nb = nb_hint
    if nb < 1:
        nb = 1
    if nb > n:
        nb = n
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        if nb_hint != CHOL_NB_PINNED:
            raise Error(
                "potrf_lower: NUMERIC_IDENTICAL refuses the block-size hint"
                " nb="
                + String(nb_hint)
                + ". NB is a NUMERIC parameter of profile "
                + CHOL_PROFILE
                + ", not a tuning knob: it partitions the k axis of the"
                " trailing update, so two values of it bracket the same"
                " sum differently and return two different factors."
                " IDENTITY_PATHS' rule has three moves and this one is"
                " PIN, so the pinned value "
                + String(CHOL_NB_PINNED)
                + " is used at every shape and a hint asking for another"
                " is refused rather than ignored. DEVIATION 1630. To"
                " explore block sizes, build NUMERIC_FAST and drop the"
                " cross-vendor claim; to change the profile's value,"
                " that is a v2 and not an argument."
            )
        return CHOL_NB_PINNED
    return nb


def chol_validate_jitter(jitter: Float32) raises:
    """**DEVIATION 1637.** Under IDENTICAL the ridge is one of two pinned
    values. Under FAST any finite non-negative value is accepted, because
    FAST makes no cross-vendor claim and a caller exploring conditioning
    needs the freedom."""
    if jitter != jitter:
        raise Error("add_jitter: jitter is NaN; refused by name")
    if jitter < Float32(0.0):
        raise Error(
            "add_jitter: jitter must be non-negative, got a negative value"
            " with bits 0x"
            + chol_hex32_bits(jitter)
            + "; a negative ridge subtracts from the diagonal and turns a"
            " positive-definite matrix indefinite"
        )
    var big = bitcast[DType.float32](UInt32(0x7F800000))
    if jitter == big:
        raise Error("add_jitter: jitter is +inf; refused by name")
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        var pinned = chol_jitter_pinned()
        var ok = bitcast[DType.uint32](jitter) == bitcast[DType.uint32](
            CHOL_JITTER_NONE
        ) or bitcast[DType.uint32](jitter) == bitcast[DType.uint32](pinned)
        if not ok:
            raise Error(
                "add_jitter: NUMERIC_IDENTICAL refuses the unpinned jitter"
                " 0x"
                + chol_hex32_bits(jitter)
                + ". The ridge is part of profile "
                + CHOL_PROFILE
                + " and not a caller's free choice: three lanes call this"
                " and if each picks its own value then 'the same fit gives"
                " the same model' depends on a number nobody writes down."
                " The two pinned values are 0x00000000 (no ridge) and"
                " 0x"
                + chol_hex32_bits(pinned)
                + " (2^-20). DEVIATION 1637. For a larger ridge, apply the"
                " pinned one more than once and record how many times; to"
                " change the value, that is a v2."
            )


def chol_hex32_bits(v: Float32) -> String:
    """Eight lowercase hex digits of a float32's bit pattern. Every error
    message here names a float by its BITS, never by `String(Float32)`."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def chol_validate_matrix(a: List[Float32], n: Int, what: String) raises:
    """**DEVIATION 1638.** Finite and symmetric, on the HOST, before any
    upload. Names the cell and both values by bits.

    Symmetry is relative: `|a_ij - a_ji| <= CHOL_SYM_REL_TOL * max(|a_ij|,
    |a_ji|)`. At `a_ij == a_ji == 0` that requires exact equality of the
    magnitudes, which both zeros satisfy -- see this file's DEVIATION 1638
    banner for why a signed-zero asymmetry is invisible here and why that is
    correct rather than a hole.
    """
    if n <= 0:
        raise Error(
            "cholesky: " + what + " must have a positive dimension, got n="
            + String(n)
        )
    if len(a) != n * n:
        raise Error(
            "cholesky: "
            + what
            + " holds "
            + String(len(a))
            + " floats, an "
            + String(n)
            + " x "
            + String(n)
            + " row-major matrix needs "
            + String(n * n)
        )
    for i in range(n):
        for j in range(n):
            var v = a[i * n + j]
            if v != v:
                raise Error(
                    "cholesky: "
                    + what
                    + " contains NaN at ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]; refused by name before any launch, because a NaN"
                    " carries the VENDOR's payload and every stage here is"
                    " a certified card stage (IDENTITY_PATHS row 39)"
                )
            if v > Float32(3.4028234663852886e38) or v < Float32(
                -3.4028234663852886e38
            ):
                raise Error(
                    "cholesky: "
                    + what
                    + " contains infinity at ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]; refused by name"
                )
    var tol = chol_sym_rel_tol()
    for i in range(n):
        for j in range(i):
            var lo = a[i * n + j]
            var up = a[j * n + i]
            var d = lo - up
            if d < Float32(0.0):
                d = -d
            var m = lo
            if m < Float32(0.0):
                m = -m
            var mu = up
            if mu < Float32(0.0):
                mu = -mu
            if mu > m:
                m = mu
            if d > tol * m:
                raise Error(
                    "cholesky: "
                    + what
                    + " is not symmetric at ["
                    + String(i)
                    + ", "
                    + String(j)
                    + "]: lower 0x"
                    + chol_hex32_bits(lo)
                    + " upper 0x"
                    + chol_hex32_bits(up)
                    + ", relative difference exceeds CHOL_SYM_REL_TOL"
                    " (2^-20). Refused by name rather than silently reading"
                    " only the lower triangle, which is what LAPACK and"
                    " cuSOLVER do and which answers a question the caller"
                    " did not ask. DEVIATION 1638"
                )


# ===========================================================================
# THE KERNELS
# ===========================================================================


def jitter_diag_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    jitter: Float32,
):
    """`A[i][i] += jitter`, one thread per diagonal entry. ABSOLUTE, never
    relative; DEVIATION 1637 and `CHOL_SAB_JITTER_RELATIVE`."""
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var d = ftz(a.unsafe_load(i * n + i))
    a.unsafe_store(i * n + i, ftz(d + jitter))


def panel_factor_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
):
    """The unblocked factorization of the `nb x nb` diagonal block at
    `[j0, j0+nb)^2`, in place. **ONE BLOCK.** DEVIATION 1641.

    Columns SERIAL and ASCENDING; within a column, thread 0 owns the
    diagonal and thread `t` owns rows `c+1+t, c+1+t+width, ...`. Each
    element's sum is one thread's own ascending `fma` chain over the panel's
    OWN columns `[j0, jc)` -- never over columns before `j0`, which the
    previous panels' trailing updates already subtracted.

    `info` is written once, by thread 0, with `jc + 1` (LAPACK's 1-based
    order of the failing leading minor); the shared flag then makes every
    thread of the block return at the same point, so no thread is left
    waiting at a barrier that nobody else reaches.
    """
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var nb = Int(nb_in)
    var tid = Int(thread_idx.x)
    var width = Int(block_dim.x)

    var flag = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    if tid == 0:
        flag[0] = Int32(0)
    barrier()

    for c in range(nb):
        var jc = j0 + c
        if tid == 0:
            # DEVIATION 1634 (a): the value compared.
            var s = ftz(a.unsafe_load(jc * n + jc))
            for k in range(j0, jc):
                var v = ftz(a.unsafe_load(jc * n + k))
                s = ftz(identical_mul_add(-v, v, s))
            # DEVIATION 1634 (b): the comparison. `not (s > 0)` and not
            # `s <= 0`, so a NaN fails; a subnormal `s` was flushed by the
            # `ftz` chain above and fails too, on EVERY column.
            if not (s > Float32(0.0)):
                info.unsafe_store(0, Int32(jc + 1))
                flag[0] = Int32(1)
            else:
                a.unsafe_store(jc * n + jc, ftz(identical_sqrt(s)))
        barrier()
        if flag[0] != Int32(0):
            return
        var ljj = ftz(a.unsafe_load(jc * n + jc))
        var i = c + 1 + tid
        while i < nb:
            var r = j0 + i
            var t = ftz(a.unsafe_load(r * n + jc))
            for k in range(j0, jc):
                var lrk = ftz(a.unsafe_load(r * n + k))
                var lck = ftz(a.unsafe_load(jc * n + k))
                t = ftz(identical_mul_add(-lrk, lck, t))
            a.unsafe_store(r * n + jc, ftz(identical_div(t, ljj)))
            i += width
        barrier()


def pack_panel_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
    n_trail_in: Int32,
):
    """Copy `L21` (rows `[j0+nb, n)`, columns `[j0, j0+nb)`, row stride `n`)
    into a CONTIGUOUS `n_trail x nb` block.

    A copy and nothing else: no arithmetic, so no rounding, so nothing to
    pin. It exists because `identical_gemm_into` takes contiguous operands
    and `L21` is a strided window of `a`; RAFT's own rank-one update does
    exactly this with `cublasCopy` and for exactly this reason
    (`cholesky_r1_update.cuh:66-70`).
    """
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var nb = Int(nb_in)
    var n_trail = Int(n_trail_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n_trail * nb:
        return
    var i = idx // nb
    var c = idx % nb
    dst.unsafe_store(idx, a.unsafe_load((j0 + nb + i) * n + j0 + c))


def subtract_lower_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    base_in: Int32,
    n_trail_in: Int32,
):
    """`A22[i][j] -= G[i][j]` for `j <= i` only. One thread per cell.

    The strict upper half of the trailing block is left alone: it still
    holds the input's upper triangle, which nothing reads and which
    DEVIATION 1640 zeroes at the end. Subtracting there too would be
    harmless and is not done, so that the set of cells this kernel writes is
    exactly the set the factorization reads.
    """
    var n = Int(n_in)
    var base = Int(base_in)
    var n_trail = Int(n_trail_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n_trail * n_trail:
        return
    var i = idx // n_trail
    var j = idx % n_trail
    if j > i:
        return
    var cur = ftz(a.unsafe_load((base + i) * n + base + j))
    var upd = ftz(g.unsafe_load(idx))
    a.unsafe_store((base + i) * n + base + j, ftz(cur - upd))


def zero_upper_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`+0.0` into every cell strictly above the diagonal. DEVIATION 1640.

    `+0.0` and not `-0.0`, stated because the sign of a written zero is a
    bit the card hashes: the upper triangle is not part of the factor, so
    its value is a CONVENTION, and a convention has to be one value on every
    column rather than whatever the input happened to hold.
    """
    var n = Int(n_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * n:
        return
    var i = idx // n
    var j = idx % n
    if j > i:
        a.unsafe_store(idx, Float32(0.0))


def logdet_kernel(
    diag: MutPointer[Float32, MutAnyOrigin],
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """`2 * sum_j log(diag[j])`, in ONE thread, ascending. DEVIATION 1639.

    `diag` is `diag(L)`, extracted by the ported RAFT kernel
    (`cholesky/derived/matrix/detail/matrix.mojo::
    copy_vector_from_matrix_diagonal_kernel`) and recorded as the card stage
    `chol.diag` before this runs. Reading the vector rather than striding the
    matrix is what gives the ported kernel a real caller, and it gives the
    card an intermediate: two columns that disagree here disagree in ONE
    diagonal entry, and a scalar hash cannot say which one.

    One thread rather than a block fold, for the same reason the trsm is one
    thread per column: a fold shape is a summation order, and a serial
    ascending chain over `n` terms is a pure function of `n` and of nothing
    else. `n` values is not enough work to be worth pinning a tree for, and
    a `pinned_block_sum` here would be a second fold shape in a lane that
    needs zero.

    Every `log` is `identical_log` (IDENTITY_PATHS row 12): a device `log`
    is a VENDOR CHOICE in its last bit, and this scalar multiplies straight
    into a GP marginal likelihood and a GMM's responsibilities, so a
    one-ulp vendor difference here moves every downstream number.

    The doubling is `identical_mul(2.0, acc)` rather than `2.0 * acc`.
    Multiplying by two is exact, so no bit depends on the spelling; it is
    written that way so no codegen can contract it into a neighbouring
    add (row 9), and so this file has ONE spelling of a product.
    """
    if Int(block_idx.x) != 0 or Int(thread_idx.x) != 0:
        return
    var n = Int(n_in)
    var acc = Float32(0.0)
    for j in range(n):
        acc = ftz(acc + ftz(identical_log(ftz(diag.unsafe_load(j)))))
    out_scalar.unsafe_store(0, ftz(identical_mul(Float32(2.0), acc)))


# ===========================================================================
# THE WORKSPACE
# ===========================================================================


def chol_workspace_floats(n: Int, nb: Int) -> Int:
    """Floats `potrf_lower` needs beside the matrix itself, at this shape.

    Layout, with `nt = n - nb` the LARGEST trailing block (panel 0's):

        [0, nt*nb)                 the packed `L21` operand
        [nt*nb, nt*nb + nt*nt)     the product `L21 L21^T`
        the rest                   `identical_gemm_into`'s own workspace

    The offsets are fixed at the LARGEST panel's sizes rather than recomputed
    per panel, so a later panel simply uses a prefix of each region.
    `identical_gemm_into`'s docstring is emphatic that sizing a workspace for
    one plan and letting the dispatcher pick another is an out-of-bounds
    write a small shape will not show you, so the gemm share is the MAXIMUM
    over every panel this `n` will walk, not the first panel's.

    `nt*nt` is a second `n^2` buffer and that is the honest cost of computing
    the whole symmetric product (DEVIATION 1636). Never less than 1, so the
    buffer is always constructible.
    """
    var nt = n - nb
    if nt < 0:
        nt = 0
    var gws = 0
    var j0 = 0
    while j0 < n:
        var w = nb
        if j0 + w > n:
            w = n - j0
        var t = n - j0 - w
        if t > 0:
            var c = identical_gemm_workspace_max_floats(t, t, w)
            if c > gws:
                gws = c
        j0 += nb
    var need = nt * nb + nt * nt + gws
    if need < 1:
        return 1
    return need


# ===========================================================================
# THE ENTRY POINTS
# ===========================================================================


def add_jitter(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    n: Int,
    jitter: Float32,
    elem_tpb: Int = CHOL_ELEM_TPB,
    sabotage: Int = CHOL_SAB_NONE,
) raises:
    """`A += jitter * I`, in place, on the device. DEVIATION 1637.

    ASYNCHRONOUS. Refuses an unpinned jitter under IDENTICAL BEFORE the
    launch, by name, through `chol_validate_jitter`. A jitter of `+0.0` is
    accepted and still launches: the kernel then writes `ftz(d + 0.0)` into
    every diagonal cell, which flushes a subnormal diagonal even in the
    no-ridge case, and that flush is part of the profile rather than a side
    effect (a diagonal that a column's hardware would keep and another's
    would flush is the FIX_DENORMAL_PIVOT divergence one step earlier).
    """
    chol_validate_jitter(jitter)
    if n <= 0:
        raise Error("add_jitter: n must be positive, got " + String(n))
    if len(a) < n * n:
        raise Error(
            "add_jitter: the matrix buffer holds "
            + String(len(a))
            + " floats, an n = "
            + String(n)
            + " matrix needs "
            + String(n * n)
        )
    var grid = (n + elem_tpb - 1) // elem_tpb
    if sabotage == CHOL_SAB_JITTER_RELATIVE:
        ctx.enqueue_function[sabotage_jitter_diag_kernel](
            a.unsafe_ptr(),
            Int32(n),
            jitter,
            Int32(sabotage),
            grid_dim=(grid, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
        return
    ctx.enqueue_function[jitter_diag_kernel](
        a.unsafe_ptr(),
        Int32(n),
        jitter,
        grid_dim=(grid, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )


def chol_panel_tag(prefix: String, p: Int, leaf: String) -> String:
    """`chol.panel003.factored` and its siblings. THREE digits, zero padded.

    `core/identity_trace.mojo` rule 2: a tag names a POSITION IN THE
    ALGORITHM and never a property of the machine. A panel index is a
    position; the number of panels is a function of `n` and `NB`, both of
    which are inputs. Rule 1's uniqueness invariant is what the padding is
    for -- the differ aligns two traces by their tag SEQUENCES, and
    `panel10` sorting beside `panel1` is how a reader misreads a diff.
    """
    var s = String(p)
    while s.byte_length() < 3:
        s = String("0") + s
    return prefix + ".panel" + s + "." + leaf


def potrf_lower(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    n: Int,
    mut trace: IdentityTrace,
    nb_hint: Int = CHOL_NB_PINNED,
    panel_tpb: Int = CHOL_PANEL_TPB,
    elem_tpb: Int = CHOL_ELEM_TPB,
    sabotage: Int = CHOL_SAB_NONE,
    solve_tpb: Int = CHOL_SOLVE_TPB,
) raises -> CholRun:
    """**THE FACTORIZATION.** `A = L L^T` in place, lower, row-major.

    cuSOLVER's `potrf(uplo = CUBLAS_FILL_MODE_LOWER)`, blocked right-looking,
    with every numeric decision named in this file's banners. Returns a
    `CholRun` carrying LAPACK's `info` and the panel width that ACTUALLY ran.

    On entry `a` holds the `n x n` symmetric matrix; the caller has already
    validated it (`chol_validate_matrix`) and applied any ridge
    (`add_jitter`), because both are host-side decisions and this is the
    device step. On exit the lower triangle holds `L`, the strict upper
    triangle holds `+0.0` (DEVIATION 1640) and `info` says whether `L` means
    anything.

    **`info != 0` LEAVES A PARTIAL FACTOR.** Columns `[0, info-1)` hold the
    finished columns of `L` and everything from column `info-1` on is
    whatever the trailing updates last wrote. That is LAPACK's contract, it
    is what `check_pivot_failure_is_identical` hashes, and it is why the
    host entry refuses to solve against a failed factor rather than
    returning nonsense.

    **THIS FORM SYNCHRONIZES**, once per panel, to read `info` back. See
    DEVIATION 1634 for why that round trip is deliberate rather than a debt.

    `ws` must hold at least `chol_workspace_floats(n, nb)` floats where `nb`
    is what `chol_nb_for` will return -- use the helper, not a guess, for the
    reason `identical_gemm_into`'s docstring gives about a workspace that had
    slack at a small shape and corrupted a large one.
    """
    if n <= 0:
        raise Error("potrf_lower: n must be positive, got " + String(n))
    if len(a) < n * n:
        raise Error(
            "potrf_lower: the matrix buffer holds "
            + String(len(a))
            + " floats, an n = "
            + String(n)
            + " matrix needs "
            + String(n * n)
        )
    var nb = chol_nb_for(n, nb_hint)
    if sabotage == CHOL_SAB_NB_FROM_LAUNCH:
        # ARM: the numerical parameter is derived from the launch. This is
        # the defect DEVIATION 1630 forbids, and `CholRun.nb` is how the
        # check sees it: the sabotaged run REPORTS the block size it used.
        nb = panel_tpb
        if nb > n:
            nb = n
        if nb < 1:
            nb = 1
    var need = chol_workspace_floats(n, nb)
    if len(ws) < need:
        raise Error(
            "potrf_lower: the workspace holds "
            + String(len(ws))
            + " floats, n = "
            + String(n)
            + " at nb = "
            + String(nb)
            + " needs "
            + String(need)
            + " (chol_workspace_floats). Sizing a workspace for one panel"
            " and letting a later one run past it is an out-of-bounds write"
            " that a small matrix will not show you"
        )

    var nt_max = n - nb
    if nt_max < 0:
        nt_max = 0
    var off_pack = 0
    var off_g = nt_max * nb
    var off_gws = off_g + nt_max * nt_max

    var dinfo = ctx.enqueue_create_buffer[DType.int32](1)
    var hinfo = ctx.enqueue_create_host_buffer[DType.int32](1)
    hinfo.unsafe_ptr().unsafe_store(0, Int32(0))
    ctx.enqueue_copy(dst_buf=dinfo, src_ptr=hinfo.unsafe_ptr())
    ctx.synchronize()

    var info = 0
    var p = 0
    var j0 = 0
    while j0 < n:
        var w = nb
        if j0 + w > n:
            w = n - j0
        var n_trail = n - j0 - w

        # ---- the panel ------------------------------------------------
        if chol_sabotage_is_kernel_arm(sabotage):
            ctx.enqueue_function[sabotage_panel_factor_kernel](
                a.unsafe_ptr(),
                dinfo.unsafe_ptr(),
                Int32(n),
                Int32(j0),
                Int32(w),
                Int32(sabotage),
                grid_dim=(1, 1, 1),
                block_dim=(panel_tpb, 1, 1),
            )
        else:
            ctx.enqueue_function[panel_factor_kernel](
                a.unsafe_ptr(),
                dinfo.unsafe_ptr(),
                Int32(n),
                Int32(j0),
                Int32(w),
                grid_dim=(1, 1, 1),
                block_dim=(panel_tpb, 1, 1),
            )
        trace.record_device(
            ctx, chol_panel_tag("chol", p, "factored"), a, n * n
        )

        # DEVIATION 1634: read `info` back and stop. One drain per panel.
        ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
        ctx.synchronize()
        info = Int(hinfo.unsafe_ptr().unsafe_load(0))
        if info != 0:
            p += 1
            break

        if n_trail > 0:
            # ---- the panel solve, L21 = A21 . L11^{-T} -----------------
            var solve_grid = (n_trail + solve_tpb - 1) // solve_tpb
            if chol_sabotage_is_kernel_arm(sabotage):
                ctx.enqueue_function[sabotage_trsm_panel_kernel](
                    a.unsafe_ptr(),
                    Int32(n),
                    Int32(j0),
                    Int32(w),
                    Int32(n_trail),
                    Int32(sabotage),
                    grid_dim=(solve_grid, 1, 1),
                    block_dim=(solve_tpb, 1, 1),
                )
            else:
                ctx.enqueue_function[trsm_panel_kernel](
                    a.unsafe_ptr(),
                    Int32(n),
                    Int32(j0),
                    Int32(w),
                    Int32(n_trail),
                    grid_dim=(solve_grid, 1, 1),
                    block_dim=(solve_tpb, 1, 1),
                )
            trace.record_device(
                ctx, chol_panel_tag("chol", p, "solved"), a, n * n
            )

            # ---- the trailing update, A22 -= L21 L21^T ------------------
            var packed = ws.create_sub_buffer[DType.float32](
                off_pack, n_trail * w
            )
            # `identical_gemm_into` receives `L21` as BOTH operands and Mojo
            # refuses one buffer passed twice to a launch, so the second is a
            # VIEW of the same memory -- the same move
            # `hierarchy/derived/cluster/detail/connectivities.mojo` makes for
            # `X` against itself.
            var packed_b = ws.create_sub_buffer[DType.float32](
                off_pack, n_trail * w
            )
            var g = ws.create_sub_buffer[DType.float32](
                off_g, n_trail * n_trail
            )
            var gws_len = len(ws) - off_gws
            if gws_len < 1:
                gws_len = 1
            var gws = ws.create_sub_buffer[DType.float32](off_gws, gws_len)

            var pack_cells = n_trail * w
            ctx.enqueue_function[pack_panel_kernel](
                packed.unsafe_ptr(),
                a.unsafe_ptr(),
                Int32(n),
                Int32(j0),
                Int32(w),
                Int32(n_trail),
                grid_dim=((pack_cells + elem_tpb - 1) // elem_tpb, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )

            if sabotage == CHOL_SAB_VENDOR_MATMUL:
                # ARM: `linalg.matmul` through `core/gemm.mojo::gemm_nt`. Its
                # k-split is a per-vendor summation order; DEVIATION 1636.
                gemm_nt(ctx, g, packed, packed_b, n_trail, n_trail, w)
            else:
                identical_gemm_into(
                    ctx, g, packed, packed_b, gws, n_trail, n_trail, w, OP_NT
                )

            var sub_cells = n_trail * n_trail
            ctx.enqueue_function[subtract_lower_kernel](
                a.unsafe_ptr(),
                g.unsafe_ptr(),
                Int32(n),
                Int32(j0 + w),
                Int32(n_trail),
                grid_dim=((sub_cells + elem_tpb - 1) // elem_tpb, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
            trace.record_device(
                ctx, chol_panel_tag("chol", p, "trailing"), a, n * n
            )
            ctx.synchronize()
            _ = packed^
            _ = packed_b^
            _ = g^
            _ = gws^

        p += 1
        j0 += nb

    if info == 0:
        var cells = n * n
        ctx.enqueue_function[zero_upper_kernel](
            a.unsafe_ptr(),
            Int32(n),
            grid_dim=((cells + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    trace.record_device(ctx, "chol.factor", a, n * n)
    var nb_record = List[Int32]()
    nb_record.append(Int32(nb))
    nb_record.append(Int32(p))
    nb_record.append(Int32(info))
    # The card carries the NUMERIC parameter that produced it. A card whose
    # partition is not in the card cannot be compared against another card
    # whose partition nobody wrote down.
    trace.record_list_i32("chol.nb", nb_record)
    ctx.synchronize()
    _ = dinfo^
    _ = hinfo^
    return CholRun(info, nb, p)


def chol_logdet(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut dwork: DeviceBuffer[DType.float32],
    n: Int,
    mut trace: IdentityTrace,
    elem_tpb: Int = CHOL_ELEM_TPB,
    sabotage: Int = CHOL_SAB_NONE,
) raises -> Float32:
    """**DEVIATION 1639.** `log |A| = 2 * sum_j log(L[j][j])`, on the device.

    Gaussian processes need it for the marginal likelihood, kernel ridge for
    its evidence and a Gaussian mixture for every responsibility, and all
    three will otherwise compute it themselves, three times, three ways.
    Each of those ways is a fold order and a `log`, which are exactly the two
    things IDENTITY_PATHS rows 21 and 12 are about, so this is the one place
    it is computed and this is why it is an entry point rather than a
    convenience.

    `dwork` must hold at least `n + 1` floats: `[0, n)` receives `diag(L)`
    and `[n]` receives the scalar. Two stages are recorded, `chol.diag` and
    `chol.logdet`.

    SYNCHRONIZES: the scalar is read back and returned.

    THE SIGN AND THE DOMAIN. Every `L[j][j]` is strictly positive -- the
    pivot test refused everything else -- so no `log` here sees a zero, a
    negative or a subnormal, and the `-inf` and NaN branches of
    `identical_log` are unreachable on a factor `potrf_lower` returned with
    `info == 0`. A factor from a FAILED run has whatever the trailing update
    last wrote on its diagonal from column `info-1` on; the host entry
    refuses to compute a determinant from one, and this device-level form
    trusts its caller.
    """
    if n <= 0:
        raise Error("chol_logdet: n must be positive, got " + String(n))
    if len(l) < n * n:
        raise Error(
            "chol_logdet: the factor buffer holds "
            + String(len(l))
            + " floats, an n = "
            + String(n)
            + " factor needs "
            + String(n * n)
        )
    if len(dwork) < n + 1:
        raise Error(
            "chol_logdet: the work buffer holds "
            + String(len(dwork))
            + " floats; it carries diag(L) in [0, n) and the scalar at [n],"
            " so n = "
            + String(n)
            + " needs "
            + String(n + 1)
        )
    var diag = dwork.create_sub_buffer[DType.float32](0, n)
    var scalar = dwork.create_sub_buffer[DType.float32](n, 1)
    ctx.enqueue_function[copy_vector_from_matrix_diagonal_kernel](
        diag.unsafe_ptr(),
        l.unsafe_ptr(),
        Int32(n),
        Int32(n),
        grid_dim=((n + elem_tpb - 1) // elem_tpb, 1, 1),
        block_dim=(elem_tpb, 1, 1),
    )
    trace.record_device(ctx, "chol.diag", diag, n)
    if sabotage == CHOL_SAB_NONE:
        ctx.enqueue_function[logdet_kernel](
            diag.unsafe_ptr(),
            scalar.unsafe_ptr(),
            Int32(n),
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
    else:
        ctx.enqueue_function[sabotage_logdet_kernel](
            diag.unsafe_ptr(),
            scalar.unsafe_ptr(),
            Int32(n),
            Int32(sabotage),
            grid_dim=(1, 1, 1),
            block_dim=(1, 1, 1),
        )
    trace.record_device(ctx, "chol.logdet", scalar, 1)
    var h = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=scalar)
    ctx.synchronize()
    var v = h.unsafe_ptr().unsafe_load(0)
    _ = h^
    _ = diag^
    _ = scalar^
    return v
