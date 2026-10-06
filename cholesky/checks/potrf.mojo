# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Blocked right-looking FP32 Cholesky, and the profile it belongs to.

Profile `mojolearn.identical.cholesky.fp32.v1`. `A = L L^T` in place, lower
triangular, row-major, contiguous. cuSOLVER's `potrf` with `uplo = LOWER`,
except that there is no cuSOLVER source to read (DEVIATION 1631), so
nothing here cites a reference line and this file says so rather than citing a line
number it cannot have.

**NO REFERENCE FILE.** cuML and cuVS do not implement Cholesky. Every Cholesky in
either library is a cuSOLVER call -- `cuvs/src/neighbors/scann/detail/
scann_avq.cuh:179-200` (`potrf` then `potrs`) is the only factorization from
scratch in the two trees, and `cuml/src/solver/lars_impl.cuh:315-320` reaches
RAFT's rank-one UPDATE, which is itself three cuBLAS calls around a host
`std::sqrt`. cuSOLVER and cuBLAS are CLOSED; `archive/reference/VENDOR_LIBS.md`'s surviving
exception says call the platform equivalent because there is no source to read,
and `IDENTITY_PATHS.md`'s opening rule says a mode has three moves. There is
no MAX `potrf` to call, so the move here is not REPLACE-with-a-vendor-call
and it is not REFUSE. It is: write the factorization with every numeric
decision named, which is what this file is.

The one thing in the RAPIDS trees that IS portable source and IS implemented here is
`raft/linalg/detail/cholesky_r1_update.cuh`; it lives under
`cholesky/impl/`.

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
# factorization is a summation order, and `checks/numerics.mojo`'s
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
# exactly the kind `archive/reference/HOST_AND_DEVICE.md` governs, it costs one drain per
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
# product appears. It is computed by `gemm/checks/gemm_identical.mojo::
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
# path here is float32 end to end and `cholesky/checks/cholesky_oracle.
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
# reorder, exactly as `gemm/checks/gemm_identical.mojo`'s structural fact
# 2 says of its tile plans, and launch invariance across `panel_tpb` is a
# property of the kernel's SHAPE rather than of a check that happens to
# pass. `panel_tpb` decides only WHICH thread computes which row of the
# column.
#
# It is deliberately the simplest correct shape and not a fast one. A real
# panel factorization stages the panel in threadgroup memory; that would be
# a page count to pin, and a second thing to pin is a second thing to get
# wrong. The price is stated in `cholesky/README.md` under WHAT IS OWED and
# no number is claimed for the staged shape, because nobody has measured it
# against this one.
# =========================================================================
"""

from cholesky.checks.fast_trsm import FTP_MAX_NB, FTS_BLOCK, fast_trsm_panel_kernel, fast_gemm_nt_sub_lower, fast_panel_solve_inv, try_chol_shared_left
from cholesky.checks.chol_fast_tall import CHOL_FAST_TALL, CTL_NB, chol_fast_tall_panel
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.primitives.warp import shuffle_xor
from max.gpu.primitives.block import prefix_sum
from std.bit import pop_count
from std.memory import bitcast, stack_allocation
from std.time import perf_counter_ns
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.sys import llvm_intrinsic
from std.sys.info import is_apple_gpu
from cholesky.logdet_fold import logdet_blocks, logdet_part, logdet_pair, logdet_double

from core.gemm import gemm_nt
from core.identity_trace import IdentityTrace
from cholesky.checks.chol_sabotage import (
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
from cholesky.checks.trsm import CHOL_SOLVE_TPB, trsm_panel_kernel, trsm_panel_guarded_kernel
from cholesky.checks.potrf_strip import (
    CHOL_STRIP_ROUTE,
    CHOL_STRIP_W,
    CS_BM,
    CS_DIAG_TPB,
    CS_NB,
    CS_TPB,
    CS_TRSM_TPB,
    chol_strip_diag_kernel,
    chol_strip_trsm_kernel,
    chol_strip_update_kernel,
)
from gemm.checks.gemm_identical import (
    APPLE_MMA,
    APPLE_MMA_ADMIT_EXP_SUM,
    _AMMA_M64,
    _admit_exp_min,
    _admit_warp_min,
    _amma_gload,
    _amma_load_t,
    _amma_mma,
    _amma_stage,
)
from checks.rtf_seam import rtf_mul_add
from cholesky.multi_gpu import chol_device_count, chol_trailing_rows
from cholesky.impl.matrix.detail.matrix import (
    copy_vector_from_matrix_diagonal_kernel,
)
from gemm.checks.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.contract import OP_NT
from x_decomp.cells import F32Ptr
from x_decomp.fast_chol import CHOL_FAST_BLOCKED, CH_NB, launch_chol_blocked
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
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

#: FAST on Apple: `potrf_lower` runs the trailing update through the vendor
#: GEMM (MAX `matmul`, the Apple simdgroup path) instead of the pinned
#: `identical_gemm_into`, with a wider panel (CHOL_FAST_NB) for callers that ask
#: `chol_default_nb_hint()`. A wider panel needs LESS workspace
#: ((n - nb) * nb + (n - nb)^2 = (n - nb) * n), and the vendor GEMM none of
#: its own, so every caller's pinned-width workspace still covers it.
#: `-D MOJOLEARN_CHOL_FAST_PINNED` keeps the pinned schedule; `-D
#: MOJOLEARN_CHOL_FAST_NB64|128|512` are panel-width arms (M4 KernelRidge
#: n = 20,000: 64 22.5 s, 128 14.6 s, 256 11.6 s, 512 11.7 s).
comptime CHOL_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_CHOL_FAST_PINNED"]()
)
comptime CHOL_TRSM_INV = CHOL_FAST_APPLE and not is_defined[
    "MOJOLEARN_CHOL_TRSM_INV_OFF"
]()
"""CHOL_FAST_APPLE: the panel solve `L21 = A21 L11^{-T}` as `L11^{-1}`
(blocked forward solve against the identity) and one vendor GEMM
(`fast_panel_solve_inv`), instead of a simdgroup per row walking the
panel's columns in sequence."""
comptime CHOL_RECURSIVE_PANEL = CHOL_TRSM_INV and not is_defined[
    "MOJOLEARN_CHOL_RECURSIVE_PANEL_OFF"
]()
"""CHOL_FAST_APPLE: a diagonal block wider than CHOL_INNER_NB is factored
blocked (`fast_diag_factor`) instead of by the one-block unblocked kernel."""
comptime CHOL_INNER_NB = 64
comptime CHOL_INV_MIN_N = 2048
"""The explicit `L11^{-1}` rounds where the column-by-column solve is exact
on exactly representable data; below this size the solve is not where the
time goes, so small factorizations keep the exact route."""
comptime CHOL_FAST_NOSYNC = CHOL_FAST_APPLE and not is_defined["MOJOLEARN_CHOL_FAST_NOSYNC_OFF"]()
"""lane/apple-fast-gap-linalg2: see `potrf_lower`'s fast_defer. The FAST +
Apple default since 2026-10-03 (M3 A/B on top of CHOL_FAST_DEVIO: cholesky
synthetic 285 -> 271 ms, residual the same; tag gl2-chol-nosync-synthetic).
Only callers passing defer_ok (`cholesky_factor_devio`) take it.
-D MOJOLEARN_CHOL_FAST_NOSYNC_OFF keeps the per-panel drains."""
comptime CHOL_FUSED_SUB = CHOL_FAST_APPLE and not is_defined[
    "MOJOLEARN_CHOL_FUSED_SUB_OFF"
]()
"""CHOL_FAST_APPLE: the lower-blocked trailing update subtracts in the
vendor GEMM's epilogue (`fast_gemm_nt_sub_lower`) instead of storing the
product block and subtracting it in a second kernel."""


def chol_default_nb_hint() -> Int:
    """The panel width a production caller asks for: CHOL_FAST_NB under
    CHOL_FAST_APPLE (the hint is honored under FAST), the pinned width
    everywhere else."""
    comptime if CHOL_FAST_APPLE:
        return CHOL_FAST_NB
    return CHOL_NB_PINNED


comptime CHOL_FAST_CB = 2048
# HOLD-speed, 2026-10-04, NB512 source19bb1c7a1:
# w2-cholnb512-quality PASS; w2-cholnb512-synthetic260.4->265.7ms.
# One run/arm showed no benefit; NB512 stays opt-in, default width unchanged.
# NB64/NB128 have no admission implied by that NB512 result.
# See docs/apple-fast/EXPERIMENTS.md (MOJOLEARN_CHOL_FAST_NB512).
comptime CHOL_FAST_NB = 64 if is_defined["MOJOLEARN_CHOL_FAST_NB64"]() else (
    128 if is_defined["MOJOLEARN_CHOL_FAST_NB128"]() else (
        512 if is_defined["MOJOLEARN_CHOL_FAST_NB512"]() else 256
    )
)

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


comptime _CHOL_SYM_TILE = 64


def _chol_sym_ok(a: List[Float32], n: Int, tol: Float32) -> Bool:
    """Every lower cell passes `chol_validate_matrix`'s relative symmetry
    test (the same predicate), visited in 64 x 64 tiles."""
    var p = a.unsafe_ptr()
    var bi = 0
    while bi < n:
        var bj = 0
        while bj <= bi:
            var ie = min(bi + _CHOL_SYM_TILE, n)
            var je = min(bj + _CHOL_SYM_TILE, n)
            for i in range(bi, ie):
                for j in range(bj, min(je, i)):
                    var lo = p[i * n + j]
                    var up = p[j * n + i]
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
                        return False
            bj += _CHOL_SYM_TILE
        bi += _CHOL_SYM_TILE
    return True


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
    # The scan below names the FIRST asymmetric cell (row-major over the
    # lower triangle) but reads a[j, i] down a column: 0.23 s at n = 8192
    # on the M4. `_chol_sym_ok` asks the same question cache-tile by tile
    # (lane/neural-pass113); only a matrix it rejects walks the naming scan.
    if _chol_sym_ok(a, n, tol):
        return
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


def panel_factor_guarded_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
):
    """`panel_factor_kernel`, returning at once (the whole block, before
    any barrier) when an earlier panel already wrote `info`
    (CHOL_DEFER_INFO)."""
    if info[0] != Int32(0):
        return
    _panel_factor_body(a, info, n_in, j0_in, nb_in)


def panel_factor_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    nb_in: Int32,
):
    _panel_factor_body(a, info, n_in, j0_in, nb_in)


@always_inline
def _panel_factor_body(
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


def subtract_lower_2d_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    g: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    base_in: Int32,
    n_trail_in: Int32,
):
    """CHOL_FAST_APPLE: `subtract_lower_kernel` on a 2-D grid -- row =
    block y, column = block x * 256 + thread -- so no thread divides, and
    a block wholly above the diagonal returns at once."""
    var i = Int(block_idx.y)
    var j = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if j > i:
        return
    var n = Int(n_in)
    var base = Int(base_in)
    var nt = Int(n_trail_in)
    var ai = (base + i) * n + base + j
    a[ai] = a[ai] - g[i * nt + j]


def subtract_block_2d_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    gb: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    base_in: Int32,
    row0_in: Int32,
    col0_in: Int32,
    cbw_in: Int32,
):
    """CHOL_FAST_APPLE lower-blocked trailing update: the product block
    `gb` ((n_trail - row0) x cbw, row-major) holds rows >= row0 of columns
    [col0, col0 + cbw); subtract its lower-triangle cells from `a`."""
    var i = Int(row0_in) + Int(block_idx.y)
    var jj = Int(block_idx.x) * 256 + Int(thread_idx.x)
    var cbw = Int(cbw_in)
    if jj >= cbw:
        return
    var j = Int(col0_in) + jj
    if j > i:
        return
    var n = Int(n_in)
    var base = Int(base_in)
    var ai = (base + i) * n + base + j
    a[ai] = a[ai] - gb[Int(block_idx.y) * cbw + jj]


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


comptime CHOL_APPLE_MMA_SYRK = (
    APPLE_MMA
    and is_defined["MOJOLEARN_CHOL_APPLE_MMA_SYRK"]()
)
"""lane/apple-identical-neural (2026-09-26): the IDENTICAL trailing update on
Apple as ONE kernel over the lower-triangle tiles only. The shipped path
writes `G = identical_gemm_into(packed, packed_b)` for the WHOLE
n_trail x n_trail square to a workspace, then `subtract_lower_kernel` reads
it back: summed over the 625 panels of a 20,000-row factor that is about a
terabyte of traffic for a k = 32 product. Here each 64 x 64 tile that holds
a cell with j <= i computes the same contract value -- one leaf (k = 32 <=
CONTRACT_K_LEAF_MIN), its chain `rtf_mul_add` over p ascending from +0.0 on
the matrix unit where the GEMM's window admission holds and exactly
elsewhere, then `ftz` (5d) and `ftz` (the store) -- and applies
`subtract_lower_kernel`'s line `A = ftz(ftz(A) - ftz(G))` to those cells.
TRIAL ONLY (`-D MOJOLEARN_CHOL_APPLE_MMA_SYRK`): bit-identical (KernelRidge
5k and 20k dual_coef hashes equal, cholesky_check green, its sabotage caught)
but no measured gain: 12,000-row factor, three interleaved pairs, 9.45 s
median off against 9.38 s on. The trailing update is not this factor's
bottleneck on the M4."""


def chol_syrk_sub_lower_amma_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    xa: MutPointer[Float32, MutAnyOrigin],
    yb: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    base_in: Int32,
    n_trail_in: Int32,
    w_in: Int32,
):
    comptime SGM = 2
    comptime SGN = 2
    comptime FM = 4
    comptime FN = 4
    comptime KB = 16
    comptime NSG = SGM * SGN
    comptime NT = NSG * 32
    comptime BM = 8 * FM * SGM
    comptime BN = 8 * FN * SGN
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NF = FM * FN
    var n = Int(n_in)
    var base = Int(base_in)
    var m = Int(n_trail_in)
    var k = Int(w_in)
    # Lower-triangle tile (bi, bj), bj <= bi, from the linear block index.
    var t = Int(block_idx.x)
    var bi = 0
    while (bi + 1) * (bi + 2) // 2 <= t:
        bi += 1
    var bj = t - bi * (bi + 1) // 2
    var m0 = bi * BM
    var n0 = bj * BN
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var lane = tid % 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[KB * AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wmin = stack_allocation[2 * NSG, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    var acc = InlineArray[_AMMA_M64, NF](fill=_AMMA_M64(0))
    var exact_ok = True
    var windows = (k + KB - 1) // KB
    var ra = _amma_gload[BM, KB, NT](xa, k, 1, m0, m, 0, min(KB, k), tid, False)
    var rb = _amma_gload[BN, KB, NT](yb, k, 1, n0, m, 0, min(KB, k), tid, False)
    for w in range(windows):
        var k0 = w * KB
        var chunk = min(KB, k - k0)
        var ea = _amma_stage[BM, KB, NT, True, AST](at, ra, tid, False)
        var eb = _amma_stage[BN, KB, NT, False, BST](bt, rb, tid, False)
        ea = _admit_warp_min(ea)
        eb = _admit_warp_min(eb)
        if lane == 0:
            wmin[sg] = ea
            wmin[NSG + sg] = eb
        barrier()
        if w + 1 < windows:
            var k1 = k0 + KB
            ra = _amma_gload[BM, KB, NT](xa, k, 1, m0, m, k1, min(KB, k - k1), tid, False)
            rb = _amma_gload[BN, KB, NT](yb, k, 1, n0, m, k1, min(KB, k - k1), tid, False)
        var bea = UInt32(0xFF)
        var beb = UInt32(0xFF)
        comptime for q in range(NSG):
            bea = min(bea, wmin[q])
            beb = min(beb, wmin[NSG + q])
        var admitted = exact_ok and chunk == KB and (bea + beb) >= UInt32(APPLE_MMA_ADMIT_EXP_SUM)
        if not admitted:
            exact_ok = False
        if admitted:
            comptime for p8 in range(KB // 8):
                var af = InlineArray[_AMMA_M64, FM](fill=_AMMA_M64(0))
                var bf = InlineArray[_AMMA_M64, FN](fill=_AMMA_M64(0))
                comptime for fm in range(FM):
                    af[fm] = _amma_load_t(at + (8 * p8) * AST + (sgm * FM + fm) * 8, AST)
                comptime for fq in range(FN):
                    bf[fq] = _amma_load_t(bt + ((sgn * FN + fq) * 8) * BST + 8 * p8, BST)
                comptime for fm in range(FM):
                    comptime for fq in range(FN):
                        acc[fm * FN + fq] = _amma_mma(af[fm], bf[fq], acc[fm * FN + fq])
        else:
            for p in range(chunk):
                comptime for fm in range(FM):
                    var av = at[p * AST + (sgm * FM + fm) * 8 + frow]
                    comptime for fq in range(FN):
                        comptime for e in range(2):
                            var bv = bt[((sgn * FN + fq) * 8 + fcol + e) * BST + p]
                            acc[fm * FN + fq][e] = rtf_mul_add(av, bv, acc[fm * FN + fq][e])
        barrier()
    comptime for fm in range(FM):
        comptime for fq in range(FN):
            comptime for e in range(2):
                var i = m0 + (sgm * FM + fm) * 8 + frow
                var j = n0 + (sgn * FN + fq) * 8 + fcol + e
                if i < m and j < m and j <= i:
                    var g = ftz(ftz(acc[fm * FN + fq][e]))
                    comptime if is_defined["MOJOLEARN_CHOL_APPLE_MMA_SYRK_SABOTAGE"]():
                        g = g * Float32(1.0000001)
                    var cell = (base + i) * n + base + j
                    a[cell] = ftz(ftz(a[cell]) - ftz(g))


comptime CHOL_APPLE_LEFT = (
    APPLE_MMA
    and not is_defined["MOJOLEARN_CHOL_APPLE_LEFT_OFF"]()
)
"""lane/apple-identical-neural (2026-09-26): the SAME per-cell arithmetic in
LEFT-LOOKING order on Apple IDENTICAL. Right-looking, every cell (i, j),
j <= i, of every trailing block takes, after each panel p, the one step
`A = ftz(ftz(A) - ftz(G_p))`, with `G_p` the GEMM contract value of the
k = 32 product of panel p's rows i and j (one leaf: its `rtf_mul_add` chain,
then `ftz`, then `ftz`). The panels' L values never change once solved, so
the cell's whole sequence -- p = 0, 1, ... in order -- can run just before
its own column block is factored, with A in a register:
`chol_left_update_amma_kernel` does that for one column block, each G_p's
chain on the matrix unit where the GEMM's window admission holds and by
`rtf_mul_add` otherwise. Same values, same order, same bits; the trailing
square is no longer rewritten once per panel (at n = 12,000 that pass was
7.2 of the factor's 9.5 s). On a pivot failure the pending updates are
applied to every later column block, so the partial factor is the
right-looking one. Only without a trace, a sabotage or a multi-GPU owner
set, at the pinned NB = 32. `-D MOJOLEARN_CHOL_APPLE_LEFT_OFF` reverts."""


#: lane/neighbors-apple (2026-09-28): in the Apple left-looking mode the
#: factor no longer drains once per panel to read `info` (DEVIATION 1634's
#: round trip, ~4 ms per panel on Metal: 313 panels at n = 10,000). Every
#: panel's kernels return at once when `info` is already set, so after a
#: failing panel nothing more is written -- the matrix is exactly what the
#: per-panel read stopped at -- and `info` is read ONCE after the loop, which
#: then runs the same partial-factor completion for the failing panel. Same
#: kernels, same order, same words. `-D MOJOLEARN_CHOL_DEFER_INFO_OFF` keeps
#: the per-panel read.
comptime CHOL_DEFER_INFO = CHOL_APPLE_LEFT and not is_defined["MOJOLEARN_CHOL_DEFER_INFO_OFF"]()


#: The left-looking update in pairs of column blocks (see `potrf_lower`):
#: half the reads of the panels' rows. `-D MOJOLEARN_CHOL_LEFT_LOOKAHEAD_OFF`
#: updates one column block at a time.
comptime CHOL_LEFT_LOOKAHEAD = not is_defined["MOJOLEARN_CHOL_LEFT_LOOKAHEAD_OFF"]()
#: Column blocks per joint update: 2 (64 columns, 8 simdgroups of 16 rows x
#: 32 columns). 4 would need a 512-thread block whose A tile does not split
#: into whole staging slots at a 16-deep window.
comptime CHOL_LEFT_GROUP = 2
#: Refused cells a block recomputes cooperatively per window; the rest (only
#: on pathological data) are recomputed by their owners. Execution only.
comptime LEFT_LIST_CAP = 1536
#: Staging window depth of the left update (16 or 32; a 32 window is one per
#: panel, half the per-window bookkeeping). Execution only.
comptime CHOL_LEFT_KB = 32 if is_defined["MOJOLEARN_CHOL_LEFT_KB32"]() else 16

#: Per-cell window admission in `chol_left_update_amma_kernel` (see there).
#: `-D MOJOLEARN_CHOL_LEFT_BLOCK_ADMIT` keeps the block-wide test.
comptime CHOL_LEFT_CELL_ADMIT = not is_defined["MOJOLEARN_CHOL_LEFT_BLOCK_ADMIT"]()


def chol_left_update_amma_kernel[BN: Int, GUARD: Bool = False](
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    j0_in: Int32,
    w_in: Int32,
    np_in: Int32,
    row_lo_in: Int32,
    p_lo_in: Int32,
    stop: MutPointer[Int32, MutAnyOrigin],
):
    """Rows [row_lo + 64 * block, +64) x columns [j0, j0 + w), w <= BN:
    apply panels p_lo .. np-1 in order to every lower cell (j <= i).
    GUARD: return at once (the whole block, before any barrier) when
    `stop[0] != 0` -- an earlier panel's factor failed (CHOL_DEFER_INFO)."""
    comptime if GUARD:
        if stop[0] != Int32(0):
            return
    # One simdgroup per 16 rows x 32 columns: BN = 64 takes 8 simdgroups,
    # so each thread keeps the 8 fragments of the 32-wide tile.
    comptime SGN = BN // 32
    comptime NSG = 4 * SGN
    comptime NT = 32 * NSG
    comptime NB = 32
    comptime KB = CHOL_LEFT_KB
    comptime G = KB // 4
    comptime BM = 64
    comptime AST = BM + 4
    comptime BST = KB + 4
    comptime NFC = 4
    comptime FPS = 8
    var n = Int(n_in)
    var j0 = Int(j0_in)
    var w = Int(w_in)
    var np = Int(np_in)
    var m0 = Int(row_lo_in) + Int(block_idx.x) * BM
    var tid = Int(thread_idx.x)
    var sg = tid // 32
    var sgm = sg // SGN
    var sgn = sg % SGN
    var lane = tid % 32
    var qd = lane // 4
    var frow = (qd & 4) + ((lane // 2) % 4)
    var fcol = (qd & 2) * 2 + (lane % 2) * 2
    var at = stack_allocation[KB * AST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var bt = stack_allocation[BN * BST, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var wmin = stack_allocation[2 * NSG, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    var rmin = stack_allocation[BM, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    var cmin = stack_allocation[BN, Scalar[DType.uint32], address_space = AddressSpace.SHARED]()
    var clist = stack_allocation[LEFT_LIST_CAP, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var cval = stack_allocation[LEFT_LIST_CAP, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var ctot = stack_allocation[1, Scalar[DType.int32], address_space = AddressSpace.SHARED]()
    var c = SIMD[DType.float32, 2 * FPS](0.0)
    comptime for q in range(FPS):
        var fr = sgm * 2 + q // NFC
        var fc = sgn * NFC + q % NFC
        comptime for e in range(2):
            var i = m0 + fr * 8 + frow
            var j = j0 + fc * 8 + fcol + e
            if i < n and j < j0 + w:
                c[2 * q + e] = a[i * n + j]
    for p in range(Int(p_lo_in), np):
        var acc = InlineArray[_AMMA_M64, FPS](fill=_AMMA_M64(0))
        var exact_ok = True
        comptime for wi in range(NB // KB):
            var kk = p * NB + wi * KB
            var ra = _amma_gload[BM, KB, NT](a, n, 1, m0, n, kk, KB, tid, False)
            var rb = _amma_gload[BN, KB, NT](a, n, 1, j0, j0 + w, kk, KB, tid, False)
            var ea = _amma_stage[BM, KB, NT, True, AST](at, ra, tid, False)
            var eb = _amma_stage[BN, KB, NT, False, BST](bt, rb, tid, False)
            comptime if CHOL_LEFT_CELL_ADMIT:
                # Per-row / per-column minimum exponent fields of this
                # window's flushed words, from the staging registers: an A
                # row's 16 words sit in 4 consecutive threads' slots, as do
                # a B column's.
                # A row's KB words sit in G = KB / 4 consecutive threads.
                comptime for sl in range((BM * KB) // (4 * NT)):
                    var v = SIMD[DType.float32, 4](0.0)
                    comptime for u in range(4):
                        v[u] = ftz(ra[4 * sl + u])
                    var e = _admit_exp_min[4](v)
                    comptime for sh in range(3):
                        comptime if (1 << sh) < G:
                            var o = shuffle_xor(e, UInt32(1 << sh))
                            e = o if o < e else e
                    if tid % G == 0:
                        rmin[(sl * NT + tid) // G] = e
                comptime for slb in range((BN * KB) // (4 * NT)):
                    var vb = SIMD[DType.float32, 4](0.0)
                    comptime for u in range(4):
                        vb[u] = ftz(rb[4 * slb + u])
                    var eb4 = _admit_exp_min[4](vb)
                    comptime for sh in range(3):
                        comptime if (1 << sh) < G:
                            var ob = shuffle_xor(eb4, UInt32(1 << sh))
                            eb4 = ob if ob < eb4 else eb4
                    if tid % G == 0:
                        cmin[(slb * NT + tid) // G] = eb4
                ea = _admit_warp_min(ea)
                eb = _admit_warp_min(eb)
                if lane == 0:
                    wmin[sg] = ea
                    wmin[NSG + sg] = eb
                barrier()
                var bea = UInt32(0xFF)
                var beb = UInt32(0xFF)
                comptime for s in range(NSG):
                    bea = min(bea, wmin[s])
                    beb = min(beb, wmin[NSG + s])
                if (bea + beb) >= UInt32(APPLE_MMA_ADMIT_EXP_SUM):
                    # Every pair this block forms passes: the matrix unit's
                    # chain is the contract's for every cell.
                    comptime for p8 in range(KB // 8):
                        comptime for q in range(FPS):
                            var fr = sgm * 2 + q // NFC
                            var fc = sgn * NFC + q % NFC
                            var af = _amma_load_t(at + (8 * p8) * AST + fr * 8, AST)
                            var bf = _amma_load_t(bt + (fc * 8) * BST + 8 * p8, BST)
                            acc[q] = _amma_mma(af, bf, acc[q])
                else:
                    # Every fragment on the matrix unit anyway (it returns
                    # exactly the chain of Apple FMAs from whatever
                    # accumulator it starts with); then each cell whose own
                    # bound fails -- its row's minimum plus its column's, a
                    # lower bound on every Ea + Eb it forms -- is recomputed
                    # from its pre-window value with the exact step. A cell
                    # that passes cannot meet the one window where the two
                    # semantics differ, whatever came before. Refused cells
                    # are walked row by row, one simdgroup per row, so the
                    # cost follows the refused rows (outlier rows of an RBF
                    # kernel), not every fragment that holds one.
                    var mask = UInt32(0)
                    var pre = SIMD[DType.float32, 2 * FPS](0.0)
                    # A lane's cells lie in two rows; a row whose minimum
                    # plus the block's column minimum passes has no refused
                    # cell, so its cells are not looked at one by one.
                    var rv0 = rmin[(sgm * 2) * 8 + frow]
                    var rv1 = rmin[(sgm * 2 + 1) * 8 + frow]
                    var risky0 = rv0 + beb < UInt32(APPLE_MMA_ADMIT_EXP_SUM)
                    var risky1 = rv1 + beb < UInt32(APPLE_MMA_ADMIT_EXP_SUM)
                    comptime for q in range(FPS):
                        var fr = sgm * 2 + q // NFC
                        var fc = sgn * NFC + q % NFC
                        var rv = rv0 if q // NFC == 0 else rv1
                        var risky_row = risky0 if q // NFC == 0 else risky1
                        if risky_row:
                            comptime for e in range(2):
                                var cj = fc * 8 + fcol + e
                                if rv + cmin[cj] < UInt32(APPLE_MMA_ADMIT_EXP_SUM):
                                    pre[2 * q + e] = acc[q][e]
                                    mask |= UInt32(1) << UInt32(2 * q + e)
                        comptime for p8 in range(KB // 8):
                            var af = _amma_load_t(at + (8 * p8) * AST + fr * 8, AST)
                            var bf = _amma_load_t(bt + (fc * 8) * BST + 8 * p8, BST)
                            acc[q] = _amma_mma(af, bf, acc[q])
                    # The refused cells of the whole block in one list, so
                    # every thread recomputes one (a refused row no longer
                    # serializes on one simdgroup). Positions come from an
                    # exclusive prefix sum of the per-lane counts; a lane
                    # walks its cells in the same order to write and read.
                    var cnt = Int32(pop_count(mask))
                    var base = Int(prefix_sum[block_size=NT, exclusive=True](cnt))
                    var pos = base
                    comptime for q in range(FPS):
                        var fr = sgm * 2 + q // NFC
                        var fc = sgn * NFC + q % NFC
                        comptime for e in range(2):
                            if (mask >> UInt32(2 * q + e)) & UInt32(1) != UInt32(0):
                                if pos < LEFT_LIST_CAP:
                                    clist[pos] = Int32((fr * 8 + frow) * BN + fc * 8 + fcol + e)
                                    cval[pos] = pre[2 * q + e]
                                pos += 1
                    if tid == NT - 1:
                        ctot[0] = Int32(pos)
                    barrier()
                    var total = min(Int(ctot[0]), LEFT_LIST_CAP)
                    for t in range(tid, total, NT):
                        var cell = Int(clist[t])
                        var ri = cell // BN
                        var cj = cell % BN
                        var tv = cval[t]
                        for kq in range(KB):
                            tv = rtf_mul_add(at[kq * AST + ri], bt[cj * BST + kq], tv)
                        cval[t] = tv
                    barrier()
                    if mask != UInt32(0):
                        pos = base
                        comptime for q in range(FPS):
                            var fr = sgm * 2 + q // NFC
                            var fc = sgn * NFC + q % NFC
                            comptime for e in range(2):
                                if (mask >> UInt32(2 * q + e)) & UInt32(1) != UInt32(0):
                                    if pos < LEFT_LIST_CAP:
                                        acc[q][e] = cval[pos]
                                    else:
                                        # Past the list: the owner runs the
                                        # same chain itself.
                                        var tv = pre[2 * q + e]
                                        var ri = fr * 8 + frow
                                        var cj = fc * 8 + fcol + e
                                        for kq in range(KB):
                                            tv = rtf_mul_add(at[kq * AST + ri], bt[cj * BST + kq], tv)
                                        acc[q][e] = tv
                                    pos += 1
            else:
                ea = _admit_warp_min(ea)
                eb = _admit_warp_min(eb)
                if lane == 0:
                    wmin[sg] = ea
                    wmin[NSG + sg] = eb
                barrier()
                var bea = UInt32(0xFF)
                var beb = UInt32(0xFF)
                comptime for s in range(NSG):
                    bea = min(bea, wmin[s])
                    beb = min(beb, wmin[NSG + s])
                var admitted = exact_ok and (bea + beb) >= UInt32(APPLE_MMA_ADMIT_EXP_SUM)
                if not admitted:
                    exact_ok = False
                if admitted:
                    comptime for p8 in range(KB // 8):
                        comptime for q in range(FPS):
                            var fr = sgm * 2 + q // NFC
                            var fc = sgn * NFC + q % NFC
                            var af = _amma_load_t(at + (8 * p8) * AST + fr * 8, AST)
                            var bf = _amma_load_t(bt + (fc * 8) * BST + 8 * p8, BST)
                            acc[q] = _amma_mma(af, bf, acc[q])
                else:
                    for kq in range(KB):
                        comptime for q in range(FPS):
                            var fr = sgm * 2 + q // NFC
                            var fc = sgn * NFC + q % NFC
                            var av = at[kq * AST + fr * 8 + frow]
                            comptime for e in range(2):
                                var bv = bt[(fc * 8 + fcol + e) * BST + kq]
                                acc[q][e] = rtf_mul_add(av, bv, acc[q][e])
            barrier()
        comptime for q in range(FPS):
            comptime for e in range(2):
                var g = ftz(ftz(acc[q][e]))
                comptime if is_defined["MOJOLEARN_CHOL_APPLE_LEFT_SABOTAGE"]():
                    g = g * Float32(1.0000001)
                c[2 * q + e] = ftz(ftz(c[2 * q + e]) - ftz(g))
    comptime for q in range(FPS):
        var fr = sgm * 2 + q // NFC
        var fc = sgn * NFC + q % NFC
        comptime for e in range(2):
            var i = m0 + fr * 8 + frow
            var j = j0 + fc * 8 + fcol + e
            if i < n and j < j0 + w and j <= i:
                a[i * n + j] = c[2 * q + e]


def _chol_left_update(
    ctx: DeviceContext, mut a: DeviceBuffer[DType.float32], n: Int, j0: Int, w: Int, np: Int,
    p_lo: Int = 0,
) raises:
    """Columns [j0, j0 + w) (w <= 64), rows j0 .. n-1, panels p_lo .. np-1.

    COMPILE-TIME GATED (0.8.23's AMD build): the callers guard it with a
    RUNTIME `left_mode`, which only the Apple column can set, but a runtime
    guard still compiles the launch, so every GPU target got this Apple
    simdgroup-matrix kernel and gfx942's linker refused its `air.*` symbols
    (mixture, gp, kernel_methods). Off Apple the body is empty, which is
    exactly what `left_mode == False` already meant there."""
    comptime if CHOL_APPLE_LEFT:
        if np <= p_lo or w <= 0:
            return
        # The unguarded kernel never reads its stop word (compiled out).
        var unused = ctx.enqueue_create_buffer[DType.int32](1)
        _chol_left_launch[False](ctx, a, unused, n, j0, w, np, p_lo)
        _ = unused^


def _chol_left_update_guarded(
    ctx: DeviceContext, mut a: DeviceBuffer[DType.float32], mut stop: DeviceBuffer[DType.int32],
    n: Int, j0: Int, w: Int, np: Int, p_lo: Int = 0,
) raises:
    """`_chol_left_update` whose blocks return when `stop[0] != 0`
    (CHOL_DEFER_INFO: the factor's `info` word)."""
    comptime if CHOL_APPLE_LEFT:
        if np <= p_lo or w <= 0:
            return
        _chol_left_launch[True](ctx, a, stop, n, j0, w, np, p_lo)


def _chol_left_launch[GUARD: Bool](
    ctx: DeviceContext, mut a: DeviceBuffer[DType.float32], mut stop: DeviceBuffer[DType.int32],
    n: Int, j0: Int, w: Int, np: Int, p_lo: Int,
) raises:
    comptime if CHOL_APPLE_LEFT:
        # F02 shared subtract must enter the public pinned-width left route.
        # Its default-off guard leaves the promoted MMA schedule unchanged.
        if try_chol_shared_left[GUARD](ctx,a,stop,n,j0,w,np,p_lo):return
        var rows = n - j0
        if w > 32:
            ctx.enqueue_function[chol_left_update_amma_kernel[64, GUARD]](
                a.unsafe_ptr(), Int32(n), Int32(j0), Int32(w), Int32(np), Int32(j0),
                Int32(p_lo), stop.unsafe_ptr(),
                grid_dim=((rows + 63) // 64, 1, 1), block_dim=(256, 1, 1),
            )
        else:
            ctx.enqueue_function[chol_left_update_amma_kernel[32, GUARD]](
                a.unsafe_ptr(), Int32(n), Int32(j0), Int32(w), Int32(np), Int32(j0),
                Int32(p_lo), stop.unsafe_ptr(),
                grid_dim=((rows + 63) // 64, 1, 1), block_dim=(128, 1, 1),
            )


# lane fam2-decomp (2026-10-04), IDENTICAL: the strip route
# (`_potrf_lower_strips`, NVIDIA and AMD) waits ONCE per factorization. The
# upper triangle's zeros are enqueued before the one read of `info`, by a
# kernel that returns at once when `info` is set (`zero_upper_guarded_kernel`),
# so the read's wait is the last one of a successful factor; the wait after
# the read to launch `zero_upper_kernel`, and the closing wait, are gone
# (a GaussianMixture M-step factors every component: two waits a component
# saved). A failed factor takes the old tail. Same kernels on the same
# cells: the same words. -D MOJOLEARN_IDN_CHOL_ONE_WAIT_OFF (or
# -D MOJOLEARN_IDN_ALL_OFF) restores the three waits.
comptime IDN_CHOL_ONE_WAIT = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (is_defined["MOJOLEARN_IDN_CHOL_ONE_WAIT_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]())


def zero_upper_guarded_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    info: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
):
    """`zero_upper_kernel` when the factorization succeeded (`info[0] == 0`);
    nothing when it failed (IDN_CHOL_ONE_WAIT)."""
    if info[0] != Int32(0):
        return
    var n = Int(n_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * n:
        return
    var i = idx // n
    var j = idx % n
    if j > i:
        a.unsafe_store(idx, Float32(0.0))


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


comptime LOGDET_TPB = 256


def logdet_part_kernel(
    diag: MutPointer[Float32, MutAnyOrigin],
    parts: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """A thread a LOGDET_BLOCK of the diagonal: its logs folded ascending
    (cholesky/logdet_fold.mojo, DEVIATION 1639 as revised by lane
    cgr4-device-optim). Every `log` is `identical_log` (IDENTITY_PATHS row
    12): a device `log` is a vendor choice in its last bit."""
    var n = Int(n_in)
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b < logdet_blocks(n):
        parts.unsafe_store(b, logdet_part(diag, n, b))


def logdet_kernel(
    parts: MutPointer[Float32, MutAnyOrigin],
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    nb_in: Int32,
):
    """One block: the nb block partials (k = n / LOGDET_BLOCK of them, not n)
    by the aligned tree, level by level through parts[0, nb) and
    parts[nb, 2 nb), then the doubling `identical_mul(2.0, root)` (exact,
    written so no codegen contracts it). `logdet_serial` is the same tree."""
    var nb = Int(nb_in)
    var tid = Int(thread_idx.x)
    var nt = Int(block_dim.x)
    var src = 0
    var dst = nb
    var cur = nb
    while cur > 1:
        var h = cur // 2
        var nxt = (cur + 1) // 2
        for i in range(tid, nxt, nt):
            if i < h:
                parts.unsafe_store(dst + i, logdet_pair(parts.unsafe_load(src + 2 * i), parts.unsafe_load(src + 2 * i + 1)))
            else:
                parts.unsafe_store(dst + i, parts.unsafe_load(src + cur - 1))
        _logdet_barrier()
        var s2 = src
        src = dst
        dst = s2
        cur = nxt
    if tid == 0:
        var r = parts.unsafe_load(src) if nb > 0 else Float32(0.0)
        out_scalar.unsafe_store(0, logdet_double(r))


@always_inline
def _logdet_barrier():
    """A block barrier that orders DEVICE memory (the partials live there):
    Apple's `barrier()` orders threadgroup memory only."""
    comptime if is_apple_gpu():
        llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(3), Int32(1))
    else:
        barrier()


def enqueue_logdet(
    ctx: DeviceContext,
    diag: MutPointer[Float32, MutAnyOrigin],
    parts: MutPointer[Float32, MutAnyOrigin],
    out_scalar: MutPointer[Float32, MutAnyOrigin],
    n: Int,
) raises:
    """The partials (a thread a block) then the tree (one block over the
    partials). parts: 2 * logdet_blocks(n) floats."""
    var nb = logdet_blocks(n)
    ctx.enqueue_function[logdet_part_kernel](
        diag, parts, Int32(n),
        grid_dim=((nb + LOGDET_TPB - 1) // LOGDET_TPB, 1, 1),
        block_dim=(LOGDET_TPB, 1, 1),
    )
    ctx.enqueue_function[logdet_kernel](  # small-launch(nb: diagonal blocks): one block folds the n over 256 block partials by the aligned tree
        parts, out_scalar, Int32(nb), grid_dim=(1, 1, 1), block_dim=(LOGDET_TPB, 1, 1),
    )


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


def fast_diag_factor(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut dinfo: DeviceBuffer[DType.int32],
    n: Int,
    j0: Int,
    w: Int,
    mut linv: DeviceBuffer[DType.float32],
    mut praw: DeviceBuffer[DType.float32],
    mut pk: DeviceBuffer[DType.float32],
    mut shape: DeviceBuffer[DType.float32],
    panel_tpb: Int,
    elem_tpb: Int,
) raises:
    """CHOL_RECURSIVE_PANEL: the `w x w` diagonal block at `j0` factored as
    a blocked Cholesky of its own, CHOL_INNER_NB columns at a time: the
    unblocked `panel_factor_kernel` on each inner diagonal block, the inner
    panel solve through `fast_panel_solve_inv`, and the inner trailing
    update through the GEMM epilogue. `info` is the unblocked kernel's."""
    var s0 = 0
    while s0 < w:
        var sw = min(CHOL_INNER_NB, w - s0)
        ctx.enqueue_function[panel_factor_kernel](  # small-launch(n: leading dimension only): factors the w x w diagonal panel block, columns serial, rows across the block; n is the row stride
            a.unsafe_ptr(), dinfo.unsafe_ptr(), Int32(n), Int32(j0 + s0),
            Int32(sw),
            grid_dim=(1, 1, 1), block_dim=(panel_tpb, 1, 1),
        )
        var rem = w - s0 - sw
        if rem > 0:
            ctx.enqueue_function[pack_panel_kernel](
                praw.unsafe_ptr(), a.unsafe_ptr(), Int32(n), Int32(j0 + s0),
                Int32(sw), Int32(rem),
                grid_dim=((rem * sw + elem_tpb - 1) // elem_tpb, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
            fast_panel_solve_inv(ctx, a, n, j0 + s0, sw, rem, linv, praw, pk)
            var xa = pk.create_sub_buffer[DType.float32](0, rem * sw)
            var yb = pk.create_sub_buffer[DType.float32](0, rem * sw)
            var sh = shape.create_sub_buffer[DType.float32](0, rem * rem)
            fast_gemm_nt_sub_lower(
                ctx, a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                n, j0 + s0 + sw, 0, sh, xa, yb, rem, rem, sw,
            )
            _ = xa^
            _ = yb^
            _ = sh^
        s0 += sw


def _strip_update(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut dinfo: DeviceBuffer[DType.int32],
    n: Int,
    col0: Int,
    col_end: Int,
    p_lo: Int,
    p_hi: Int,
    guard: Bool,
) raises:
    """Panels p_lo .. p_hi - 1 onto the lower cells of columns
    [col0, col_end), rows col0 .. n - 1 (`chol_strip_update_kernel`)."""
    if p_hi <= p_lo or col_end <= col0:
        return
    var gx = (col_end - col0 + CS_BM - 1) // CS_BM
    var gy = (n - col0 + CS_BM - 1) // CS_BM
    if guard:
        ctx.enqueue_function[chol_strip_update_kernel[True]](
            a.unsafe_ptr(), dinfo.unsafe_ptr(), Int32(n), Int32(col0),
            Int32(col_end), Int32(p_lo), Int32(p_hi),
            grid_dim=(gx, gy, 1), block_dim=(CS_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[chol_strip_update_kernel[False]](
            a.unsafe_ptr(), dinfo.unsafe_ptr(), Int32(n), Int32(col0),
            Int32(col_end), Int32(p_lo), Int32(p_hi),
            grid_dim=(gx, gy, 1), block_dim=(CS_TPB, 1, 1),
        )


def _potrf_lower_strips(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    n: Int,
    elem_tpb: Int,
    mut trace: IdentityTrace,
) raises -> CholRun:
    """`potrf_lower` at the pinned width in strips of CHOL_STRIP_W columns
    (`cholesky/checks/potrf_strip.mojo` states the argument). Strip
    [J, J + sw): one launch applies panels 0 .. J/32 - 1 to its lower cells,
    left-looking; then each of its panels is factored, solved and applied
    to the strip's later columns. `info` is read ONCE, after the loop; every
    launch returns at once when it is set, so the matrix is what the
    per-panel read stopped at, and the later strips then take the panels
    before the failing one: the right-looking partial factor."""
    var dinfo = ctx.enqueue_create_buffer[DType.int32](1)
    var hinfo = ctx.enqueue_create_host_buffer[DType.int32](1)
    hinfo.unsafe_ptr().unsafe_store(0, Int32(0))
    ctx.enqueue_copy(dst_buf=dinfo, src_ptr=hinfo.unsafe_ptr())
    var j_strip = 0
    while j_strip < n:
        var sw = min(CHOL_STRIP_W, n - j_strip)
        var s_end = j_strip + sw
        _strip_update(ctx, a, dinfo, n, j_strip, s_end, 0, j_strip // CS_NB, True)
        var q0 = j_strip
        while q0 < s_end:
            var w = min(CS_NB, n - q0)
            var n_trail = n - q0 - w
            ctx.enqueue_function[chol_strip_diag_kernel](  # small-launch(n: leading dimension only): factors the w x w diagonal block (w <= CS_NB) in threadgroup memory, n is the row stride
                a.unsafe_ptr(), dinfo.unsafe_ptr(), Int32(n), Int32(q0), Int32(w),
                grid_dim=(1, 1, 1), block_dim=(CS_DIAG_TPB, 1, 1),
            )
            if n_trail > 0:
                ctx.enqueue_function[chol_strip_trsm_kernel](
                    a.unsafe_ptr(), dinfo.unsafe_ptr(), Int32(n), Int32(q0),
                    Int32(n_trail),
                    grid_dim=((n_trail + CS_TRSM_TPB - 1) // CS_TRSM_TPB, 1, 1),
                    block_dim=(CS_TRSM_TPB, 1, 1),
                )
                var p = q0 // CS_NB
                _strip_update(ctx, a, dinfo, n, q0 + w, s_end, p, p + 1, True)
            q0 += CS_NB
        j_strip = s_end
    comptime if IDN_CHOL_ONE_WAIT:
        ctx.enqueue_function[zero_upper_guarded_kernel](
            a.unsafe_ptr(),
            dinfo.unsafe_ptr(),
            Int32(n),
            grid_dim=((n * n + elem_tpb - 1) // elem_tpb, 1, 1),
            block_dim=(elem_tpb, 1, 1),
        )
    ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
    ctx.synchronize()
    var info = Int(hinfo.unsafe_ptr().unsafe_load(0))
    var n_panels = (n + CS_NB - 1) // CS_NB
    # everything enqueued has been waited for unless a launch follows the read
    var drained = False
    if info != 0:
        var pf = (info - 1) // CS_NB
        var f_end = min((pf * CS_NB // CHOL_STRIP_W + 1) * CHOL_STRIP_W, n)
        _strip_update(ctx, a, dinfo, n, f_end, n, 0, pf, False)
        n_panels = pf + 1
    else:
        comptime if IDN_CHOL_ONE_WAIT:
            drained = not trace.enabled
        else:
            var cells = n * n
            ctx.enqueue_function[zero_upper_kernel](
                a.unsafe_ptr(),
                Int32(n),
                grid_dim=((cells + elem_tpb - 1) // elem_tpb, 1, 1),
                block_dim=(elem_tpb, 1, 1),
            )
    trace.record_device(ctx, "chol.factor", a, n * n)
    var nb_record = List[Int32]()
    nb_record.append(Int32(CS_NB))
    nb_record.append(Int32(n_panels))
    nb_record.append(Int32(info))
    trace.record_list_i32("chol.nb", nb_record)
    if not drained:
        ctx.synchronize()
    _ = dinfo^
    _ = hinfo^
    return CholRun(info, CS_NB, n_panels)


def _potrf_lower_fast_blocked(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    n: Int,
) raises -> CholRun:
    """`potrf_lower` through x_decomp/fast_chol.mojo `launch_chol_blocked`
    (CHOL_FAST_BLOCKED, default off, FAST + Apple; recovered from
    lane/apple-fast-decomp-linalg@74d52352b): the launches enqueued, `info`
    read once after them. The panel width that ran is CH_NB."""
    var dinfo = ctx.enqueue_create_buffer[DType.float32](1)
    var hinfo = ctx.enqueue_create_host_buffer[DType.float32](1)
    launch_chol_blocked(
        ctx,
        F32Ptr(unsafe_from_address=Int(a.unsafe_ptr())),
        F32Ptr(unsafe_from_address=Int(dinfo.unsafe_ptr())),
        n,
    )
    ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
    ctx.synchronize()
    var info = Int(hinfo.unsafe_ptr().unsafe_load(0))
    _ = dinfo^
    _ = hinfo^
    return CholRun(info, CH_NB, (n + CH_NB - 1) // CH_NB)


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
    defer_ok: Bool = False,
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
    var need: Int
    comptime if CHOL_FAST_APPLE:
        need = (n - nb) * n
        if need < 1:
            need = 1
    else:
        need = chol_workspace_floats(n, nb)
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

    # The strip schedule (cholesky/checks/potrf_strip.mojo): the same cells,
    # the same steps, the same order. Only without a trace, a sabotage or a
    # multi-GPU owner set, at the pinned width. MOJOLEARN_CHOL_STRIP_OFF=1
    # runs the panel-by-panel loop below (the A/B arm).
    comptime if CHOL_STRIP_ROUTE:
        if (
            sabotage == CHOL_SAB_NONE
            and not trace.enabled
            and chol_device_count() == 1
            and nb == CS_NB
            and String(getenv("MOJOLEARN_CHOL_STRIP_OFF")) != "1"
        ):
            return _potrf_lower_strips(ctx, a, n, elem_tpb, trace)
    # -D MOJOLEARN_CHOL_FAST_BLOCKED (x_decomp/fast_chol.mojo, default off,
    # FAST + Apple): the blocked right-looking route instead of the
    # CHOL_FAST_NB route below. Only without a sabotage, a trace or a
    # multi-GPU owner set, past one panel, and for a `defer_ok` caller: on a
    # non-positive pivot the route continues the sweep (x_decomp's rule), and
    # a `defer_ok` caller (cholesky/estimator.mojo `cholesky_factor_devio`)
    # redoes a failed factor without `defer_ok`, so LAPACK's partial factor
    # still comes from the route below (that redo is CHOL_FAST_NOSYNC's, so
    # the route needs it on).
    comptime if CHOL_FAST_BLOCKED and CHOL_FAST_NOSYNC:
        if (
            defer_ok
            and sabotage == CHOL_SAB_NONE
            and not trace.enabled
            and chol_device_count() == 1
            and n > CH_NB
        ):
            return _potrf_lower_fast_blocked(ctx, a, n)

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
    var inv_linv = ctx.enqueue_create_buffer[DType.float32](
        nb * nb if CHOL_TRSM_INV else 1
    )
    var inv_praw = ctx.enqueue_create_buffer[DType.float32](
        max(nt_max * nb, 1) if CHOL_TRSM_INV else 1
    )
    var inv_pk = ctx.enqueue_create_buffer[DType.float32](
        nb * nb if CHOL_RECURSIVE_PANEL else 1
    )
    var inv_shape = ctx.enqueue_create_buffer[DType.float32](
        nb * nb if CHOL_RECURSIVE_PANEL else 1
    )
    # CHOL_FAST_TALL (FAST+Apple default): the inner step's L11^{-1}
    var tall_linv = ctx.enqueue_create_buffer[DType.float32](
        CTL_NB * CTL_NB if CHOL_FAST_TALL else 1
    )
    # MOJOLEARN_CHOL_TIMING=1: per-stage host times (synchronizing), printed
    # once at the end. Diagnostics only; off by default and bit-inert.
    var ctim = String(getenv("MOJOLEARN_CHOL_TIMING")) == "1"
    var left_mode = False
    comptime if CHOL_APPLE_LEFT:
        left_mode = (
            sabotage == CHOL_SAB_NONE and not trace.enabled
            and chol_device_count() == 1 and nb == 32
        )
    # CHOL_DEFER_INFO: only in the left-looking mode (no trace, no
    # sabotage, one owner) and not under MOJOLEARN_CHOL_TIMING.
    var defer = False
    comptime if CHOL_DEFER_INFO:
        defer = left_mode and not ctim
    # CHOL_FAST_NOSYNC (FAST + Apple default; _OFF reverts): a caller
    # that can redo a failed factor (defer_ok: `cholesky_factor_devio`
    # re-runs from its staged input with defer_ok False) gets no drain per
    # panel -- neither the info read nor the end-of-trailing wait -- and one
    # info read after the loop. A failed factor's matrix is then not the
    # partial one, which is why the caller redoes it.
    var fast_defer = False
    comptime if CHOL_FAST_NOSYNC:
        fast_defer = defer_ok and not defer and not ctim and sabotage == CHOL_SAB_NONE and not trace.enabled
    var tf = 0
    var ts = 0
    var tt = 0
    var tq = 0
    var tu = 0
    var tk = Int(perf_counter_ns())
    while j0 < n:
        var w = nb
        if j0 + w > n:
            w = n - j0
        var n_trail = n - j0 - w
        if ctim:
            ctx.synchronize()
            tq += Int(perf_counter_ns()) - tk
            tk = Int(perf_counter_ns())
        if left_mode and p > 0:
            comptime if CHOL_LEFT_LOOKAHEAD:
                # Pairs of column blocks: an even block p takes panels
                # 0 .. p-1 together with block p+1 (one pass over those
                # panels' rows for 64 columns), and block p+1 takes panel p
                # alone once p is factored. Every cell still takes panels
                # 0, 1, ... in order.
                comptime LW = CHOL_LEFT_GROUP
                var r = p % LW
                if r == 0:
                    if defer:
                        _chol_left_update_guarded(ctx, a, dinfo, n, j0, min(LW * nb, n - j0), p)
                    else:
                        _chol_left_update(ctx, a, n, j0, min(LW * nb, n - j0), p)
                else:
                    if defer:
                        _chol_left_update_guarded(ctx, a, dinfo, n, j0, w, p, p - r)
                    else:
                        _chol_left_update(ctx, a, n, j0, w, p, p - r)
            else:
                if defer:
                    _chol_left_update_guarded(ctx, a, dinfo, n, j0, w, p)
                else:
                    _chol_left_update(ctx, a, n, j0, w, p)
            if ctim:
                ctx.synchronize()
                tu += Int(perf_counter_ns()) - tk
                tk = Int(perf_counter_ns())

        # ---- the panel ------------------------------------------------
        # CHOL_FAST_TALL (FAST+Apple default, cholesky/checks/chol_fast_tall.mojo;
        # rollback -D MOJOLEARN_CHOL_FAST_TALL_OFF): only on the fast_defer sweep (the
        # public Cholesky.fit, which redoes a failed factor on this loop's
        # main route). The column panel is factored whole -- L11 and L21 --
        # so the panel solve below is skipped; the trailing update is main's.
        var tall_done = False
        comptime if CHOL_FAST_TALL:
            if fast_defer:
                chol_fast_tall_panel(ctx, a, dinfo, tall_linv, n, j0, w)
                tall_done = True
        if tall_done:
            pass
        elif chol_sabotage_is_kernel_arm(sabotage):
            ctx.enqueue_function[sabotage_panel_factor_kernel](  # small-launch(n: leading dimension only): the negative-control copy of the w x w panel factor, reached only with a sabotage id
                a.unsafe_ptr(),
                dinfo.unsafe_ptr(),
                Int32(n),
                Int32(j0),
                Int32(w),
                Int32(sabotage),
                grid_dim=(1, 1, 1),
                block_dim=(panel_tpb, 1, 1),
            )
        elif CHOL_RECURSIVE_PANEL and w > CHOL_INNER_NB and n >= CHOL_INV_MIN_N:
            fast_diag_factor(
                ctx, a, dinfo, n, j0, w, inv_linv, inv_praw, inv_pk,
                inv_shape, panel_tpb, elem_tpb,
            )
        elif defer:
            ctx.enqueue_function[panel_factor_guarded_kernel](  # small-launch(n: leading dimension only): factors the w x w diagonal panel block, columns serial, rows across the block; n is the row stride
                a.unsafe_ptr(),
                dinfo.unsafe_ptr(),
                Int32(n),
                Int32(j0),
                Int32(w),
                grid_dim=(1, 1, 1),
                block_dim=(panel_tpb, 1, 1),
            )
        else:
            ctx.enqueue_function[panel_factor_kernel](  # small-launch(n: leading dimension only): factors the w x w diagonal panel block, columns serial, rows across the block; n is the row stride
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

        # DEVIATION 1634: read `info` back and stop. One drain per panel
        # (CHOL_DEFER_INFO: once, after the loop).
        if not defer and not fast_defer:
            ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
            ctx.synchronize()
            info = Int(hinfo.unsafe_ptr().unsafe_load(0))
        if ctim:
            tf += Int(perf_counter_ns()) - tk
            tk = Int(perf_counter_ns())
        if info != 0:
            if left_mode:
                # The right-looking partial factor: every later column block
                # has taken panels 0 .. p-1 (not panel p, which failed).
                var q0 = j0 + nb
                var qb = p + 1
                while q0 < n:
                    var have = 0
                    comptime if CHOL_LEFT_LOOKAHEAD:
                        # A later block of the same group already holds
                        # panels 0 .. (group start - 1), unless the group
                        # starts at panel 0 (no joint update ran).
                        var g0 = (p // CHOL_LEFT_GROUP) * CHOL_LEFT_GROUP
                        if qb < g0 + CHOL_LEFT_GROUP and g0 > 0:
                            have = g0
                    _chol_left_update(ctx, a, n, q0, min(nb, n - q0), p, have)
                    q0 += nb
                    qb += 1
                ctx.synchronize()
            p += 1
            break

        var packed_by_solve = False
        if n_trail > 0:
            # ---- the panel solve, L21 = A21 . L11^{-T} -----------------
            var solve_grid = (n_trail + solve_tpb - 1) // solve_tpb
            if tall_done:
                pass  # CHOL_FAST_TALL: L21 is already in `a`
            elif chol_sabotage_is_kernel_arm(sabotage):
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
            elif CHOL_TRSM_INV and w <= FTS_BLOCK and n >= CHOL_INV_MIN_N:
                ctx.enqueue_function[pack_panel_kernel](
                    inv_praw.unsafe_ptr(),
                    a.unsafe_ptr(),
                    Int32(n),
                    Int32(j0),
                    Int32(w),
                    Int32(n_trail),
                    grid_dim=((n_trail * w + elem_tpb - 1) // elem_tpb, 1, 1),
                    block_dim=(elem_tpb, 1, 1),
                )
                var pk_inv = ws.create_sub_buffer[DType.float32](
                    off_pack, n_trail * w
                )
                fast_panel_solve_inv(
                    ctx, a, n, j0, w, n_trail, inv_linv, inv_praw, pk_inv
                )
                _ = pk_inv^
                packed_by_solve = True
            elif CHOL_FAST_APPLE and w <= FTP_MAX_NB:
                ctx.enqueue_function[fast_trsm_panel_kernel](
                    a.unsafe_ptr(), Int32(n), Int32(j0), Int32(w), Int32(n_trail),
                    grid_dim=((n_trail * 32 + 255) // 256, 1, 1),
                    block_dim=(256, 1, 1),
                )
            elif defer:
                ctx.enqueue_function[trsm_panel_guarded_kernel](
                    a.unsafe_ptr(),
                    dinfo.unsafe_ptr(),
                    Int32(n),
                    Int32(j0),
                    Int32(w),
                    Int32(n_trail),
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
            # `hierarchy/impl/cluster/detail/connectivities.mojo` makes for
            # `X` against itself.
            var packed_b = ws.create_sub_buffer[DType.float32](
                off_pack, n_trail * w
            )
            var g = ws.create_sub_buffer[DType.float32](
                off_g, n_trail * n_trail
            )
            if ctim:
                ctx.synchronize()
                ts += Int(perf_counter_ns()) - tk
                tk = Int(perf_counter_ns())
            if not left_mode:
                var gws_len = len(ws) - off_gws
                if gws_len < 1:
                    gws_len = 1
                var gws = ws.create_sub_buffer[DType.float32](off_gws, gws_len)

                var pack_cells = n_trail * w
                if packed_by_solve:
                    pack_cells = 0
                if pack_cells > 0:
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

                var vendor = sabotage == CHOL_SAB_VENDOR_MATMUL
                comptime if CHOL_FAST_APPLE:
                    vendor = True
                var lower_blocked = False
                comptime if CHOL_FAST_APPLE:
                    lower_blocked = sabotage == CHOL_SAB_NONE and n_trail > CHOL_FAST_CB
                if lower_blocked:
                    # rows >= cb of each CHOL_FAST_CB-wide column block only:
                    # about half the product of the full square
                    var cb = 0
                    while cb < n_trail:
                        var cbw = min(CHOL_FAST_CB, n_trail - cb)
                        var rows_b = n_trail - cb
                        var xa = packed.create_sub_buffer[DType.float32](cb * w, rows_b * w)
                        var yb = packed_b.create_sub_buffer[DType.float32](cb * w, cbw * w)
                        var gb = g.create_sub_buffer[DType.float32](0, rows_b * cbw)
                        comptime if CHOL_FUSED_SUB:
                            fast_gemm_nt_sub_lower(
                                ctx, a.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                                n, j0 + w, cb, gb, xa, yb, rows_b, cbw, w,
                            )
                        else:
                            gemm_nt(ctx, gb, xa, yb, rows_b, cbw, w)
                            ctx.enqueue_function[subtract_block_2d_kernel](
                                a.unsafe_ptr(), gb.unsafe_ptr(), Int32(n),
                                Int32(j0 + w), Int32(cb), Int32(cb), Int32(cbw),
                                grid_dim=((cbw + 255) // 256, rows_b, 1),
                                block_dim=(256, 1, 1),
                            )
                        _ = xa^
                        _ = yb^
                        _ = gb^
                        cb += cbw
                elif vendor:
                    # ARM: `linalg.matmul` through `core/gemm.mojo::gemm_nt`. Its
                    # k-split is a per-vendor summation order; DEVIATION 1636.
                    gemm_nt(ctx, g, packed, packed_b, n_trail, n_trail, w)
                else:
                    # Whole output rows across owners when the operation-level
                    # driver is enabled (cholesky/multi_gpu.mojo); same cells.
                    var owners = chol_device_count()
                    if owners > 1 and n_trail > 1:
                        chol_trailing_rows(ctx, g, packed, n_trail, w, owners)
                    else:
                        var fused = False
                        comptime if CHOL_APPLE_MMA_SYRK:
                            if sabotage == CHOL_SAB_NONE:
                                fused = True
                                var tb = (n_trail + 63) // 64
                                ctx.enqueue_function[chol_syrk_sub_lower_amma_kernel](
                                    a.unsafe_ptr(), packed.unsafe_ptr(), packed_b.unsafe_ptr(),
                                    Int32(n), Int32(j0 + w), Int32(n_trail), Int32(w),
                                    grid_dim=(tb * (tb + 1) // 2, 1, 1), block_dim=(128, 1, 1),
                                )
                                lower_blocked = True  # the subtraction is done
                        if not fused:
                            identical_gemm_into(
                                ctx, g, packed, packed_b, gws, n_trail, n_trail, w, OP_NT
                            )

                var sub_cells = n_trail * n_trail
                if lower_blocked:
                    sub_cells = 0
                comptime if CHOL_FAST_APPLE:
                    if sub_cells > 0:
                        ctx.enqueue_function[subtract_lower_2d_kernel](
                            a.unsafe_ptr(), g.unsafe_ptr(), Int32(n), Int32(j0 + w),
                            Int32(n_trail),
                            grid_dim=((n_trail + 255) // 256, n_trail, 1),
                            block_dim=(256, 1, 1),
                        )
                    sub_cells = 0
                if sub_cells > 0:
                    ctx.enqueue_function[subtract_lower_kernel](
                        a.unsafe_ptr(),
                        g.unsafe_ptr(),
                        Int32(n),
                        Int32(j0 + w),
                        Int32(n_trail),
                        grid_dim=((sub_cells + elem_tpb - 1) // elem_tpb, 1, 1),
                        block_dim=(elem_tpb, 1, 1),
                    )
                if ctim:
                    ctx.synchronize()
                    tt += Int(perf_counter_ns()) - tk
                    tk = Int(perf_counter_ns())
                trace.record_device(
                    ctx, chol_panel_tag("chol", p, "trailing"), a, n * n
                )
                if not fast_defer:
                    ctx.synchronize()
                _ = packed^
                _ = packed_b^
                _ = g^
                _ = gws^

        p += 1
        j0 += nb

    if fast_defer:
        ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
        ctx.synchronize()
        info = Int(hinfo.unsafe_ptr().unsafe_load(0))
    _ = tall_linv^  # CHOL_FAST_TALL: alive past every enqueued use
    if defer:
        # CHOL_DEFER_INFO: the one read of `info`. After a failing panel pf
        # every later kernel returned at once, so the matrix is what the
        # per-panel read stopped at; complete the partial factor exactly as
        # the loop's `info != 0` branch does for that panel.
        ctx.enqueue_copy(dst_ptr=hinfo.unsafe_ptr(), src_buf=dinfo)
        ctx.synchronize()
        info = Int(hinfo.unsafe_ptr().unsafe_load(0))
        if info != 0:
            var pf = (info - 1) // nb
            var q0 = pf * nb + nb
            var qb = pf + 1
            while q0 < n:
                var have = 0
                comptime if CHOL_LEFT_LOOKAHEAD:
                    var g0 = (pf // CHOL_LEFT_GROUP) * CHOL_LEFT_GROUP
                    if qb < g0 + CHOL_LEFT_GROUP and g0 > 0:
                        have = g0
                _chol_left_update(ctx, a, n, q0, min(nb, n - q0), pf, have)
                q0 += nb
                qb += 1
            ctx.synchronize()
            p = pf + 1

    if ctim:
        print("CHOL_TIMING n=" + String(n) + " nb=" + String(nb) + " left_update_ms=" + String(Float64(tu) / 1e6) + " factor+info_sync_ms=" + String(Float64(tf) / 1e6)
              + " panel_solve_ms=" + String(Float64(ts) / 1e6) + " trailing_ms=" + String(Float64(tt) / 1e6)
              + " trace+sync_ms=" + String(Float64(tq) / 1e6))
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
    var parts = ctx.enqueue_create_buffer[DType.float32](2 * logdet_blocks(n))
    if sabotage == CHOL_SAB_NONE:
        enqueue_logdet(
            ctx,
            MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(diag.unsafe_ptr())),
            MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(parts.unsafe_ptr())),
            MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(scalar.unsafe_ptr())),
            n,
        )
    else:
        ctx.enqueue_function[sabotage_logdet_kernel](  # small-launch(n: sabotage arm only): the negative control of the log-det fold, reached only with a sabotage id, never by a fit
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
    _ = parts^
    return v
