# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PCA by covariance eigendecomposition.

PORT OF `cuml/cpp/src/pca/pca.cuh` and the `calEig` /
`truncCompExpVars` pair from `cuml/cpp/src/tsvd/tsvd.cuh` at cuML `00094f7`
(branch-25.08), against RAFT `661a3b8`. Partial. Do not improve.

**THE FILE PATHS ABOVE WERE WRONG UNTIL THIS ROUND.** They said
`raft/linalg/detail/pca.cuh` and `raft/linalg/detail/tsvd.cuh`. Neither file
has ever existed: `git log --all -- cpp/include/raft/linalg/detail/pca.cuh`
in the RAFT checkout returns nothing. PCA lives in cuML, not in RAFT. Every
line number cited from those two paths was therefore invented, and the
corrections below were made by reading the real files.

**RAFT's PCA is NOT randomized SVD**, which is what I expected to find and is
worth stating because it changes what the port costs. `pcaFit` is six steps
(`cuml/cpp/src/pca/pca.cuh:104-139`):

    1  mean over columns                     -> mu           O(rows)
    2  center the input IN PLACE             ->              O(rows)
    3  covariance, Xc^T Xc / (n_rows - 1)    -> cov          O(rows * cols^2)
    4  eigendecompose cov, take the top k    -> components   O(cols^3)
    5  singular_vals = sqrt(var * (n-1))     ->              O(k)
    6  RESTORE the input by adding mu back   ->              O(rows)

**Only steps 1, 2, 3 and 6 touch rows at all.** Everything after step 3 works
on an `n_cols x n_cols` matrix. That is the structural reason this algorithm
suits a GPU so well: one bandwidth-bound pass and one big arithmetic-dense
product, then a small dense problem that does not care where it runs.

The `input`-unchanged CONTRACT is the one to not drop: `input` is an in-out
parameter that must end the call unchanged, and a fit that leaves the
caller's matrix centered is wrong in a way nothing in the fit itself will
reveal. HOW the contract is met now depends on the arm (DEVIATION 42): on
the split-K arm the centering is FUSED into the Gram kernel's read
(`core/gram_splitk.mojo::gram_centered_splitk_into` -- RAFT's own
`stable=false` @todo, `raft/stats/detail/cov.cuh:67-69`, that cuBLAS never
let them ship), so steps 2 and 6 are not launched at all and x is
bit-identical afterwards, not merely restored to within rounding. On the
fallback arm steps 2 and 6 run exactly as cuML ships them, and there step 6
is still the launch that is easy to drop and invisible when dropped
(`check_covariance_fused_and_fallback_restore` watches both arms).

COV_EIG_DQ vs COV_EIG_JACOBI: THE BRANCH IS THREE LINES WIDE AND NOTHING ELSE
------------------------------------------------------------------------------
Their default is `COV_EIG_DQ` (`params.hpp:53`; `pca.pyx:394-395` maps both
`'full'` and `'auto'` to it) and this port ships a Jacobi. The obvious worry
is that the two arms differ in more than the eigensolver. **They do not.**
`calEig` is `tsvd.cuh:98-126` and the whole branch is `:108-120`:

    if (prms.algorithm == COV_EIG_JACOBI) eigJacobi(..., prms.tol, prms.n_iterations);
    else                                  eigDC(...);
    colReverse(components, ...);      // :122  -- BOTH arms
    transpose(components, n_cols);    // :123  -- BOTH arms
    rowReverse(explained_var, ...);   // :125  -- BOTH arms

Scaling, `truncCompExpVars`, the `seqRoot` factor, the component order and
the (absent) sign handling are all downstream of the join and are identical.
The only thing the arm changes besides the numerics of the decomposition is
that `prms.tol` and `prms.n_iterations` are DEAD on the DQ arm and live on
the Jacobi arm. Both arms end in closed cuSOLVER, so neither is more
portable; there is nothing to port from either and a hand-written Jacobi is
the only option here.

ONE STEP OF `pcaFit` WE DO NOT PERFORM: `set_neg_zero`
------------------------------------------------------
`pcaFit` calls `seqRoot(explained_var, singular_vals, n_rows - 1,
n_components, stream, true)` (`pca.cuh:136`) and that trailing `true` is
`set_neg_zero`: any eigenvalue that came back NEGATIVE becomes 0 instead of
`sqrt(negative)` (`raft/matrix/detail/math.cuh:86-95`). A covariance matrix
is positive semi-definite in exact arithmetic, so a negative eigenvalue is
always round-off -- and always REACHABLE, on a rank-deficient or badly scaled
design. Ours computes `sqrt(lam * scale)` with no clamp, so it can return NaN
where cuML returns 0. `tsvdFit` passes NO `set_neg_zero` (`tsvd.cuh:237`,
default false), so the clamp belongs to the PCA path only. NOT FIXED here:
it is an arithmetic change and this lane is read-mostly; see the
dispatch-audit lane report.

ORDER IS PART OF THE ANSWER
---------------------------
`calEig` gets ascending eigenvalues from cuSOLVER and then calls
`raft::matrix::colReverse` (`tsvd.cuh:122`) to make components descend by
explained variance. Copied. PCA components in the wrong order are not a
lesser answer, they are a different one.

SIGN IS ALSO PART OF THE ANSWER, AND HERE WE DO NOT MATCH THEM
--------------------------------------------------------------
An eigenvector is defined up to sign, so any implementation must PICK one or
its output is not reproducible.

WHAT THIS DOCSTRING USED TO SAY, WHICH WAS INVENTED: "RAFT calls
`sign_flip_components` with a `flip_signs_based_on_U` switch
(`detail/pca.cuh:189`). This ports the default arm." **Neither
`sign_flip_components` nor `flip_signs_based_on_U` exists anywhere in RAFT or
cuML.** `grep -rn` over both checkouts returns nothing. The whole passage was
written from a recollection of their code rather than from their code.

WHAT THEY ACTUALLY DO, read from the files:

  - `pcaFit` (`cuml/cpp/src/pca/pca.cuh:104`) -- the entry `PCA.fit()`
    reaches (`pca.pyx:547`) -- **does not flip signs at all.** Its component
    signs are whatever cuSOLVER happened to return.
  - `pcaFitTransform` (`pca.cuh:160`) calls `ML::signFlip`
    (`cuml/cpp/src/tsvd/tsvd.cuh:140`) after the transform, and that one is
    **U-BASED**: it takes the argmax-by-absolute-value down each column of
    the TRANSFORMED data and flips the score column and the matching
    component row together (`tsvd.cuh:151-176`).
  - **AND NO PYTHON PATH REACHES `pcaFitTransform`.** `PCA.fit_transform` is
    `return self.fit(X).transform(X)` (`pca.pyx:582-588`), and the only
    caller of `pcaFitTransform` in either checkout is
    `cpp/tests/sg/pca_test.cu:165`. So for a cuML PCA user the sign of a
    component is unconditionally whatever cuSOLVER returned; there is no flip
    anywhere on the reachable path. tSVD is NOT the same:
    `TruncatedSVD.fit_transform` does call `tsvdFitTransform`
    (`tsvd.pyx:414`), which does flip, at `tsvd.cuh:270`.
  - `raft::matrix::signFlip` / `signFlipKernel`
    (`raft/matrix/detail/math.cuh:367`) is a V-based flip that does exist,
    but **no cuML PCA or tSVD path calls it**; only RAFT's own unit test does
    (`raft/cpp/tests/matrix/math.cu:176`).

OURS: the V-based flip, unconditionally, in `pcaFit`'s position. That is a
DEVIATION on two counts -- they do not flip on this path, and where they do
flip it is U-based. The reason is that a fit whose signs are "whatever the
eigensolver returned" is not reproducible across solvers, and ours is not
cuSOLVER. scikit-learn 1.9 resolves it the same way and in the same
direction: `_fit_full` calls `svd_flip(U, Vt, u_based_decision=False)`, which
is exactly "largest-magnitude entry of each component positive". Verified
against the installed sklearn, not assumed.

That flip is `sign_flip_kernel` below, a transliteration of `signFlipKernel`
(`raft/matrix/detail/math.cuh:367`), and it runs ON THE DEVICE. It replaced a
host loop over the copied-back eigenvectors.

THE RULE, STATED AS A TOTAL ORDER (DEVIATION 525, IDENTITY_PATHS row 30)
------------------------------------------------------------------------
"Largest-magnitude entry positive" is INCOMPLETE and shipping it as though
it were complete is the defect this section was audited for. Two entries can
share the largest magnitude with opposite signs, and then "the" largest
entry does not exist; resolved by whichever lane got there first, the sign
of a whole component becomes a function of a schedule. So the convention
this file implements is the completed one, in three clauses, and all three
are load-bearing:

  1. Let `M` be the largest ABSOLUTE value in the component. NaNs do not
     compete for it (they lose every comparison), and the search starts from
     `+0.0`, so `M >= 0` always.
  2. Let `i` be the LOWEST INDEX whose absolute value equals `M`. This is
     the tie-break, and it is not an invention: it is `cub::ArgMax`'s rule,
     it is cuML's own thrust loop (`tsvd.cuh:160`, `if (val > max)` keeps
     the first strict improvement), and it is `np.argmax`'s and therefore
     scikit-learn's. Four implementations of the same tie rule.
  3. Negate the component iff `entry[i] < 0.0`. Note `<`, not "the sign bit
     is set": `-0.0 < 0.0` is FALSE, so a component whose largest-magnitude
     entry is a zero of either sign is never flipped.

The rule is a pure function of the component's own values. Nothing in it
reads an index range, a lane id, a block count or a mode.

ZERO, NEGATIVE ZERO AND NaN, RULED ON RATHER THAN AVOIDED
----------------------------------------------------------
IDENTITY_PATHS row 13 is the precedent and the reason this paragraph exists:
`-0.0` and `+0.0` compare EQUAL, so which one survives a float min/max is
decided by fold ORDER, and in the ET range kernel the surviving sign reached
the model. Both zeros do reach clause 1's maximum here. **Neither can decide
anything**, and the reason is structural:

  - clause 1 folds `abs(...)`, and its result is consumed only by clause 2's
    `==`, which cannot tell `+0.0` from `-0.0`. Whichever the fold keeps,
    clause 2 selects the same index.
  - clause 3's test is `< 0.0`, which both zeros fail.

That is a BIT-level statement, not a formality, because the flip is a
negation and `-(+0.0)` is `-0.0`. An all-zero component therefore comes back
bit for bit as it went in; had clause 3 tested the sign bit instead, an
all-zero component would come back as a row of `0x80000000` and which one
you got would be decided by a fold order. The other side is pinned too: a
component that IS flipped has its `+0.0` entries come back as `-0.0`, which
is the rule's answer rather than a race's.

An all-NaN component has no defined maximum -- clause 1's `>` admits no NaN,
so `M` is the `+0.0` starting value that no real entry attains -- and it is
left untouched rather than flipped on a sentinel.

WHY THERE IS NO `NUMERIC_IDENTICAL` ARM HERE, AND WHY THAT IS THE ANSWER
-------------------------------------------------------------------------
Two block collectives compute this rule (`block_max` over the magnitudes,
`block_min` over the indices attaining it), and IDENTITY_PATHS row 20 is
about exactly that call shape: `max.gpu.primitives.block.sum` folds across
lanes at the HARDWARE width -- 32 on Apple and NVIDIA, 64 on AMD's wavefront
-- so a fold's shape is a per-vendor number and row 20 had to REPLACE the
sums it found.

**A fold shape cannot move a min or a max.** A sum is non-associative in
floating point; a max is a SELECTION over a total order, exactly associative
and commutative, so a 32-wide tree and a 64-wide tree return the same bits.
The only two escapes from that total order are the `-0.0`/`+0.0` pair and
NaN, and the paragraph above shows both are neutralised inside the rule
rather than left to the fold. So the pathway is already pinned by
construction and an `IDENTICAL` arm would be a second spelling of the same
answer.

The gate is also refused for a second, independent reason: mode-gating this
would MOVE SHIPPED BITS. The flip has been unconditional since `b495627`, so
`NUMERIC_FAST` -- the default, the arm every benchmark and every wheel is
built from -- already carries it. Turning it off under FAST to "match the
mirror" would make the shipped fit's component signs depend on the
eigensolver's rounding again, which is a regression with no measurement
behind it and is the opposite of what a numerics mode is for. The deviation
from cuML is recorded in `NOT_IMPLEMENTED.tsv` and in `DERIVATION_MAP.tsv`; it is not
re-litigated by a build flag.

WHAT IS STILL NOT PINNED, AND IT IS NOT THIS RULE
--------------------------------------------------
The rule is a pure function of the eigenvector's values, so it is exactly as
cross-vendor as those values are. Two entries whose magnitudes differ by one
ULP are not a tie and this rule will not treat them as one: if a vendor's
eigensolver puts them the other way round, clause 2 selects the other index
and the component's sign can invert. That is an amplification of an
upstream difference, not a defect here -- sklearn's `svd_flip` has it too,
and no sign convention that depends on the values can avoid it. It is named
because a cross-vendor comparison that finds a flipped component should look
at the eigensolver first.
"""

# `block_dim` joined this import for `whiten_scale_kernel`, which is the
# only kernel in this file that strides by the launch's own block size
# rather than by a pinned comptime constant; it has no fold, so the block
# size is scheduling and reading it back is safe here.
from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
# DEVIATION 528: NOT `max.gpu.primitives.block`. Its reductions carry
# `constraint failed: Block size must be a greater than warp size`, so at
# `SIGNFLIP_TPB = 32` they REFUSE TO COMPILE on a 64-wide wavefront and this
# whole section failed to build on the MI325X (E2's AMD leg, 2026-08-23).
# These two return the same bits on every width -- a selection, not a sum --
# so nothing here is mode-gated; see `core/pinned_reduce.mojo`'s block
# comment for the +/-0.0 and NaN caveat and why this call site is clear of it.
from core.pinned_reduce import (
    pinned_block_max as block_max,
    pinned_block_min as block_min,
)
from max.gpu.sync import barrier
from std.memory import stack_allocation

from core.gemm import gemm_nt, gemm_tn
# The whitening rescale at the bottom of this file is the FIRST mode-gated
# arithmetic this lane owns: everything above it inherits its identity from
# `core/gram_splitk.mojo` and `core/gemm.mojo`. DEVIATION 584.
from checks.numerics import ftz, identical_div, identical_mul
from core.gram_splitk import gram_centered_splitk_into, gram_splitk_applies
from core.column_stats import (
    STATS_TPB,
    column_mean_kernel,
    scale_in_place_kernel,
    shift_columns_kernel,
)
from decomposition.checks.jacobi_eigh import jacobi_eigh
from decomposition.checks.jacobi_eigh_device import (
    JACOBI_SWEEPS,
    JACOBI_TOL,
    JACOBI_TPB,
    jacobi_eigh_kernel,
)


@fieldwise_init
struct PCAResult(Movable):
    """What `pca_fit` writes back on the host side.

    `components` is `n_components x n_cols` row major, matching theirs and
    scikit-learn's `components_`.
    """

    var components: List[Float64]
    var explained_var: List[Float64]
    var explained_var_ratio: List[Float64]
    var singular_vals: List[Float64]
    var noise_var: Float64


def compute_covariance(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut x_alias2: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    restore_input: Bool = True,
) raises:
    """Steps 1, 2, 3 and 6. The only part of PCA that scales with rows.

    TWO ARMS, ONE PREDICATE (DEVIATION 42). The branch below must take the
    fused arm exactly when `gemm_tn` would take split-K for this shape, so
    it asks the SAME `gram_splitk_applies(m, n, k)` that `gemm_tn` asks --
    one predicate, both readers, no target test of our own. On the split-K
    arm the centering is fused into the Gram kernel's read and x is NEVER
    MODIFIED, so steps 2 and 6 are not launched (RAFT's own `stable=false`
    design, `raft/stats/detail/cov.cuh:67-69`, that cuBLAS's missing
    epilogue hook kept them from shipping; bit-identical to center-then-gemm
    by `check_gram_centered_fused`). The fallback arm is their shipped
    `stable=true` path exactly: meanCenter in place (`cov.cuh:61`), GEMM,
    meanAdd restore (`pca.cuh:138`).
    """
    ctx.enqueue_function[column_mean_kernel](
        mu.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    var cells = n_rows * n_cols
    var fused = gram_splitk_applies(n_cols, n_cols, n_rows)
    if fused:
        # `x_alias` is pure scratch on every arm (the transpose arm
        # overwrites it wholesale), so it doubles as the partials workspace
        # exactly as `gemm_tn` feeds `xt` to `gemm_tn_splitk_into`.
        gram_centered_splitk_into(
            ctx, cov, x, mu, x_alias, n_cols, n_rows
        )
    else:
        ctx.enqueue_function[shift_columns_kernel](
            x.unsafe_ptr(),
            mu.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            Float32(-1.0),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
        # `raft::stats::cov` is a GEMM with `CUBLAS_OP_T, CUBLAS_OP_N`,
        # served by `gemm_tn`'s dispatch -- which re-asks the same predicate,
        # answers False the same way, and lands on the transpose + matmul
        # arm. This was the SECOND contraction-shaped kernel in `core/` and
        # it is why PCA did not move for four benchmark rounds while the
        # other one was being tuned.
        gemm_tn(ctx, cov, x, x_alias, x_alias2, n_cols, n_cols, n_rows)
    # MAX's matmul has no `alpha`, so cuBLAS's scale becomes its own pass
    # over `n_cols^2` elements on both arms. A deviation in launch count,
    # not arithmetic.
    ctx.enqueue_function[scale_in_place_kernel](
        cov.unsafe_ptr(),
        Int32(n_cols * n_cols),
        Float32(1.0) / Float32(n_rows - 1),
        grid_dim=((n_cols * n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    if restore_input and not fused:
        # `raft::stats::meanAdd`, `pca.cuh:138`, ON THE ARM THAT CENTERED IN
        # PLACE only. Do not drop this HERE. The fused arm never modified x,
        # so there is nothing to restore there -- running this launch on
        # that arm would ADD mu to pristine data and corrupt it, which is
        # why the guard is `and not fused` and not a comment.
        ctx.enqueue_function[shift_columns_kernel](
            x.unsafe_ptr(),
            mu.unsafe_ptr(),
            Int32(n_rows),
            Int32(n_cols),
            Float32(1.0),
            grid_dim=((cells + 255) // 256, 1, 1),
            block_dim=(256, 1, 1),
        )
    ctx.synchronize()


def pca_validate(n_rows: Int, n_cols: Int, n_components: Int) raises:
    """The four shape refusals `pca_fit` makes first (cuML's `pcaFit`
    asserts, in their order). Split out 2026-08-23 so the traced surface
    (`decomposition/estimator.mojo`) can run the two halves of the fit
    itself and record the covariance BETWEEN them (IDENTITY_PATHS row 38);
    `pca_fit` is unchanged for every other caller."""
    if n_cols <= 1:
        raise Error("Parameter n_cols: number of columns cannot be less than two")
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if n_components <= 0:
        raise Error(
            "Parameter n_components: number of components cannot be less than one"
        )
    if n_components > n_cols:
        raise Error("n_components cannot exceed n_cols")


def pca_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut x_alias2: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut cov: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
    restore_input: Bool = True,
) raises -> PCAResult:
    """`pca_fit`, all six steps.

    The eigen half runs ON THE DEVICE, as cuSOLVER's `syevj` does for them:
    `jacobi_eigh_kernel` then `sign_flip_kernel`, both on the
    `n_cols x n_cols` matrix. What still comes back to the host is the
    ordering and the truncation; see `eig_and_truncate`.
    `decomposition/checks/jacobi_eigh.mojo` is the host reference the
    device solver is checked against, not the path a fit takes.
    """
    pca_validate(n_rows, n_cols, n_components)

    # Mojo refuses one buffer as two mutable kernel arguments (PORTING.md 24),
    # and X^T X names X twice, so the caller supplies an aliased copy.
    compute_covariance(
        ctx, x, x_alias, x_alias2, mu, cov, n_rows, n_cols, restore_input
    )

    return eig_and_truncate(
        ctx, cov, n_cols, n_components, n_rows - 1
    )


# LAUNCH GEOMETRY, NOT A PROBLEM BOUND. RAFT dispatches this `TPB` on the
# column LENGTH -- 32 for `D <= 32`, then 64, 128, 256
# (`raft/matrix/detail/math.cuh:400-409`) -- purely to give a long column
# more threads. Every loop in the kernel below is strided by `SIGNFLIP_TPB`,
# so any `n` is covered at any of those values and 32 is a throughput choice.
# Matching their dispatch ladder is an OPEN item, recorded in the lane
# report: it changes how fast a 128-column flip runs, not what it returns.
comptime SIGNFLIP_TPB = 32


def sign_flip_kernel(
    v: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """PORT OF `signFlipKernel`, `raft/matrix/detail/math.cuh:367`.

    One block per component. Finds the entry of largest ABSOLUTE value in the
    component, breaks a tie for that magnitude by taking the LOWEST INDEX,
    and negates the whole component if that entry is `< 0.0`. Those three
    clauses are the whole convention and the module docstring states them as
    one (DEVIATION 525); it is scikit-learn's
    `svd_flip(..., u_based_decision=False)` and it is NOT cuML's, which
    records that `pcaFit` does not flip at all and `pcaFitTransform` flips
    on U.

    The rule reads nothing but the component's values -- no lane id, no
    block count, no mode -- which is what makes it identical across vendors.
    `decomposition/checks/pca_check.mojo` holds it to that: the device
    answer must equal a fold-free host scan BITWISE, and the tie, the zero
    cases and the NaN case are each planted rather than hoped for.

    This REPLACED a host loop in `eig_and_truncate` that did the same thing on
    the copied-back eigenvectors. Same convention, same tie-break, on the
    device where theirs is.

    LAYOUT: theirs is column major, so a column is `D` CONTIGUOUS elements at
    `blockIdx.x * D`. `jacobi_eigh_kernel` writes eigenvector `c` into COLUMN
    `c` of a ROW MAJOR `n x n` matrix, so ours is the same column with a
    stride: entry `f` of component `c` is at `f * n + c`. That is the only
    change to their kernel.

    WHY TWO PASSES AND NOT `cub::BlockReduce<cub::KeyValuePair<int, T>>`.
    Theirs reduces a (index, value) pair with `cub::ArgMax`.
    `max.gpu.primitives.block.max` reduces VALUES only, and
    VENDOR_LIBRARIES.md records no block reduce over a custom operator or a
    pair type, so the argmax is split: `block_max` over `|entry|`, then
    `block_min` over the indices that attain it. The alternative was a
    warp-shuffle reduction carrying the index alongside the value, which
    `std.gpu.primitives.warp` can now express. Two passes won because the
    block collective is correct for any `block_size` without a cross-warp
    stage of our own, and because 32 columns of at most 32 entries is not a
    place to spend a hand-rolled reduction. The extra pass is O(n) on a
    matrix whose eigendecomposition already cost O(n^3).

    TIE-BREAK, WHICH IS PART OF THE CONVENTION. `cub::ArgMax` keeps the LOWER
    index on a tie, cuML's own thrust loop keeps the first strict
    improvement (`cuml/cpp/src/tsvd/tsvd.cuh:160`, `if (val > max)`), and
    `np.argmax` -- so scikit-learn's `svd_flip` -- returns the first
    occurrence. Taking the MINIMUM index among the entries attaining the
    maximum reproduces all three. Without it, a component holding both `+x`
    and `-x` at the largest magnitude would have no defined sign, which is
    exactly the irreproducibility the sign flip exists to remove.

    It is TOTAL, not merely two-valued: indices are distinct integers, so a
    three-way or n-way tie has one lowest index just as a two-way one does.
    The check plants a three-way tie for that reason, and plants ties both
    ACROSS lanes (indices 3 and 4 at `SIGNFLIP_TPB = 32`) and WITHIN one
    lane's strided slice (3 and 35), because those are two different things
    that could decide it and only one of them is a fold.
    """
    var n = Int(n_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    # `sh[0]` the largest magnitude, `sh[1]` the index holding it, `sh[2]` the
    # sign. Three slots and no reuse, so no read of one pass can race a write
    # of the next. Their `__shared__ bool need_sign_flip` is `sh[2]`.
    var sh = stack_allocation[
        3,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # PASS 1: the largest magnitude in this component.
    var local = Float32(0.0)
    var f = tid
    while f < n:
        var m = abs(v.unsafe_load(f * n + col))
        if m > local:
            local = m
        f += SIGNFLIP_TPB
    var reduced_max = block_max[block_size=SIGNFLIP_TPB](local)
    # Only thread 0 is promised the reduction's result by CUB's contract, so
    # it is published through shared memory rather than assumed broadcast.
    if tid == 0:
        sh[0] = reduced_max
    barrier()
    var biggest = sh[0]

    # PASS 2: the LOWEST index attaining it. `n` is the sentinel for a thread
    # whose slice holds none. `biggest` is bit for bit one of the compared
    # values WHENEVER ANY REAL ENTRY ATTAINS IT -- a max is a selection --
    # but that is not unconditional and the sentence that used to claim it
    # was is deleted: on an all-NaN component nothing attains the `+0.0`
    # starting value, so `first` stays at the sentinel `n` and PASS 3 leaves
    # `sh[2]` at +1. Untouched is the right answer there; the wrong one would
    # be to flip on a sentinel.
    #
    # `+0.0` and `-0.0` both reach this `==` and it cannot tell them apart,
    # which is the point rather than a gap: whichever zero PASS 1's fold kept
    # (IDENTITY_PATHS row 13's hazard), this pass selects the same index.
    var cand = Float32(n)
    f = tid
    while f < n:
        var fv = Float32(f)
        if abs(v.unsafe_load(f * n + col)) == biggest:
            if fv < cand:
                cand = fv
        f += SIGNFLIP_TPB
    # Indices travel through the reduction as `Float32`, which is exact only
    # below 2^24. Named rather than guarded because the bound is not
    # reachable: `n` here is `n_cols`, and the matrix this kernel is handed
    # is `n_cols x n_cols`, so `n_cols = 2^24` is 2^48 floats.
    var reduced_first = block_min[block_size=SIGNFLIP_TPB](cand)
    if tid == 0:
        sh[1] = reduced_first
        sh[2] = Float32(1.0)
    barrier()
    var first = sh[1]

    # PASS 3: the thread owning that index reads its SIGN. AT MOST one thread
    # matches -- exactly one when any real entry attained the maximum, and
    # none on an all-NaN component, where `first` is the sentinel `n` and no
    # `f < n` can equal it. (The claim that used to stand here, "exactly one
    # thread matches, because `biggest` is attained by at least one real
    # entry", is false in that case and is deleted rather than qualified.)
    #
    # The test is `< Float32(0.0)` and NOT a sign-bit test, deliberately:
    # `-0.0 < 0.0` is FALSE, so a component whose largest-magnitude entry is
    # a zero is never flipped and an all-zero component comes back bit for
    # bit as it arrived. A sign-bit test would negate it instead, turning a
    # row of `0x00000000` into a row of `0x80000000` on the strength of a
    # zero's sign. `check_sign_flip_rule_and_ties` asserts both zero columns
    # on the bits, and that sabotage is the one it catches.
    f = tid
    while f < n:
        if Float32(f) == first:
            if v.unsafe_load(f * n + col) < Float32(0.0):
                sh[2] = Float32(-1.0)
            else:
                sh[2] = Float32(1.0)
        f += SIGNFLIP_TPB
    barrier()

    # `if (need_sign_flip)`, `math.cuh:378`.
    if sh[2] < Float32(0.0):
        f = tid
        while f < n:
            v.unsafe_store(f * n + col, -v.unsafe_load(f * n + col))
            f += SIGNFLIP_TPB


def eig_and_truncate(
    ctx: DeviceContext,
    mut cov: DeviceBuffer[DType.float32],
    n_cols: Int,
    n_components: Int,
    singular_scale: Int,
) raises -> PCAResult:
    """`calEig` + `truncCompExpVars`, shared by PCA and truncated SVD.

    Both callers reach this with an `n_cols x n_cols` symmetric matrix and
    differ only in how they built it: PCA from CENTERED data scaled by
    `1 / (n_rows - 1)`, truncated SVD from RAW data with no scaling. Their
    two files call the same `calEig`, so this is one function here too.

    `singular_scale` is the `n_rows - 1` factor `pca_fit` multiplies by
    before taking the square root (`pca.cuh:136`). `tsvd_fit` passes
    1, because its eigenvalues are already the squared singular values.
    """
    # `calEig` -> cuSOLVER `syevj`, WHICH RUNS ON THE DEVICE. So this runs
    # on the device too. The first version of this port put it on the host,
    # which was inside HOST_AND_DEVICE.md's O(rows) rule but was NOT a mirror
    # of their host/device split, and mirroring that split is the standing
    # rule. `jacobi_eigh.mojo` survives as the reference this is checked
    # against.
    var vec_buf = ctx.enqueue_create_buffer[DType.float32](n_cols * n_cols)
    var info_buf = ctx.enqueue_create_buffer[DType.float32](3)
    ctx.synchronize()
    # `tol` and `sweeps` are `raft::linalg::eigJacobi`'s own defaults
    # (`raft/linalg/eig.cuh:108-109`) AND cuML's Python defaults for the
    # Jacobi arm (`pca.pyx:356-358`: `iterated_power=15`, `tol=1e-7`, both
    # copied into `params` at `pca.pyx:412-413`). They used to be a hardcoded
    # `80` and `1e-10`, neither of which came from their code.
    #
    # WHICH DOOR THOSE DEFAULTS COME FROM MATTERS. cuML's C++ `paramsSolver`
    # defaults `tol = 0.0` and `n_iterations = 15` (`params.hpp:44-45`), so a
    # C++ caller who asks for COV_EIG_JACOBI gets a tolerance of ZERO and
    # always runs all 15 sweeps. We match the PYTHON door, which is the one a
    # user comes in through, not the C++ one.
    ctx.enqueue_function[jacobi_eigh_kernel](
        cov.unsafe_ptr(),
        vec_buf.unsafe_ptr(),
        info_buf.unsafe_ptr(),
        Int32(n_cols),
        Int32(JACOBI_SWEEPS),
        Float32(JACOBI_TOL),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )

    # `signFlipKernel` (`raft/matrix/detail/math.cuh:367`), which is a
    # V-based flip NO cuML PCA path calls; see the module docstring for the
    # deviation and for what `pcaFit` and `pcaFitTransform` actually do.
    # (The name `sign_flip_components` and the path `detail/pca.cuh:189` stood
    # here until this round. Both are invented; neither exists upstream.)
    # One block per
    # eigenvector, and the whole `n_cols x n_cols` basis is fixed in one
    # launch before anything is copied back, which is what REPLACED the host
    # loop that used to run per SELECTED component further down. The
    # convention is unchanged: largest-magnitude entry positive.
    #
    # It runs over every column rather than only the `n_components` kept,
    # because the sign of a column does not depend on the ordering and doing
    # it here means the host side never touches signs at all. Theirs flips
    # `n_components` columns for the same reason: the flip is per column.
    #
    # PINNED ONCE, HERE, FOR BOTH CALLERS (DEVIATION 525). `eig_and_truncate`
    # is `calEig` + `truncCompExpVars` and `tsvd_fit` reaches it too, so
    # truncated SVD carries the same convention with the same tie-break by
    # construction rather than by a second copy of the rule. cuML's two
    # entries do NOT agree with each other on this -- `pcaFit` never flips
    # and `tsvdFitTransform` flips on U (`tsvd.cuh:270`) -- and one shared
    # rule is the deliberate divergence, not an accident of sharing a
    # function.
    #
    # ORDER MATTERS AND IT IS THIS ONE. The flip runs on the eigenvectors as
    # the solver left them, BEFORE the descending-eigenvalue permutation
    # below (`raft::matrix::colReverse`, `tsvd.cuh:122`, which for a Jacobi
    # is a sort rather than a reverse). It commutes with that permutation --
    # the rule is per column and reads only that column -- so putting it
    # first is a choice about where the host stops caring, not a numeric
    # one.
    ctx.enqueue_function[sign_flip_kernel](
        vec_buf.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(SIGNFLIP_TPB, 1, 1),
    )
    ctx.synchronize()

    # Only the O(n_cols^2) post-processing comes back: ordering, truncation
    # and the variance ratios. RAFT does that part on device too
    # (`colReverse`, `truncZeroOrigin`, `matrix::ratio`), so this is still
    # a departure and it is named in UNWIRED.md rather than glossed. It is
    # O(cols^2), never O(rows).
    #
    # The SIGN convention is no longer in that list: `sign_flip_kernel` above
    # applied it on the device, so `h_vec` arrives already flipped.
    var h_cov = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    var h_vec = ctx.enqueue_create_host_buffer[DType.float32](n_cols * n_cols)
    var h_info = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=h_cov.unsafe_ptr(), src_buf=cov)
    ctx.enqueue_copy(dst_ptr=h_vec.unsafe_ptr(), src_buf=vec_buf)
    ctx.enqueue_copy(dst_ptr=h_info.unsafe_ptr(), src_buf=info_buf)
    ctx.synchronize()

    # `eigDC`, the arm cuML's default `svd_solver='auto'` reaches
    # (`pca.pyx:395` maps 'auto' to COV_EIG_DQ, `tsvd.cuh:119` calls eigDC),
    # aborts on a non-zero `dev_info` with "eigensolver couldn't converge to a
    # solution" (`raft/linalg/detail/eig.cuh:149-151`; the identical ASSERT at
    # `:79-81` is `eigDC_legacy`'s and nothing calls it). Their JACOBI arm
    # silently does
    # not (`eig.cuh:310`, `executed_sweeps` fetched and never read). We follow
    # the default arm: an unconverged eigendecomposition returned as if it
    # were one is the same defect as the 32-feature cap, a wrong answer with
    # no error.
    if h_info.unsafe_ptr().unsafe_load(0) == Float32(0.0):
        raise Error(
            "the device Jacobi did not converge in "
            + String(JACOBI_SWEEPS)
            + " sweeps at n_cols = "
            + String(n_cols)
            + ": ||offdiag(A)||_F / ||A||_F is still "
            + String(h_info.unsafe_ptr().unsafe_load(1))
            + " against a tolerance of "
            + String(JACOBI_TOL)
            + ". cuSOLVER's syevj has the same failure mode and the same"
            " remedy, which is more sweeps. A non-symmetric covariance"
            " produces this too; see check_covariance_is_symmetric."
        )

    var a = List[Float64]()
    for i in range(n_cols * n_cols):
        a.append(Float64(h_cov.unsafe_ptr().unsafe_load(i)))
    var vecs = List[Float64]()
    for i in range(n_cols * n_cols):
        vecs.append(Float64(h_vec.unsafe_ptr().unsafe_load(i)))

    # `calEig` + `colReverse`: order DESCENDING by eigenvalue.
    #
    # A SORT, WHERE RAFT ONLY REVERSES, AND WHY THAT IS NOT A REGRESSION.
    # `calEig` calls cuSOLVER `syevj`, which returns eigenvalues ALREADY
    # ASCENDING, so `tsvd.cuh:122` reverses with `raft::matrix::colReverse`
    # and `tsvd.cuh:125` reverses the variances with `rowReverse`: O(n) on
    # the device. **Cyclic Jacobi does not order anything.** Its output sits
    # in whatever order the rotations left it, so a reverse here would return
    # an arbitrary permutation, which for PCA is a DIFFERENT answer and not a
    # worse one. The sort is not a slower way to do their reverse, it is the
    # step their eigensolver already did for them.
    #
    # An O(n_cols^2) selection sort, ON THE HOST, and that is now a real
    # cost rather than a rounding error. The note that used to sit here said
    # it "cannot grow while that cap holds"; the cap is gone, so the sentence
    # is deleted and the number is stated instead: 8128 index comparisons at
    # n_cols = 128 and 32640 at 256, against an eigendecomposition that
    # already cost O(n_cols^3) on the device. It is a permutation of INDICES
    # with no arithmetic in it, so nothing here can move a number, and
    # `HOST_AND_DEVICE.md`'s rule is about O(rows), which this is not.
    #
    # It stops being the right shape when `n_cols` reaches the low thousands.
    #
    # **`nn.argsort` IS NOT THE REPLACEMENT AND THIS COMMENT USED TO SAY IT
    # WAS.** `nn.argsort[target="gpu"]` is correct at 256 elements and
    # non-monotone at 257 and every larger size tried, with the first
    # inversion always at output position 256; it raises nothing and returns a
    # well-formed permutation, so it would silently reorder the components at
    # exactly the sizes this note is about. VENDOR_LIBRARIES.md listed it as
    # device-capable on the strength of its signature. OPEN, and it needs a
    # device sort we can trust, not this one.
    var order = List[Int]()
    for i in range(n_cols):
        order.append(i)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if a[order[j] * n_cols + order[j]] > a[order[i] * n_cols + order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t

    var total = 0.0
    for i in range(n_cols):
        total += a[i * n_cols + i]

    var components = List[Float64]()
    var explained_var = List[Float64]()
    var explained_var_ratio = List[Float64]()
    var singular_vals = List[Float64]()

    for c in range(n_components):
        var src = order[c]
        var lam = a[src * n_cols + src]

        # The sign convention is already applied: `sign_flip_kernel` fixed
        # every column of `vec_buf` on the device before the copy back, so
        # the host only reads. This is where a host `abs()` loop per selected
        # component used to be.
        for f in range(n_cols):
            components.append(vecs[f * n_cols + src])
        explained_var.append(lam)
        explained_var_ratio.append(lam / total if total != 0.0 else 0.0)
        # `weighted_sqrt(explained_var, n_rows - 1)`, `pca.cuh:136`.
        singular_vals.append(sqrt(lam * Float64(singular_scale)))

    # `noise_vars`: the mean of the DISCARDED eigenvalues, or zero if none
    # were discarded (`truncCompExpVars`, `cuml/cpp/src/pca/pca.cuh:74-83`).
    #
    # THEIR GUARD IS `n_components < n_cols && n_components < n_rows`, and
    # ours had only the first half. `n_rows` is not a parameter here because
    # `eig_and_truncate` is shared with `tsvdFit`, so it arrives folded into
    # `singular_scale`, which `pcaFit` passes as `n_rows - 1` and `tsvdFit`
    # passes as 1. `n_components < n_rows` is therefore
    # `n_components <= singular_scale` on the PCA path. The tSVD path has no
    # noise variance in cuML at all (`tsvdFit` does not compute one), so
    # passing 1 makes the guard reject every truncation there, which is the
    # right answer for the wrong-looking reason and is why it is spelled out.
    var noise = 0.0
    if n_components < n_cols and n_components <= singular_scale:
        for c in range(n_components, n_cols):
            noise += a[order[c] * n_cols + order[c]]
        noise /= Float64(n_cols - n_components)

    return PCAResult(
        components^, explained_var^, explained_var_ratio^, singular_vals^, noise
    )


def pca_transform(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    mut components: DeviceBuffer[DType.float32],
    mut out: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
) raises:
    """`pca_transform`: center, then project onto the components.

    The projection is `Xc . components^T`, which is exactly the shape
    `core/gemm.mojo` already computes, so the transform needs no new kernel.
    The input is centered and restored around it, same as the fit.
    """
    var cells = n_rows * n_cols
    ctx.enqueue_function[shift_columns_kernel](
        x.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Float32(-1.0),
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    gemm_nt(
        ctx,
        out,
        x,
        components,
        n_rows,
        n_components,
        n_cols,
    )
    ctx.enqueue_function[shift_columns_kernel](
        x.unsafe_ptr(),
        mu.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Float32(1.0),
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()


# ==========================================================================
# WHITENING. DEVIATIONS 580, 581, 582, 583, 584 (and 585 for the sabotage
# file). NUMBERED 580-585 AND NOT 552-557, which is where they started:
# this lane's own 500-550 band was already full, and 552-557 were taken by
# the glm, kde and neighbors lanes DURING this session. 580-585 was verified
# unused across the whole tree at the moment of writing; a merge that finds
# a collision should renumber here rather than there, because nothing
# outside `decomposition/` and `python/mojolearn/decomposition.py` cites
# these six.
# ==========================================================================
#
# WHAT IT IS. `whiten=True` asks that the transformed columns come out with
# UNIT VARIANCE instead of with the variances the fit found. Component `c`
# carries variance `explained_var[c] = s_c^2 / (n_fit_rows - 1)`, so dividing
# score column `c` by `sqrt(explained_var[c])` normalizes it, and that is the
# whole operation. It is a rescale, not a second decomposition: `components_`,
# `explained_variance_` and `singular_values_` are what the fit produced and
# whitening never touches them, in cuML, in scikit-learn or here.
#
# WHERE THE RESCALE IS APPLIED, AND WHY NOT ON THE OUTPUT. cuML folds it into
# a COPY OF THE COMPONENTS before the projection: multiply the copy by
# `sqrt(n_rows - 1)`, then divide row `c` by `singular_vals[c]`
# (`cuml/cpp/src/pca/pca.cuh:292-302` forward and `:232-243` inverse at the
# 25.08 pin; the same two calls moved into `raft/linalg/detail/pca.cuh:233-243`
# and `:301-311` at 26.08, and the two texts were diffed). scikit-learn
# instead divides the OUTPUT by `sqrt(explained_variance_)`
# (`sklearn/decomposition/_base.py:160-168`). The two agree in exact
# arithmetic and differ in the last bits, and this port takes cuML's, for two
# reasons that both matter here: it is the design source this file mirrors,
# and it is `O(n_components * n_cols)` where scikit-learn's is
# `O(n_rows * n_components)` -- the components matrix is the small one, and
# `HOST_AND_DEVICE.md`'s rule is about what scales with rows.
#
# SO THE PROJECTION KERNEL IS UNCHANGED. `pca_transform` still calls
# `gemm_nt` on a components buffer; whitening only decides WHICH components
# buffer. Nothing on the unwhitened path moves in any numeric mode, and that
# is true BY CONSTRUCTION rather than by a gate: not one line of
# `pca_transform`, `pca_fit`, `eig_and_truncate` or `compute_covariance`
# changed for this, so `whiten=False` executes the same instructions it
# executed before whitening existed. A gate asserting it would be asserting
# that an unedited function is unedited.
#
# DEVIATION 580 -- THE ROW COUNT IS THE FIT'S, NOT THE TRANSFORM'S, AND THIS
# IS A DELIBERATE DIVERGENCE FROM cuML's DENSE PATH. Their scalar is
# `sqrt(prms.n_rows - 1)` (`pca.cuh:293` forward, `:233` inverse) and `pca.pyx` sets
# `params.n_rows = _n_rows`, THE ROW COUNT OF THE MATRIX BEING TRANSFORMED
# (`pca.pyx:770` for transform, `:672` for inverse_transform). So cuML's
# dense `transform` returns a DIFFERENT whitened answer for the same rows
# depending on how many other rows were handed in with them, and
# `pca.transform(X[:100])` does not agree with `pca.transform(X)[:100]`.
# cuML's OWN SPARSE path does not do this -- `_sparse_transform` uses
# `self.n_samples_`, the fit's count (`pca.pyx:716-718`), and so does
# `_sparse_inverse_transform` (`:598-604`) -- and neither does scikit-learn,
# whose `explained_variance_` is a fit-time quantity. Two of the three agree
# and the third disagrees with itself, so this is their bug and not their
# design; `PORTING_RULES.md`'s "do not port their bugs" applies and we take
# the fit's `n_rows`. Held by `check_whiten_row_subset_agrees`, which is also
# the sabotage gate for it.
#
# DEVIATION 581 -- ONE SKIP THRESHOLD FOR BOTH DIRECTIONS. Their forward pass
# skips the division when `abs(s) < 1e-10`
# (`raft/matrix/detail/math.cuh:265-266`, `matrixVectorBinaryDivSkipZero`
# with `return_zero = false`, which returns the entry UNCHANGED); their
# inverse pass skips the multiplication only when `s == 0` exactly
# (`:219-220`, `matrixVectorBinaryMultSkipZero`). For a singular value in
# `(0, 1e-10)` the forward pass therefore does not divide while the inverse
# pass does multiply, and `inverse_transform(transform(X))` comes back scaled
# by `s` in that direction. One threshold in both directions makes the pair
# exact inverses on the skipped rows, which is the property
# `check_whiten_round_trip` can actually state. `1e-10` is theirs and is kept
# (float32 `0x2EDBE6FF`); it is far above the largest subnormal, so the
# division below never sees a subnormal divisor.
#
# DEVIATION 582 -- NOT scikit-learn's CLAMP. They clip the scale up to
# `finfo(dtype).eps` instead of skipping (`_base.py:164-168`), which turns a
# null direction's near-zero score into a number of order `1/eps`. cuML's
# skip leaves that direction scaled by `sqrt(n - 1)` alone, so a score that
# was near zero stays near zero. Both are arbitrary in a rank-deficient
# direction; the skip is the one that does not amplify, and it is the design
# source's. Named because scikit-learn's comment at `_base.py:161` calls out
# `covariance_eigh` BY NAME as the solver that produces those directions, and
# `covariance_eigh` is the only solver this class has.
#
# DEVIATION 583 -- THE SCALAR IS COMPUTED ON THE HOST IN FLOAT64 AND ROUNDED
# ONCE, which is what `math_t(sqrt(prms.n_rows - 1))` does in C++ (an integer
# widened to double, `std::sqrt` on a double, one narrowing conversion). It
# is not a device seam and not a per-vendor quantity: IEEE-754 requires
# float64 `sqrt` and float64 division to be correctly rounded, so every host
# this library builds on produces the same 32 bits from the same `n_rows`.
#
# DEVIATION 584 -- THE ARITHMETIC IS PINNED AT THE SEAM. The rescale is one
# multiply and one divide per cell, and both go through the mode-gated
# spellings: `identical_mul` (row 9's contraction pin, so no codegen may fuse
# the scalar multiply into the neighbouring divide's operand) and
# `identical_div` (row 49's division seam, `portable_divf` under IDENTICAL).
# The intermediate is stored through `ftz` before the second operation, which
# is row 10's requirement for a pinned expression, and the result is stored
# through `ftz` because the components buffer is a seam `gemm_nt` reads.
# NOTE that a division is NOT a tier cost here: `identical_div` is bit-inert
# on Apple (DEVIATION 740) and exists so a denormal-honoring column can be
# aligned to the flushing ones. Whitening runs in all three tiers.


#: cuML's skip-zero threshold, `Type(1e-10)` in
#: `raft/matrix/detail/math.cuh:251` and `:265`. Float32 `0x2EDBE6FF`.
comptime WHITEN_SKIP_ZERO = 1.0e-10


def whiten_scalar(n_fit_rows: Int, inverse: Bool) -> Float32:
    """`sqrt(n_fit_rows - 1)` forward, `1 / sqrt(n_fit_rows - 1)` inverse.

    `pca.cuh:293` and `:233-234` respectively, including their guard
    `prms.n_rows - 1 > 0 ? ... : 0` on the inverse. `pca_validate` already
    refuses `n_rows <= 1` so the guard cannot fire on a fitted model; it is
    kept because dropping a guard that upstream wrote is a silent change of
    behavior on the one input it was written for.

    DEVIATION 583: float64 on the host, one narrowing at the end.
    """
    var d = Float64(n_fit_rows - 1)
    if d <= 0.0:
        return Float32(0.0)
    var r = sqrt(d)
    if inverse:
        return Float32(1.0 / r)
    return Float32(r)


def whiten_scale_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    singular: MutPointer[Float32, MutAnyOrigin],
    n_cells_in: Int32,
    n_cols_in: Int32,
    scalar_in: Float32,
    divide_in: Int32,
):
    """`dst = whiten(src)`: cuML's two passes over the components copy, fused.

    `src` and `dst` are `n_components x n_cols` ROW MAJOR, the layout
    `PCAResult.components` documents and `gemm_nt` reads, so row `c` is the
    component whose singular value is `singular[c]`. Theirs is column major
    and broadcasts along columns; the operation is the same one.

    THEIRS IS TWO KERNELS AND THIS IS ONE, which is a launch-count deviation
    and NOT an arithmetic one: `scalarMultiply` writes the scaled copy to
    memory and `matrixVectorBinaryDivSkipZero` reads it back, so their
    intermediate is a float32 round trip through DRAM. `ftz(identical_mul(
    ...))` below is the same value with the same rounding -- a float32 store
    followed by a float32 load is the identity -- with the store elided. The
    flush is what makes that exact rather than nearly exact, because a
    denormal intermediate would be flushed by the store on an FTZ column and
    kept in a register on a column that honors denormals; `ftz` removes that
    difference, which is precisely IDENTITY_PATHS row 10's argument for
    storing intermediates through it.

    `divide_in != 0` is the forward direction (transform); `0` is the inverse
    (`inverse_transform`), where theirs multiplies by the singular value
    instead. DEVIATION 581 gives both directions the same skip test.

    NO FOLD, NO SHARED MEMORY, ONE CELL PER THREAD. The block size is a
    scheduling number here, not a numeric one, so it is not read from
    `kernel_matrix.mojo`: there is nothing in this kernel whose value could
    depend on how the cells were partitioned across threads or blocks.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_cells_in):
        var c = i // Int(n_cols_in)
        var s = singular.unsafe_load(c)
        # Pass 1, `scalarMultiply`. `identical_mul` and not `*` because the
        # divide (or the second multiply) is the very next operation on this
        # value and row 9's contraction pin is about exactly that adjacency.
        var v = ftz(identical_mul(src.unsafe_load(i), scalar_in))
        # Pass 2, `matrixVector{Div,Mult}SkipZero`, with DEVIATION 581's one
        # threshold. `abs` because theirs is `raft::abs(b)`; a singular value
        # out of `eig_and_truncate` is a square root and cannot be negative,
        # so this can only ever fire on the non-negative side, and the test
        # is written their way rather than narrowed to it.
        if abs(s) < Float32(WHITEN_SKIP_ZERO):
            dst.unsafe_store(i, v)
        elif divide_in != Int32(0):
            dst.unsafe_store(i, ftz(identical_div(v, s)))
        else:
            dst.unsafe_store(i, ftz(identical_mul(v, s)))


def whiten_components(
    ctx: DeviceContext,
    mut src: DeviceBuffer[DType.float32],
    mut dst: DeviceBuffer[DType.float32],
    mut singular: DeviceBuffer[DType.float32],
    n_components: Int,
    n_cols: Int,
    n_fit_rows: Int,
    inverse: Bool,
) raises:
    """Launch `whiten_scale_kernel` over the whole components matrix.

    `dst` is a separate buffer because cuML makes a copy too
    (`rmm::device_uvector<math_t> components_copy` at `pca.cuh:229` and
    `:289`): the caller's `components_` must survive the transform unchanged,
    the same contract `check_input_restored` holds the fit to. Their Python
    layer's sparse arm instead scales `self.components_` in place and scales
    it back afterwards (`pca.pyx:716-724`), which is not round-trip exact in
    float32; the copy is the version that is.
    """
    var cells = n_components * n_cols
    var scalar = whiten_scalar(n_fit_rows, inverse)
    var divide = Int32(1)
    if inverse:
        divide = Int32(0)
    ctx.enqueue_function[whiten_scale_kernel](
        dst.unsafe_ptr(),
        src.unsafe_ptr(),
        singular.unsafe_ptr(),
        Int32(cells),
        Int32(n_cols),
        scalar,
        divide,
        grid_dim=((cells + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()
