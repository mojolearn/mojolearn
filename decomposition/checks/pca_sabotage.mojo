# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The whitening rescale with its pins BROKEN ON PURPOSE, for the check.

NOT A PORT, NOT REACHED by any driver, by `decomposition/estimator.mojo` or
by the identity card. One copy of
`decomposition/impl/linalg/detail/pca.mojo::whiten_scale_kernel` carrying
arms that `decomposition/checks/pca_check.mojo` selects through the `arm`
argument and that nothing else can reach. The shipped kernel is launched
whenever `arm == PCA_SAB_NONE`, so no shipped bit depends on this file.

Same construction, and for the same reason, as
`cholesky/checks/chol_sabotage.mojo` and `hierarchy/checks/sabotage_tile.mojo`:
a sabotage arm does not belong in a production kernel, and a sabotage that
requires editing source cannot be run by an orchestrator that is forbidden to
edit source.

DEVIATION 585.

WHY EVERY ARM EXISTS. The rule in this tree is that a gate which has never
been shown capable of failing does not count, so each arithmetic decision the
whitening kernel makes gets an arm that is a PLAUSIBLE way to make that
decision wrongly, and the check records what the arm must move and what it
actually moved.

`PCA_SAB_WHITEN_NO_SCALE`       drops the `sqrt(n_fit_rows - 1)` multiply.
                                This is the shape of "I read scikit-learn's
                                `X /= sqrt(explained_variance_)` and divided
                                by the singular value instead", which is off
                                by exactly that factor. The whitened columns
                                come out with variance `1 / (n_fit_rows - 1)`
                                rather than 1, so the unit-variance property
                                gate MUST fail.
`PCA_SAB_WHITEN_NO_DIV`         drops the per-component division entirely, so
                                the transform is the unwhitened one scaled by
                                a constant. Column variances become
                                `explained_var[c] * (n_fit_rows - 1)`, which
                                the unit-variance gate MUST reject, and it
                                rejects each column separately so a single
                                overall scale cannot pass it.
`PCA_SAB_WHITEN_NO_SKIP`        removes the `abs(s) < 1e-10` test and always
                                divides. On a rank-deficient fixture whose
                                trailing singular value is exactly zero this
                                divides by zero and the components become
                                infinities, so the finiteness gate MUST fail.
                                This is the arm that proves the guard is live
                                rather than decorative.
`PCA_SAB_WHITEN_RECIPROCAL`     multiplies by `1 / s` instead of dividing by
                                `s`: two roundings where the pin has one.
                                RAFT itself has this shape elsewhere
                                (`raft/matrix/detail/matrix.cuh:290`
                                `getDiagonalInverseMatrix`), which is why it
                                is a plausible mistake rather than an
                                invented one. It must move BITS, and the gate
                                is a bitwise cell count rather than a
                                tolerance, because on a well-conditioned
                                fixture it will not move a tolerance.
`PCA_SAB_WHITEN_STD_DIV`        plain `/` instead of `identical_div`. EXPECTED
                                INERT ON APPLE and on any column that flushes
                                denormals in hardware -- `portable_divf` is
                                bit-inert there by construction (DEVIATION
                                740) -- so the check asserts nothing about
                                the cell count for this arm and RECORDS it
                                instead. It moves on a denormal-honoring
                                column, which is the column the pin exists
                                for, and the planted-bits fixture is what
                                makes that visible.
`PCA_SAB_WHITEN_NO_FTZ`         drops both flushes. Same expectation and same
                                treatment as the arm above: inert where the
                                hardware already flushes, live where it does
                                not. Driven by the planted-bit fixture, whose
                                subnormal cells are the only ones that can
                                separate the two spellings.
`PCA_SAB_WHITEN_TRANSFORM_ROWS` a DRIVER arm, not a kernel arm. The check
                                passes the row count of the matrix BEING
                                TRANSFORMED as `n_fit_rows`, which is cuML's
                                dense path exactly (`pca.pyx:770` sets
                                `params.n_rows = _n_rows`). DEVIATION 580
                                exists to not do that, and the row-subset
                                gate MUST fail under this arm; that is the
                                evidence the deviation is real and reached.
"""

from std.gpu import block_dim, block_idx, thread_idx

from checks.numerics import ftz, identical_div, identical_mul
from decomposition.impl.linalg.detail.pca import WHITEN_SKIP_ZERO


comptime PCA_SAB_NONE = 0
comptime PCA_SAB_WHITEN_NO_SCALE = 1
comptime PCA_SAB_WHITEN_NO_DIV = 2
comptime PCA_SAB_WHITEN_NO_SKIP = 3
comptime PCA_SAB_WHITEN_RECIPROCAL = 4
comptime PCA_SAB_WHITEN_STD_DIV = 5
comptime PCA_SAB_WHITEN_NO_FTZ = 6
comptime PCA_SAB_WHITEN_TRANSFORM_ROWS = 7
comptime PCA_SAB_COUNT = 8


def pca_sabotage_name(arm: Int) -> String:
    if arm == PCA_SAB_NONE:
        return String("PCA_SAB_NONE")
    if arm == PCA_SAB_WHITEN_NO_SCALE:
        return String("PCA_SAB_WHITEN_NO_SCALE")
    if arm == PCA_SAB_WHITEN_NO_DIV:
        return String("PCA_SAB_WHITEN_NO_DIV")
    if arm == PCA_SAB_WHITEN_NO_SKIP:
        return String("PCA_SAB_WHITEN_NO_SKIP")
    if arm == PCA_SAB_WHITEN_RECIPROCAL:
        return String("PCA_SAB_WHITEN_RECIPROCAL")
    if arm == PCA_SAB_WHITEN_STD_DIV:
        return String("PCA_SAB_WHITEN_STD_DIV")
    if arm == PCA_SAB_WHITEN_NO_FTZ:
        return String("PCA_SAB_WHITEN_NO_FTZ")
    if arm == PCA_SAB_WHITEN_TRANSFORM_ROWS:
        return String("PCA_SAB_WHITEN_TRANSFORM_ROWS")
    return String("PCA_SAB_UNKNOWN")


def pca_sabotage_is_kernel_arm(arm: Int) -> Bool:
    """False for the driver arms, which the check applies by changing what it
    PASSES rather than by launching a different kernel."""
    return arm != PCA_SAB_NONE and arm != PCA_SAB_WHITEN_TRANSFORM_ROWS


def sabotage_whiten_scale_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    singular: MutPointer[Float32, MutAnyOrigin],
    n_cells_in: Int32,
    n_cols_in: Int32,
    scalar_in: Float32,
    divide_in: Int32,
    arm_in: Int32,
):
    """`whiten_scale_kernel` with one decision broken, chosen by `arm_in`.

    Deliberately a COPY rather than the real kernel with a flag: the shipped
    kernel must contain no branch that a sabotage can reach, and a reader of
    `pca.mojo` must not have to hold this file in their head.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_cells_in):
        return
    var arm = Int(arm_in)
    var c = i // Int(n_cols_in)
    var s = singular.unsafe_load(c)
    var a = src.unsafe_load(i)

    # --- pass 1, the scalar multiply -------------------------------------
    var v = Float32(0.0)
    if arm == PCA_SAB_WHITEN_NO_SCALE:
        v = a
    elif arm == PCA_SAB_WHITEN_NO_FTZ:
        v = identical_mul(a, scalar_in)
    else:
        v = ftz(identical_mul(a, scalar_in))

    # --- pass 2, the per-component rescale --------------------------------
    if arm == PCA_SAB_WHITEN_NO_DIV:
        dst.unsafe_store(i, v)
        return
    var skip = abs(s) < Float32(WHITEN_SKIP_ZERO)
    if arm == PCA_SAB_WHITEN_NO_SKIP:
        skip = False
    if skip:
        dst.unsafe_store(i, v)
        return
    if divide_in == Int32(0):
        if arm == PCA_SAB_WHITEN_NO_FTZ:
            dst.unsafe_store(i, identical_mul(v, s))
        else:
            dst.unsafe_store(i, ftz(identical_mul(v, s)))
        return
    if arm == PCA_SAB_WHITEN_RECIPROCAL:
        dst.unsafe_store(i, ftz(identical_mul(v, ftz(identical_div(Float32(1.0), s)))))
    elif arm == PCA_SAB_WHITEN_STD_DIV:
        dst.unsafe_store(i, ftz(v / s))
    elif arm == PCA_SAB_WHITEN_NO_FTZ:
        dst.unsafe_store(i, identical_div(v, s))
    else:
        dst.unsafe_store(i, ftz(identical_div(v, s)))
