"""`raft::linalg::choleskyRank1Update`, the public entry.

PORT of `raft/linalg/cholesky_r1_update.cuh` at RAFT `ebf9268`
(`upstream/raft-v26.08.00`). Their header is a forwarder to
`detail/cholesky_r1_update.cuh` plus the long doc comment that is the only
prose specification of the routine anywhere in the three checkouts; both are
mirrored, the doc because it states the contract their code assumes and does
not check. **COPY, DO NOT IMPROVE.**

THEIR CONTRACT, in their words (`:18-49`), and what it means here
-----------------------------------------------------------------
- On entry `L` is the factor of `A`, both `(n-1) x (n-1)`; the new column of
  `A'` is stored as the `n`-th column of `L` for UPPER, or as the `n`-th ROW
  for LOWER. On exit `L` is the factor of `A'`, and the elements of `A_new`
  have been OVERWRITTEN by the new row of `L`.
- "If the matrix is not positive definite, or very ill conditioned then the
  new diagonal element of `L` would be NaN. In such a case an exception is
  thrown. The `eps` argument can be used to override this behavior."
- "NOTE: The new mdspan-based API will not be provided for this function."

THE INCREMENTAL PROPERTY, and why this lane mirrors the file at all
--------------------------------------------------------------------
A caller that grows a matrix one row at a time and needs its factor at every
step -- LARS's active set (`cuml/src/solver/lars_impl.cuh:240-320`), a
Gaussian process adding a design point, a kernel-ridge solver adding a
landmark -- can either refactor from scratch at `O(n^3)` per step or update
at `O(n^2)`. That is the routine's whole reason to exist.

**IT ALSO GIVES THIS LANE ITS SHARPEST GATE.** The rank-one update and
`potrf_lower` compute the same factor by two different sequences of
operations. Under IDENTICAL they must agree BIT FOR BIT, and
`check_r1_update_equals_potrf` asserts exactly that, rank by rank, on the
planted fixture. Two spellings of one arithmetic agreeing is the shape
`kde/mojo_only/kde_oracle.mojo`'s header argues for, and here both spellings
are on the DEVICE, so it is a stronger claim than an oracle comparison: a
mistake in a shared helper would move both, but a mistake in the panel's
bracketing or in the update's would move only one.

THE ONE THING THAT CANNOT BE ASSERTED, stated so nobody assumes it
------------------------------------------------------------------
The two agree because at `n <= CHOL_NB_PINNED` the blocked factorization is
ONE panel and its column loop performs, for column `c`, exactly the
operations the rank-one update performs at rank `c+1`: the same ascending
`fma` chain over the same values in the same order, the same `ftz` seams, the
same divide, the same `identical_sqrt`. **Past `n = CHOL_NB_PINNED` they
DIVERGE and they are entitled to** -- the blocked path subtracts a panel's
contribution through the trailing update, in one bracket, where the rank-one
path subtracts each column's contribution separately inside the trsm. That is
DEVIATION 1630's argument applied to two algorithms instead of two block
sizes, and it is why `check_r1_update_equals_potrf` asserts only up to the
panel width and REPORTS beyond it rather than raising.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from cholesky.mojo_only.chol_sabotage import CHOL_SAB_NONE
from cholesky.mojo_only.potrf import CHOL_ELEM_TPB
from cholesky.mojo_only.trsm import CHOL_SOLVE_TPB
from cholesky.ported.linalg.detail.cholesky_r1_update import (
    cholesky_rank1_update as _detail_rank1_update,
    cholesky_rank1_update_workspace_floats as _detail_workspace_floats,
)


def chol_rank1_update_workspace_floats(n: Int) -> Int:
    """Their `n_bytes` query (`detail/:53-56`), in floats. Call it with the
    LARGEST rank the caller will reach, once, exactly as
    `lars_impl.cuh:311-315` does before its loop."""
    return _detail_workspace_floats(n)


def chol_rank1_update(
    ctx: DeviceContext,
    mut l: DeviceBuffer[DType.float32],
    mut workspace: DeviceBuffer[DType.float32],
    n: Int,
    ld: Int,
    eps: Float32,
    mut trace: IdentityTrace,
    tag: StringSlice = "chol.r1",
    tpb: Int = CHOL_SOLVE_TPB,
    elem_tpb: Int = CHOL_ELEM_TPB,
    sabotage: Int = CHOL_SAB_NONE,
) raises:
    """`raft::linalg::choleskyRank1Update`, LOWER arm, float32.

    A forwarder, exactly as theirs is (`:96-110` is a one-line call into
    `detail::choleskyRank1Update`). The reason the forwarder exists in their
    tree is the public/detail split, and the reason it exists here is that
    `cholesky/PORTED_MAP.tsv` maps one of our files to one of theirs and a
    header with no counterpart would be an unrecorded merge.

    `eps` semantics are theirs with DEVIATION 1633's two changes; pass any
    negative value for their default "refuse rather than clamp".
    """
    _detail_rank1_update(
        ctx, l, workspace, n, ld, eps, trace, tag, tpb, elem_tpb, sabotage
    )
