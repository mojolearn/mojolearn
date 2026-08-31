# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The ONE `n_rows`-length reduction the solver section owns under IDENTICAL,
and it is not this section's: it is `mojolearn.identical.gemm.fp32.v1` at
`m = n = 1, k = n_rows` (an `OP_NT` cell, `gemm/IDENTICAL_FP32_CONTRACT.md`
section 0.1: "`gemv` is `OP_NT` at `n == 1` and is NOT a fourth operation").

NOT A PORT. cuML's coordinate descent performs four reductions over the
rows -- the column norms (`cd.cuh:172`, RAFT `coalescedReduction`), the
per-coordinate `dot(X[:, ci], residual)` (`cd.cuh:206`, cuBLAS `gemv`), and
under `fit_intercept` the column means and the label mean
(`preprocess.cuh:58,65`, RAFT `coalescedReduction` again) -- and ships them
on three different fold shapes, two of which are chosen by the SM count
(`solver/ported/linalg/coalesced_reduction.mojo`'s header) and one of which
is a closed library. Under IDENTICAL all four are THIS function, so the
section has exactly one fold to reason about and it is the gemm lane's,
already contracted, already gated on 62 shapes and launch-invariant
(`gemm/README.md`). The brief for this lane says it in one line: "CALL it
rather than writing a second fold."

WHAT THAT BUYS. Contract section 6: the leaf partition is a pure function
of `k = n_rows` -- `L = contract_leaf_size(n_rows)`, `P = ceil(n_rows/L)`
-- and section 7: serial ascending inside a leaf through `identical_mul_add`
with every seam through `ftz`, then a fixed balanced tree over the `P`
partials. Not of the block size, the grid, the plan, the vendor, or how
many other columns shared the launch. The host oracle
(`solver/mojo_only/cd_oracle.mojo`) computes the same cell through
`gemm_oracle_cell`, imported, so the two cannot hold different opinions
about the fold.

THE SUM IS THE DOT WITH ONES. A column mean needs `sum(x)`, and the profile
has no plain-sum entry. `fma(x, 1.0, acc)` is `x + acc` EXACTLY (a multiply
by one is exact and `fma` rounds once, after the exact product), so the
profile dot against an all-ones vector IS the profile sum, with the same
leaf partition and the same tree, and no second fold is spelled. The ones
vector is `n_rows` floats allocated once per fit.

FAST does not come here. Under FAST the four reductions are the vendor
spellings: `gemv_gpu` for the dot (cuBLAS mirror, `core/gemm.mojo::gemv_n`)
and the ported `coalescedSumMediumKernel` for the norms and means.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from gemm.mojo_only.gemm_identical import (
    identical_gemm_with_plan,
    identical_gemm_workspace_floats,
    choose_gemm_plan,
    GEMM_PLAN_COUNT,
)
from gemm.mojo_only.gemm_oracle import (
    OP_NT,
    contract_leaf_size,
    gemm_oracle_cell,
    gemm_oracle_serial_cell,
)


def profile_dot_workspace_floats(k: Int) -> Int:
    """The largest workspace ANY plan needs at `1 x 1 x k`, so a caller can
    allocate once and pick any plan (the launch-invariance gate picks
    several). Never less than 1."""
    var w = 1
    for plan in range(GEMM_PLAN_COUNT):
        var p = identical_gemm_workspace_floats(1, 1, k, plan)
        if p > w:
            w = p
    return w


def profile_dot_into(
    ctx: DeviceContext,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    k: Int,
    plan: Int,
) raises:
    """`c[0] = sum_p a[p] * b[p]` under `mojolearn.identical.gemm.fp32.v1`.
    ASYNCHRONOUS: the caller owns every buffer and synchronizes.

    `a`, `b` are contiguous `k`-vectors -- sub-buffer views of a column are
    fine and are how `cd.mojo` calls it. `plan < 0` lets `choose_gemm_plan`
    pick; any other value names one of the gemm lane's execution plans,
    which by contract cannot move a bit and which the launch-invariance
    gate exercises. `ws` must hold `profile_dot_workspace_floats(k)`.
    """
    var p = plan
    if p < 0:
        p = choose_gemm_plan(1, 1, k)
    identical_gemm_with_plan(ctx, c, a, b, ws, 1, 1, k, OP_NT, p)


def profile_dot_host(a: List[Float32], b: List[Float32], k: Int) -> Float32:
    """The host value of the same cell: `gemm_oracle_cell` at the contract's
    own leaf size. `a` and `b` hold at least `k` floats from index 0."""
    return gemm_oracle_cell(a, b, OP_NT, 0, 0, 1, 1, k, contract_leaf_size(k))


def serial_dot_host(a: List[Float32], b: List[Float32], k: Int) -> Float32:
    """The SERIAL ascending chain over all `k` -- the one-leaf case, and
    what the profile returns at `k <= 128`. Reported beside the profile
    value by the oracle so a reader can see the fold move (they differ at
    `P > 1`, which is the witness that the tree is reached)."""
    return gemm_oracle_serial_cell(a, b, OP_NT, 0, 0, 1, 1, k)


def column_as_list(x: List[Float32], off: Int, k: Int) -> List[Float32]:
    """`x[off : off + k]` as its own list, for the two host dots above."""
    var out = List[Float32]()
    out.reserve(k)
    for p in range(k):
        out.append(x[off + p])
    return out^
