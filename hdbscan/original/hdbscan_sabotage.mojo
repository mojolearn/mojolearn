# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The sabotage arms, and the two helpers they corrupt.

NOT A PORT, NOT REACHED by any driver, by `estimator.mojo`, or by
`hdbscan_main.mojo`: every one of those passes `HDB_SAB_NONE` and the
arms below are selected only by `hdbscan/original/hdbscan_check.mojo`.
The shape is `hierarchy/original/edge_order.mojo`'s sabotage block and
`hierarchy/original/sabotage_tile.mojo`'s copied kernel, for the same
reason those exist: a check that cannot SHOW a gate fail has established
that its code runs, not that its gate holds (COMMON_BRIEF rule 6,
`[[reached-but-inert]]`).

Each arm breaks EXACTLY ONE pin. The arm number is threaded through the
ported call chain as an `Int32` argument that defaults to `HDB_SAB_NONE`,
so the production path never reads a sabotage branch it did not take, and
a reader can grep `sabotage` to find every site.

WHAT IS AND IS NOT HERE. The MAX ITSELF lives here (`mr_max3`), because
IDENTITY_PATHS row 39's hazard is the max and a sabotage that cannot
reach the max cannot test it. The rest of the arms are one branch each in
the file that owns the pin, and this file only names them.
"""

from original.numerics import identical_fmax, identical_mul
from std.math import max as hardware_max


comptime HDB_SAB_NONE = 0

comptime HDB_SAB_MR_TWO_WAY = 1
"""Mutual reachability drops `core_dists[col]` from the three-way max, so
`mr(a, b)` becomes `max(core[a], alpha*d(a,b))` and is no longer
SYMMETRIC. Their functor (`cuvs .../reachability.cuh:127-130`) is
`max(core_dists[col], max(core_dists[row], alpha * value))` precisely so
that `mr(a,b) == mr(b,a)`, which is what makes the undirected MST edge
order well defined at all. MUST FAIL the mutual-reachability gate and the
label gate on every fixture whose core distances are not all equal."""

comptime HDB_SAB_HW_MAX = 2
"""The three-way max is the STDLIB `max` rather than the total-order
selection (DEVIATION 1601). IDENTITY_PATHS row 39 MEASURED
`max(+0.0, -0.0)` as `-0.0` on Apple (the second operand) and `+0.0` on
NVIDIA and AMD, so on the planted signed-zero mutual-reachability fixture
this arm returns a DIFFERENT BIT PATTERN from the pinned spelling on
Apple, and a different one again on NVIDIA. It is RECORDED rather than
asserted because LLVM is permitted to fold a `maxnum` into a
compare-select whose tie answer is `+0.0` (the `HW_MAX_CLAMP` lesson in
`numerics.mojo::portable_fmaxf`'s block), so an inert result on one
toolchain is a fact about the fold, not about the pin."""

comptime HDB_SAB_CORE_KTH_PLUS_ONE = 3
"""`core_distances` reads column `min_samples` instead of
`min_samples - 1` (`reachability.cuh:60-62`). The k-th order statistic
becomes the (k+1)-th. MUST FAIL the core-distance gate on every fixture
with two distinct neighbour distances."""

comptime HDB_SAB_CONDENSE_DFS = 4
"""`_build_condensed_hierarchy` walks the dendrogram DEPTH FIRST instead
of level by level (`condense.cuh:38-66` is a level-by-level BFS whose
ORDER decides `next_label`). The condensed tree's SHAPE is the same; its
NUMBERING is not, and every downstream array is indexed by that
numbering. MUST FAIL the condensed-tree gate on any tree deeper than two
levels."""

comptime HDB_SAB_STABILITY_DESCENDING = 5
"""The per-cluster stability fold walks its segment DESCENDING instead of
ascending (DEVIATION 1603). A summation order, so it moves bits wherever
a segment holds three or more terms that do not sum associatively.
RECORDED with the count: a fixture whose segments are all short cannot
separate the two orders and that is a property of the fixture."""

comptime HDB_SAB_EOM_NO_UPDATE = 6
"""Excess of Mass skips the write-back `stability[node] = subtree_
stability` on a deselected node (`select.cuh:227`). Their loop runs from
the leaves toward the root and an ancestor's subtree sum READS the
updated children, so dropping the write-back changes every selection
above the first deselected node. MUST FAIL the selection gate on the
density-gradient fixture."""

comptime HDB_SAB_SKIP_GUARDS = 7
"""Skip DEVIATION 1607's non-finite refusals, so the check can show what
reaches a recorded stage without them (a vendor-payload NaN, an infinite
lambda). RECORDED, never asserted: this arm exists to print the byte the
guard keeps out."""

comptime HDB_SAB_LAMBDA_STD_DIV = 8
"""`lambda = 1 / distance` through the hardware divide instead of
`identical_div` (DEVIATION 1606). Apple's divide is correctly rounded, so
this arm is EXPECTED NOT TO MOVE a bit on this device and the check
REPORTS; it is the arm that would move on a column whose divide is
approximate, exactly as `LINK_SAB_STD_SQRT` is for `sqrt`."""


@always_inline
def mr_max3(
    core_row: Float32, core_col: Float32, scaled: Float32, sabotage: Int32
) -> Float32:
    """`ReachabilityPostProcess::operator()`, with the pin and its arms.

    THEIRS (`cuvs cpp/src/neighbors/detail/reachability.cuh:127-130`):

        return max(core_dists[col], max(core_dists[row], alpha * value));

    OURS is the same expression with `max` spelled `identical_fmax`
    (DEVIATION 1601) and the operand order unchanged, so a reader can
    diff the two line for line. `identical_fmax` under IDENTICAL is
    `portable_fmaxf`, an INTEGER total-order selection that flushes its
    operands, canonicalizes NaN and orders `-0.0` below `+0.0` on every
    vendor; under FAST it is the stdlib `max`, whose zero tie is the
    vendor's own answer, which is what FAST is for.

    THE OPERAND ORDER IS THEIRS AND IS NOT LOAD BEARING UNDER THE PIN. A
    total-order max is commutative and associative, so the three-way
    result is a pure function of the SET `{core_row, core_col, scaled}`.
    That is the property that makes `mr(a,b) == mr(b,a)` hold bit for bit
    and therefore makes the undirected edge order of
    `hierarchy/original/edge_order.mojo` well defined on this graph.
    Under FAST the order IS load bearing on a `(+0, -0)` pair and the
    lane makes no cross-vendor claim there.
    """
    if sabotage == HDB_SAB_MR_TWO_WAY:
        return identical_fmax(core_row, scaled)
    if sabotage == HDB_SAB_HW_MAX:
        # The HARDWARE max, imported at module scope under a distinct name.
        # Mojo forbids an import inside a branch, and aliasing it keeps the
        # sabotage's intent legible beside `identical_fmax` below.
        return hardware_max(core_col, hardware_max(core_row, scaled))
    return identical_fmax(core_col, identical_fmax(core_row, scaled))


@always_inline
def mr_scale(inv_alpha: Float32, value: Float32) -> Float32:
    """`alpha * value` of their functor, where their `alpha` field is the
    caller's `1 / alpha` (`reachability.cuh:222` passes
    `(value_t)1.0 / alpha`). Spelled `identical_mul` so no codegen may
    contract it into the surrounding max chain (DEVIATION 826's reason,
    `numerics.mojo`). At the shipped `alpha = 1.0` the multiplier is
    exactly `1.0` and the product is bit-equal to `value`, so this seam
    moves no bit on the default path."""
    return identical_mul(inv_alpha, value)
