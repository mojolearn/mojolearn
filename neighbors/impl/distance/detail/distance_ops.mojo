# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""cuVS's per-pair distance ops, in ONE place, for every lane that needs one.

Reference: cuVS `cpp/src/distance/detail/distance_ops/{cosine,lp_unexp,l2_unexp,
l1,l_inf}.cuh` (cuVS `94c2819`), plus the `DistanceType` enumerators from
`cpp/include/cuvs/distance/distance.h:22-69`. Partial.

WHY THIS FILE EXISTS AND WHY IT IS HERE
---------------------------------------
cuVS keeps ONE `distance_ops/` directory and every consumer -- the pairwise
matrix, brute-force k-NN, k-means, KDE -- reaches the same op structs. This
tree had the same arithmetic in two places: `kde/impl/distance/
distance_ops.mojo` carried `l2_unexp` / `l1` / `l_inf`, and the k-NN lane
carried the expanded L2 identity inline in
`neighbors/checks/pinned_distance_tile.mojo`. Adding cosine and Minkowski
to both would have made four one-off branches out of two. This is the one
op library; `kde/impl/distance/distance_ops.mojo` now imports its three
cores from here rather than spelling them a second time (the bodies moved
CHARACTER FOR CHARACTER, so no kde bit moves; `check-kde` is the proof).

It lives under `neighbors/` because `neighbors/` is where cuVS's brute
force lives and brute force is the heaviest consumer; the direction of the
existing dependency is already kde -> neighbors (`kde/impl/distance/
distance.mojo` imports `neighbors/checks/pinned_distance_tile.mojo`), and
reversing it anywhere would make a cycle.

WHAT AN OP IS, IN THEIR SHAPE
------------------------------
Every cuVS op struct has exactly three parts and this file keeps all three
so a reader can diff:

    static constexpr bool use_norms   ->  `metric_uses_norms`
    DI void core(acc, x, y)           ->  `<name>_core`
    DI void epilog(acc, regxn, regyn) ->  `<name>_epilog`

plus, for `lp_unexp_distance_op` only, a data member `DataT p` set by the
constructor (`lp_unexp.cuh:36-38`). There is no `runtime_params` machinery
in cuVS; `p` is a plain captured member, so it is a plain kernel argument
here.

THE TWO NEW OPS
---------------
    cosine.cuh:63-70    core:   acc += x * y                  (a DOT PRODUCT)
    cosine.cuh:79-89    epilog: acc = 1.0 - acc/(regxn*regyn)
    cosine.cuh:50       use_norms = true, and the norm is the TRUE L2 norm
                        (`distance.cuh:215-216` passes `raft::sqrt_op{}`),
                        not the squared norm the L2 ops want. Their comment
                        at `knn_brute_force.cuh:122` is the whole rule:
                        "cosine needs the l2norm, where as l2 distances
                        needs the squared norm".

    lp_unexp.cuh:54-58  core:   diff = abs(x - y); acc += pow(diff, p)
    lp_unexp.cuh:67     one_over_p = 1.0f / p, formed ONCE
    lp_unexp.cuh:72     epilog: acc = pow(acc, one_over_p)
    lp_unexp.cuh:41     use_norms = false -- there is NO expanded form of
                        Minkowski at general p and none can be invented:
                        `sum |x-y|^p` does not factor into per-row terms
                        for any p except 2. That is not a limitation of
                        this implementation, it is why cuVS's own op is called
                        `lp_UNEXP`.

WHICH METRICS CAN USE THE EXPANDED TRICK, WHICH CANNOT
-------------------------------------------------------
This decides the kernel shape and is worth stating flatly.

    L2Expanded / L2SqrtExpanded   EXPANDED. ||x||^2 + ||y||^2 - 2 x.y over
                                  a GEMM-shaped inner product.
    CosineExpanded                EXPANDED, DIFFERENTLY. The inner product
                                  is the same GEMM; only the epilogue
                                  changes, and the norms are sqrt'd rather
                                  than squared. cuVS's own brute force says
                                  so out loud: `knn_brute_force.cuh:141`
                                  REWRITES the metric to `InnerProduct`
                                  before the tile call and applies
                                  `1 - dot/(nx*ny)` by hand at `:221`.
    L1 / Linf / L2SqrtUnexpanded  UNEXPANDED. Nothing factors.
    LpUnexpanded                  UNEXPANDED AT EVERY p. See above.

So cosine rides the SAME two arms the expanded L2 path already has (vendor
matmul + epilogue under FAST, the pinned per-cell fold under IDENTICAL) and
Minkowski rides the per-cell fold in BOTH modes, exactly as `l1` and `linf`
already do in `kde/`.

THE NUMERIC CONTRACT, PER OP (IDENTITY_PATHS rows 9, 10, 12, 49)
-----------------------------------------------------------------
There is NO tier on which a metric here is refused. Every primitive the two
new ops need was already pinned and gated in `checks/numerics.mojo` before
this file was written, and none of them is invented here:

    acc += x * y            row 9   `identical_mul_add`   (fma under IDENTICAL)
    a / b                   row 49  `identical_div`       (DEVIATION 740)
    xn * yn                 row 9   `identical_mul`       (DEVIATION 826)
    pow(z, q)               row 12  `identical_pow`       (DEVIATION 258;
                                    `portable_powf` = `portable_expf(q *
                                    portable_logf(z))`, one arithmetic on
                                    every column)
    every stored partial    row 10  `ftz`

`pow` being a transcendental is NOT an obstacle and buys no deviation of
its own: row 12's whole point is that `identical_exp` / `identical_log` are
ONE arithmetic built from correctly rounded basic ops, and `identical_pow`
is their composition. What general p DOES cost is ACCURACY, not sameness:
`exp(p log z)` is a few ulp where a native `powf` is one or two, and the
error grows with |p log z|. That is measured rather than asserted --
`neighbors/checks/metric_check.mojo::check_metric_matches_float64_reference`
prints the worst |float32 - float64| per metric and asserts a tolerance --
and it is the same trade `portable_powf`'s existing consumer (the Bayesian
bootstrap temperature) already accepted.

DEVIATION 552 (2026-09-01): p IS RESTRICTED TO THE FINITE POSITIVE NORMALS.
REFERENCE: `lp_unexp_distance_op` takes any `DataT p`. At `p = 0` their
`one_over_p` is `1/0 = +inf` and `pow(acc, inf)` is 0, 1 or inf by
magnitude; at `p < 0` the op is not a metric at all; at `p = inf` the
mathematical limit is Chebyshev but `pow` gives inf or NaN. cuVS never
checks, and `cuvs::neighbors::brute_force::index`'s DEFAULT `metric_arg_`
is `0` (`brute_force.cu:35`), so their own default-constructed index
computes `pow(diff, 0) == 1` for every feature and returns `k^(1/0)` =
garbage. HERE, `p <= 0`, non-finite p and subnormal p are refused BY NAME
at the host entry, before any launch. That is input validation, which is the
one place a refusal is the correct answer, and it is the same class as
kde's DEVIATION 604. `p = 1` and `p = 2` are NOT special-cased into L1 and
L2: the Lp op is what the caller asked for and running a different op
would make `metric='minkowski', p=2` and `metric='euclidean'` two spellings
of one code path, which is exactly the aliasing that hides a bug.

DEVIATION 553 (2026-09-01): A ZERO-NORM ROW UNDER COSINE IS REFUSED.
REFERENCE: `cosine.cuh:86` is a bare `1.0 - acc/(regxn[i]*regyn[j])` with no
guard, and `knn_brute_force.cuh:221` duplicates it by hand with no guard
either. A row of all zeros has `||x|| = 0`, so the divide is `0/0 = NaN`
(or `+-inf` if the dot product is a signed zero of the other sign), and
that NaN then enters a top-k selection. In RAFT's radix selector a NaN's
`twiddle_in` key sorts ABOVE every finite distance, so the query silently
loses a real neighbour to a garbage one; in the FAISS warp queue the
comparison is `<` and a NaN is never less than anything, so it sorts to the
other end. Two selectors, two different wrong answers, no error either way.
HERE, a zero-norm row is refused by name at the host entry (`cosine_zero_norm_
row` below finds it), because cosine distance to the origin is undefined
and returning a vendor-shaped NaN for it is worse than saying so. This is
NOT implementing their bug and it is NOT improving their algorithm: the
arithmetic on every row they can answer is theirs, bit for bit.

WHAT IS NOT IMPLEMENTED
-------------------
- Their `Policy4x4` `pairwise_matrix_cuda` register tile. The kernel below
  gives ONE THREAD to ONE OUTPUT CELL and walks the feature axis ascending
  inside it, which is the discipline `neighbors/checks/
  pinned_distance_tile.mojo` and `kde/impl/distance/distance_ops.mojo`
  already adopted, and for the same reason: the summation order is then a
  pure function of `n_features` and of nothing else. The tile is speed
  only. Recorded in `neighbors/NOT_IMPLEMENTED.tsv`.
- `cosine_cutlass_op` / `get_cutlass_op` (`cosine.cuh:26-34, 92-95`). A
  CUTLASS epilogue hook has no counterpart here.
- `expensive_inner_loop` (`lp_unexp.cuh:44`). An unroll hint for nvcc.
- The `half` arms of both `core` and `epilog`. Float32 only.
- Their `shared_mem_size<Policy>()` statics, which size the tile this file
  does not have.

A NOTE ON ONE UPSTREAM BUG WE DO NOT CARRY, AND ONE WE DO
----------------------------------------------------------
DO NOT CARRY: `cosine.cuh:83` guards the half arm with
`std::is_same_v<AccT, float> && std::is_same_v<AccT, half>`, comparing
`AccT` to itself twice, so it is always false and `:84` is dead. Line 65 in
the same file gets the same test right. This implementation is float32 only so the
arm does not exist here at all.
DO CARRY: cosine's epilogue has NO clamp where the expanded L2 op has two
(`l2_exp.cuh:132-134`, the `val > 0` mask and the self-neighbour
`val*val < eps && regxn == regyn` mask). We do not add one. A cosine
distance slightly below zero or slightly above two is round-off in THEIR
formula, and clamping it would be a different answer from theirs on
exactly the fixtures a user would notice. `metric_check.mojo` plants a
self-neighbour and records what comes out instead of hiding it.

THE SEVEN METRICS (lane/neighbors-rest, 2026-09-15)
----------------------------------------------------
Reference: cuVS `distance_ops/{canberra,correlation,jensen_shannon,hamming,
russel_rao}.cuh`, `distance.cuh`'s CorrelationExpanded setup and cuVS brute
force's `select_min = is_min_close(metric)` for InnerProduct; BrayCurtis
from scikit-learn (`scipy.spatial.distance.braycurtis`). Every one is a
core plus an epilogue in ONE thread per cell, features ascending, each
operand `ftz`'d as loaded, every primitive from `checks/numerics.mojo`. None
reads a norm buffer, so `tiled_brute_force_knn`'s `use_norms = false` arm
takes all seven in both modes and no new arm exists.

    canberra       acc += add != 0 ? |x-y| / add : 0,  add = |x| + |y|
                   (`canberra.cuh`'s `(add != 0) * diff / (add + (add == 0))`
                   is that value: at add = 0, diff = 0 too)
    braycurtis     sum |x-y| / sum |x+y|
    correlation    1 - (k sxy - sx sy) / sqrt((k sxx - sx^2)(k syy - sy^2))
    jensenshannon  acc += -x (logM - log x) - y (logM - log y), m = (x+y)/2,
                   a zero m or x or y reads log as 0; sqrt(acc / 2)
    hamming        count(x != y) * (1/k)
    russellrao     (k - count(x != 0 and y != 0)) * (1/k)
    inner_product  the negation of sum x y, stored; see DEVIATION 2901

DEVIATION 2898 (2026-09-15): CORRELATION'S ROW SUMS ARE FOLDED IN THE CELL.
REFERENCE: `distance.cuh:118-205` precomputes each row's sum (`reduce`) and
squared L2 norm (`norm<L2Norm>`) through RAFT's reductions and the op reads
them from registers. HERE the five accumulators (sx, sy, sxx, syy, sxy) are
running folds over the same ascending feature axis in the cell's thread, so
each row statistic is a pure function of the row and of nothing else. The
epilogue is theirs, grouping included: `numer / sqrt(Q * R)`. A row whose
`Q = k sxx - sx^2` computes to zero or below (a constant row; its
correlation is 0/0) is refused by name at the host entry, as cosine refuses
an all-zero row (DEVIATION 553), and so is a pair of rows whose `Q * R`
flushes to zero (the smallest index Q times the smallest query Q, which is
the smallest product because a correctly rounded product is monotone).
REFERENCE BEHAVIOR NOT CARRIED: no guard, a NaN distance into the selector.

DEVIATION 2899 (2026-09-15): JENSEN-SHANNON REFUSES NEGATIVE ENTRIES AND
RECTIFIES BEFORE THE ROOT. REFERENCE: `jensen_shannon.cuh` takes `log` of
whatever arrives and roots `0.5 * acc` bare. A negative entry makes `log`
NaN, and round-off can leave `acc` a hair below zero on a near-identical
pair, whose root is NaN. HERE a negative entry (tested on the raw value,
`v < 0`) in the index or the queries is refused by name, and `0.5 * acc`
at or below zero roots as zero. Rows are NOT renormalized to sum to one;
the reference does not, so the answer is scipy's only on probability rows.

DEVIATION 2900 (2026-09-15): RUSSELL-RAO READS ITS OPERANDS AS BOOLEANS.
REFERENCE: `russel_rao.cuh` accumulates `x * y`, which is scikit-learn's
boolean count only on 0/1 data. scikit-learn converts to bool (nonzero is
True) before counting, and that is what a caller typing the name gets
there, so the core counts `x != 0 and y != 0`. On 0/1 data the two agree.
BrayCurtis's `0 / 0` (two all-zero rows) reads 0, as Canberra's per-feature
`0 / 0` does in the reference; a zero denominator with a nonzero numerator
is `+inf`, the IEEE quotient.

DEVIATION 2901 (2026-09-15): INNER PRODUCT'S SELECT-MAX IS AN ASCENDING
SELECT OVER THE STORED NEGATION. REFERENCE: `select_min = false` for
InnerProduct, the largest products first. Every selector in this tree is a
select-min over a total order, so the cell stores `0 - acc` (exact, and
`+0` for a zero product, never `-0`: `acc` cannot be `-0` because the fold
starts at `+0`) and the host entry writes `0 - v` back after the sort, which
is `acc` bit for bit. Ties keep the lower index first, as for every metric.
A distance weighting over a similarity has no meaning and is refused by
name (`refuse_similarity_weights`).
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast

from checks.kernel_matrix import (
    K_LIB_ROW_NORM,
    TARGET_COLUMN,
    lib_block_size_for,
)
from checks.numerics import (
    ftz,
    identical_div,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_pow,
    identical_sqrt,
)
from core.pinned_reduce import pinned_block_sum


# ===========================================================================
# THEIR ENUM, THEIR VALUES. `cuvs/distance/distance.h:22-69`.
#
# These are the actual cuVS enumerators, not a local renumbering. Only the
# rows this tree reaches are named; the value gaps (6 = InnerProduct, 8 =
# Canberra) are theirs and are left as holes ON PURPOSE so that a later
# implementation of one of them cannot silently take a number that already means
# something else.
#
# `kde/impl/distance/distance_ops.mojo` used to define its own
# `DIST_L2_SQRT_UNEXPANDED = 0 / DIST_L2_EXPANDED = 1 / DIST_L1 = 2 /
# DIST_LINF = 3`, which were a local invention. It now imports these. Every
# use in that lane was BY NAME (`kde_oracle.mojo`, `kde_check.mojo`,
# `kernel_density.mojo::metric_from_name`, the trace header), so the
# renumbering is invisible to every gate and to the identity card, which
# records the metric NAME.
# ===========================================================================

comptime DIST_L2_EXPANDED = 0
comptime DIST_L2_SQRT_EXPANDED = 1
comptime DIST_COSINE_EXPANDED = 2
comptime DIST_L1 = 3
comptime DIST_L2_UNEXPANDED = 4
comptime DIST_L2_SQRT_UNEXPANDED = 5
comptime DIST_LINF = 7
comptime DIST_LP_UNEXPANDED = 9

# THE SEVEN METRICS OF lane/neighbors-rest (2026-09-15), at their cuVS
# values (`c/include/cuvs/distance/distance.h`). BrayCurtis has an
# enumerator and no op file in the reference (cuML's VALID_METRICS carries
# it as a TODO); its op here is scikit-learn's definition. See THE SEVEN
# METRICS below.
comptime DIST_INNER_PRODUCT = 6
comptime DIST_CANBERRA = 8
comptime DIST_CORRELATION_EXPANDED = 10
comptime DIST_BRAY_CURTIS = 14
comptime DIST_JENSEN_SHANNON = 15
comptime DIST_HAMMING_UNEXPANDED = 16
comptime DIST_RUSSEL_RAO_EXPANDED = 18

#: SCHEDULING: one thread owns one output cell; the block width moves no
#: bit. Same value and same contract as `PAIRWISE_ELEM_TPB` and
#: `PINNED_TILE_TPB`, kept separate so a launch-invariance gate can move
#: one without moving the others.
comptime METRIC_ELEM_TPB = 256

#: `core/row_norms.mojo`'s `NORM_TPB`, read from the SAME kernel-matrix row
#: through the SAME accessor, so the two norm kernels cannot drift in fold
#: width. DEVIATION 508 pins it there; nothing is restated here.
#: See `cosine_row_norm_kernel`.
comptime COSINE_NORM_TPB = lib_block_size_for[K_LIB_ROW_NORM, TARGET_COLUMN]()


def metric_uses_norms(metric: Int) -> Bool:
    """Their `static constexpr bool use_norms`, as a value test.

    True for the three expanded ops (`l2_exp.cuh:91`, `cosine.cuh:50`) and
    false for everything else (`lp_unexp.cuh:41`, `l1.cuh`, `l_inf.cuh`,
    `l2_unexp.cuh`).
    """
    return (
        metric == DIST_L2_EXPANDED
        or metric == DIST_L2_SQRT_EXPANDED
        or metric == DIST_COSINE_EXPANDED
    )


def metric_norm_takes_sqrt(metric: Int) -> Bool:
    """`knn_brute_force.cuh:122-131` and `distance.cuh:215-216`: cosine's
    norm goes through `raft::sqrt_op{}` and the L2 ops' does not.

    Getting this backward gives a plausible, wrong answer on every row,
    which is why it is a named function and not a comment. `core/
    row_norms.mojo`'s own docstring says the same thing about the same
    flag.
    """
    return metric == DIST_COSINE_EXPANDED


def metric_is_known(metric: Int) -> Bool:
    """The fifteen enumerators this tree computes. Anything else is a caller
    error and the host entries refuse it by value."""
    return (
        metric == DIST_L2_EXPANDED
        or metric == DIST_L2_SQRT_EXPANDED
        or metric == DIST_COSINE_EXPANDED
        or metric == DIST_L1
        or metric == DIST_L2_UNEXPANDED
        or metric == DIST_L2_SQRT_UNEXPANDED
        or metric == DIST_LINF
        or metric == DIST_LP_UNEXPANDED
        or metric_is_neighbors_rest(metric)
    )


def metric_is_neighbors_rest(metric: Int) -> Bool:
    """The seven metrics of lane/neighbors-rest, every one computed whole in
    one thread per cell with no norm (`use_norms = false` here, including
    correlation; see THE SEVEN METRICS)."""
    return (
        metric == DIST_INNER_PRODUCT
        or metric == DIST_CANBERRA
        or metric == DIST_CORRELATION_EXPANDED
        or metric == DIST_BRAY_CURTIS
        or metric == DIST_JENSEN_SHANNON
        or metric == DIST_HAMMING_UNEXPANDED
        or metric == DIST_RUSSEL_RAO_EXPANDED
    )


def metric_value_name(metric: Int) -> String:
    """Their enumerator spelling, for messages and cards."""
    if metric == DIST_L2_EXPANDED:
        return String("L2Expanded")
    if metric == DIST_L2_SQRT_EXPANDED:
        return String("L2SqrtExpanded")
    if metric == DIST_COSINE_EXPANDED:
        return String("CosineExpanded")
    if metric == DIST_L1:
        return String("L1")
    if metric == DIST_L2_UNEXPANDED:
        return String("L2Unexpanded")
    if metric == DIST_L2_SQRT_UNEXPANDED:
        return String("L2SqrtUnexpanded")
    if metric == DIST_LINF:
        return String("Linf")
    if metric == DIST_LP_UNEXPANDED:
        return String("LpUnexpanded")
    if metric == DIST_INNER_PRODUCT:
        return String("InnerProduct")
    if metric == DIST_CANBERRA:
        return String("Canberra")
    if metric == DIST_CORRELATION_EXPANDED:
        return String("CorrelationExpanded")
    if metric == DIST_BRAY_CURTIS:
        return String("BrayCurtis")
    if metric == DIST_JENSEN_SHANNON:
        return String("JensenShannon")
    if metric == DIST_HAMMING_UNEXPANDED:
        return String("HammingUnexpanded")
    if metric == DIST_RUSSEL_RAO_EXPANDED:
        return String("RusselRaoExpanded")
    return String("?")


# ===========================================================================
# THE HOST-SIDE VALIDATION THE TWO NEW OPS NEED. Host only; no device bit
# depends on either, and both refuse BEFORE any upload.
# ===========================================================================


def validate_metric_arg(metric: Int, metric_arg: Float32) raises:
    """DEVIATION 552. `metric_arg` is Minkowski's `p` and nothing else
    reads it (`distance.cuh:193` names it `DataT)  // unused` on the cosine
    overload, and every other overload discards it the same way).

    Refused: `p` non-finite, `p <= 0`, and a SUBNORMAL `p`. The subnormal
    clause is not pedantry: `identical_pow` is `exp(p * log(z))` and a
    subnormal `p` is flushed to zero on an FTZ column and not on a
    denormal-honoring one, so the same call would return `z^0 = 1` on Metal
    and `z^tiny` elsewhere. Refusing is the only answer that is the same on
    three columns.
    """
    if metric != DIST_LP_UNEXPANDED:
        # Their overloads accept and discard it; so do we, so a caller that
        # passes the cuML default 2.0 with a non-Lp metric is not refused.
        return
    if metric_arg != metric_arg:
        raise Error(
            "mojolearn distance: metric='minkowski' needs a finite positive"
            " p, got NaN (DEVIATION 552)"
        )
    if metric_arg <= Float32(0.0):
        raise Error(
            "mojolearn distance: metric='minkowski' needs p > 0, got "
            + String(metric_arg)
            + " (DEVIATION 552; p = 0 makes 1/p infinite and p < 0 is not a"
            " metric)"
        )
    if metric_arg == bitcast[DType.float32](UInt32(0x7F800000)):
        raise Error(
            "mojolearn distance: metric='minkowski' needs a finite p; p ="
            " infinity is Chebyshev, which is metric='chebyshev'"
            " (DEVIATION 552)"
        )
    if metric_arg < Float32(1.1754943508222875e-38):
        raise Error(
            "mojolearn distance: metric='minkowski' p = "
            + String(metric_arg)
            + " is subnormal; the flush policy differs by vendor so this"
            " cannot be one arithmetic (DEVIATION 552)"
        )


def cosine_zero_norm_row(
    x: List[Float32], n_rows: Int, n_features: Int
) -> Int:
    """DEVIATION 553. The index of the first row of `x` (row-major
    `n_rows x n_features`) whose every feature is `+-0.0`, or `-1`.

    A row of zeros has `||x|| = 0` and cosine's epilogue divides by it.
    Tested on the RAW BITS rather than on a computed norm: a row that is
    not all-zero can still have a norm that UNDERFLOWS to zero in float32
    (every feature below 2^-75, say), and that row is caught by the norm
    check the caller does afterwards, on the device's own norm, rather
    than by guessing here. This function answers the question that needs
    no arithmetic at all.
    """
    for i in range(n_rows):
        var all_zero = True
        for f in range(n_features):
            var v = x[i * n_features + f]
            if v != Float32(0.0):
                all_zero = False
                break
        if all_zero:
            return i
    return -1


def cosine_zero_norm_row_ptr(
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
) -> Int:
    """`cosine_zero_norm_row` over a HOST POINTER rather than a `List`.

    The k-NN lane's boundary is host pointers (`neighbors/estimator.mojo::
    knn_search_traced`) and the KDE lane's is host lists; the rule and the
    loop are the same, so the two spellings sit side by side rather than
    one calling the other through a copy of the whole matrix.
    """
    for i in range(n_rows):
        var all_zero = True
        for f in range(n_features):
            if x.unsafe_load(i * n_features + f) != Float32(0.0):
                all_zero = False
                break
        if all_zero:
            return i
    return -1


def refuse_similarity_weights(metric: Int, weighted: Bool) raises:
    """DEVIATION 2901: `weights='distance'` inverts a DISTANCE; an inner
    product is a similarity, larger is nearer, so `1/d` would weight the
    farthest neighbour most. Refused by name for a vote or a mean."""
    if weighted and metric == DIST_INNER_PRODUCT:
        raise Error(
            "mojolearn k-NN: weights='distance' with metric='inner_product'"
            " is refused; the distance weighting is 1/d and an inner product"
            " is a similarity (larger is nearer), so it would weight the"
            " farthest neighbour most (DEVIATION 2901). Use weights='uniform'"
        )


def _js_negative_message(what: String, row: Int) -> String:
    """DEVIATION 2899's refusal, in ONE place: the pointer walk and the
    `List` walk below raise the same sentence."""
    return (
        "knn_search: metric='jensenshannon' but " + what + " row "
        + String(row) + " holds a negative entry; the divergence takes the"
        " log of each entry and is undefined below zero (DEVIATION 2899)"
    )


def _correlation_constant_message(what: String, row: Int, q: Float32) -> String:
    """DEVIATION 2898's constant-row refusal, in ONE place."""
    return (
        "knn_search: metric='correlation' but " + what + " row " + String(row)
        + " has zero variance (k*sum(x^2) - sum(x)^2 computes to " + String(q)
        + "); correlation divides by it and is undefined for a constant row"
        " (DEVIATION 2898)"
    )


def _correlation_underflow_message(qi: Float32, qq: Float32) -> String:
    """DEVIATION 2898's flushed-product refusal, in ONE place."""
    return (
        "knn_search: metric='correlation' but the product of the smallest"
        " index and query row variances (" + String(qi) + " * " + String(qq)
        + ") flushes to zero in float32, so a distance would divide by zero"
        " (DEVIATION 2898)"
    )


@always_inline
def _correlation_row_statistic(
    v_sum: Float32, v_sq: Float32, n_features: Int
) -> Float32:
    """`correlation_row_q` over a row's finished sums."""
    return correlation_row_q(v_sum, v_sq, n_features)


def _neighbors_rest_row_refusal(
    metric: Int, x: MutPointer[Float32, MutUntrackedOrigin], n_rows: Int,
    n_features: Int, what: String, mut q_min: Float32,
) raises:
    """The input refusals of DEVIATIONS 2898 and 2899 for one matrix, index
    or queries, on the raw rows. `q_min` receives the smallest correlation
    row statistic `Q` (unchanged for any other metric)."""
    if metric == DIST_JENSEN_SHANNON:
        for i in range(n_rows):
            for f in range(n_features):
                if x.unsafe_load(i * n_features + f) < Float32(0.0):
                    raise Error(_js_negative_message(what, i))
    if metric == DIST_CORRELATION_EXPANDED:
        for i in range(n_rows):
            var sx = Float32(0.0)
            var sxx = Float32(0.0)
            for f in range(n_features):
                var v = ftz(x.unsafe_load(i * n_features + f))
                sx = correlation_sum_core(sx, v)
                sxx = inner_product_core(sxx, v, v)
            var q = _correlation_row_statistic(sx, sxx, n_features)
            if q <= Float32(0.0):
                raise Error(_correlation_constant_message(what, i, q))
            if q < q_min:
                q_min = q


def _neighbors_rest_row_refusal_list(
    metric: Int, x: List[Float32], n_rows: Int, n_features: Int,
    what: String, mut q_min: Float32,
) raises:
    """`_neighbors_rest_row_refusal` over a `List`, which is the host
    oracle's boundary (`cosine_zero_norm_row` and its `_ptr` twin sit side
    by side for the same reason). Same tests, same sentences."""
    if metric == DIST_JENSEN_SHANNON:
        for i in range(n_rows):
            for f in range(n_features):
                if x[i * n_features + f] < Float32(0.0):
                    raise Error(_js_negative_message(what, i))
    if metric == DIST_CORRELATION_EXPANDED:
        for i in range(n_rows):
            var sx = Float32(0.0)
            var sxx = Float32(0.0)
            for f in range(n_features):
                var v = ftz(x[i * n_features + f])
                sx = correlation_sum_core(sx, v)
                sxx = inner_product_core(sxx, v, v)
            var q = _correlation_row_statistic(sx, sxx, n_features)
            if q <= Float32(0.0):
                raise Error(_correlation_constant_message(what, i, q))
            if q < q_min:
                q_min = q


def _neighbors_rest_variance_product(qi: Float32, qq: Float32) raises:
    """The pair test of DEVIATION 2898: the smallest index variance times
    the smallest query variance, which is the smallest product because a
    correctly rounded product is monotone in each operand."""
    if ftz(identical_mul(qi, qq)) <= Float32(0.0):
        raise Error(_correlation_underflow_message(qi, qq))


def refuse_neighbors_rest_inputs(
    metric: Int,
    index: MutPointer[Float32, MutUntrackedOrigin], n_index: Int,
    queries: MutPointer[Float32, MutUntrackedOrigin], n_queries: Int,
    n_features: Int,
) raises:
    """Every input refusal the seven metrics carry, index first, before any
    distance. The GPU entry (`knn_search_traced`) calls this one and the
    host oracle calls the `_list` twin below; both walk the same tests and
    raise the same sentences."""
    if not metric_is_neighbors_rest(metric):
        return
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    var qi = inf
    var qq = inf
    _neighbors_rest_row_refusal(metric, index, n_index, n_features, "index", qi)
    _neighbors_rest_row_refusal(
        metric, queries, n_queries, n_features, "query", qq
    )
    if metric == DIST_CORRELATION_EXPANDED:
        _neighbors_rest_variance_product(qi, qq)


def refuse_neighbors_rest_inputs_list(
    metric: Int, index: List[Float32], n_index: Int,
    queries: List[Float32], n_queries: Int, n_features: Int,
) raises:
    """`refuse_neighbors_rest_inputs` over `List`s, for the host oracle."""
    if not metric_is_neighbors_rest(metric):
        return
    var inf = bitcast[DType.float32](UInt32(0x7F800000))
    var qi = inf
    var qq = inf
    _neighbors_rest_row_refusal_list(
        metric, index, n_index, n_features, "index", qi
    )
    _neighbors_rest_row_refusal_list(
        metric, queries, n_queries, n_features, "query", qq
    )
    if metric == DIST_CORRELATION_EXPANDED:
        _neighbors_rest_variance_product(qi, qq)


# ===========================================================================
# THE OP CORES AND EPILOGUES. One function per cuVS `core()` / `epilog()`.
#
# The three unexpanded cores below MOVED HERE from
# `kde/impl/distance/distance_ops.mojo` on 2026-09-01, body for body and
# comment for comment. Nothing about their arithmetic changed and
# `check-kde`'s device hashes are the proof; that lane now imports them.
# ===========================================================================


@always_inline
def l2_unexp_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`l2_unexp.cuh:62-63`. `diff = x - y; acc += diff * diff`."""
    var diff = ftz(x - y)
    return ftz(identical_mul_add(diff, diff, acc))


@always_inline
def l1_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`l1.cuh:32`. `acc += abs(x - y)`."""
    return ftz(acc + abs(ftz(x - y)))


@always_inline
def linf_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`l_inf.cuh:33-34`. `diff = abs(x - y); acc = max(acc, diff)`.

    Written as their `max`: the larger survives, and on a tie either one
    (they are the same bits -- both non-negative, so no zero-sign
    question). IDENTITY_PATHS row 39 carries the full argument; the strict
    `>` is load-bearing and `kde/checks/kde_check.mojo::
    check_kde_row39_signed_zero_rowmax` is its gate.
    """
    var diff = abs(ftz(x - y))
    if diff > acc:  # row 39: strict `>`, operands non-negative
        return diff
    return acc


@always_inline
def inner_product_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`cosine.cuh:68` and `l2_exp.cuh`'s core, which are the same
    statement: `acc += x * y`.

    ONE multiply-add, pinned for row 9. This is also the arithmetic
    `neighbors/checks/pinned_distance_tile.mojo:99` spells inline for the
    expanded L2 arm; that file is left alone rather than re-pointed here,
    because it is a gated IDENTICAL path and its bits are cited by name in
    IDENTITY_PATHS row 24. When someone re-gates it, this is the call it
    should make.
    """
    return ftz(identical_mul_add(x, y, acc))


@always_inline
def lp_unexp_core(
    acc: Float32, x: Float32, y: Float32, p: Float32
) -> Float32:
    """`lp_unexp.cuh:56-57`. `diff = abs(x - y); acc += pow(diff, p)`.

    `identical_pow(0, p)` is `0` for every `p > 0` (`portable_powf`'s
    explicit zero branch), which is the value the mathematics wants and is
    NOT what `exp(p * log(0)) = exp(-inf)` would give if the branch were
    missing. `p > 0` is guaranteed by `validate_metric_arg`.
    """
    var diff = abs(ftz(x - y))
    return ftz(acc + ftz(identical_pow(diff, p)))


@always_inline
def lp_unexp_epilog(acc: Float32, one_over_p: Float32) -> Float32:
    """`lp_unexp.cuh:72`. `acc = pow(acc, 1/p)`, with `one_over_p` formed
    ONCE by the caller exactly as their `:67` does."""
    return ftz(identical_pow(acc, one_over_p))


@always_inline
def cosine_epilog(acc: Float32, xn: Float32, yn: Float32) -> Float32:
    """`cosine.cuh:86` / `knn_brute_force.cuh:221`, which are the same
    expression written twice in the reference: `1.0 - acc / (xn * yn)`.

    THE GROUPING IS THEIRS AND IS NOT NEGOTIABLE. `acc / (xn * yn)` is one
    product then one divide; `(acc / xn) / yn` is two divides and a
    different float. `xn` and `yn` are the TRUE L2 norms (sqrt applied),
    not the squared ones.

    No clamp. See the module docstring: `l2_exp` has two and cosine has
    none in the reference, and adding one here would be a different answer from
    theirs on exactly the self-neighbour fixtures a user would notice.
    """
    var denom = ftz(identical_mul(xn, yn))
    return ftz(Float32(1.0) - ftz(identical_div(acc, denom)))


@always_inline
def l2_exp_epilog(
    acc: Float32, xn: Float32, yn: Float32, is_sqrt: Bool
) -> Float32:
    """`l2_exp.cuh:125` plus the `val > 0` half of `:132`, and `:142`'s
    sqrt. The SELF-NEIGHBOUR half of their clamp (`:134`, `val*val < eps &&
    regxn == regyn`) is NOT implemented and is a row in
    `neighbors/NOT_IMPLEMENTED.tsv`; `core/expand_distances.mojo` records
    the same omission and this function is written to agree with it bit for
    bit so the two arms of the k-NN tile cannot drift.
    """
    var dist = ftz(
        identical_mul_add(Float32(-2.0), acc, ftz(ftz(xn) + ftz(yn)))
    )
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    if is_sqrt:
        dist = ftz(identical_sqrt(dist))
    return dist


# THE SEVEN METRICS' CORES AND EPILOGUES (lane/neighbors-rest). Host
# callable, like the cores above; `core/knn_host_predict.mojo` calls them.


@always_inline
def canberra_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`canberra.cuh`: `acc += (add != 0) * diff / (add + (add == 0))`,
    which is `diff / add` when `add != 0` and 0 otherwise."""
    var diff = abs(ftz(x - y))
    var add = ftz(abs(x) + abs(y))
    if add != Float32(0.0):
        return ftz(acc + ftz(identical_div(diff, add)))
    return acc


@always_inline
def bray_curtis_num_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """scikit-learn's numerator, `sum |x - y|`."""
    return ftz(acc + abs(ftz(x - y)))


@always_inline
def bray_curtis_den_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """scikit-learn's denominator, `sum |x + y|`."""
    return ftz(acc + abs(ftz(x + y)))


@always_inline
def bray_curtis_epilog(num: Float32, den: Float32) -> Float32:
    """`num / den`; `0 / 0` reads 0 and `num / 0` is `+inf` (DEVIATION
    2900)."""
    if den == Float32(0.0):
        if num == Float32(0.0):
            return Float32(0.0)
        return bitcast[DType.float32](UInt32(0x7F800000))
    return ftz(identical_div(num, den))


@always_inline
def correlation_sum_core(acc: Float32, v: Float32) -> Float32:
    """A row sum's fold step (DEVIATION 2898)."""
    return ftz(acc + v)


@always_inline
def correlation_row_q(s: Float32, ss: Float32, k: Int) -> Float32:
    """`correlation.cuh`'s `Q_denom = k * regx2n - regxn * regxn`, from the
    row's sum `s` and sum of squares `ss`."""
    var kf = Float32(k)
    return ftz(ftz(identical_mul(kf, ss)) - ftz(identical_mul(s, s)))


@always_inline
def correlation_epilog(
    sx: Float32, sy: Float32, sxx: Float32, syy: Float32, sxy: Float32, k: Int
) -> Float32:
    """`correlation.cuh`'s epilogue, their grouping:
    `1 - (k*acc - xn*yn) / sqrt(Q_denom * R_denom)`."""
    var kf = Float32(k)
    var numer = ftz(ftz(identical_mul(kf, sxy)) - ftz(identical_mul(sx, sy)))
    var q = correlation_row_q(sx, sxx, k)
    var r = correlation_row_q(sy, syy, k)
    var root = ftz(identical_sqrt(ftz(identical_mul(q, r))))
    return ftz(Float32(1.0) - ftz(identical_div(numer, root)))


@always_inline
def jensen_shannon_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`jensen_shannon.cuh`'s core, with its zero substitutions as branches:
    `log(m + m_zero)` read as 0 when `m == 0`, `log(x + x_zero)` as 0 when
    `x == 0` (the term is then `-0 * (...)`, which a zero `x` makes 0)."""
    var m = ftz(identical_mul(Float32(0.5), ftz(x + y)))
    var log_m = Float32(0.0)
    if m != Float32(0.0):
        log_m = ftz(identical_log(m))
    var log_x = Float32(0.0)
    if x != Float32(0.0):
        log_x = ftz(identical_log(x))
    var log_y = Float32(0.0)
    if y != Float32(0.0):
        log_y = ftz(identical_log(y))
    var tx = ftz(identical_mul(-x, ftz(log_m - log_x)))
    var ty = ftz(identical_mul(-y, ftz(log_m - log_y)))
    return ftz(acc + ftz(tx + ty))


@always_inline
def jensen_shannon_epilog(acc: Float32) -> Float32:
    """`sqrt(0.5 * acc)`, rectified at zero (DEVIATION 2899)."""
    var half = ftz(identical_mul(Float32(0.5), acc))
    if half <= Float32(0.0):
        return Float32(0.0)
    return ftz(identical_sqrt(half))


@always_inline
def hamming_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """`hamming.cuh`: `acc += (x != y)`."""
    if x != y:
        return ftz(acc + Float32(1.0))
    return acc


@always_inline
def russell_rao_core(acc: Float32, x: Float32, y: Float32) -> Float32:
    """The count of features where both operands are nonzero (DEVIATION
    2900)."""
    if x != Float32(0.0) and y != Float32(0.0):
        return ftz(acc + Float32(1.0))
    return acc


@always_inline
def one_over_features(k: Int) -> Float32:
    """`hamming.cuh`'s `AccT(1.0) / k` and `russel_rao.cuh`'s `1.0f / k_`."""
    return ftz(identical_div(Float32(1.0), Float32(k)))


@always_inline
def hamming_epilog(acc: Float32, k: Int) -> Float32:
    """`acc *= one_over_k`."""
    return ftz(identical_mul(acc, one_over_features(k)))


@always_inline
def russell_rao_epilog(acc: Float32, k: Int) -> Float32:
    """`(k - acc) * one_over_k`."""
    return ftz(identical_mul(ftz(Float32(k) - acc), one_over_features(k)))


@always_inline
def inner_product_epilog(acc: Float32) -> Float32:
    """DEVIATION 2901: the stored value is `0 - acc`."""
    return ftz(Float32(0.0) - acc)


@always_inline
def inner_product_restore(v: Float32) -> Float32:
    """DEVIATION 2901: the reported value, `0 - v`, which is `acc`."""
    return Float32(0.0) - v


# ===========================================================================
# THE NORM. `rowNorm<L2Norm, true>(..., raft::sqrt_op{})`.
# ===========================================================================


def cosine_row_norm_kernel(
    out_norm: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_cols_in: Int32,
):
    """`raft::linalg::rowNorm<L2Norm, true>(norm, a, k, rows, stream,
    raft::sqrt_op{})` -- `distance.cuh:215-216` and
    `knn_brute_force.cuh:125-131`. One block per row.

    THIS IS `core/row_norms.mojo::row_norm_kernel` WITH ONE CHANGE, AND
    THE CHANGE IS THE REASON THE FILE EXISTS. That kernel's `take_sqrt`
    arm ends `total = ftz(sqrt(total))` with the STDLIB sqrt (`core/
    row_norms.mojo:102`), which is DEVIATION 258's exact defect: Mojo's
    `std.math.sqrt` lowers to an APPROXIMATE PTX sqrt on NVIDIA (180,714
    of 2^20 hashed patterns off by one ulp, `check-ieee-arith` on the
    H100), where Metal and HIP are correctly rounded. An IDENTICAL build
    of cosine that took that arm would agree on two columns and differ on
    the third, which is the whole failure DEVIATION 550 fixed for the
    k-NN tile's own sqrt. Everything else here -- `NORM_TPB` from the
    kernel matrix (DEVIATION 508), `identical_mul_add` for the
    contraction (row 9), `pinned_block_sum` for the fold shape (row 19's
    move 2), `ftz` at every seam (row 10) -- is that kernel, unchanged.

    THE SQUARED ARM IS NOT DUPLICATED HERE. `take_sqrt = False` callers
    keep calling `core/row_norms.mojo` and no bit of theirs moves. This
    file only owns the arm cosine needs.

    RUN OWED, and it is a defect in another lane rather than in this one:
    `core/row_norms.mojo:102` should become `identical_sqrt` too, which
    would move `cluster/`'s cosine k-means bits on the NVIDIA column under
    IDENTICAL and is therefore that lane's call, not this one's. Recorded
    in `neighbors/NOT_IMPLEMENTED.tsv`.
    """
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var col = tid
    while col < n_cols:
        var v = ftz(a.unsafe_load(row * n_cols + col))
        acc = ftz(identical_mul_add(v, v, acc))
        col += COSINE_NORM_TPB

    var s0 = pinned_block_sum[COSINE_NORM_TPB](acc)

    if tid == 0:
        var total = ftz(s0)
        if total <= Float32(0.0):
            total = Float32(0.0)
        out_norm.unsafe_store(row, ftz(identical_sqrt(total)))


# ===========================================================================
# THE KERNEL. Their `pairwise_matrix_cuda`, at one thread per output cell.
# ===========================================================================


def metric_distance_kernel(
    dist: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    m_in: Int32,
    n_in: Int32,
    k_in: Int32,
    metric_in: Int32,
    metric_arg_in: Float32,
):
    """`dist[i][j] = op(x_i, y_j)` for every implemented `DistanceType`, one
    thread per cell, the feature axis walked ASCENDING in that thread.

    Stands in for `pairwise_matrix_dispatch` (`distance.cuh:224-225`,
    `:665-666`) plus every `distance_impl` overload's op construction. The
    runtime `metric_in` is their compile-time `distance_tag`: their switch
    (`distance-inl.cuh:261-311`) turns a runtime enum into a template
    argument, and Mojo has no such trick at a kernel boundary, so the
    branch lives inside the thread. It is uniform across the whole grid --
    every thread takes the same arm -- so there is no divergence to pay
    for, only the branch itself.

    `x_norm` / `y_norm` are read ONLY when `metric_uses_norms(metric)`.
    The caller must have filled them with the SQUARED norm for the two L2
    expanded metrics and with the TRUE L2 norm for cosine
    (`metric_norm_takes_sqrt`). A caller that gets that backward gets a
    plausible wrong answer, which is why
    `neighbors/checks/metric_check.mojo::check_cosine_norm_flag_is_load_
    bearing` sabotages exactly that flag.

    `metric_arg_in` is Minkowski's `p` and is ignored by every other arm,
    which is their `DataT)  // unused` at `distance.cuh:193`.

    An unknown metric writes a quiet NaN, so a host oracle gate cannot
    mistake a mis-dispatch for a result. Same contract as
    `kde/impl/distance/distance_ops.mojo:169`.
    """
    var m = Int(m_in)
    var n = Int(n_in)
    var k = Int(k_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= m * n:
        return
    var i = idx // n
    var j = idx % n
    var metric = Int(metric_in)

    var acc = Float32(0.0)

    if metric == DIST_L1:
        for f in range(k):
            acc = l1_core(
                acc,
                ftz(x.unsafe_load(i * k + f)),
                ftz(y.unsafe_load(j * k + f)),
            )
        dist.unsafe_store(idx, acc)
        return

    if metric == DIST_LINF:
        for f in range(k):
            acc = linf_core(
                acc,
                ftz(x.unsafe_load(i * k + f)),
                ftz(y.unsafe_load(j * k + f)),
            )
        dist.unsafe_store(idx, acc)
        return

    if metric == DIST_L2_UNEXPANDED or metric == DIST_L2_SQRT_UNEXPANDED:
        for f in range(k):
            acc = l2_unexp_core(
                acc,
                ftz(x.unsafe_load(i * k + f)),
                ftz(y.unsafe_load(j * k + f)),
            )
        # `l2_unexp.cuh:74-80`: the epilog applies sqrt only for the
        # `L2SqrtUnexpanded` tag.
        if metric == DIST_L2_SQRT_UNEXPANDED:
            acc = ftz(identical_sqrt(acc))
        dist.unsafe_store(idx, acc)
        return

    if metric == DIST_LP_UNEXPANDED:
        var p = metric_arg_in
        for f in range(k):
            acc = lp_unexp_core(
                acc,
                ftz(x.unsafe_load(i * k + f)),
                ftz(y.unsafe_load(j * k + f)),
                p,
            )
        # `:67`: `one_over_p` is formed ONCE, and here that is once per
        # cell rather than once per register tile. Same value, same bits:
        # it is a pure function of `p`.
        var one_over_p = ftz(identical_div(Float32(1.0), p))
        dist.unsafe_store(idx, lp_unexp_epilog(acc, one_over_p))
        return

    if (
        metric == DIST_COSINE_EXPANDED
        or metric == DIST_L2_EXPANDED
        or metric == DIST_L2_SQRT_EXPANDED
    ):
        # The shared `core()`: a dot product. `cosine.cuh:68` and
        # `l2_exp.cuh`'s core are the same statement, which is exactly
        # why `knn_brute_force.cuh:141` can rewrite BOTH to
        # `InnerProduct` and fix them up in an epilogue.
        for f in range(k):
            acc = inner_product_core(
                acc,
                ftz(x.unsafe_load(i * k + f)),
                ftz(y.unsafe_load(j * k + f)),
            )
        var xn = ftz(x_norm.unsafe_load(i))
        var yn = ftz(y_norm.unsafe_load(j))
        if metric == DIST_COSINE_EXPANDED:
            dist.unsafe_store(idx, cosine_epilog(acc, xn, yn))
        else:
            dist.unsafe_store(
                idx,
                l2_exp_epilog(acc, xn, yn, metric == DIST_L2_SQRT_EXPANDED),
            )
        return

    if metric_is_neighbors_rest(metric):
        dist.unsafe_store(idx, neighbors_rest_cell(x, i, y, j, k, metric))
        return

    dist.unsafe_store(idx, bitcast[DType.float32](UInt32(0x7FC00000)))


@always_inline
def neighbors_rest_cell(
    x: MutPointer[Float32, MutAnyOrigin],
    i: Int,
    y: MutPointer[Float32, MutAnyOrigin],
    j: Int,
    k: Int,
    metric: Int,
) -> Float32:
    """One cell of the seven metrics: `x` row `i` against `y` row `j`, the
    feature axis ascending, each operand `ftz`'d as loaded, the op's core,
    then its epilogue (THE SEVEN METRICS)."""
    var acc = Float32(0.0)
    if metric == DIST_CORRELATION_EXPANDED:
        var sx = Float32(0.0)
        var sy = Float32(0.0)
        var sxx = Float32(0.0)
        var syy = Float32(0.0)
        for f in range(k):
            var xv = ftz(x.unsafe_load(i * k + f))
            var yv = ftz(y.unsafe_load(j * k + f))
            sx = correlation_sum_core(sx, xv)
            sy = correlation_sum_core(sy, yv)
            sxx = inner_product_core(sxx, xv, xv)
            syy = inner_product_core(syy, yv, yv)
            acc = inner_product_core(acc, xv, yv)
        return correlation_epilog(sx, sy, sxx, syy, acc, k)
    if metric == DIST_BRAY_CURTIS:
        var den = Float32(0.0)
        for f in range(k):
            var xv = ftz(x.unsafe_load(i * k + f))
            var yv = ftz(y.unsafe_load(j * k + f))
            acc = bray_curtis_num_core(acc, xv, yv)
            den = bray_curtis_den_core(den, xv, yv)
        return bray_curtis_epilog(acc, den)
    for f in range(k):
        var xv = ftz(x.unsafe_load(i * k + f))
        var yv = ftz(y.unsafe_load(j * k + f))
        if metric == DIST_CANBERRA:
            acc = canberra_core(acc, xv, yv)
        elif metric == DIST_JENSEN_SHANNON:
            acc = jensen_shannon_core(acc, xv, yv)
        elif metric == DIST_HAMMING_UNEXPANDED:
            acc = hamming_core(acc, xv, yv)
        elif metric == DIST_RUSSEL_RAO_EXPANDED:
            acc = russell_rao_core(acc, xv, yv)
        else:
            acc = inner_product_core(acc, xv, yv)
    if metric == DIST_CANBERRA:
        return acc
    if metric == DIST_JENSEN_SHANNON:
        return jensen_shannon_epilog(acc)
    if metric == DIST_HAMMING_UNEXPANDED:
        return hamming_epilog(acc, k)
    if metric == DIST_RUSSEL_RAO_EXPANDED:
        return russell_rao_epilog(acc, k)
    return inner_product_epilog(acc)


def cosine_epilog_kernel(
    z: MutPointer[Float32, MutAnyOrigin],
    x_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`knn_brute_force.cuh:215-223`, the hand-inlined cosine fix-up over a
    materialized INNER PRODUCT tile: `z[i][j] <- 1 - z[i][j] / (xn * yn)`.

    This is the FAST arm's second pass, the exact counterpart of
    `core/expand_distances.mojo::expand_distances_kernel` for the L2
    family, and it exists for the same reason that one does: on the tiled
    k-NN path the product comes out of a vendor matmul, and the epilogue
    cannot be fused into a reduction because the top-k needs every distance
    to survive. Under IDENTICAL nothing reaches this kernel --
    `metric_distance_kernel` does the whole cell in one thread -- because a
    vendor matmul's k-split is a summation order nothing here can pin
    (`neighbors/checks/pinned_distance_tile.mojo` carries the argument).
    """
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return
    var row = idx // n_cols
    var col = idx % n_cols
    z.unsafe_store(
        idx,
        cosine_epilog(
            ftz(z.unsafe_load(idx)),
            ftz(x_norm.unsafe_load(row)),
            ftz(y_norm.unsafe_load(col)),
        ),
    )
