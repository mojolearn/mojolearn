# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""RAFT `cpp/include/raft/stats/detail/silhouette_score.cuh` (ebf9268): the
pieces the BATCHED path reaches -- `SilOp` (:156-169) and `countLabels`
(:100-136). The unbatched `silhouette_score` driver (:187-308:
`pairwise_distance` of the whole n x n matrix, `reduce_cols_by_key`,
`populateAKernel`, `matrixVectorOp(DivOp)`, `reduce(min_op)`) is NOT
PORTED: cuML's Python surface (`silhouette_score.pyx:_silhouette_coeff`)
always calls the batched entry with `chunksize = 40000` by default, so
their dispatch never takes it (PORTING_RULES 0b-i). UNPORTED.tsv.

SilOp, THEIRS (:156-169), per sample with `a` its mean intra-cluster
distance and `b` the least mean distance to another cluster:

    if (a == 0 && b == 0 || a == b) return 0;
    else if (a == -1)               return 0;     (the unbatched path's singleton mark)
    else if (a > b)                 return (b - a) / a;
    else                            return (b - a) / b;

sklearn `silhouette_samples`: `(b - a) / max(a, b)` then `nan_to_num`.
THE +0.0 / -0.0 HAZARD ON `max(a, b)`: sklearn's spelling divides `b - a`
by `max(a, b)`; on an exact tie `b - a` is `+0.0` (x - x is +0.0 in
round-to-nearest) and the quotient is +0.0, and on `a == b == 0` it is
0/0 = NaN, mapped to 0 by `nan_to_num`. RAFT never calls `max`: its first
branch returns `+0.0` for both cases, so no vendor's `max(+0, -0)`
convention (IDENTITY_PATHS row 39: -0.0 on Apple, +0.0 on NVIDIA/AMD)
can reach the score, and the 0/0 is never COMPUTED (the tie branch is
taken before the division; no NaN of any payload exists to be recorded).
`-0.0` cannot arise as `a` or `b` here at all -- both are sums of
nonnegative terms seeded `+0.0` (or `numeric_limits::max()`), and a
`-0.0` minus a `+0.0` never happens. Ours is RAFT's spelling, branch for
branch, plus DEVIATION 656 below.

ROW 39 AUDIT (2026-08-23), why the recorded score is never -0.0 and never
NaN on any finite X:
  * `a` and `b` are seeded `+0.0` and accumulate `ftz(part)` where every
    part is a tree sum of terms `ftz(d / denom)` with `d >= +0.0` (a
    square root of a `+0.0`-seeded sum of squares) and `denom >= 1`, padded
    with `+0.0`. Under round-to-nearest `x + y` is `-0.0` only when BOTH
    operands are `-0.0`, so no sum here is ever `-0.0`; the only zero a
    or b can hold is `+0.0`. Neither is NaN: the terms are in `[+0.0,
    +inf]` and an all-nonnegative sum never forms `inf - inf`.
  * `b - a` on the tie branch is never reached (a == b returns first). Off
    the tie, a and b are each `+0.0` or at least `sqrt(FLT_MIN) / n_rows`
    (the smallest nonzero `d` is `sqrt` of a normal, `1.08e-19`, divided
    by a count <= n_rows), so a nonzero `b - a` is at least one ulp of the
    smaller, ~`6.4e-27 / n_rows`: NEVER subnormal, never flushed to `-0.0`.
    And `|b - a| / max(a, b) >= ulp(max) / max ~ 6e-8` when both are
    nonzero, or exactly `1` when one of them is zero: the quotient never
    underflows, so `ftz` never turns it into a signed zero.
  * The one NaN a FINITE X can produce is `-inf / inf`: an overflow-scale
    coordinate difference (`|x_i - x_j| > 1.8e19`) squares to `+inf`, `d`
    is `+inf`, and `a` becomes `+inf`. `b` NEVER does: the min over
    clusters is seeded `FLT_MAX` and a `+inf` candidate never wins a
    strict `<` against it (RAFT's `reduce(min_op, init MAX)` behaves the
    same), so `b <= FLT_MAX` always and `a = inf` takes the `a > b` branch:
    `(b - a) / a = -inf / inf`. (With `b = FLT_MAX` and a finite `a` the
    score is `(FLT_MAX - a) / FLT_MAX`, which rounds to exactly `1.0` for
    any `a` below half an ulp of FLT_MAX: finite, defined, RAFT's.) That
    NaN is DEVIATION 656's.
=========================================================================
DEVIATION 656 (metrics lane, 2026-08-23): A NaN SILHOUETTE SCORE IS +0.0,
AS sklearn's `nan_to_num` MAKES IT; RAFT RECORDS THE NaN.
=========================================================================
THEIRS (`SilOp`, :156-169) returns the quotient as computed, so `a = inf`
(with any `b <= FLT_MAX`) hands a NaN to the scores array, the mean, and
the caller; `silhouette_samples` in sklearn maps exactly that NaN to 0
(`np.nan_to_num(sil_samples)`, `_unsupervised.py`). OURS returns `+0.0`
for a NaN quotient. WHY: the per-sample scores are a RECORDED DEVICE STAGE
(`metrics_main.mojo`, `metrics.silhouette_samples`) and the mean is a
recorded scalar; a computed NaN's payload is the vendor's (IDENTITY_PATHS
row 39: Apple 0x7fc00000, NVIDIA 0x7fffffff, AMD 0xffc00000), so the same
finite X would give three different cards for one IEEE answer. sklearn's
value is the defined one. MEASURED: `silhouette_check.mojo::
check_silhouette_inf_distances` plants an overflow-scale fixture that
reaches the NaN arm (`a = inf`, once with a finite `b` and once with `b =
FLT_MAX`), shows the `b = inf` arm UNREACHABLE (`b` stays FLT_MAX and the
score is exactly 1.0), and reads `0x00000000` for the NaN rows in both
modes, bitwise equal to the host model. The guard is ONE compare on
the finished quotient; every finite quotient is untouched, so no bit of
any non-NaN score moves.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from metrics.ported.stats.detail.histogram import histogram
from mojo_only.numerics import ftz


@always_inline
def sil_op(a: Float32, b: Float32) -> Float32:
    """`SilOp<DataT>::operator()(a, b)` (:156-169), plus DEVIATION 656's
    NaN -> +0.0 on the finished quotient. The tie branch is taken BEFORE
    any division, so `0 / 0` is never computed (IDENTITY_PATHS row 39)."""
    if (a == Float32(0.0) and b == Float32(0.0)) or a == b:
        return Float32(0.0)
    elif a == Float32(-1.0):
        return Float32(0.0)
    var s: Float32
    if a > b:
        s = ftz(ftz(b - a) / a)
    else:
        s = ftz(ftz(b - a) / b)
    # DEVIATION 656: `-inf / inf` (a = +inf from an overflow-scale
    # coordinate difference) is the one NaN a finite X reaches; sklearn's
    # `nan_to_num` makes it 0.
    if s != s:
        return Float32(0.0)
    return s


def count_labels(
    ctx: DeviceContext,
    mut labels: DeviceBuffer[DType.int32],
    mut bin_count_array: DeviceBuffer[DType.int32],
    n_rows: Int,
    n_unique_labels: Int,
) raises:
    """`countLabels(labels, binCountArray, nRows, nUniqueLabels, ...)`
    (:100-136): `cub::DeviceHistogram::HistogramEven` with unit bins over
    `[0, nUniqueLabels)`, the same integers `histogram.mojo` produces. The
    batched path's `get_cluster_counts` (batched/silhouette_score.cuh:
    109-124) calls exactly this."""
    histogram(ctx, bin_count_array, n_unique_labels, labels, n_rows, Int32(0))
