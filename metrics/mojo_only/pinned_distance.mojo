"""The silhouette's pairwise distance, computed where the order is visible.

NOT A PORT, and the twin of `neighbors/mojo_only/pinned_distance_tile.mojo`
(DEVIATION 505): that tile is the EXPANDED L2 (`||q||^2 + ||y||^2 - 2 q.y`
from precomputed norms), which is cuVS's `L2Expanded` arm. cuML's
silhouette takes `DistanceType::L2SqrtUnexpanded` for its default
`metric='euclidean'` (`python/cuml/cuml/metrics/pairwise_distances.pyx:50`,
`silhouette_score.pyx` through `_determine_metric`), the UNEXPANDED formula
`sqrt(sum_f (x_f - y_f)^2)`, a different arithmetic, so that tile cannot be
called here and its discipline is mirrored instead:

- one thread per cell, the feature axis walked ASCENDING, each `diff^2`
  folded through `identical_mul_add` (row 9: one rounding, everywhere), each
  stored intermediate through `ftz` (row 10), the root through
  `identical_sqrt` (row 10's sqrt, DEVIATION 258: NVIDIA's stdlib sqrt is
  approximate);
- no shared-memory staging, no register tile, no k-split: nothing to pin
  but the feature order, which is a pure function of `n_cols`.

cuVS computes this with its Contractions tile kernel (`l2_unexp_distance`),
whose inner fold is `acc = fma(diff, diff, acc)` per feature in a tiled
order that is a function of the tile policy; that kernel is NOT ported
(UNPORTED.tsv) and the one-thread formula stands in under both modes.
Under FAST the helpers are the naive chain and the stdlib sqrt; under
IDENTICAL one arithmetic on every vendor.
"""

from mojo_only.numerics import ftz, identical_mul_add, identical_sqrt


@always_inline
def l2sqrt_unexpanded(
    x: MutPointer[Float32, MutAnyOrigin], i: Int, j: Int, n_cols: Int
) -> Float32:
    """`sqrt(sum_f (x[i,f] - x[j,f])^2)`, row-major `x`, ascending f."""
    var acc = Float32(0.0)
    var bi = i * n_cols
    var bj = j * n_cols
    for f in range(n_cols):
        var diff = ftz(ftz(x.unsafe_load(bi + f)) - ftz(x.unsafe_load(bj + f)))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return ftz(identical_sqrt(acc))


def host_l2sqrt_unexpanded(
    x: List[Float32], i: Int, j: Int, n_cols: Int
) -> Float32:
    """The host twin, for the oracle: the same sequence through the same
    helpers (on the host `identical_mul_add` is `std.math.fma` and
    `identical_sqrt` is `portable_sqrtf` under IDENTICAL)."""
    var acc = Float32(0.0)
    var bi = i * n_cols
    var bj = j * n_cols
    for f in range(n_cols):
        var diff = ftz(ftz(x[bi + f]) - ftz(x[bj + f]))
        acc = ftz(identical_mul_add(diff, diff, acc))
    return ftz(identical_sqrt(acc))
