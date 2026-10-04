# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST bundle for Ordered boosting (lane/apple-fast-sym-ordered,
2026-10-03; default since lane/apple-fast-ordall). One module so the fit
(`ordered_boosting.mojo`) and the structure searcher
(`oblivious_tree_doc_parallel_structure_searcher.mojo`) read the same gate.

`ORD_ALL` is compiled only under the FAST tier on an Apple GPU; IDENTICAL,
DETERMINISTIC and non-Apple FAST compile main's statements unchanged.
`-D MOJOLEARN_ORD_ALL_OFF` compiles it out. At run time the fit turns it on
only when the compressed index carries MORE than `ORD_ALL_MIN_FEATURES`
features (`ord_all_on`); narrower data keeps main's Ordered path.

WHY THE WIDTH GATE (M3 A/Bs, one run per arm, 2026-10-03, gbdt-ordered):
    istella (220 features): 75,758 -> 63,554 ms (-16.1%), auc .979518 ->
        .979529, logloss .190603 -> .190385 (better on both).
    taxi (11 features): 56,473 -> 48,163 ms (-14.7%) but auc .628875 ->
        .628572, logloss .52909 -> .529215 (worse). FAST never degrades
        quality, so narrow data keeps the current path. Of the pieces
        alone, only the fold-order index moved taxi (-5.9%) and it is
        what drops taxi's auc; the others alone were noise (STD_PARALLEL
        istella -1.8%, TREE_LEAN istella ~-0.8%, FOLD_BINS_ONE taxi
        -1.0%), so they no longer have their own defines.

The bundle's four pieces, all on together when `ord_all_on`:
    fold bins in one launch: the fold layout's initial bins from a
        device-resident partition-start table (uploaded once a fit)
        instead of `write_fold_based_initial_bins`' staging buffer, fill
        and copy PER PARTITION and its drain, every tree.
    wide score-noise reduce: the noise sum and the scale's two magnitudes
        over REDUCE_LANES_BLOCK blocks of ORDERED_BLOCK threads (the
        library block sum) instead of the eight 32-lane blocks of
        `_ord_std_lanes_kernel`; no per-tree `MOJOLEARN_ORD_STD_SPLIT` read.
    fold-order index: the compressed index at the fold POSITION (gathered
        once per learn permutation, cached in the pool) as the compiled
        default; no per-level doc-id gather, no `d_observations` buffer.
    lean tree buffers: both planes, the score-noise partials, the tree's
        bins and the searcher's observation scratch allocated once a fit
        and rewritten each tree; the partition sort's three dead host
        readbacks per permutation skipped under the batched estimation.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: the bundle is compiled in (FAST + Apple, not opted out)
comptime ORD_ALL = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_ORD_ALL_OFF"]()
)
#: the bundle runs only above this many compressed-index features. A RANGE
#: rule, not a board row: the bundle's savings are per-feature-group work
#: (fold bins, gathers, score-noise reduce), so they grow with width, while
#: the fold-order index's quality cost showed on narrow data. 32 lies
#: between the two measured widths (about 3x above one, 7x below the other),
#: so the edge is not fitted to either row. The two measured points
#: (module docstring) are 11 and 220 features only: NEEDS NEIGHBOR-SHAPE
#: VALIDATION (24, 32, 33, 48, 64, 128 features) before the edge is trusted.
comptime ORD_ALL_MIN_FEATURES = 32


def ord_all_on(n_features: Int) -> Bool:
    """True when the bundle is compiled in and the data is wide enough."""
    comptime if ORD_ALL:
        return n_features > ORD_ALL_MIN_FEATURES
    else:
        return False
