# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Apple FAST experiment switches for Ordered boosting (lane/apple-fast-sym-ordered,
2026-10-03). Every switch is False unless the build is the FAST tier on an
Apple GPU AND its define is passed; IDENTICAL, DETERMINISTIC and non-Apple
FAST compile main's statements unchanged. One module so the fit
(`ordered_boosting.mojo`) and the structure searcher
(`oblivious_tree_doc_parallel_structure_searcher.mojo`) read the same flags.

    -D MOJOLEARN_ORD_FOLD_BINS_ONE   the fold layout's initial bins in ONE
        launch from a device-resident partition-start table (uploaded once
        a fit), instead of `write_fold_based_initial_bins`' staging buffer,
        fill and copy PER PARTITION (2 x n_folds allocations, 4 x n_folds
        launches) and its drain, every tree.
    -D MOJOLEARN_ORD_STD_PARALLEL    the score-noise sum and the scale's two
        magnitudes as a wide reduce (REDUCE_LANES_BLOCK blocks of
        ORDERED_BLOCK threads, the library block sum) instead of the eight
        32-lane blocks of `_ord_std_lanes_kernel`; also drops the per-tree
        `MOJOLEARN_ORD_STD_SPLIT` env read.
    -D MOJOLEARN_ORD_FOLD_INDEX      the fold-order compressed index
        (`ordered_fold_index`, today an env read per tree, default off) as
        the compiled default: the levels read the compressed index at the
        fold POSITION (gathered once per learn permutation, cached in the
        pool), no per-level doc-id gather launch, no `d_observations`
        buffer.
    -D MOJOLEARN_ORD_TREE_LEAN       the per-tree buffers of the fit (both
        planes, the score-noise partials, the tree's bins, the searcher's
        observation scratch) allocated once a fit and rewritten each tree;
        the partition sort's three dead host readbacks per permutation
        skipped when the batched estimation never reads them.
    -D MOJOLEARN_ORD_ALL             all four.
"""

from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST

#: the tier and the vendor every switch below is gated on
comptime ORD_FAST_APPLE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)
comptime ORD_ALL = ORD_FAST_APPLE and is_defined["MOJOLEARN_ORD_ALL"]()
comptime ORD_FOLD_BINS_ONE = ORD_FAST_APPLE and (
    is_defined["MOJOLEARN_ORD_FOLD_BINS_ONE"]() or ORD_ALL
)
comptime ORD_STD_PARALLEL = ORD_FAST_APPLE and (
    is_defined["MOJOLEARN_ORD_STD_PARALLEL"]() or ORD_ALL
)
comptime ORD_FOLD_INDEX = ORD_FAST_APPLE and (
    is_defined["MOJOLEARN_ORD_FOLD_INDEX"]() or ORD_ALL
)
comptime ORD_TREE_LEAN = ORD_FAST_APPLE and (
    is_defined["MOJOLEARN_ORD_TREE_LEAN"]() or ORD_ALL
)
