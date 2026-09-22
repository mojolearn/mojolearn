# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Do a parent node and its two children draw the SAME feature columns?

    tools/with_build_lock.sh pixi run mojo run -I . \\
        ensemble/checks/sibling_column_overlap_check.mojo

WHY THIS CHECK EXISTS. The NVIDIA forest speed lane set
out to port LightGBM's histogram-subtraction trick to the forest path: build
the smaller child's histogram, derive the sibling as `parent - child`. The
arithmetic premise holds here in a way it does not for LightGBM, because
every forest bin field is an integer (`bins.mojo:316,412,500,605`:
`count: UInt32`, `label_sum: Int32`, `weight: Int32`), so a subtraction is
exact rather than drifting.

THE ALGORITHMIC PREMISE IS THE ONE THAT FAILS, and this check is what
measures the failure rather than arguing it. `parent - child` is only
meaningful if the parent's histogram covers the same COLUMNS the child's
does. In this forest it does not. The per-node feature sampler seeds on the
node's index IN THE TREE:

    rng_seed = fnv1a32_hash(seed, treeid, nodeid)   # builder_kernels.cuh:88

with `nodeid = work_items[node_idx].idx`, the tree index
(`kernels/builder_kernels.mojo:211,216-221`). A parent and each of its two
children are three different tree indices, so they seed three independent
permutations and slice three different column sets out of them. The
histogram is indexed by the node's own column SLOT, not by a global feature
id (`kernels/builder_kernels_impl.mojo:2457`,
`column_samples[nid * n_sampled_cols + slot]`), so slot `j` of the parent and
slot `j` of the child are not even the same feature.

WHAT IS PRINTED: for a range of `(n_features, max_features)` shapes, the
mean number of a child's `k` columns that its parent also holds, over many
parent/child triples. That mean is the CEILING on the fraction of a child's
histogram that subtraction could supply. Everything above it has to be built
from the data no matter what, so the ceiling is what decides the lane.

The `k == n` row is included deliberately and is the one case that does NOT
fail: when `max_features` is 1.0 every node draws a permutation of the SAME
full set, so the overlap is total and subtraction is available behind a
column remap. That row is the lane's stated exception, and printing it is
what stops "subtraction is impossible" being recorded as a flat claim when
it is a conditional one.

THE SABOTAGE ARM, `-D FOREST_OVERLAP_SAB_SHARED_SEED=1`, and it must be seen
to FAIL before the clean arm means anything. It drops `nodeid` from the
sampler's seed, which is exactly the counterfactual world in which
subtraction would work: every node then draws the SAME permutation, so a
parent holds every column each child holds and the measured overlap becomes
`k` on every row. If the clean arm and the sabotage arm printed the same
numbers, this file would be measuring something other than the nodeid
dependence, and the sabotage is what rules that out. The `k == n` row cannot
distinguish the two arms (it is `k` either way by construction) and is kept
only as a liveness assertion, which is a weaker claim and is labelled as
one.
"""

from std.sys.compile import is_defined

from ensemble.decisiontree.batched_levelalgo.random_utils import (
    fnv1a32_hash_seed_tree_node,
)
from core.shuffle_iterator import shuffled_feature

comptime SAB_SHARED_SEED = is_defined["FOREST_OVERLAP_SAB_SHARED_SEED"]()
"""Seed the sampler WITHOUT the node id, the counterfactual described above."""


def columns_for_node(
    nodeid: UInt32, treeid: Int32, seed: UInt64, n: Int, k: Int
) -> List[Int]:
    """The column set a node draws, exactly as `sampled_column_at` builds it
    (`kernels/builder_kernels.mojo:315-335`) with `sample_offset` 0, the
    round-0 value every level starts at."""
    var rng_seed = fnv1a32_hash_seed_tree_node(seed, treeid, nodeid)
    comptime if SAB_SHARED_SEED:
        # Every node hashes node id 0, so every node draws one permutation.
        rng_seed = fnv1a32_hash_seed_tree_node(seed, treeid, UInt32(0))
    var out = List[Int]()
    for column_index in range(k):
        out.append(shuffled_feature(n, rng_seed, 0, column_index))
    return out^


def overlap_count(a: List[Int], b: List[Int]) -> Int:
    """How many of `b`'s columns appear in `a`. The sets are tiny (`k` is
    `sqrt(n)` in the benchmarked configuration), so this is the honest
    quadratic scan rather than a hash set whose iteration order would be one
    more thing to pin."""
    var hits = 0
    for i in range(len(b)):
        for j in range(len(a)):
            if a[j] == b[i]:
                hits += 1
                break
    return hits


def report_shape(n: Int, k: Int, n_triples: Int, seed: UInt64) -> Float64:
    """Mean parent/child column overlap over `n_triples` parent nodes.

    A node at tree index `p` has its children appended consecutively, so the
    triples walk `p` and takes `2p+1` / `2p+2` as the children. That is the
    shape a breadth-first tree actually produces; the exact indices do not
    matter to the result because the seed is a hash, but using real ones
    keeps the check honest about what it is modelling."""
    var total = 0
    var pairs = 0
    for p in range(1, n_triples + 1):
        var parent = columns_for_node(UInt32(p), Int32(0), seed, n, k)
        var left = columns_for_node(UInt32(2 * p + 1), Int32(0), seed, n, k)
        var right = columns_for_node(UInt32(2 * p + 2), Int32(0), seed, n, k)
        total += overlap_count(parent, left)
        total += overlap_count(parent, right)
        pairs += 2
    return Float64(total) / Float64(pairs)


def main() raises:
    print("PARENT/CHILD COLUMN OVERLAP, forest per-node feature sampler")
    print("seed 7, treeid 0, sample_offset 0, 4096 parent nodes per shape")
    print("")
    print(
        "  n_features  k=max_features  mean overlap  ceiling  independent-draw"
    )
    print(
        "                                  of k        k*k/n      expectation"
    )

    var seed = UInt64(7)
    var n_triples = 4096

    # HIGGS is the forest benchmark's dataset (28 float32 features,
    # `bench/OPPONENT_REFERENCE.md`), and `max_features` sqrt is the
    # configuration the cuML RandomForest row was taken at.
    var shapes_n = [28, 28, 28, 54, 100, 784, 28]
    var shapes_k = [5, 6, 14, 7, 10, 28, 28]

    var full_row_overlap = Float64(-1.0)
    for s in range(len(shapes_n)):
        var n = shapes_n[s]
        var k = shapes_k[s]
        var mean = report_shape(n, k, n_triples, seed)
        var expect = Float64(k) * Float64(k) / Float64(n)
        if expect > Float64(k):
            expect = Float64(k)
        print(
            "  ",
            n,
            "         ",
            k,
            "            ",
            mean,
            "    ",
            mean / Float64(k),
            "    ",
            expect,
        )
        if n == k:
            full_row_overlap = mean

    print("")
    # LIVENESS. When every node samples all `n` columns the sets are
    # permutations of one another, so the overlap must be exactly `k`. If the
    # loop above read nothing, or `shuffled_feature` returned a constant, or
    # the overlap scan never fired, this row would not be `k` and the check
    # refuses rather than printing a clean zero.
    if full_row_overlap != Float64(28):
        raise Error(
            "LIVENESS FAILED: with k == n == 28 every node draws a"
            " permutation of the same full column set, so the mean overlap"
            " must be exactly 28. Measured "
            + String(full_row_overlap)
            + ". The check is not reaching the sampler."
        )
    print("LIVENESS: k == n == 28 row is exactly 28, the sampler is reached.")
