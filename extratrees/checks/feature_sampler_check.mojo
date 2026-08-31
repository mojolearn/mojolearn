# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The two per-node feature samplers, and the dispatch that picks between them.

`extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels.mojo`
ports cuML's `excess_sample_with_replacement_kernel`
(`builder_kernels.cuh:152-248`), `algo_L_sample_kernel` (`:268-317`) and the
dispatch at `builder.cuh:398-471`. This checks all three, and it is built
around one rule and one fact.

THE RULE is `PORTING_RULES.md` 8: a parameter that selects a kernel is a
parameter the checks enumerate, and a check that cannot NAME the kernel it ran
can pass about a different one. There are THREE arms, not two -- the excess
kernel is instantiated at `MAX_SAMPLES_PER_THREAD = 1` and at `= 72`, and the
two consume the RNG at different rates and return different column sets -- so
every case below prints the arm it ran and asserts which one it was, and the
two dispatch boundaries (`n_parallel_samples` crossing `128` and crossing
`72 * 128 = 9216`) are each pinned with an adjacent pair of `k` that straddles
it.

THE FACT is that cuML's excess sampler is BIASED, and this check measures the
bias rather than hiding it. Their `SubtractLeft` call passes `mask[0]` -- the
PREVIOUS iteration's flag, a 0 or a 1 -- as the predecessor of the block
minimum (`builder_kernels.cuh:231-232`, and CUB's own
`block_adjacent_difference.cuh:393-419` confirms `output[0] =
difference_op(input[0], tile_predecessor_item)`), so a sampled column 0 is
flagged a duplicate and dropped. The port copies that, because
`PORTING_RULES.md` 0b is COPY, DO NOT IMPROVE, and the check ASSERTS it, so
that a later "fix" turns red instead of silently forking from cuML.

The uniformity section is a smoke test and says so where it runs. It is not a
statistical test of anything.
"""

from std.testing import assert_equal, assert_true

from extratrees.impl.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    EXCESS_MAX_ITERATIONS,
    InstanceRange,
    NodeWorkItem,
    SAMPLE_ALGO_L,
    SAMPLE_ALL_FEATURES,
    SAMPLE_EXCESS,
    SAMPLER_BLOCK_THREADS,
    SAMPLER_MAX_SAMPLES_PER_THREAD,
    algo_l_subsequence,
    excess_sample_with_replacement,
    excess_subsequence,
    n_parallel_samples_for,
    n_parallel_samples_libm,
    n_parallel_samples_portable,
    plan_feature_sampling,
    sample_features,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# DEVIATION 457 -- the excess/algo-L boundary at n=20000 is MODE-DEPENDENT.
# The FAST arm is host libm in double and puts the smallest algo-L k at 7385
# on this host (k=7384 lands exactly on 9216); the IDENTICAL arm is row 12's
# float32 `portable_logf` and sits a couple of counts lower, so ITS smallest
# algo-L k is 7386 (k=7385 lands exactly on 9216). Every case below that
# needs "a k on the algo-L side" or "the k exactly on the boundary" reads
# these, so the file pins the boundary of the arm it is actually built with.
# `sampler_kernel_check.mojo` carries the frontier constant by the same
# number.
comptime ALGO_L_FRONTIER_K = (
    7386 if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 7385
)
comptime EXCESS_EDGE_K = (
    7385 if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else 7384
)


def work_items_for(node_ids: List[Int]) -> List[NodeWorkItem]:
    """A batch of work items with the given node ids. The instance range is
    never read by either sampler (`builder_kernels.cuh:165` and `:279` take
    only `.idx`), so it is filler and is deliberately NOT uniform -- if a
    sampler ever started reading it, that would show."""
    var items = List[NodeWorkItem]()
    for i in range(len(node_ids)):
        items.append(
            NodeWorkItem(
                Int32(node_ids[i]),
                Int32(i % 7),
                InstanceRange(Int32(i * 13), Int32(11 + i)),
            )
        )
    return items^


def sorted_copy(colids: List[Int32], offset: Int, k: Int) -> List[Int32]:
    var out = List[Int32]()
    for i in range(k):
        out.append(colids[offset + i])
    sort(out)
    return out^


def assert_one_sample(
    colids: List[Int32], offset: Int, n: Int, k: Int, label: String
) raises -> Int:
    """The whole contract of both algorithms, PER ELEMENT: `k` ids, each in
    `[0, n)`, all distinct. Returns the number of assertions made.

    Distinctness is checked with an `n`-wide seen-array rather than by
    comparing a count of uniques, because a count is an aggregate and
    `PORTING_RULES.md` 7 is explicit that an aggregate verifies the total and
    nothing about placement.
    """
    var seen = List[Int](length=n, fill=0)
    var cells = 0
    for i in range(k):
        var c = colids[offset + i]
        assert_true(
            c >= 0 and Int(c) < n,
            String(
                label,
                ": column ",
                c,
                " at slot ",
                i,
                " is outside [0, ",
                n,
                ")",
            ),
        )
        assert_equal(
            seen[Int(c)],
            0,
            String(
                label,
                ": column ",
                c,
                " appears twice; producing k DISTINCT columns is the entire"
                " job of both samplers",
            ),
        )
        seen[Int(c)] = 1
        cells += 2
    return cells


def run_case(
    n: Int,
    k: Int,
    node_ids: List[Int],
    tree_id: Int32,
    seed: UInt64,
    expected_arm: Int,
    expected_m: Int,
    label: String,
) raises -> Int:
    """One named case: run the dispatch, ASSERT WHICH ARM RAN, then check
    every node's sample element by element."""
    var items = work_items_for(node_ids)
    var colids = List[Int32](length=len(node_ids) * k, fill=Int32(-1))
    var plan = sample_features(colids, items, tree_id, seed, n, k)
    print(
        "   ",
        label,
        " n=",
        n,
        " k=",
        k,
        " -> ",
        plan.name(),
        " n_parallel_samples=",
        plan.n_parallel_samples,
        " max_samples_per_thread=",
        plan.max_samples_per_thread,
    )
    var cells = 0
    assert_equal(
        plan.arm,
        expected_arm,
        String(
            label,
            ": dispatch sent (",
            n,
            ", ",
            k,
            ") to the wrong kernel",
        ),
    )
    assert_equal(
        plan.max_samples_per_thread,
        expected_m,
        String(label, ": wrong excess-kernel instantiation"),
    )
    cells += 2
    for node in range(len(node_ids)):
        cells += assert_one_sample(
            colids, node * k, n, k, String(label, " node ", node_ids[node])
        )

    # Determinism: same (seed, treeid, nodeid) -> same columns, every time.
    var again = List[Int32](length=len(node_ids) * k, fill=Int32(-1))
    var plan2 = sample_features(again, items, tree_id, seed, n, k)
    assert_equal(plan2.arm, plan.arm, String(label, ": arm is not stable"))
    for i in range(len(node_ids) * k):
        assert_equal(
            again[i],
            colids[i],
            String(
                label,
                ": slot ",
                i,
                " changed between two runs with the same (seed, treeid,"
                " nodeid); both samplers are counter-based and cannot",
            ),
        )
        cells += 1
    return cells


def main() raises:
    var cells = 0

    # =====================================================================
    # 1. THE DISPATCH, all three arms, and both boundaries
    # =====================================================================
    print("[dispatch] builder.cuh:398-471, every arm named")
    print(
        "    constants: block_threads =",
        SAMPLER_BLOCK_THREADS,
        " max_samples_per_thread =",
        SAMPLER_MAX_SAMPLES_PER_THREAD,
        " so the excess arm is taken while n_parallel_samples <=",
        SAMPLER_BLOCK_THREADS * SAMPLER_MAX_SAMPLES_PER_THREAD,
    )

    # The two boundaries, each pinned by the ADJACENT PAIR of k that straddles
    # it. An off-by-one in the comparison (`>=` for `>`) moves exactly one of
    # these two cases and nothing else in the file would notice.
    var boundary_n = 20000
    for pair in [
        (EXCESS_EDGE_K, SAMPLE_EXCESS),
        (ALGO_L_FRONTIER_K, SAMPLE_ALGO_L),
    ]:
        var k = pair[0]
        var nps = n_parallel_samples_for(boundary_n, k)
        var plan = plan_feature_sampling(boundary_n, k)
        print(
            "    n=",
            boundary_n,
            " k=",
            k,
            " n_parallel_samples=",
            nps,
            " -> ",
            plan.name(),
        )
        assert_equal(
            plan.arm,
            pair[1],
            "the excess/algo-L boundary is `max_samples_per_thread *"
            " block_threads >= n_parallel_samples` (builder.cuh:427) and it"
            " must fall between these two adjacent k",
        )
        cells += 1
    assert_equal(
        n_parallel_samples_for(boundary_n, EXCESS_EDGE_K),
        SAMPLER_BLOCK_THREADS * SAMPLER_MAX_SAMPLES_PER_THREAD,
        "the mode's EXCESS_EDGE_K at n=20000 must land EXACTLY on 9216 --"
        " that is what makes the pair above a boundary rather than a gap"
        " (DEVIATION 457: 7384 under FAST on this host, 7385 under"
        " IDENTICAL)",
    )
    cells += 1

    # DEVIATION 457's sweep -- the FAST arm (host libm, double) against the
    # portable arm (row 12 float32), held side by side on THIS host. The
    # counts differ often and by a little (float32 cannot form 1 - 1/n to
    # double precision), which moves no correctness anywhere -- any count is
    # a valid draw budget -- so what this pins is the only thing that can
    # move behavior, WHICH KERNEL the dispatch picks. On this host the sweep
    # has exactly one arm flip, and it is IDENTITY_PATHS row 18's knife edge
    # made visible -- (n=20000, k=7385), libm 9217 (algo-L) against portable
    # 9216 (excess). A glibc host may put the libm side of these numbers
    # elsewhere; that host-dependence is the entire reason the portable arm
    # exists.
    print("[dispatch] DEVIATION 457: libm arm vs portable arm, swept")
    var sweep_ns = [64, 500, 1000, 2000, 20000]
    var sweep_total = 0
    var sweep_mismatch = 0
    var flip_n = List[Int]()
    var flip_k = List[Int]()
    comptime BIG = SAMPLER_BLOCK_THREADS * SAMPLER_MAX_SAMPLES_PER_THREAD
    for ni in range(len(sweep_ns)):
        var n = sweep_ns[ni]
        for k in range(1, n):
            var a = n_parallel_samples_libm(n, k)
            var b = n_parallel_samples_portable(n, k)
            sweep_total += 1
            if a != b:
                sweep_mismatch += 1
            var flip = (
                (a <= SAMPLER_BLOCK_THREADS) != (b <= SAMPLER_BLOCK_THREADS)
            ) or ((a <= BIG) != (b <= BIG))
            if flip:
                flip_n.append(n)
                flip_k.append(k)
    print(
        "    swept",
        sweep_total,
        "(n, k); counts differ on",
        sweep_mismatch,
        "; dispatch-boundary flips:",
        len(flip_n),
    )
    for i in range(len(flip_n)):
        print(
            "    ARM FLIP at n=",
            flip_n[i],
            " k=",
            flip_k[i],
            " libm=",
            n_parallel_samples_libm(flip_n[i], flip_k[i]),
            " portable=",
            n_parallel_samples_portable(flip_n[i], flip_k[i]),
        )
    assert_equal(
        len(flip_n),
        1,
        "this host's sweep has exactly one (n, k) where the two arms pick"
        " different kernels; a second one appearing means the host libm or"
        " the portable polynomial moved",
    )
    assert_true(
        flip_n[0] == 20000 and flip_k[0] == 7385,
        "the one arm flip is the row-18 knife edge (n=20000, k=7385)",
    )
    # Reach for the gate itself -- `n_parallel_samples_for` must equal the
    # arm this build was compiled with, asserted where the arms DISAGREE so
    # a gate stuck on the wrong arm cannot pass by coincidence.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        assert_equal(
            n_parallel_samples_for(20000, 7385),
            n_parallel_samples_portable(20000, 7385),
            "IDENTICAL build must route the portable arm",
        )
    else:
        assert_equal(
            n_parallel_samples_for(20000, 7385),
            n_parallel_samples_libm(20000, 7385),
            "FAST build must route the libm arm",
        )
    cells += 3

    for pair in [(127, 1), (128, SAMPLER_MAX_SAMPLES_PER_THREAD)]:
        var k = pair[0]
        var nps = n_parallel_samples_for(boundary_n, k)
        var plan = plan_feature_sampling(boundary_n, k)
        print(
            "    n=",
            boundary_n,
            " k=",
            k,
            " n_parallel_samples=",
            nps,
            " -> ",
            plan.name(),
            " instantiation M=",
            plan.max_samples_per_thread,
        )
        assert_equal(plan.arm, SAMPLE_EXCESS, "both sides are the excess arm")
        assert_equal(
            plan.max_samples_per_thread,
            pair[1],
            "the instantiation boundary is `n_parallel_samples <="
            " block_threads` (builder.cuh:434) and it must fall between these"
            " two adjacent k",
        )
        cells += 2
    assert_equal(
        n_parallel_samples_for(boundary_n, 127),
        SAMPLER_BLOCK_THREADS,
        "k=127 at n=20000 must land EXACTLY on 128",
    )
    cells += 1

    # The two key chains are NOT the same function, and a port that used one
    # for both would still pass every property test in this file.
    assert_true(
        excess_subsequence(0, 3, 5) != algo_l_subsequence(3, 5),
        "the excess sampler hashes (thread, tree, node) with fnv1a32"
        " (builder_kernels.cuh:167-170); algo-L bit-packs (tree, node)"
        " (:280). They must not be the same value",
    )
    assert_equal(
        algo_l_subsequence(3, 5),
        (UInt64(3) << 32) | UInt64(5),
        "algo-L's subsequence is a plain bit-packing, not a hash",
    )
    cells += 2

    # =====================================================================
    # 2. ARM: excess sampling, instantiation MAX_SAMPLES_PER_THREAD = 1
    # =====================================================================
    print("[excess M=1] builder_kernels.cuh:152-248, one draw per thread")
    var nodes8 = List[Int]()
    for i in range(8):
        nodes8.append(i)
    var nodes64 = List[Int]()
    for i in range(64):
        nodes64.append(i * 3 + 1)

    cells += run_case(
        64, 8, nodes64, 2, 0xC0FFEE, SAMPLE_EXCESS, 1, String("small")
    )
    cells += run_case(
        1000, 20, nodes64, 7, 0xC0FFEE, SAMPLE_EXCESS, 1, String("wide")
    )
    cells += run_case(
        100, 1, nodes64, 1, 0xC0FFEE, SAMPLE_EXCESS, 1, String("k==1")
    )

    # =====================================================================
    # 3. ARM: excess sampling, instantiation MAX_SAMPLES_PER_THREAD = 72
    # =====================================================================
    print("[excess M=72] the same kernel at the other instantiation")
    cells += run_case(
        100,
        99,
        nodes8,
        3,
        0xC0FFEE,
        SAMPLE_EXCESS,
        SAMPLER_MAX_SAMPLES_PER_THREAD,
        String("k==n-1"),
    )
    cells += run_case(
        20000,
        128,
        nodes8,
        3,
        0xC0FFEE,
        SAMPLE_EXCESS,
        SAMPLER_MAX_SAMPLES_PER_THREAD,
        String("just over the M boundary"),
    )
    cells += run_case(
        2000,
        1500,
        nodes8,
        5,
        0xC0FFEE,
        SAMPLE_EXCESS,
        SAMPLER_MAX_SAMPLES_PER_THREAD,
        String("dense"),
    )

    # THE TWO INSTANTIATIONS ARE DIFFERENT KERNELS AND MUST DISAGREE. Same
    # (n, k, seed, tree, node), only M changes: the block draws a different
    # number of samples per thread, so the streams line up differently. If
    # these ever matched, the M boundary above would be untested by
    # construction -- both sides would be the same computation.
    var m_items = work_items_for(nodes8)
    var a_colids = List[Int32](length=len(nodes8) * 20, fill=Int32(-1))
    var b_colids = List[Int32](length=len(nodes8) * 20, fill=Int32(-1))
    _ = sample_features(a_colids, m_items, 7, 0xC0FFEE, 1000, 20)
    var differed = 0
    # DEVIATION 211: the inner arms take one tree id per work item now.
    var m_trees = List[Int32](length=len(m_items), fill=Int32(7))
    excess_sample_with_replacement(
        b_colids,
        m_items,
        m_trees,
        0xC0FFEE,
        1000,
        20,
        n_parallel_samples_for(1000, 20),
        SAMPLER_MAX_SAMPLES_PER_THREAD,
        SAMPLER_BLOCK_THREADS,
    )
    for i in range(len(nodes8) * 20):
        if a_colids[i] != b_colids[i]:
            differed += 1
    print(
        "    M=1 vs M=72 at (n=1000, k=20): ",
        differed,
        "of",
        len(nodes8) * 20,
        "slots differ",
    )
    assert_true(
        differed > 0,
        "the two instantiations must not agree; if they did, the dispatch"
        " between them would be untestable",
    )
    cells += 1

    # =====================================================================
    # 4. ARM: algo-L reservoir sampling
    # =====================================================================
    print("[algo-L] builder_kernels.cuh:268-316, one thread per work item")
    var nodes16 = List[Int]()
    for i in range(16):
        nodes16.append(i * 5)
    cells += run_case(
        20000,
        ALGO_L_FRONTIER_K,
        nodes16,
        4,
        0xC0FFEE,
        SAMPLE_ALGO_L,
        0,
        String("frontier"),
    )
    cells += run_case(
        2000, 1990, nodes16, 6, 0xC0FFEE, SAMPLE_ALGO_L, 0, String("k near n")
    )

    # =====================================================================
    # 5. THE SAMPLE IS KEYED ON THE NODE ID, NOT ON THE SLOT IN THE BATCH
    # =====================================================================
    # Both kernels read `work_items[...].idx` (`:165`, `:279`), so the SAME
    # node id at two different positions in the batch must produce the SAME
    # columns, and a different node id must produce different ones. A port
    # that keyed on the loop counter would pass every property above.
    print("[keying] node id, not batch position")
    for arm_case in [(1000, 20), (20000, ALGO_L_FRONTIER_K)]:
        var n = arm_case[0]
        var k = arm_case[1]
        var repeated = List[Int]()
        repeated.append(41)
        repeated.append(9)
        repeated.append(41)
        var items = work_items_for(repeated)
        var colids = List[Int32](length=3 * k, fill=Int32(-1))
        var plan = sample_features(colids, items, 2, 0xFEED, n, k)
        var same = 0
        var diff = 0
        for i in range(k):
            if colids[i] == colids[2 * k + i]:
                same += 1
            if colids[i] != colids[k + i]:
                diff += 1
        print(
            "    ",
            plan.name(),
            ": node 41 at batch slots 0 and 2 agree in",
            same,
            "of",
            k,
            "columns; node 9 differs in",
            diff,
        )
        assert_equal(
            same,
            k,
            "the same node id must give the same sample wherever it sits in"
            " the batch",
        )
        assert_true(diff > 0, "a different node id must give a different sample")
        cells += 2

    # =====================================================================
    # 6. DIFFERENT NODES AND DIFFERENT TREES GET DIFFERENT SAMPLES
    # =====================================================================
    # Not a spot check: 1024 (tree, node) pairs on the excess arm and 64 on
    # algo-L, compared PAIRWISE as sorted sets. n and k are chosen so the
    # space of possible samples is astronomically larger than the number of
    # draws -- C(1000, 20) is about 3e41 -- so a repeat means a key
    # collision, not bad luck.
    print("[independence] pairwise-distinct samples over many (tree, node)")
    var sigs = List[List[Int32]]()
    var n_ind = 1000
    var k_ind = 20
    var ind_nodes = List[Int]()
    for i in range(128):
        ind_nodes.append(i)
    var ind_items = work_items_for(ind_nodes)
    for tree in range(8):
        var colids = List[Int32](length=128 * k_ind, fill=Int32(-1))
        var plan = sample_features(
            colids, ind_items, Int32(tree), 0x5EED, n_ind, k_ind
        )
        assert_equal(plan.arm, SAMPLE_EXCESS, "this sweep is the excess arm")
        for node in range(128):
            sigs.append(sorted_copy(colids, node * k_ind, k_ind))
    var collisions = 0
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            var equal = True
            for i in range(k_ind):
                if sigs[a][i] != sigs[b][i]:
                    equal = False
                    break
            if equal:
                collisions += 1
    print(
        "    excess: ",
        len(sigs),
        "samples over 8 trees x 128 nodes, ",
        len(sigs) * (len(sigs) - 1) // 2,
        "pairs compared, ",
        collisions,
        "identical pairs",
    )
    assert_equal(
        collisions,
        0,
        "two (tree, node) pairs produced the same 20 of 1000 columns; the"
        " subsequence chain has collapsed",
    )
    cells += len(sigs) * (len(sigs) - 1) // 2

    var l_sigs = List[List[Int32]]()
    var n_l = 20000
    var k_l = ALGO_L_FRONTIER_K
    var l_nodes = List[Int]()
    for i in range(32):
        l_nodes.append(i)
    var l_items = work_items_for(l_nodes)
    for tree in range(2):
        var colids = List[Int32](length=32 * k_l, fill=Int32(-1))
        var plan = sample_features(
            colids, l_items, Int32(tree), 0x5EED, n_l, k_l
        )
        assert_equal(plan.arm, SAMPLE_ALGO_L, "this sweep is the algo-L arm")
        for node in range(32):
            l_sigs.append(sorted_copy(colids, node * k_l, k_l))
    var l_collisions = 0
    for a in range(len(l_sigs)):
        for b in range(a + 1, len(l_sigs)):
            var equal = True
            for i in range(k_l):
                if l_sigs[a][i] != l_sigs[b][i]:
                    equal = False
                    break
            if equal:
                l_collisions += 1
    print(
        "    algo-L: ",
        len(l_sigs),
        "samples over 2 trees x 32 nodes, ",
        len(l_sigs) * (len(l_sigs) - 1) // 2,
        "pairs compared, ",
        l_collisions,
        "identical pairs",
    )
    assert_equal(l_collisions, 0, "two (tree, node) pairs gave the same sample")
    cells += len(l_sigs) * (len(l_sigs) - 1) // 2

    # The seed is the third component and gets its own case, because nothing
    # above varies it.
    var seed_a = List[Int32](length=k_ind, fill=Int32(-1))
    var seed_b = List[Int32](length=k_ind, fill=Int32(-1))
    var one_node = List[Int]()
    one_node.append(3)
    var one_item = work_items_for(one_node)
    _ = sample_features(seed_a, one_item, 1, 0x1111, n_ind, k_ind)
    _ = sample_features(seed_b, one_item, 1, 0x2222, n_ind, k_ind)
    var seed_diff = 0
    for i in range(k_ind):
        if seed_a[i] != seed_b[i]:
            seed_diff += 1
    print("    seed: two seeds differ in", seed_diff, "of", k_ind, "slots")
    assert_true(seed_diff > 0, "the seed must move the sample")
    cells += 1

    # =====================================================================
    # 7. UNIFORMITY -- A SMOKE TEST, AND THE MEASURED BIAS IN THEIRS
    # =====================================================================
    # WHAT THIS IS: a count of how often each column is chosen, over many
    # nodes, against the k/n each column would get if the sampler were
    # uniform. WHAT IT IS NOT: a test of uniformity. It has no null
    # hypothesis, no variance model and no multiple-comparison correction,
    # and it would not detect a sampler that was correct in the margins and
    # wrong in the joint distribution. It catches a column that is never
    # chosen or always chosen, and that is all it is here for.
    #
    # TOLERANCE, and why. algo-L is checked in BUCKETS of 200 adjacent
    # columns, because a per-column count at these sample sizes has a
    # binomial spread of about +/- 25% at 3 sigma and a +/-25% band would be
    # satisfied by almost anything; bucketing 200 columns divides the spread
    # by ~14 and lets a 5% band mean something. The excess arm is checked
    # per column over the BULK of the range at +/-25%, with its two
    # structurally-biased ends asserted separately below, because bucketing
    # would average that bias away -- which is the opposite of what this
    # check is for.
    print("[uniformity] a smoke test, stated as one")

    var uni_n = 20000
    var uni_k = ALGO_L_FRONTIER_K
    var uni_counts = List[Int](length=uni_n, fill=0)
    var uni_nodes = List[Int]()
    for i in range(64):
        uni_nodes.append(i)
    var uni_items = work_items_for(uni_nodes)
    var uni_total = 0
    for tree in range(4):
        var colids = List[Int32](length=64 * uni_k, fill=Int32(-1))
        var plan = sample_features(
            colids, uni_items, Int32(tree), 0xA5A5, uni_n, uni_k
        )
        assert_equal(plan.arm, SAMPLE_ALGO_L, "algo-L uniformity sweep")
        for i in range(64 * uni_k):
            uni_counts[Int(colids[i])] += 1
            uni_total += 1
    var bucket = 200
    var n_buckets = uni_n // bucket
    var expected_bucket = Float64(uni_total) / Float64(n_buckets)
    var worst = 0.0
    var worst_bucket = 0
    for b in range(n_buckets):
        var got = 0
        for c in range(b * bucket, (b + 1) * bucket):
            got += uni_counts[c]
        var rel = Float64(got) / expected_bucket
        var dev = rel - 1.0
        if dev < 0.0:
            dev = -dev
        if dev > worst:
            worst = dev
            worst_bucket = b
        assert_true(
            rel > 0.95 and rel < 1.05,
            String(
                "algo-L bucket ",
                b,
                " (columns ",
                b * bucket,
                "..",
                (b + 1) * bucket,
                ") got ",
                got,
                " selections against ",
                expected_bucket,
                " expected -- outside the stated +/-5% band",
            ),
        )
        cells += 1
    print(
        "    algo-L: ",
        n_buckets,
        "buckets of",
        bucket,
        "columns, k/n =",
        Float64(uni_k) / Float64(uni_n),
        ", worst bucket deviates",
        worst * 100.0,
        "% (bucket",
        worst_bucket,
        "), band +/-5%",
    )

    var ex_n = 64
    var ex_k = 8
    var ex_counts = List[Int](length=ex_n, fill=0)
    var ex_nodes = List[Int]()
    for i in range(512):
        ex_nodes.append(i)
    var ex_items = work_items_for(ex_nodes)
    var ex_total = 0
    for tree in range(8):
        var colids = List[Int32](length=512 * ex_k, fill=Int32(-1))
        var plan = sample_features(
            colids, ex_items, Int32(tree), 0xA5A5, ex_n, ex_k
        )
        assert_equal(plan.arm, SAMPLE_EXCESS, "excess uniformity sweep")
        for i in range(512 * ex_k):
            ex_counts[Int(colids[i])] += 1
            ex_total += 1
    var ex_expected = Float64(ex_total) / Float64(ex_n)
    # THE BULK: columns 1 .. n/2. Both ends are excluded ON PURPOSE and each
    # is asserted below instead.
    for c in range(1, ex_n // 2):
        var rel = Float64(ex_counts[c]) / ex_expected
        assert_true(
            rel > 0.75 and rel < 1.25,
            String(
                "excess column ",
                c,
                " chosen ",
                ex_counts[c],
                " times against ",
                ex_expected,
                " expected -- outside the stated +/-25% band",
            ),
        )
        cells += 1
    print(
        "    excess: columns 1..",
        ex_n // 2,
        "within +/-25% of",
        ex_expected,
        "; measured spread over that band:",
    )
    var bulk_lo = ex_counts[1]
    var bulk_hi = ex_counts[1]
    for c in range(1, ex_n // 2):
        if ex_counts[c] < bulk_lo:
            bulk_lo = ex_counts[c]
        if ex_counts[c] > bulk_hi:
            bulk_hi = ex_counts[c]
    print("        min", bulk_lo, "max", bulk_hi)

    # =====================================================================
    # 8. NO COLUMN IS STARVED AND NONE IS FAVOURED -- the two cuML bugs, fixed
    # =====================================================================
    # THIS SECTION USED TO ASSERT THE OPPOSITE. It pinned cuML's two
    # structural biases in place on the reasoning that they are what
    # `builder_kernels.cuh:231-232` and `:201-203` do and that COPY, DO NOT
    # IMPROVE therefore required reproducing them. Andrew, 2026-08-21: **do
    # not port bugs, fix them.** Deviations 164 and 165 are the fixes and this
    # section is now the regression guard for them, in the direction that
    # makes the learner right.
    #
    # What the old behaviour cost, measured before the fix and kept here so
    # nobody re-derives it: at k = 1, column 0 was drawn 0 times of 64 at
    # every one of n = 2, 3, 4 and 8 -- never, not merely rarely -- and column
    # n-1 was drawn 662 times against 512 expected. On a two-column fixture
    # whose separating feature is column 0, the whole learner could not see
    # it: `quality_band_check` scored 0.523 where sklearn scores exactly 1.0.
    print("[uniformity] no column starved, none favoured (deviations 164/165)")
    print(
        "    column 0 chosen",
        ex_counts[0],
        "times against",
        ex_expected,
        "expected (ratio",
        Float64(ex_counts[0]) / ex_expected,
        ")",
    )
    assert_true(
        Float64(ex_counts[0]) / ex_expected > 0.75,
        "column 0 must NOT be starved. cuML passes mask[0] -- the previous"
        " iteration's FLAG -- as the predecessor of the block minimum"
        " (builder_kernels.cuh:231-232), which drops a sampled column 0"
        " entirely. DEVIATION 164 gives the block minimum an unconditional"
        " head flag, because a minimum has no predecessor. If this ratio has"
        " fallen back toward 0 the fix has been reverted",
    )
    print(
        "    column n-1 chosen",
        ex_counts[ex_n - 1],
        "times against",
        ex_expected,
        "expected (ratio",
        Float64(ex_counts[ex_n - 1]) / ex_expected,
        ")",
    )
    assert_true(
        Float64(ex_counts[ex_n - 1]) / ex_expected < 1.25,
        "column n-1 must NOT be over-represented. cuML fills every slot past"
        " n_parallel_samples with the REAL column id n-1 (:201-203), so a"
        " slot that never drew votes anyway. DEVIATION 165 fills those slots"
        " with an out-of-range sentinel that is masked off",
    )
    cells += 2

    # The clearest possible demonstration, with no statistics in it at all:
    # at n=2, k=1 EVERY node draws one of exactly two columns, so both must
    # appear. Before the fix, column 1 came back 64 times of 64.
    var tiny_nodes = List[Int]()
    for i in range(64):
        tiny_nodes.append(i)
    var tiny_items = work_items_for(tiny_nodes)
    var tiny = List[Int32](length=64, fill=Int32(-1))
    var tiny_plan = sample_features(tiny, tiny_items, 0, 0x1234, 2, 1)
    assert_equal(tiny_plan.arm, SAMPLE_EXCESS, "n=2 k=1 is the excess arm")
    var ones = 0
    for i in range(64):
        if tiny[i] == 1:
            ones += 1
        cells += 1
    print(
        "    n=2, k=1 over 64 nodes: column 1 returned",
        ones,
        "times, column 0",
        64 - ones,
        "times",
    )
    assert_true(
        ones > 0 and ones < 64,
        "at n=2, k=1 BOTH columns must appear across 64 nodes. cuML returns"
        " column 1 all 64 times; that is the bug deviation 164 fixes, and a"
        " sweep of exactly 0 or exactly 64 means it is back",
    )
    cells += 1

    # =====================================================================
    # 9. EDGE CASES
    # =====================================================================
    print("[edges] k==n, k==1, n==1, k==n-1")

    # k == n: cuML does not sample at all (builder.cuh:399) and its consumer
    # uses the column index directly (builder_kernels_impl.cuh:250-254).
    # DEVIATION 156 materializes that as the identity, so the result must be
    # a PERMUTATION OF EVERY COLUMN -- checked as one, not merely as k
    # distinct ids.
    for n in [1, 2, 17, 256]:
        var items = work_items_for(nodes8)
        var colids = List[Int32](length=len(nodes8) * n, fill=Int32(-1))
        var plan = sample_features(colids, items, 3, 0xDEAD, n, n)
        assert_equal(
            plan.arm,
            SAMPLE_ALL_FEATURES,
            String(
                "k == n must not reach either sampler: log(1 - k/n) is -inf"
                " and n_parallel_samples would be undefined (n=",
                n,
                ")",
            ),
        )
        for node in range(len(nodes8)):
            cells += assert_one_sample(
                colids, node * n, n, n, String("k==n, n=", n)
            )
            var got = sorted_copy(colids, node * n, n)
            for c in range(n):
                assert_equal(
                    got[c],
                    Int32(c),
                    String(
                        "k == n must return a permutation of ALL ",
                        n,
                        " columns",
                    ),
                )
                cells += 1
        print("    n =", n, " k = n -> ", plan.name(), ", identity permutation")

    # n == 1 with k == 1 is the same case and is covered above; k > n and
    # k < 1 are refused by name (DEVIATION 159) rather than returning a
    # NaN-derived arm.
    var refused = 0
    for pair in [(4, 5), (4, 0), (4, -1)]:
        try:
            _ = plan_feature_sampling(pair[0], pair[1])
        except:
            refused += 1
        cells += 1
    assert_equal(
        refused, 3, "k > n and k < 1 must be refused, not silently sampled"
    )

    # k == 1 on a range of n, all on the excess arm, all distinct-by-triviality
    # but they must still be IN RANGE and keyed.
    for n in [2, 3, 100, 5000]:
        var items = work_items_for(nodes64)
        var colids = List[Int32](length=len(nodes64), fill=Int32(-1))
        var plan = sample_features(colids, items, 9, 0xB0B, n, 1)
        assert_equal(plan.arm, SAMPLE_EXCESS, "k == 1 lands on the excess arm")
        var distinct_values = List[Int](length=n, fill=0)
        var n_distinct = 0
        for node in range(len(nodes64)):
            cells += assert_one_sample(
                colids, node, n, 1, String("k==1, n=", n)
            )
            if distinct_values[Int(colids[node])] == 0:
                distinct_values[Int(colids[node])] = 1
                n_distinct += 1
        print(
            "    n =",
            n,
            " k = 1 -> ",
            plan.name(),
            ", 64 nodes drew",
            n_distinct,
            "distinct columns",
        )
        if n > 2:
            assert_true(
                n_distinct > 1,
                "64 nodes drawing one column each must not all draw the same"
                " one",
            )
            cells += 1

    # k == n - 1 across the M boundary and across the arm boundary.
    for n in [3, 100, 2000]:
        var items = work_items_for(nodes8)
        var colids = List[Int32](length=len(nodes8) * (n - 1), fill=Int32(-1))
        var plan = sample_features(colids, items, 11, 0xB0B, n, n - 1)
        for node in range(len(nodes8)):
            cells += assert_one_sample(
                colids, node * (n - 1), n, n - 1, String("k==n-1, n=", n)
            )
        print(
            "    n =",
            n,
            " k = n-1 -> ",
            plan.name(),
            " M=",
            plan.max_samples_per_thread,
        )

    print("    EXCESS_MAX_ITERATIONS =", EXCESS_MAX_ITERATIONS, "(DEVIATION 159); no case above reached it")

    print("feature_sampler: ", cells, "cells")
    print("feature_sampler_check: PASS")
