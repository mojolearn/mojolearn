"""The work-item structs the builder and its kernels share.

A PORT of the host-visible half of cuML
`cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh`, pinned at
`00094f7` in `~/CascadeProjects/upstream/cuml`:

| ours | theirs |
|---|---|
| `InstanceRange` | `builder_kernels.cuh:34-37` |
| `NodeWorkItem` | `builder_kernels.cuh:39-43` |
| `WorkloadInfo` | `builder_kernels.cuh:49-57` |
| `split_not_valid` | `builder_kernels.cuh:59-67` |
| `excess_sample_with_replacement` | `builder_kernels.cuh:152-248` |
| `algo_l_sample` | `builder_kernels.cuh:268-316` |
| `n_parallel_samples_for` / `plan_feature_sampling` | `builder.cuh:398-471` |

NOT IN THIS FILE, DELIBERATELY, AND WHERE THEY WENT
---------------------------------------------------
- `fnv1a32` (`builder_kernels.cuh:98-113`) and the `PCGenerator` it keys are in
  `extratrees/mojo_only/pcg_rng.mojo`, with RAFT as the upstream for the
  generator itself. They are a RAFT primitive plus four lines of cuML glue, and
  this tree keeps RAFT primitives in `mojo_only/` the way `cluster/` does. The
  samplers below IMPORT them from there; the key chain they build is cuML's
  own three-component one, not this lane's four-component `key_for`.
- `adaptive_sample_kernel` (`:318-347`) is a THIRD sampler in their file --
  Knuth's selection sampling (Algorithm S), which walks all `N` features once
  and accepts feature `i` when `(M - selected) * 2**32 > toss * (N - i)`. It is
  DEAD CODE in cuML: `grep -rn adaptive_sample_kernel cpp/` at `00094f7`
  returns only its own definition, and `builder.cuh:398-471` dispatches only
  between the two below. Not ported, for that reason; see `UNPORTED.tsv`.
- `lower_bound` (`:110-125`) is a QUANTILE lookup. There are no quantiles in
  this directory and there never will be; see `UNPORTED.tsv` on
  `quantiles.cuh`.

WHAT `WorkloadInfo` IS FOR, because the name does not say it
------------------------------------------------------------
It is cuML's answer to the ragged-frontier problem: a batch of nodes has wildly
different row counts, so instead of one block per node they flatten the batch
into a list of blocks and give each block a row telling it which node it serves
and which slice of that node's rows it owns (`builder.cuh:378-393` fills it,
`builder_kernels_impl.cuh:236-241` reads it). A "large" node is one needing
more than one block, and only large nodes get a global-memory scratch slot.
This lane keeps that shape: it is the piece that makes a breadth-first frontier
efficient, and it is theirs, not ours.
"""

from extratrees.ported.decisiontree.batched_levelalgo.split import Split


@fieldwise_init
struct InstanceRange(ImplicitlyCopyable, Movable):
    """The range of instances belonging to a node, as a range in
    `Dataset.row_ids`. `builder_kernels.cuh:34-37`."""

    var begin: Int32
    var count: Int32


@fieldwise_init
struct NodeWorkItem(ImplicitlyCopyable, Movable):
    """One node awaiting a split. `builder_kernels.cuh:39-43`."""

    var idx: Int32
    """Index of the work item in the tree. Theirs is `size_t idx`."""

    var depth: Int32
    var instances: InstanceRange


@fieldwise_init
struct WorkloadInfo(ImplicitlyCopyable, Movable):
    """Which node a threadblock serves, and which slice of it.
    `builder_kernels.cuh:49-57`."""

    var nodeid: Int32
    """Node in the batch this threadblock works on."""

    var large_nodeid: Int32
    """Counts only LARGE nodes -- those needing more than one block along the
    x dimension, and therefore a global-memory scratch slot."""

    var offset_blockid: Int32
    """This block's offset among all blocks working on this node."""

    var num_blocks: Int32
    """Total blocks working on this node."""


def split_not_valid(
    split: Split,
    min_impurity_decrease: Float32,
    min_samples_leaf: Int32,
    num_rows: Int32,
) -> Bool:
    """`builder_kernels.cuh:59-67`, transcribed.

    Theirs: `split.best_metric_val <= min_impurity_decrease ||
    split.nLeft < min_samples_leaf ||
    (IdxT(num_rows) - split.nLeft) < min_samples_leaf`.

    Note the FIRST clause is `<=`, so a split whose gain exactly equals
    `min_impurity_decrease` is rejected, and with their default of 0 a
    zero-gain split is rejected too. sklearn's test is the other way round --
    `min_impurity_decrease` is checked as `>=` against a differently scaled
    quantity in `_tree.pyx` -- which is why this lane reports the gain in BOTH
    forms rather than assuming they are the same number.
    """
    return (
        split.best_metric_val <= min_impurity_decrease
        or split.n_left < min_samples_leaf
        or (num_rows - split.n_left) < min_samples_leaf
    )


# ===========================================================================
# THE PER-NODE FEATURE SAMPLERS
#
# cuML has two of them and a dispatch that chooses between them by problem
# size. Both are ported below, as HOST functions, for the reason
# `builder_kernels_impl.mojo` gives for `partition_samples`: a deterministic
# host transcription is the oracle the device version gets checked against,
# and `PORTING_RULES.md` 0b-ii says an oracle is not a "CPU path".
#
# The dispatch is `builder.cuh:398-471` and it is ported VERBATIM in
# `plan_feature_sampling`, including both constants and the reason for the
# log formula. `PORTING_RULES.md` 0b-i: porting the wrong arm of a dispatch is
# invisible and has already cost this project a measured 20x.
# ===========================================================================

from std.ffi import external_call

from extratrees.mojo_only.pcg_rng import (
    FNV1A32_BASIS,
    PCGenerator,
    fnv1a32,
    uniform_int_u64,
)


# --- libm, not `std.math` --------------------------------------------------
# DEVIATION 158. `std.math.log` on this toolchain carries ~5e-8 ABSOLUTE error
# against libm, and BOTH samplers turn a log into an integer decision:
# `plan_feature_sampling` takes `ceil` of a ratio of logs, and `algo_l_sample`
# TRUNCATES a ratio of logs to an int to decide which column to jump to. One
# ulp there is a different column, not a rounder number.


def _log64(x: Float64) -> Float64:
    """`std::log(double)`, which is what `raft::log` becomes for a `double`
    argument on the host (`raft/core/math.hpp:324-331`)."""
    return external_call["log", Float64](x)


def _logf32(x: Float32) -> Float32:
    """`std::log(float)`, i.e. `logf`. `raft::log` is generic in its argument
    type, so a `float` argument stays in float -- see `algo_l_sample`, where
    that matters."""
    return external_call["logf", Float32](x)


def _expf32(x: Float32) -> Float32:
    """`std::exp(float)`, i.e. `expf`. Same genericity as `_logf32`."""
    return external_call["expf", Float32](x)


def _ceil64(x: Float64) -> Float64:
    """`std::ceil(double)`, `builder.cuh:420`."""
    return external_call["ceil", Float64](x)


# --- the two key chains, which are NOT the same ----------------------------


def excess_subsequence(
    thread_id: UInt32, tree_id: UInt32, node_id: UInt32
) -> UInt64:
    """The excess sampler's PCG subsequence. `builder_kernels.cuh:167-170`::

        uint64_t subsequence(fnv1a32_basis);
        subsequence = fnv1a32(subsequence, uint32_t(threadIdx.x));
        subsequence = fnv1a32(subsequence, uint32_t(treeid));
        subsequence = fnv1a32(subsequence, uint32_t(nodeid));

    THREE components, in that order, from `fnv1a32_basis`. This lane's
    `key_for` in `extratrees/mojo_only/pcg_rng.mojo` chains FOUR, because
    DEVIATION 130 needs a per-FEATURE threshold key. That extension belongs to
    the threshold draw and to nothing else: the sampler's key is cuML's, so
    that a node's candidate column set is bit-identical to theirs.

    Note their `subsequence` is a `uint64_t` holding a 32-bit hash -- the high
    32 bits are always zero -- and `PCGenerator` shifts it left by one and
    ORs in a 1 (`rng_device.cuh:673`), so nothing is lost. Widening the hash
    would change every stream.
    """
    var subsequence = FNV1A32_BASIS
    subsequence = fnv1a32(subsequence, thread_id)
    subsequence = fnv1a32(subsequence, tree_id)
    subsequence = fnv1a32(subsequence, node_id)
    return subsequence.cast[DType.uint64]()


def algo_l_subsequence(tree_id: UInt32, node_id: UInt32) -> UInt64:
    """The reservoir sampler's PCG subsequence. `builder_kernels.cuh:280`::

        uint64_t subsequence = (uint64_t(treeid) << 32) | uint64_t(nodeid);

    NOT HASHED. The two samplers key differently and a port that used one
    chain for both would produce a correct-looking sample from the wrong
    stream, which no property test can see. Theirs is a plain bit-packing
    because algo-L runs ONE thread per node, so there is no `threadIdx` to
    mix in and no collision to avoid.
    """
    return (tree_id.cast[DType.uint64]() << 32) | node_id.cast[DType.uint64]()


# --- the dispatch, `builder.cuh:398-471` -----------------------------------

comptime SAMPLER_BLOCK_THREADS: Int = 128
"""`builder.cuh:401`, `constexpr int block_threads = 128`."""

comptime SAMPLER_MAX_SAMPLES_PER_THREAD: Int = 72
"""`builder.cuh:402`, `constexpr int max_samples_per_thread = 72;  // register
spillage if more than this limit`."""

comptime SAMPLE_ALL_FEATURES: Int = 0
"""No sampling: every column is a candidate."""

comptime SAMPLE_EXCESS: Int = 1
"""`excess_sample_with_replacement_kernel`, `builder_kernels.cuh:152`."""

comptime SAMPLE_ALGO_L: Int = 2
"""`algo_L_sample_kernel`, `builder_kernels.cuh:268`."""


def sampler_arm_name(arm: Int) -> String:
    """So a check and a benchmark can PRINT which kernel ran; rule 8."""
    if arm == SAMPLE_ALL_FEATURES:
        return String("all-features")
    if arm == SAMPLE_EXCESS:
        return String("excess-with-replacement")
    if arm == SAMPLE_ALGO_L:
        return String("algo-L-reservoir")
    return String("unknown")


@fieldwise_init
struct FeatureSamplerPlan(ImplicitlyCopyable, Movable):
    """Which sampler runs for an `(n, k)`, and with which instantiation.

    OURS as a STRUCT -- theirs is three locals and two `if`s in `doSplit` --
    but every field is one of their values, and it exists so the arm is a
    thing a check can name rather than something inferred from an output.
    """

    var arm: Int
    var n_parallel_samples: Int
    """`builder.cuh:419-421`. Zero on the arms that do not use it."""

    var max_samples_per_thread: Int
    """1 or 72: the `MAX_SAMPLES_PER_THREAD` template argument of the excess
    kernel, chosen at `builder.cuh:434-455`. Zero off that arm."""

    var block_threads: Int

    def name(self) -> String:
        return sampler_arm_name(self.arm)


def n_parallel_samples_for(n: Int, k: Int) -> Int:
    """`builder.cuh:419-421`, transcribed with their comment intact.

    Their code::

        IdxT n_parallel_samples =
          std::ceil(raft::log(1 - double(dataset.n_sampled_cols) / double(dataset.N)) /
                    (raft::log(1 - 1.f / double(dataset.N))));

    and their reason, `builder.cuh:416-418`, quoted because it is the whole
    justification for the excess strategy:

        number of samples we'll need to sample (in parallel, with
        replacement), to expect 'k' unique samples from 'n' is given by the
        following equation: log(1 - k/n)/log(1 - 1/n)
        ref: https://stats.stackexchange.com/questions/296005/
             the-expected-number-of-unique-elements-drawn-with-replacement

    TYPES, because they decide the answer. Both logs are DOUBLE: `1 -
    double(k)/double(N)` is double, and in `1 - 1.f/double(N)` the `1.f`
    converts to double before the divide, so it is `1.0 - 1.0/double(N)` --
    the `f` suffix is a red herring and rounding it in float32 first would
    move the ratio. `raft::log` on a double host argument is `std::log`
    (`raft/core/math.hpp:324-331`).

    NOT DEFINED AT `k == n`: `log(0)` is `-inf` and the quotient is `+inf`,
    which `std::ceil` passes through and the conversion to `IdxT` makes
    undefined behaviour. Their `doSplit` cannot reach it -- feature sampling
    is guarded by `n_sampled_cols != N` at `builder.cuh:399` -- and neither
    can `plan_feature_sampling`, which takes that guard first.
    """
    var ratio = _log64(1.0 - Float64(k) / Float64(n))
    var per_draw = _log64(1.0 - 1.0 / Float64(n))
    return Int(_ceil64(ratio / per_draw))


def plan_feature_sampling(n: Int, k: Int) raises -> FeatureSamplerPlan:
    """THE DISPATCH. `builder.cuh:398-471`, every branch.

    Their shape, with the guard that encloses it::

        if (dataset.n_sampled_cols != dataset.N) {            // :399
          constexpr int block_threads          = 128;          // :401
          constexpr int max_samples_per_thread = 72;           // :402
          IdxT n_parallel_samples = std::ceil(...);            // :419-421
          if (max_samples_per_thread * block_threads >= n_parallel_samples) {
            if (n_parallel_samples <= block_threads)
              excess_sample_with_replacement_kernel<IdxT, 1, block_threads>
            else
              excess_sample_with_replacement_kernel<IdxT, max_samples_per_thread, block_threads>
          } else {
            algo_L_sample_kernel<<<...>>>
          }
        }

    So there are THREE arms a check must enumerate, not two: the excess
    kernel is instantiated at two different `MAX_SAMPLES_PER_THREAD` and the
    two instantiations consume different numbers of draws per thread and
    therefore return DIFFERENT column sets for the same `(seed, tree, node)`.

    Their reason for the outer test, `builder.cuh:403-414`, quoted because it
    is a SHARED-MEMORY argument and nothing else -- there is no claim here
    that either sampler is faster, and this port makes none either:

        our required shared memory is a function of number of samples we'll
        need to sample (in parallel, with replacement) in excess to get 'k'
        uniques out of 'n' features. estimated static shared memory required
        by cub's block-wide collectives: max_samples_per_thread *
        block_threads * sizeof(IdxT)

        The maximum items to sample (the constant `max_samples_per_thread` to
        be set at compile-time) is calibrated so that:
        1. There is no register spills and accesses to global memory
        2. The required static shared memory (ie, `max_samples_per_thread *
           block_threads * sizeof(IdxT)` does not exceed 46KB.
    """
    if k < 1 or k > n:
        raise Error(
            String(
                "feature sampling needs 1 <= k <= n; got k=",
                k,
                " n=",
                n,
                ". cuML's max_features clamp makes this unreachable there,"
                " so there is no upstream behaviour to mirror -- DEVIATION"
                " 159.",
            )
        )
    if k == n:
        # `builder.cuh:399`, `if (dataset.n_sampled_cols != dataset.N)`. The
        # sampler does not run and `colids` is never written; the consumer
        # branches on the same test and uses the column index directly
        # (`builder_kernels_impl.cuh:250-254`). DEVIATION 156.
        return FeatureSamplerPlan(
            SAMPLE_ALL_FEATURES, 0, 0, SAMPLER_BLOCK_THREADS
        )

    var n_parallel_samples = n_parallel_samples_for(n, k)
    if SAMPLER_MAX_SAMPLES_PER_THREAD * SAMPLER_BLOCK_THREADS >= (
        n_parallel_samples
    ):
        var max_samples_per_thread = SAMPLER_MAX_SAMPLES_PER_THREAD
        if n_parallel_samples <= SAMPLER_BLOCK_THREADS:
            # `builder.cuh:434-436`: "each thread randomly samples only 1
            # sample", template argument 1.
            max_samples_per_thread = 1
        return FeatureSamplerPlan(
            SAMPLE_EXCESS,
            n_parallel_samples,
            max_samples_per_thread,
            SAMPLER_BLOCK_THREADS,
        )
    return FeatureSamplerPlan(
        SAMPLE_ALGO_L, n_parallel_samples, 0, SAMPLER_BLOCK_THREADS
    )


# --- arm 1: excess sampling with replacement -------------------------------

comptime EXCESS_MAX_ITERATIONS: Int = 1024
"""OURS, DEVIATION 159. Their `do { } while (n_uniques < k)` has no bound."""


def excess_sample_with_replacement(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_id: Int32,
    seed: UInt64,
    n: Int,
    k: Int,
    n_parallel_samples: Int,
    max_samples_per_thread: Int = SAMPLER_MAX_SAMPLES_PER_THREAD,
    block_threads: Int = SAMPLER_BLOCK_THREADS,
) raises:
    """`excess_sample_with_replacement_kernel`, `builder_kernels.cuh:152-248`.

    Their summary, `:144-150`: sample `n_parallel_samples` columns in parallel
    WITH replacement, sort the block's samples, flag the head of each run of
    equal values, prefix-sum the flags to get gather indices, and if fewer
    than `k` heads came out, redraw only the non-heads and go round again.
    The mask survives the iteration, so the uniques already found are kept and
    the loop makes forward progress.

    Transcribed as a host function with the block made explicit, exactly as
    `partition_samples` does in `builder_kernels_impl.mojo`. DEVIATION 157
    states what the block collectives became. `colids` is `[len(work_items),
    k]` row-major, their `:259`.

    THE FIVE THINGS THAT ARE NOT GUESSABLE FROM THE SUMMARY
    -------------------------------------------------------
    1. ONE GENERATOR PER THREAD, made once outside the loop (`:172`) and
       carried across iterations. Threads that redraw keep advancing their own
       stream; a per-iteration generator would repeat.

    2. THE ITEMS MIGRATE BETWEEN THREADS. `cub::BlockRadixSort::Sort` leaves
       the block's `BLOCK_THREADS * MAX_SAMPLES_PER_THREAD` items in BLOCKED
       arrangement -- thread `t` gets sorted ranks `[t*M, (t+1)*M)`. So after
       the first sort, the slot a thread redraws is no longer the slot it
       drew. That is why the flat array below is indexed by SLOT and the
       generator by `slot / M`, and it is why a per-thread-local port that
       kept each thread's own samples together would be a different
       algorithm.

    3. SLOTS PAST `n_parallel_samples` ARE NOT IDLE, they are set to `n - 1`
       (`:201-203`). They are real values that get sorted, deduped and can be
       WRITTEN OUT: column `n - 1` is present in the block's sample on every
       iteration, so it is selected whenever the loop stops at exactly `k`
       uniques. Only the FIRST `n_parallel_samples` slots draw, and the
       condition is on the CTA-wide slot index, not the thread's.

    4. THE PREDECESSOR OF THE FIRST ITEM IS `mask[0]`, NOT AN ITEM.
       `:231-232` calls `SubtractLeft(items, mask, CustomDifference<IdxT>(),
       mask[0])`, and CUB's implementation
       (`cccl/cub/cub/block/block_adjacent_difference.cuh:393-419`) does
       `output[0] = difference_op(input[0], tile_predecessor_item)` on thread
       0. So the block MINIMUM is compared against the previous iteration's
       flag -- a 0 or a 1 -- and is flagged a DUPLICATE when it happens to
       equal it. On the first iteration that flag is 0, so a sampled column 0
       is dropped; if a later iteration leaves `mask[0] == 1`, column 1 is
       dropped instead. This is theirs, it is measurable (see
       `feature_sampler_check.mojo`, which shows `n=2, k=1` returning column 1
       every time), and it is transcribed rather than fixed: `PORTING_RULES.md`
       0b is COPY, DO NOT IMPROVE. It costs uniformity at the low columns; it
       cannot produce a duplicate or an out-of-range column, because it only
       ever CLEARS the head flag of the smallest item.

    5. THE OUTPUT IS THE `k` SMALLEST UNIQUES, not a random `k` of them
       (`:243-246`): the gather index is the exclusive prefix sum of the head
       flags, and everything with index `>= k` is simply not written.

    `CustomDifference` (`:133-142`) returns 0 for equal and 1 otherwise, so it
    is symmetric and the argument order CUB passes it in cannot matter here.
    """
    var n_slots = block_threads * max_samples_per_thread
    var items = List[Int32](length=n_slots, fill=0)
    var mask = List[Int32](length=n_slots, fill=0)
    var col_indices = List[Int32](length=n_slots, fill=0)

    for block_id in range(len(work_items)):
        # `:163`, `if (blockIdx.x >= work_items_size) return;` -- the grid is
        # sized to the batch (`builder.cuh:429-430`), so the host loop IS that
        # bound and there is no partial block to reject.
        var node_id = UInt32(work_items[block_id].idx)  # `:165`

        var gens = List[PCGenerator]()
        for thread_id in range(block_threads):
            gens.append(
                PCGenerator(
                    seed,
                    excess_subsequence(
                        UInt32(thread_id), UInt32(tree_id), node_id
                    ),
                    UInt64(0),
                )
            )

        # `:184-186`, mask starts all zero so every slot draws on iteration 1.
        for slot in range(n_slots):
            mask[slot] = 0
            items[slot] = 0
            col_indices[slot] = 0

        # `:180`, `IdxT n_uniques = 0;`. Their initializer is dead -- the
        # scan below assigns it before the `while` ever reads it -- so this
        # declares without one rather than carry a value Mojo would warn about.
        var n_uniques: Int
        var iterations = 0
        while True:  # `:188` .. `:241`, a do-while
            iterations += 1
            if iterations > EXCESS_MAX_ITERATIONS:
                raise Error(
                    String(
                        "excess sampler did not reach k=",
                        k,
                        " uniques of n=",
                        n,
                        " in ",
                        EXCESS_MAX_ITERATIONS,
                        " iterations. Theirs has no such bound and would"
                        " hang; DEVIATION 159.",
                    )
                )

            # `:190-207`. `cta_sample_idx` starts at `M * threadIdx.x` and
            # advances with the thread-local index, so it is exactly the
            # blocked-arrangement slot number.
            for thread_id in range(block_threads):
                for local in range(max_samples_per_thread):
                    var slot = max_samples_per_thread * thread_id + local
                    if mask[slot] == 0 and slot < n_parallel_samples:
                        items[slot] = Int32(
                            uniform_int_u64(gens[thread_id], 0, UInt64(n))
                        )
                    elif mask[slot] == 0:
                        items[slot] = Int32(n - 1)
                    # else: `continue` -- keep the previous iteration's unique.

            # `:224`, `BlockRadixSortT(...).Sort(items)`. Ascending over the
            # whole block, blocked arrangement out. The keys ARE the payload,
            # so equal keys are indistinguishable and any ascending sort gives
            # the identical array a radix sort would; DEVIATION 157.
            sort(items)

            # `:231-232`, `SubtractLeft(items, mask, CustomDifference, mask[0])`.
            # The predecessor is read before the call, so it is the PREVIOUS
            # iteration's flag; see point 4 in the docstring. CUB writes items
            # `ITEMS-1 .. 1` and then item 0, which this mirrors literally
            # even though nothing depends on the order.
            var tile_predecessor_item = mask[0]
            for slot in range(n_slots - 1, 0, -1):
                mask[slot] = 0 if items[slot] == items[slot - 1] else 1
            mask[0] = 0 if items[0] == tile_predecessor_item else 1

            # `:237`, `BlockScanT(...).ExclusiveSum(mask, col_indices, n_uniques)`.
            var running: Int32 = 0
            for slot in range(n_slots):
                col_indices[slot] = running
                running += mask[slot]
            n_uniques = Int(running)

            if n_uniques >= k:  # `:241`, `while (n_uniques < k)`
                break

        # `:244-247`
        var col_offset = k * block_id
        for slot in range(n_slots):
            if mask[slot] != 0 and Int(col_indices[slot]) < k:
                colids[col_offset + Int(col_indices[slot])] = items[slot]


# --- arm 2: reservoir sampling, algorithm L --------------------------------


def algo_l_sample(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_id: Int32,
    seed: UInt64,
    n: Int,
    k: Int,
):
    """`algo_L_sample_kernel`, `builder_kernels.cuh:268-316`.

    Their header, `:250-266`: one THREAD per work item (not one block), select
    `k` of `n` without replacement by reservoir sampling, "algo L" of
    https://en.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm .
    `colids` is `[work_items_size, k]` row-major, their `:259`.

    Transcribed with the thread loop made explicit, the same way as the excess
    sampler above. There is nothing block-wide in it -- no collective, no
    shared memory -- so the host form is the kernel with `tid` spelled out.

    WHAT A FROM-MEMORY ALGORITHM L WOULD GET WRONG
    ----------------------------------------------
    * THE FILL LOOP LEAVES `col` AT `k - 1`, NOT `k` (`:293-301`): it is a
      `while (1)` that writes `colids[col]` and breaks when `col == k - 1`.
      Textbook algorithm L sets `i = k` before the jump loop; theirs relies on
      the `+ 1` in the jump (`:306`) to make up the difference. Writing `k`
      here would skip a column.
    * `W` IS DRAWN BEFORE THE RESERVOIR IS FILLED (`:290-291`), so the first
      float of the stream is the `W` seed and not a jump.
    * THE TYPES ARE MIXED ON PURPOSE and they decide which column is picked.
      `fp_uniform_val` is a `float`; `raft::log`/`raft::exp` of a float are
      `logf`/`expf`; `k` is a `size_t`, so `raft::log(fp)/k` converts `k` to
      FLOAT and divides in float; `W` is a `double`, so `raft::log(1 - W)` is
      the double `std::log`, the numerator `raft::log(fp)` is promoted, and
      the division is double -- and then `static_cast<int>` TRUNCATES it.
      Computing that whole line in float, or in double, moves the truncation
      boundary and lands on a different column.
    * THE REPLACEMENT SLOT IS DRAWN FROM `[0, k)` (`:283-286`), through the
      SAME `UniformIntDistParams<IdxT, uint64_t>` overload as the excess
      sampler, i.e. the 64-bit Lemire path (`rng_device.cuh:208-230`), which
      burns two 32-bit draws per attempt. The 32-bit overload would consume
      the stream at a different rate and every subsequent column would differ.
    * DISTINCTNESS IS STRUCTURAL, not tested for: `col` strictly increases and
      each new `col` OVERWRITES one reservoir slot, so no value can appear
      twice. Nothing in the kernel checks it.

    `k == n` returns the identity `0..n-1` in order, because the fill covers
    every column and the first jump is at least 1 past `k - 1 == n - 1`. The
    dispatch never routes `k == n` here (`plan_feature_sampling`), but the
    property is what makes the fill loop's off-by-one visible.
    """
    for tid in range(len(work_items)):
        # `:277-278`. One thread per work item; the host loop is the grid.
        var node_id = UInt32(work_items[tid].idx)  # `:279`
        var gen = PCGenerator(
            seed, algo_l_subsequence(UInt32(tree_id), node_id), UInt64(0)
        )  # `:280-281`

        # `:287-291`
        var fp_uniform_val = gen.next_float()
        var W = Float64(_expf32(_logf32(fp_uniform_val) / Float32(k)))

        # `:293-301`
        var col = 0
        while True:
            colids[tid * k + col] = Int32(col)
            if col == k - 1:
                break
            else:
                col += 1

        # `:303-315`
        while col < n:
            fp_uniform_val = gen.next_float()
            col += (
                Int(Float64(_logf32(fp_uniform_val)) / _log64(1.0 - W)) + 1
            )
            if col < n:
                var int_uniform_val = Int(
                    uniform_int_u64(gen, 0, UInt64(k))
                )
                colids[tid * k + int_uniform_val] = Int32(col)
                fp_uniform_val = gen.next_float()
                W *= Float64(_expf32(_logf32(fp_uniform_val) / Float32(k)))


# --- the driver, `builder.cuh:398-471` -------------------------------------


def sample_features(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_id: Int32,
    seed: UInt64,
    n: Int,
    k: Int,
) raises -> FeatureSamplerPlan:
    """Fill `colids` with `k` distinct columns per work item, and SAY WHICH
    KERNEL DID IT.

    `doSplit`'s feature-sampling block, `builder.cuh:398-471`. The returned
    plan is how a caller, a check or a benchmark names the arm that ran; rule
    8 -- "a harness that cannot name the kernel it ran can publish a number
    about a different one".

    `colids` must already be `len(work_items) * k` long.
    """
    var plan = plan_feature_sampling(n, k)
    if plan.arm == SAMPLE_ALL_FEATURES:
        # DEVIATION 156.
        for item in range(len(work_items)):
            for c in range(k):
                colids[item * k + c] = Int32(c)
    elif plan.arm == SAMPLE_EXCESS:
        excess_sample_with_replacement(
            colids,
            work_items,
            tree_id,
            seed,
            n,
            k,
            plan.n_parallel_samples,
            plan.max_samples_per_thread,
            plan.block_threads,
        )
    else:
        algo_l_sample(colids, work_items, tree_id, seed, n, k)
    return plan


# ===========================================================================
# DEVIATION BLOCK
# ===========================================================================
#
# 156. The `k == n` guard moved into the sampler, and materializes an identity.
#
#   Theirs. The guard is in the CALLER: `builder.cuh:399`, `if
#   (dataset.n_sampled_cols != dataset.N) { ...sample... }`. When they are
#   equal, neither kernel launches and `colids` is left untouched -- it holds
#   whatever the previous batch wrote. The consumer tests the same condition
#   itself (`builder_kernels_impl.cuh:250-254`) and uses `colStart +
#   blockIdx.y` directly instead of reading `colids`.
#
#   Ours. `plan_feature_sampling` reports a third arm, `SAMPLE_ALL_FEATURES`,
#   and `sample_features` fills `colids` with `0..k-1` for every work item.
#
#   Why. The consumer branch does not exist in this lane yet -- the score pass
#   of DEVIATION 137 is unported -- so there is nothing to carry their
#   two-place test. Putting it in one place keeps `colids` meaning the same
#   thing on every arm, which is what lets one check assert the same
#   properties across all three.
#
#   Why it is safe. The candidate column set is identical either way: theirs
#   is `0..N-1` implicitly, ours is `0..N-1` written down. No column that
#   their build would consider is missed and none is added.
#
#   Price. One write of `len(work_items) * n` int32s that theirs does not do,
#   on the one configuration where sampling is off. It is a HOST write in a
#   host oracle. When the device version lands, whoever writes it must decide
#   whether to keep the fill or restore their two-place branch, and if they
#   restore it they must restore BOTH halves -- the guard alone, without the
#   consumer's `if`, reads an uninitialized `colids`.
#
# ---------------------------------------------------------------------------
#
# 157. The block collectives are explicit loops, and the block is a parameter.
#
#   Theirs. `excess_sample_with_replacement_kernel` is one CUDA block per node
#   using three CUB block-wide collectives over a `BLOCK_THREADS x
#   MAX_SAMPLES_PER_THREAD` register array: `cub::BlockRadixSort::Sort`
#   (`:210`, `:224`), `cub::BlockAdjacentDifference::SubtractLeft` (`:212`,
#   `:231-232`) and `cub::BlockScan::ExclusiveSum` (`:214`, `:237`), sharing
#   one `union` of temp storage (`:216-221`) with a `__syncthreads()` between
#   each.
#
#   Ours. A host function over one flat `List` of `block_threads *
#   max_samples_per_thread` slots in their blocked arrangement, with the sort
#   done by Mojo's `sort`, the adjacent difference by a descending loop that
#   mirrors CUB's own write order, and the exclusive sum by the running prefix
#   it is defined to be. `block_threads` and `max_samples_per_thread` are
#   ARGUMENTS, not template parameters, so the dispatch's choice between
#   instantiation 1 and instantiation 72 is a value a check can enumerate.
#   `algo_l_sample` gets the same treatment for its `tid`, though it has no
#   collectives to unwind.
#
#   Why. Identical to the reason `partition_samples` is a host function
#   (`builder_kernels_impl.mojo`): this is the ORACLE the device kernel will be
#   checked against, and `PORTING_RULES.md` 0b-ii says an oracle is not a CPU
#   path and stays. Both samplers are deterministic given the block width --
#   nothing in either depends on warp scheduling or on the order the sort
#   visits equal keys -- so the host form reproduces the device form's exact
#   output, not merely an equivalent sample.
#
#   Why the sort substitution cannot change an answer. `BlockRadixSort::Sort`
#   sorts KEYS ONLY here (`:224` passes one array), and the keys are the
#   column ids themselves. Two equal keys are indistinguishable, so every
#   ascending sort produces the identical array; there is no payload whose
#   permutation could differ. This is the one collective that is not
#   transcribed instruction for instruction, and it is the one where that is
#   provably free.
#
#   Price. `n_slots` int32s of host memory per block instead of registers, and
#   an O(n log n) sort where theirs is a radix pass. No timing claim is
#   attached to either and none will be until the perf round; when the device
#   kernel lands it will need `max.gpu.primitives.block`, and the sort will
#   need to become a real block radix sort, which is a KERNEL MATRIX row.
#
# ---------------------------------------------------------------------------
#
# 158. `log`, `exp` and `ceil` come from libm through FFI, not `std.math`.
#
#   Theirs. `raft::log` / `raft::exp` (`raft/core/math.hpp:324-331`) are
#   `std::log` / `std::exp` on the host and `::log` / `::exp` on device, and
#   `std::ceil` at `builder.cuh:420`.
#
#   Ours. `external_call["log"|"logf"|"expf"|"ceil"]`, the same libm the C++
#   host build links against, keeping the float/double split of their
#   expressions exactly (see the type notes in `algo_l_sample`).
#
#   Why. This repository's standing finding: `std.math.log` carries ~5e-8
#   ABSOLUTE error against libm, which is enough to re-decide a tie. Both of
#   these call sites turn a log into an INTEGER: `n_parallel_samples_for`
#   takes `ceil` of a ratio of logs, so an error near an integer flips the
#   sample count and, at the `9216` and `128` boundaries, flips which KERNEL
#   runs; `algo_l_sample` truncates `log(u)/log(1-W)` to an int, so an error
#   near an integer picks a different column. Neither is a rounding-quality
#   question.
#
#   Price. Four un-inlinable calls per use and a dependency on the host libm.
#   The device version cannot use FFI and will have to re-establish this: the
#   jump line is the one to check, and the check to run against it is
#   `feature_sampler_check.mojo`'s determinism and distinctness cases at the
#   `(n, k)` pairs it already names.
#
# ---------------------------------------------------------------------------
#
# 159. Index widths, and a bound on their unbounded loop.
#
#   Theirs. `IdxT` is `int` throughout the sampler: `n_parallel_samples`,
#   `n_uniques`, `items[]`, `mask[]`, `col_indices[]`. `n`, `k` and algo-L's
#   `col` are `size_t`. `do { } while (n_uniques < k)` (`:188-241`) has no
#   iteration bound, and `plan_feature_sampling`'s inputs are trusted -- `k >
#   n` would make `log(1 - k/n)` a NaN and `k < 1` is meaningless.
#
#   Ours. Column ids stay `Int32` (their `IdxT`, and the width `colids`
#   actually is), while counts and loop indices are Mojo's `Int`, which is
#   64-bit. `EXCESS_MAX_ITERATIONS = 1024` bounds the do-while and RAISES
#   instead of hanging, and `plan_feature_sampling` REFUSES `k < 1` or `k > n`
#   by name rather than returning a NaN-derived arm.
#
#   Why. `n_parallel_samples` is about `n * ln(n/(n-k))`, so overflowing
#   `int32` needs `n` around `1e8` columns -- a dataset that cannot be held --
#   and every value the two agree on is identical; this is a width choice with
#   no reachable disagreement. The bound and the refusal exist because this is
#   a HOST oracle that a check runs unattended: a hang is a failure mode a
#   check cannot report, and cuML reaches neither state because
#   `max_features` is clamped into `[1, N]` before `doSplit` sees it.
#
#   Price. Three departures from their control flow that a device port must
#   NOT carry over blindly: an `Error` cannot be raised from a kernel, so the
#   device version needs the bound expressed some other way, or dropped with
#   the argument that the dispatch guarantees termination written down.
