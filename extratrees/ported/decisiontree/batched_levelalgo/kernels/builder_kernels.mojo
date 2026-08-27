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
    """`builder_kernels.cuh:59-67` -- with DEVIATION 216 on the first clause.

    Theirs: `split.best_metric_val <= min_impurity_decrease || ...` -- the
    `<=` rejects a split whose gain exactly equals the threshold, so at the
    default 0 cuML turns every ZERO-GAIN node into a leaf. sklearn accepts
    equality (`>=` in its own orientation), so at the default it SPLITS
    zero-gain nodes, and the split rule this lane implements is sklearn's
    (the reference for the formulation, per the lane header). The difference
    is not cosmetic on integer-valued targets, where exact zero gain is
    common: on year (integer years, `max_features=all`, depth 8) cuML's gate
    left our trees ~13% smaller than sklearn's at EQUAL train MSE and
    measurably worse TEST MSE -- the gate was discarding refining splits.
    DEVIATION 216 changes the clause to strict `<`, sklearn's boundary; an
    invalid candidate's `MIN_FINITE` sentinel is still rejected by it. For
    NONZERO `min_impurity_decrease` the two libraries also SCALE the gain
    differently; that parity question is unchanged by this edit and still
    documented at the estimator boundary.
    """
    return (
        split.best_metric_val < min_impurity_decrease
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
from std.sys.compile import is_defined

from extratrees.mojo_only.pcg_rng import (
    FNV1A32_BASIS,
    PCGenerator,
    fmix32,
    fnv1a32,
    uniform_int_u64,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    identical_exp,
    identical_log,
    portable_logf,
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
    `key_for` in `extratrees/mojo_only/pcg_rng.mojo` chains FOUR -- a salt
    and then a per-FEATURE slot, because DEVIATION 130 needs a per-feature
    threshold key. That extension belongs to the threshold draw and to
    nothing else: the sampler's key is cuML's, so that a node's candidate
    column set is bit-identical to theirs. (DEVIATION 465: until 2026-08-26
    `key_for` in fact chained THREE -- byte-identical to this function at
    `feature_id == thread_id` -- so the two streams collided on covtype's
    low column ids; the salt link is what makes this paragraph true.)

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


comptime EXCESS_SELECTION_SALT = UInt32(0x5E1EC7ED)

comptime ET_SELECTION_HASH_FINALIZED = not is_defined[
    "MOJOLEARN_ET_RAW_SELECTION_HASH"
]()
"""A/B arm for DEVIATION 464: build with `-D MOJOLEARN_ET_RAW_SELECTION_HASH=1`
to restore the pre-464 un-finalized fnv chain, host and device together (so
`sampler_kernel_check`'s host-vs-device control stays bit-identical in BOTH
builds)."""


def excess_selection_hash_raw(
    tree_id: UInt32, node_id: UInt32, col: UInt32
) -> UInt32:
    """The bare fnv1a32 chain -- the pre-DEVIATION-464 selection key, kept as
    the sabotage/A-B arm. See `excess_selection_hash` for why it is not the
    shipping form: for `col < 256` this chain is affine in `perm(col)` with a
    multiplier whose top bits rotate by ~5/16 of the circle, so ranking by it
    is a 16-cluster Weyl walk, not a permutation."""
    var h = fnv1a32(FNV1A32_BASIS, EXCESS_SELECTION_SALT)
    h = fnv1a32(h, tree_id)
    h = fnv1a32(h, node_id)
    h = fnv1a32(h, col)
    return h


def excess_selection_hash(
    tree_id: UInt32, node_id: UInt32, col: UInt32
) -> UInt32:
    """DEVIATION 215's selection key: which `k` of the uniques survive when
    the excess sampler drew MORE than `k`.

    cuML keeps the `k` SMALLEST unique column ids (`:243-246`) -- a real
    selection bias, not a tie-break: at higgs's `(n=28, k=5)` the dispatch
    draws only 6 samples, and simulating their rule puts column 27 in the
    sample at 0.38x column 0's rate. That is cuML's own bug, and this lane's
    standing rule is to fix their bugs, not port them (164 and 165 fixed two
    others in this same kernel). The fix: rank the uniques by THIS keyed hash
    instead of by column id, and keep the `k` smallest BY HASH -- a uniform
    `k`-subset of the uniques, deterministic from `(tree, node, col)` alone,
    identical on host and device, and free of any cross-thread draw order.
    The salt keeps this stream disjoint from `excess_subsequence`'s and
    `key_for`'s.

    DEVIATION 464 (2026-08-26): the combined word now runs through `fmix32`.
    The bare chain had NO avalanche on the final `col` round: for `col < 256`
    the three trailing zero-byte rounds collapse to `K + perm(col) * C mod
    2^32` with `C = FNV1A32_PRIME^3` and `C / 2^32` within 2^-9 of 5/16 -- a
    16-cluster Weyl set, not a permutation. Rank-by-hash under that map keeps
    every MARGINAL uniform (each column's rate is fine, which is why the
    (n=28, k=5) marginal gate stayed green) but distorts the JOINT k-of-
    uniques selection -- which columns co-occur -- on every node that takes
    the overshoot branch (~58% of covtype nodes). `fmix32` is a bijection,
    so the marginals stay uniform and the joint selection becomes what
    DEVIATION 215 claimed it was. The un-finalized form survives as
    `excess_selection_hash_raw` (`SAMP_SAB_RAW_SELECTION_HASH`, and the
    `MOJOLEARN_ET_RAW_SELECTION_HASH` build define).
    """
    var h = excess_selection_hash_raw(tree_id, node_id, col)
    comptime if ET_SELECTION_HASH_FINALIZED:
        return fmix32(h)
    return h


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


def NON_DRAWING_SENTINEL(n: Int) -> Int32:
    """A value above every valid column id, for a slot that did not draw.

    DEVIATION 165. Any value `>= n` works; `n` itself is the smallest such and
    keeps the sorted array's tail adjacent to the real ids, which makes the
    masking loop a single comparison rather than a range test.
    """
    return Int32(n)


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

    DEVIATION 457 -- this dispatch value is MODE-GATED. The libm arm below
    is host-vendor arithmetic, so cross-vendor is cross-HOST here -- macOS
    libm and glibc can disagree in `log`'s last bits, and near an integer
    the `ceil` turns that last bit into a different draw count, and at the
    128/9216 boundaries into a DIFFERENT KERNEL (IDENTITY_PATHS row 18's ET
    half). Under IDENTICAL the count comes from `n_parallel_samples_portable`
    instead, one arithmetic on every host. Under FAST the libm arm runs
    verbatim -- bit for bit the code that has always been here.
    """
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return n_parallel_samples_portable(n, k)
    return n_parallel_samples_libm(n, k)


def n_parallel_samples_libm(n: Int, k: Int) -> Int:
    """DEVIATION 158's arm of `n_parallel_samples_for` -- both logs through
    HOST libm in double, cuML's types exactly. This is the body that lived
    inline in `n_parallel_samples_for` until DEVIATION 457 named it, so a
    check can hold it against the portable arm on one host; the FAST gate
    above compiles to exactly this call and nothing else."""
    var ratio = _log64(1.0 - Float64(k) / Float64(n))
    var per_draw = _log64(1.0 - 1.0 / Float64(n))
    return Int(_ceil64(ratio / per_draw))


def n_parallel_samples_portable(n: Int, k: Int) -> Int:
    """DEVIATION 457's arm -- the same `ceil(log(1 - k/n) / log(1 - 1/n))`
    with both logs through row 12's `portable_logf`, so every HOST computes
    the same count from the same `(n, k)`.

    Every seam that is NOT a log is already one arithmetic everywhere and
    needs no replacement -- `Float32(k)` and `Float32(n)` are exact below
    2^24 columns, the divide and the subtract are single correctly rounded
    IEEE operations on every host, the quotient of the two logs is one more
    correctly rounded divide, and `ceil` of a float is exact (its result is
    integral, there is nothing to round). The two logs were the only
    host-vendor seams, and they are the two calls replaced.

    THE PRICE, MEASURED (this is row 18's knife edge made visible, not a
    bug) -- float32 cannot form `1 - 1/n` to double precision, so this arm's
    count sits a few counts below the libm arm's at large `n` (at n=20000
    the sweep in `feature_sampler_check` sees 17278 of 19999 k differ, all
    within [-36, -1]) and at exactly one swept k per large n the difference
    crosses a dispatch boundary -- (n=20000, k=7385) is fast=9217/algo-L,
    portable=9216/excess. Either sampler is a correct draw of k distinct
    columns at ANY count, so this moves WHICH KERNEL, never correctness;
    IDENTICAL bits differ from FAST bits BY DESIGN, and the property
    purchased is that IDENTICAL's count is one value on every host."""
    from std.math import ceil

    var ratio = portable_logf(Float32(1.0) - Float32(k) / Float32(n))
    var per_draw = portable_logf(Float32(1.0) - Float32(1.0) / Float32(n))
    return Int(ceil(ratio / per_draw))


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
    tree_ids: List[Int32],
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
       (`:201-203`) -- AND THAT IS THE SAME BUG WEARING DIFFERENT CLOTHES,
       ALSO FIXED HERE. See DEVIATION BLOCK 165. A slot that did not draw
       votes anyway, as column `n - 1`, so that column is present in the
       block's sample on every iteration and is selected whenever the loop
       stops at exactly `k` uniques. Measured before the fix: column `n - 1`
       chosen 662 times against 512 expected. Only the FIRST
       `n_parallel_samples` slots draw, and the condition is on the CTA-wide
       slot index, not the thread's.

    4. THE PREDECESSOR OF THE FIRST ITEM IS `mask[0]`, NOT AN ITEM -- AND
       THAT IS A BUG IN cuML, FIXED HERE. See DEVIATION BLOCK 164 below.
       `:231-232` calls `SubtractLeft(items, mask, CustomDifference<IdxT>(),
       mask[0])`, and CUB's implementation
       (`cccl/cub/cub/block/block_adjacent_difference.cuh:393-419`) does
       `output[0] = difference_op(input[0], tile_predecessor_item)` on thread
       0. So the block MINIMUM is compared against the previous iteration's
       FLAG -- a 0 or a 1 -- and is flagged a duplicate when it equals it.
       Measured before the fix: at `k = 1`, column 0 was drawn 0 times of 64
       at every one of n = 2, 3, 4 and 8. Not under-drawn. NEVER drawn.

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
                        UInt32(thread_id),
                        # DEVIATION 211: per-ITEM tree id, so one batch may
                        # span trees. Same key, same draws.
                        UInt32(Int(tree_ids[block_id])),
                        node_id,
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
                        # DEVIATION 165: theirs writes `n - 1` here, a REAL
                        # column id, so a slot that did not draw votes anyway.
                        # Ours writes an out-of-range sentinel that is masked
                        # off below, so a non-drawing slot casts no vote.
                        items[slot] = NON_DRAWING_SENTINEL(n)
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
            for slot in range(n_slots - 1, 0, -1):
                mask[slot] = 0 if items[slot] == items[slot - 1] else 1
            # DEVIATION 164: the block minimum has NO predecessor, so its head
            # flag is unconditionally 1. Theirs compares it against `mask[0]`,
            # the previous iteration's FLAG, and drops it when they are equal.
            mask[0] = 1

            # DEVIATION 165: a slot that did not draw casts no vote. The
            # sentinel sorts above every real column id, so these are the tail
            # of the sorted array; clearing their head flags keeps them out of
            # `n_uniques` and out of the gather.
            for slot in range(n_slots):
                if items[slot] >= Int32(n):
                    mask[slot] = 0

            # `:237`, `BlockScanT(...).ExclusiveSum(mask, col_indices, n_uniques)`.
            var running: Int32 = 0
            for slot in range(n_slots):
                col_indices[slot] = running
                running += mask[slot]
            n_uniques = Int(running)

            if n_uniques >= k:  # `:241`, `while (n_uniques < k)`
                break

        # `:244-247` -- with DEVIATION 215 replacing WHICH `k` survive.
        # cuML's gather keeps the `k` smallest unique COLUMN IDS, a selection
        # bias (see `excess_selection_hash`). When the loop landed on exactly
        # `k` uniques there is nothing to select and their gather is kept
        # verbatim; when it overshot, the `k` kept are the smallest by the
        # keyed hash -- a uniform `k`-subset -- and the slot order is hash
        # order, which no consumer reads meaning into (candidates are
        # order-independent by the total-order reduction).
        var col_offset = k * block_id
        if n_uniques <= k:
            for slot in range(n_slots):
                if mask[slot] != 0 and Int(col_indices[slot]) < k:
                    colids[col_offset + Int(col_indices[slot])] = items[slot]
        else:
            var tree_u = UInt32(Int(tree_ids[block_id]))
            for slot in range(n_slots):
                if mask[slot] == 0:
                    continue
                var col = items[slot]
                var h = excess_selection_hash(
                    tree_u, node_id, UInt32(Int(col))
                )
                var rank = 0
                for s in range(n_slots):
                    if mask[s] == 0:
                        continue
                    var c2 = items[s]
                    var h2 = excess_selection_hash(
                        tree_u, node_id, UInt32(Int(c2))
                    )
                    if h2 < h or (h2 == h and c2 < col):
                        rank += 1
                if rank < k:
                    colids[col_offset + rank] = col


# --- arm 2: reservoir sampling, algorithm L --------------------------------


def algo_l_sample(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_ids: List[Int32],
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
        # DEVIATION 211: per-ITEM tree id, so one batch may span trees.
        var gen = PCGenerator(
            seed,
            algo_l_subsequence(UInt32(Int(tree_ids[tid])), node_id),
            UInt64(0),
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
    """One-tree form of `sample_features_pertree` below: every work item
    belongs to `tree_id`. The host trainer and the sampler checks call this;
    the batched forest trainer (DEVIATION 211) calls the per-item form."""
    var tree_ids = List[Int32](length=len(work_items), fill=tree_id)
    return sample_features_pertree(colids, work_items, tree_ids, seed, n, k)


def sample_features_pertree(
    mut colids: List[Int32],
    work_items: List[NodeWorkItem],
    tree_ids: List[Int32],
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

    `colids` must already be `len(work_items) * k` long; `tree_ids` holds one
    tree id PER WORK ITEM (DEVIATION 211: a batch may span trees, and each
    item's draws are keyed by ITS tree). The arm is chosen by `(n, k)` alone,
    so a merged batch takes the same arm every single-tree batch would.
    """
    if len(tree_ids) != len(work_items):
        raise Error(
            "sample_features_pertree: "
            + String(len(work_items))
            + " work items but "
            + String(len(tree_ids))
            + " tree ids"
        )
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
            tree_ids,
            seed,
            n,
            k,
            plan.n_parallel_samples,
            plan.max_samples_per_thread,
            plan.block_threads,
        )
    else:
        algo_l_sample(colids, work_items, tree_ids, seed, n, k)
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
#   ADDENDUM 2026-08-23 -- libm itself is a HOST-vendor arithmetic, which is
#   the half of the story this entry could not fix. DEVIATION 457 gates
#   `n_parallel_samples_for` on the numeric mode (IDENTICAL computes the
#   count through row 12's `portable_logf`), and DEVIATION 456 does the same
#   for the device kernel's float seams. Under FAST every call this entry
#   describes runs verbatim.
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


# ===========================================================================
# THE SAME TWO SAMPLERS, ON THE DEVICE, WHERE cuML RUNS THEM
#
# `PORTING_RULES.md` rule 2: "If they do something on the GPU in the control
# plane, we do it on the GPU." cuML LAUNCHES its sampler
# (`builder.cuh:434-461`); everything above this line is a HOST transcription,
# which rule 0b-ii admits as an ORACLE and nothing else. These are the
# kernels, and the host functions above are what they are checked against,
# slot for slot, by `extratrees/mojo_only/sampler_kernel_check.mojo`.
#
# WHAT IS *NOT* MOVED, AND WHY THAT IS ALSO RULE 2
# -------------------------------------------------
# `plan_feature_sampling` STAYS ON THE HOST because it is on the host in cuML
# too: `n_parallel_samples` is a `std::ceil` of a ratio of `std::log`s
# evaluated in `Builder::doSplit` (`builder.cuh:419-421`), and the choice
# between the three arms is a C++ `if` around three different `<<<...>>>`
# launches (`:434-466`). Moving that decision onto the device would be a
# deviation FROM them, not toward them. Rule 2's second sentence -- "if they
# keep a decision on the device so the host never learns it, we keep it on the
# device" -- does not bite: their host learns this decision, because their
# host makes it.
#
# THREE MEASURED WALLS, RECORDED HERE BECAUSE THEY SHAPE EVERY LINE BELOW
# -----------------------------------------------------------------------
# 1. `cub::BlockRadixSort` has no MAX counterpart. `max.gpu.primitives.block`
#    offers `sum`, `min`, `max`, `broadcast` and `prefix_sum`, and nothing
#    else. Hand-written under `PORTING_RULES.md` 0b-i; DEVIATION 195.
# 2. Their sort's storage does not FIT. Their own calibration is
#    `max_samples_per_thread * block_threads * sizeof(IdxT)` = 36864 bytes and
#    their comment budgets 46KB (`builder.cuh:403-414`); this box refuses at
#    32768. DEVIATION 196.
# 3. Metal has no `double` AT ALL, and `algo_L_sample_kernel` is a double
#    algorithm. DEVIATION 199.
# ===========================================================================

from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv, exp, log
from std.memory import stack_allocation
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceContext
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier


# --- what the device writes about itself -----------------------------------

comptime SAMPLER_UNVISITED: Int32 = -3
"""The seed the CALLER must write into every cell of `report`. It is not a
legal outcome: a cell still holding it means NO BLOCK SERVED THAT WORK ITEM,
which is a grid too small or a kernel that never ran. `sampler_kernel_check`
asserts there are none, and that assertion is the REACH evidence for these
kernels -- rule 15, a digest cannot tell a working change from a no-op."""

comptime SAMPLER_OVERRUN: Int32 = -2
"""`EXCESS_MAX_ITERATIONS` was reached. DEVIATION 159 bounds their unbounded
`do { } while (n_uniques < k)` and RAISES on the host; a kernel cannot raise,
so the device reports it as a status and writes no columns."""


# --- sabotage selectors, one per MECHANISM ---------------------------------
# A kernel ARGUMENT and not a comptime parameter, for the reason the
# `RANGE_SAB_*`, `SCORE_SAB_*` and `PART_SAB_*` families in
# `builder_kernels_impl.mojo` state: a sabotage compiled into a different
# binary proves nothing about the binary that ships, so every arm of
# `sampler_kernel_check.mojo` runs THESE kernels.

comptime SAMP_SAB_NONE: Int32 = 0
"""No sabotage. The shipping path."""

comptime SAMP_SAB_SORT_DESCENDING: Int32 = 1
"""Flip the bitonic comparator direction, so the network sorts DESCENDING.
The array is still fully sorted and every duplicate is still adjacent, so the
dedupe and the scan still "work" -- what changes is WHICH `k` uniques survive
the `col_indices < k` cut (`:245`), because the gather takes a prefix of the
sorted order. Point 5 of `excess_sample_with_replacement`'s docstring: the
output is the `k` SMALLEST uniques, not a random `k`."""

comptime SAMP_SAB_NO_DEDUPE: Int32 = 2
"""Skip the adjacent-difference test, so EVERY slot is a head. The gather then
emits repeated columns and `n_uniques` is the slot count rather than the
unique count. This is the mechanism `CustomDifference` (`:133-142`) is."""

comptime SAMP_SAB_AGG_ONE_PER_THREAD: Int32 = 3
"""Scan ONE item per thread instead of `MAX_SAMPLES_PER_THREAD` of them --
the classic mis-port of a CUB collective that has an `ITEMS_PER_THREAD`
template argument. Both the exclusive prefix and the aggregate are taken over
each thread's slot 0 alone.

STRUCTURALLY INERT AT `MAX_SAMPLES_PER_THREAD == 1`, and that is not a hole in
the fixture: at M=1 a thread HAS one item, so "one item per thread" and "all
of this thread's items" are the same expression. The check runs this arm on
the M=72 instantiations, where the mechanism exists, and says so."""

comptime SAMP_SAB_KEY_NO_THREAD: Int32 = 4
"""Drop `threadIdx.x` from the fnv1a32 chain (`:168`), so every thread of the
block draws the IDENTICAL stream. The block still samples, still sorts, still
dedupes -- it just has `BLOCK_THREADS` times fewer distinct values to find,
which is the failure a port that keyed on the block alone would have."""

comptime SAMP_SAB_LOOP_ANY_UNIQUE: Int32 = 5
"""`while (n_uniques < k)` (`:241`) becomes `while (n_uniques < 1)`, so the
do-while stops after one pass whatever it found. The excess strategy IS that
loop: without it, `n_parallel_samples` draws with replacement are not enough
to guarantee `k` uniques and the tail of `colids` is never written."""

comptime SAMP_SAB_CUML_MASK0: Int32 = 6
"""RESTORE cuML's bug 164: compare the block minimum against `mask[0]`, the
PREVIOUS iteration's FLAG, instead of giving it the unconditional head flag a
minimum has by definition. The fix is what makes column 0 reachable at all;
this arm is how the check proves the FIX is load-bearing on the device path
and not just on the host."""

comptime SAMP_SAB_CUML_FILLER: Int32 = 7
"""RESTORE cuML's bug 165: fill a slot past `n_parallel_samples` with `n - 1`,
a REAL column id, instead of the out-of-range `NON_DRAWING_SENTINEL`. A slot
that did not draw then votes anyway, every iteration, for one specific
column."""

comptime SAMP_SAB_ALL_REVERSED: Int32 = 8
"""`sample_all_features_kernel` writes `k - 1 - c` instead of `c`. DEVIATION
156 materializes an IDENTITY, in order; a reversal has the same column SET and
is invisible to any check that only looks at the set."""

comptime SAMP_SAB_SMALLEST_K: Int32 = 9
"""cuML's ORIGINAL selection (`:243-246`): when the loop drew more than `k`
uniques, keep the `k` smallest COLUMN IDS instead of DEVIATION 215's uniform
keyed-hash subset. This is the bug arm: it under-samples high-indexed columns
(0.38x at higgs's `(28, 5)`), and the uniformity gate in
`sampler_kernel_check` must go red under it -- that is what proves the gate
watches the selection distribution and not just the set's validity."""

comptime SAMP_SAB_RAW_SELECTION_HASH: Int32 = 10
"""DEVIATION 464's pre-fix arm: rank the overshoot's uniques by the
UN-FINALIZED fnv chain (`excess_selection_hash_raw`) instead of the
avalanche-finalized hash. Against the host oracle this differs exactly on
overshoot nodes (a slot comparison sees it there); the (n=28, k=5)
MARGINAL uniformity gate does NOT see it -- the raw hash's marginals are
uniform, its defect is the JOINT selection, which no existing gate asserts
(the distribution gate only ever asserted marginals). It is an A/B and
reach arm, not a required-RED arm of the marginal gate."""


# --- host-side sizing, so the caller never guesses a length ----------------


def next_pow2(x: Int) -> Int:
    """The smallest power of two `>= x`, and `1` for `x <= 1`."""
    var p = 1
    while p < x:
        p <<= 1
    return p


def excess_sort_span(plan: FeatureSamplerPlan) -> Int:
    """How many slots the bitonic network covers: `next_pow2` of
    `n_parallel_samples`, NOT of `block_threads * max_samples_per_thread`.

    WHY THE SHORTER SPAN IS THE SAME ANSWER, which is the whole reason
    DEVIATION 196 is affordable. Slots at or above `n_parallel_samples` are
    filled with `NON_DRAWING_SENTINEL(n) == n` on iteration 1 and can never
    hold anything else: a slot only keeps a value when its mask is 1, and
    DEVIATION 165 clears the mask of every item `>= n`. So the tail is a
    constant run of the array's MAXIMUM, already in its sorted position, and
    sorting `[0, next_pow2(n_parallel_samples))` leaves the full
    `block_threads * max_samples_per_thread` array in exactly the order a sort
    of the whole thing would.
    """
    return next_pow2(plan.n_parallel_samples)


def excess_scratch_stride(plan: FeatureSamplerPlan) -> Int:
    """Int32s of scratch per BLOCK. The sort span can exceed the slot count
    (`n_parallel_samples` may sit just above a power of two), so the stride is
    the larger of the two and the overhang is padded with the sentinel."""
    var n_slots = plan.block_threads * plan.max_samples_per_thread
    var span = excess_sort_span(plan)
    return n_slots if n_slots > span else span


def sampler_scratch_len(work_items_size: Int, n: Int, k: Int) raises -> Int:
    """Int32s the caller must allocate for `d_scratch`. Never 0 -- a
    zero-length device buffer is not a legal allocation, and the two arms that
    do not sort still want a name to pass."""
    var plan = plan_feature_sampling(n, k)
    if plan.arm != SAMPLE_EXCESS:
        return 1
    return work_items_size * excess_scratch_stride(plan)


def sampler_report_len(work_items_size: Int) -> Int:
    """Int32s the caller must allocate for `d_report`, and seed with
    `SAMPLER_UNVISITED`. Three per work item: `[arm, instantiation,
    iterations]`. The kernel writes all three; no host arithmetic contributes
    to any of them, which is what makes them evidence."""
    return 3 * work_items_size


def device_has_float64() -> Bool:
    """Can a kernel on this box hold a `double`?

    Mojo 1.0 exposes no fp64 capability query, so this is written as the one
    fact that is true today: of the three GPU targets Mojo supports, Apple is
    the one whose backend refuses the TYPE. It is a HOST predicate and it
    gates a HOST dispatch decision -- there is no `if apple` anywhere in a
    kernel, and when Mojo grows a real query this becomes one line. See
    DEVIATION 199 for the measured refusal.
    """
    return not has_apple_gpu_accelerator()


# --- arm 1 on device: `excess_sample_with_replacement_kernel` --------------


def excess_sample_kernel[
    BLOCK_THREADS: Int, MAX_SAMPLES_PER_THREAD: Int
](
    colids: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    report: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    work_items_size_in: Int32,
    seed: UInt64,
    n_in: Int32,
    k_in: Int32,
    n_parallel_samples_in: Int32,
    scratch_stride_in: Int32,
    sabotage_in: Int32,
):
    """`excess_sample_with_replacement_kernel`, `builder_kernels.cuh:152-248`,
    ON THE DEVICE.

    GRID. `grid_dim = (work_items_size, 1, 1)`, `block_dim = (BLOCK_THREADS,
    1, 1)`, which is `builder.cuh:434-455`'s `grid.x = work_items.size()` with
    `block_threads` threads -- ONE BLOCK PER WORK ITEM. `block_idx.x` indexes
    `work_items`, `colids`' row and `scratch`' region alike, exactly as their
    `blockIdx.x` does. The `if (blockIdx.x >= work_items_size) return` of
    `:163` is kept even though the grid is sized to the batch.

    THE ANSWER IS `excess_sample_with_replacement`'S, SLOT FOR SLOT, and the
    check compares it that way rather than comparing two samples for
    equivalence. Nothing here depends on scheduling: the draws are keyed per
    `(thread, tree, node)`, the sort is a fixed network over the same slots,
    the dedupe is a pure function of adjacent slots, and the two scan results
    are block collectives. Given `BLOCK_THREADS` and `MAX_SAMPLES_PER_THREAD`
    the output is determined.

    WHAT LIVES WHERE, because their three register arrays could not all stay
    registers (DEVIATION 195, DEVIATION 196):

    * `items` -> `scratch`, in GLOBAL memory, `excess_scratch_stride(plan)`
      Int32s per block. It is the array the block sorts, so every thread must
      see every slot, and 36864 bytes of it will not fit in this box's 32768
      bytes of threadgroup memory.
    * `mask` -> a PRIVATE `stack_allocation[MAX_SAMPLES_PER_THREAD]` per
      thread, which is where theirs is. It is indexed by the thread's own
      slots and never read across threads: the ITEMS migrate between threads
      under the sort (point 2 of the host docstring), the SLOTS do not.
    * `col_indices` -> not stored at all. It is the exclusive prefix sum of
      `mask`, and the gather recomputes it in one pass from the block scan's
      per-thread prefix.

    CALLER OBLIGATIONS:

    * `colids` -- `work_items_size * k` Int32s. Every cell is written on the
      shipping path unless the block reports `SAMPLER_OVERRUN`.
    * `scratch` -- `sampler_scratch_len(work_items_size, n, k)` Int32s. Its
      contents are scratch, not output, and it needs no seed.
    * `report` -- `sampler_report_len(work_items_size)` Int32s, SEEDED WITH
      `SAMPLER_UNVISITED`.
    * `n_parallel_samples_in` and `scratch_stride_in` come from
      `plan_feature_sampling` and `excess_scratch_stride`; passing anything
      else silently changes the algorithm, which is why
      `sample_features_device` below is the intended caller.

    `sabotage_in` selects the arms; `SAMP_SAB_NONE` is the shipping path.
    """
    var b = Int(block_idx.x)
    if b >= Int(work_items_size_in):  # `:163`
        return
    var tid = Int(thread_idx.x)
    var sab = sabotage_in
    var n = Int(n_in)
    var k = Int(k_in)
    var nps = Int(n_parallel_samples_in)
    comptime M = MAX_SAMPLES_PER_THREAD
    comptime N_SLOTS = BLOCK_THREADS * MAX_SAMPLES_PER_THREAD

    # `:165` -- `work_items[blockIdx.x].idx`, ONE FIELD through the pointer.
    # DEVIATION 162: a whole-struct load in a kernel kills the Metal compiler.
    var node_id = work_items[unsafe_offset=b].idx.cast[DType.uint32]()

    var items = scratch.unsafe_offset(b * Int(scratch_stride_in))

    # `:167-172`. The shipping path calls the SAME `excess_subsequence` the
    # host oracle calls, so the key chain cannot drift between the two; only
    # the sabotage is spelled out here.
    # DEVIATION 211: the tree id is per NODE (`tree_ids[b]`), not per launch.
    var tree_u = tree_ids[unsafe_offset=b].cast[DType.uint32]()
    var subsequence: UInt64
    if sab == SAMP_SAB_KEY_NO_THREAD:
        var h = FNV1A32_BASIS
        h = fnv1a32(h, tree_u)
        h = fnv1a32(h, node_id)
        subsequence = h.cast[DType.uint64]()
    else:
        subsequence = excess_subsequence(UInt32(tid), tree_u, node_id)
    var gen = PCGenerator(seed, subsequence, UInt64(0))

    # `:184-186` -- their `for (i) mask[i] = 0`, so every slot draws on
    # iteration 1. PRIVATE, not shared: see the docstring.
    var mask = stack_allocation[M, Scalar[DType.int32]]()
    for l in range(M):
        mask[unsafe_offset=l] = Int32(0)

    # The bitonic network needs a power of two. `excess_sort_span` on the
    # host computes the same number and sizes `scratch` from it.
    var span = 1
    while span < nps:
        span <<= 1

    # The overhang, when `span > N_SLOTS`. Written once: the sort can only
    # ever put the array's maximum back into it, and the draw loop below
    # never touches a slot at or above `N_SLOTS`.
    var pad = N_SLOTS + tid
    while pad < span:
        items[unsafe_offset=pad] = NON_DRAWING_SENTINEL(n)
        pad += BLOCK_THREADS

    var n_uniques = 0
    var thread_prefix = 0
    var iters = 0
    var overrun = False

    while True:  # `:188` .. `:241`, a do-while
        iters += 1
        if iters > EXCESS_MAX_ITERATIONS:
            # DEVIATION 159's bound, as a STATUS. A kernel cannot raise and a
            # hang is the one failure a check cannot report. `iters` is
            # block-uniform, so the whole block leaves together and no
            # collective below is entered by a subset of the threads.
            overrun = True
            break

        # `:190-207`. `cta_sample_idx` starts at `MAX_SAMPLES_PER_THREAD *
        # threadIdx.x`, so it IS the blocked-arrangement slot number, and the
        # `< n_parallel_samples` test is on THAT and not on the thread.
        for l in range(M):
            var slot = M * tid + l
            if mask[unsafe_offset=l] == Int32(0) and slot < nps:
                items[unsafe_offset=slot] = Int32(
                    uniform_int_u64(gen, 0, UInt64(n))
                )
            elif mask[unsafe_offset=l] == Int32(0):
                if sab == SAMP_SAB_CUML_FILLER:
                    items[unsafe_offset=slot] = Int32(n - 1)  # their `:201-203`
                else:
                    items[unsafe_offset=slot] = NON_DRAWING_SENTINEL(n)
            # else: their `continue` -- keep the previous iteration's unique.
        barrier()

        # `:224`, `BlockRadixSortT(...).Sort(items)`. DEVIATION 195: a
        # hand-written bitonic network in global scratch. Zero warp
        # intrinsics, no assumed wavefront width, one `barrier()` per stage,
        # and each pair `(i, i ^ j)` is touched by exactly one thread.
        var kk = 2
        while kk <= span:
            var j = kk >> 1
            while j > 0:
                var i = tid
                while i < span:
                    var partner = i ^ j
                    if partner > i:
                        var a = items[unsafe_offset=i]
                        var c = items[unsafe_offset=partner]
                        var up = (i & kk) == 0
                        if sab == SAMP_SAB_SORT_DESCENDING:
                            up = not up
                        var swap = (a > c) if up else (a < c)
                        if swap:
                            items[unsafe_offset=i] = c
                            items[unsafe_offset=partner] = a
                    i += BLOCK_THREADS
                barrier()
                j >>= 1
            kk <<= 1

        # `:231-232`, the adjacent difference. Slot `s` reads slot `s - 1`,
        # which belongs to the PREVIOUS thread -- legal because `items` is in
        # global scratch and the sort's last `barrier()` ordered it.
        var local_sum = Int32(0)
        for l in range(M):
            var slot = M * tid + l
            var here = items[unsafe_offset=slot]
            var m: Int32
            if slot == 0:
                if sab == SAMP_SAB_CUML_MASK0:
                    # Their `SubtractLeft(..., mask[0])`: the block MINIMUM
                    # against the previous iteration's FLAG. `mask[0]` still
                    # holds it -- CUB writes item 0 last and so does this
                    # loop.
                    m = (
                        Int32(0) if here
                        == mask[unsafe_offset=0] else Int32(1)
                    )
                else:
                    # DEVIATION 164. A minimum has no predecessor.
                    m = Int32(1)
            else:
                var prev = items[unsafe_offset = slot - 1]
                m = Int32(0) if here == prev else Int32(1)
            if sab == SAMP_SAB_NO_DEDUPE:
                m = Int32(1)
            # DEVIATION 165. A slot that did not draw casts no vote. Inert on
            # SAMP_SAB_CUML_FILLER, where no item is ever `>= n`.
            if here >= Int32(n):
                m = Int32(0)
            mask[unsafe_offset=l] = m
            local_sum += m
        barrier()

        # `:237`, `BlockScanT(...).ExclusiveSum(mask, col_indices, n_uniques)`
        # over `MAX_SAMPLES_PER_THREAD` items per thread. Mojo 1.0 has no
        # primitive returning both the prefix and the aggregate, and none that
        # takes more than one item per thread, so the per-thread totals are
        # summed in registers first and then scanned -- which is what CUB's
        # multi-item BlockScan does internally. The aggregate must be
        # BLOCK-UNIFORM because it drives the loop condition, which
        # `block_sum`'s default `broadcast=True` gives.
        var scan_in = local_sum
        if sab == SAMP_SAB_AGG_ONE_PER_THREAD:
            scan_in = mask[unsafe_offset=0]
        thread_prefix = Int(
            block_prefix_sum[block_size=BLOCK_THREADS, exclusive=True](scan_in)
        )
        barrier()
        n_uniques = Int(block_sum[block_size=BLOCK_THREADS](scan_in))
        barrier()

        var need = k  # `:241`, `while (n_uniques < k)`
        if sab == SAMP_SAB_LOOP_ANY_UNIQUE:
            need = 1
        if n_uniques >= need:
            break

    # `:244-247`. `col_indices[i]` is the exclusive prefix sum of `mask`, so
    # it is this thread's block-scan prefix plus its own running count.
    # DEVIATION 215 replaces WHICH `k` survive an overshoot: cuML keeps the
    # `k` smallest unique column ids -- a selection bias (see
    # `excess_selection_hash`) -- and the fixed rule keeps the `k` smallest
    # by keyed hash. `n_uniques == k` has nothing to select, so their gather
    # runs verbatim there and under the SMALLEST_K sabotage arm. The rank
    # scan reads `items` from the block's GLOBAL scratch (deviation 195 put
    # it there) and recomputes each slot's headness by the same adjacent
    # rule the dedupe used; the sort's final barrier ordered the scratch.
    if not overrun:
        var col_offset = k * b
        if n_uniques <= k or sab == SAMP_SAB_SMALLEST_K:
            var running = Int32(thread_prefix)
            for l in range(M):
                var slot = M * tid + l
                var ci = running
                running += mask[unsafe_offset=l]
                if mask[unsafe_offset=l] != Int32(0) and Int(ci) < k:
                    colids[unsafe_offset = col_offset + Int(ci)] = items[
                        unsafe_offset=slot
                    ]
        else:
            # DEVIATION 464: the shipping rank is the avalanche-finalized
            # hash; SAMP_SAB_RAW_SELECTION_HASH restores the bare fnv chain.
            var raw_hash = sab == SAMP_SAB_RAW_SELECTION_HASH
            for l in range(M):
                var slot = M * tid + l
                if mask[unsafe_offset=l] == Int32(0):
                    continue
                var col = items[unsafe_offset=slot]
                var h: UInt32
                if raw_hash:
                    h = excess_selection_hash_raw(
                        tree_u, node_id, col.cast[DType.uint32]()
                    )
                else:
                    h = excess_selection_hash(
                        tree_u, node_id, col.cast[DType.uint32]()
                    )
                var rank = 0
                for s in range(N_SLOTS):
                    var c2 = items[unsafe_offset=s]
                    if c2 >= Int32(n):
                        continue
                    var is_head: Bool
                    if s == 0:
                        is_head = True
                    else:
                        is_head = c2 != items[unsafe_offset = s - 1]
                    if not is_head:
                        continue
                    var h2: UInt32
                    if raw_hash:
                        h2 = excess_selection_hash_raw(
                            tree_u, node_id, c2.cast[DType.uint32]()
                        )
                    else:
                        h2 = excess_selection_hash(
                            tree_u, node_id, c2.cast[DType.uint32]()
                        )
                    if h2 < h or (h2 == h and c2 < col):
                        rank += 1
                if rank < k:
                    colids[unsafe_offset = col_offset + rank] = col

    if tid == 0:
        report[unsafe_offset = 3 * b + 0] = Int32(SAMPLE_EXCESS)
        report[unsafe_offset = 3 * b + 1] = Int32(M)
        report[unsafe_offset = 3 * b + 2] = (
            SAMPLER_OVERRUN if overrun else Int32(iters)
        )


# --- arm 3 on device: DEVIATION 156's identity fill ------------------------


def sample_all_features_kernel(
    colids: MutPointer[Int32, MutAnyOrigin],
    report: MutPointer[Int32, MutAnyOrigin],
    work_items_size_in: Int32,
    k_in: Int32,
    sabotage_in: Int32,
):
    """`k == n`: every column is a candidate. DEVIATION 156.

    NO cuML COUNTERPART AS A KERNEL -- `builder.cuh:399` simply does not
    launch, and `colids` keeps whatever the previous batch left in it while
    the consumer branches on the same test. DEVIATION 156 records why this
    lane materializes the identity instead, and its "Price" paragraph asks
    whoever writes the device version to decide between the fill and their
    two-place branch. THIS IS THAT DECISION: the fill stays, on the device,
    because the consumer's half of their branch still does not exist here (the
    score pass of DEVIATION 137 reads `colids` unconditionally,
    `builder_kernels_impl.mojo::node_feature_range_kernel`) and one meaning
    for `colids` is what lets one check assert the same properties on all
    three arms.

    GRID. Flat: `grid_dim = ceildiv(work_items_size * k, TPB)`, `block_dim =
    TPB`. There is nothing per-node about it.
    """
    var k = Int(k_in)
    var items_size = Int(work_items_size_in)
    var g = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if g < items_size * k:
        var c = g % k
        if sabotage_in == SAMP_SAB_ALL_REVERSED:
            c = k - 1 - c
        colids[unsafe_offset=g] = Int32(c)
    if g < items_size:
        report[unsafe_offset = 3 * g + 0] = Int32(SAMPLE_ALL_FEATURES)
        report[unsafe_offset = 3 * g + 1] = Int32(0)
        report[unsafe_offset = 3 * g + 2] = Int32(0)


# --- arm 2 on device: `algo_L_sample_kernel` -------------------------------


# `std.math`, not libm through FFI: a kernel cannot make an FFI call, so
# DEVIATION 158's whole argument has to be re-made on the device and cannot
# be won by libm. These three exist so the substitution is NAMED at each of
# the four call sites below rather than hidden inside them, and so that a
# reader comparing the kernel against `algo_l_sample` above sees the
# float/double split of cuML's expressions preserved: `logf`/`expf` on the
# `float` arguments, `log` on the `double` one. DEVIATION 199.
#
# DEVIATION 456 -- the two FLOAT seams now route through row 12's mode
# gate. Under FAST each is the `std.math` call it always was, verbatim, so
# FAST bits cannot move; under IDENTICAL each is `portable_logf` /
# `portable_expf`, one arithmetic on every backend, which closes the
# "SECOND reason" DEVIATION 199 records for the float32 half of this
# kernel. The DOUBLE seam `_dev_log64` is NOT routed -- the portable pair
# is float32, and carrying `W` or `log(1 - W)` in float32 is the
# cancellation fork 199 rejects by name -- so on a double-capable target
# the algo-L device arm is still not bit-pinned across vendors or against
# the host oracle until a portable DOUBLE log exists. See 199's addendum.


def _dev_logf32(x: Float32) -> Float32:
    """`raft::log(float)`. `_logf32`'s device twin. DEVIATION 456 -- FAST
    is `std.math.log` verbatim, IDENTICAL is `portable_logf`."""
    return identical_log(x)


def _dev_expf32(x: Float32) -> Float32:
    """`raft::exp(float)`. `_expf32`'s device twin. DEVIATION 456 -- FAST
    is `std.math.exp` verbatim, IDENTICAL is `portable_expf`."""
    return identical_exp(x)


def _dev_log64(x: Float64) -> Float64:
    """`raft::log(double)`. `_log64`'s device twin, and the one the Metal
    backend refuses. NOT routed through row 12's gate -- the portable pair
    is float32 only, and DEVIATION 199 rejects a float32 stand-in for this
    seam by name -- so under BOTH modes it is the stdlib's device `log`."""
    return log(x)



def algo_l_sample_kernel[
    BLOCK_THREADS: Int
](
    colids: MutPointer[Int32, MutAnyOrigin],
    report: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    tree_ids: MutPointer[Int32, MutAnyOrigin],
    work_items_size_in: Int32,
    seed: UInt64,
    n_in: Int32,
    k_in: Int32,
    sabotage_in: Int32,
):
    """`algo_L_sample_kernel`, `builder_kernels.cuh:268-316`, ON THE DEVICE.

    **THIS KERNEL CANNOT BE ENQUEUED ON AN APPLE GPU AND HAS THEREFORE NEVER
    RUN. DEVIATION 199.** It carries `W` as a `double` because cuML does, and
    the Metal backend refuses the type outright -- not the operation, the
    TYPE. `sample_features_device` REFUSES this arm by name rather than
    substituting a `float` one, because at the `k/n` the dispatch actually
    routes here `1 - W` is around `1e-4` and float32 loses it to cancellation:
    that would be a different algorithm wearing this one's name, not a
    rounding difference. Read DEVIATION 199 before touching it.

    GRID, AND IT IS NOT THE OTHER KERNEL'S. `builder.cuh:459-466`:
    `grid.x = (work_items.size() + 127) / 128` with `block_threads` threads,
    and `:277`, `int tid = threadIdx.x + blockIdx.x * blockDim.x`. ONE THREAD
    per work item, not one block -- there is nothing block-wide in algorithm
    L, no collective and no shared memory. So `grid_dim =
    ceildiv(work_items_size, BLOCK_THREADS)`, `block_dim = BLOCK_THREADS`,
    and `report` is written by EVERY thread that serves an item rather than by
    thread 0 of a block.

    Note their grid divides by a literal `128` while the block width is
    `block_threads`, which is also 128; this transcription divides by
    `BLOCK_THREADS`, so the two cannot drift if the constant ever moves.

    Everything else is `algo_l_sample` above, whose docstring lists the five
    things a from-memory algorithm L gets wrong. The one thing that IS
    different: `std.math.log` / `std.math.exp` instead of DEVIATION 158's
    libm-through-FFI, because a kernel cannot make an FFI call. DEVIATION 158
    already names this as the thing a device port must re-establish, and it is
    a SECOND reason this arm is not bit-identical to the host oracle even on a
    target that has `double`.
    """
    var tid = Int(block_idx.x) * BLOCK_THREADS + Int(thread_idx.x)
    if tid >= Int(work_items_size_in):  # `:278`
        return
    var n = Int(n_in)
    var k = Int(k_in)

    var node_id = work_items[unsafe_offset=tid].idx.cast[DType.uint32]()  # `:279`
    # DEVIATION 211: per-node tree id.
    var gen = PCGenerator(
        seed,
        algo_l_subsequence(
            tree_ids[unsafe_offset=tid].cast[DType.uint32](), node_id
        ),
        UInt64(0),
    )  # `:280-281`

    # `:287-291`
    var fp_uniform_val = gen.next_float()
    var W = Float64(_dev_expf32(_dev_logf32(fp_uniform_val) / Float32(k)))

    # `:293-301`. The fill leaves `col` at `k - 1`, NOT `k`.
    var col = 0
    while True:
        colids[unsafe_offset = tid * k + col] = Int32(col)
        if col == k - 1:
            break
        else:
            col += 1

    # `:303-315`
    while col < n:
        fp_uniform_val = gen.next_float()
        col += (
            Int(Float64(_dev_logf32(fp_uniform_val)) / _dev_log64(1.0 - W)) + 1
        )
        if col < n:
            var int_uniform_val = Int(uniform_int_u64(gen, 0, UInt64(k)))
            colids[unsafe_offset = tid * k + int_uniform_val] = Int32(col)
            fp_uniform_val = gen.next_float()
            W *= Float64(_dev_expf32(_dev_logf32(fp_uniform_val) / Float32(k)))

    report[unsafe_offset = 3 * tid + 0] = Int32(SAMPLE_ALGO_L)
    report[unsafe_offset = 3 * tid + 1] = Int32(0)
    report[unsafe_offset = 3 * tid + 2] = Int32(0)


# --- the device dispatch, `builder.cuh:398-471` ----------------------------


def sample_features_device[
    oc: MutOrigin, os: MutOrigin, orp: MutOrigin, ow: MutOrigin, ot: MutOrigin, //
](
    ctx: DeviceContext,
    d_colids: MutPointer[Int32, oc],
    d_scratch: MutPointer[Int32, os],
    d_report: MutPointer[Int32, orp],
    d_work_items: MutPointer[NodeWorkItem, ow],
    d_tree_ids: MutPointer[Int32, ot],
    work_items_size: Int,
    seed: UInt64,
    n: Int,
    k: Int,
    sabotage: Int32 = SAMP_SAB_NONE,
) raises -> FeatureSamplerPlan:
    """`sample_features`, but the sampling happens where cuML does it.

    Same contract as the host driver and the same returned plan, so a caller
    can name the arm that ran (rule 8) without reading `d_report` -- though
    `d_report` is the DEVICE's own statement of the same thing and is what a
    check must believe.

    THE DISPATCH ITSELF STAYS ON THE HOST, and that is theirs: see the header
    comment above `excess_sample_kernel`.

    Buffers, all device pointers, none of them seeded by this function:
      * `d_colids`     `work_items_size * k` Int32
      * `d_scratch`    `sampler_scratch_len(work_items_size, n, k)` Int32
      * `d_report`     `sampler_report_len(work_items_size)` Int32, SEEDED
                       with `SAMPLER_UNVISITED`
      * `d_work_items` `work_items_size` `NodeWorkItem`

    Enqueues and does NOT synchronize.
    """
    var plan = plan_feature_sampling(n, k)
    if plan.arm == SAMPLE_ALL_FEATURES:
        comptime TPB = 128
        var total = work_items_size * k
        ctx.enqueue_function[sample_all_features_kernel](
            d_colids,
            d_report,
            Int32(work_items_size),
            Int32(k),
            sabotage,
            grid_dim=ceildiv(total, TPB),
            block_dim=TPB,
        )
        return plan
    if plan.arm == SAMPLE_EXCESS:
        var stride = excess_scratch_stride(plan)
        if plan.max_samples_per_thread == 1:
            ctx.enqueue_function[
                excess_sample_kernel[SAMPLER_BLOCK_THREADS, 1]
            ](
                d_colids,
                d_scratch,
                d_report,
                d_work_items,
                d_tree_ids,
                Int32(work_items_size),
                seed,
                Int32(n),
                Int32(k),
                Int32(plan.n_parallel_samples),
                Int32(stride),
                sabotage,
                grid_dim=(work_items_size, 1, 1),
                block_dim=(SAMPLER_BLOCK_THREADS, 1, 1),
            )
        else:
            ctx.enqueue_function[
                excess_sample_kernel[
                    SAMPLER_BLOCK_THREADS, SAMPLER_MAX_SAMPLES_PER_THREAD
                ]
            ](
                d_colids,
                d_scratch,
                d_report,
                d_work_items,
                d_tree_ids,
                Int32(work_items_size),
                seed,
                Int32(n),
                Int32(k),
                Int32(plan.n_parallel_samples),
                Int32(stride),
                sabotage,
                grid_dim=(work_items_size, 1, 1),
                block_dim=(SAMPLER_BLOCK_THREADS, 1, 1),
            )
        return plan

    # SAMPLE_ALGO_L. DEVIATION 199.
    comptime if not device_has_float64():
        raise Error(
            String(
                "the algo-L reservoir arm has no device path on this target:"
                " cuML's algo_L_sample_kernel carries W, log(1 - W) and the"
                " truncated jump in DOUBLE (builder_kernels.cuh:291, :306),"
                " and this backend refuses the type. Refused by name rather"
                " than silently substituted in float32, which at k/n=",
                Float64(k) / Float64(n),
                " loses 1 - W to cancellation. DEVIATION 199. n=",
                n,
                " k=",
                k,
            )
        )
    else:
        ctx.enqueue_function[algo_l_sample_kernel[SAMPLER_BLOCK_THREADS]](
            d_colids,
            d_report,
            d_work_items,
            d_tree_ids,
            Int32(work_items_size),
            seed,
            Int32(n),
            Int32(k),
            sabotage,
            grid_dim=ceildiv(work_items_size, SAMPLER_BLOCK_THREADS),
            block_dim=SAMPLER_BLOCK_THREADS,
        )
    return plan


# ===========================================================================
# DEVIATION BLOCK -- THE DEVICE SAMPLERS
#
# `extratrees/DEVIATIONS.md` is owned by another session this round, so these
# five live here, in the file they describe, in the same shape the entries
# above use. They are numbered 195-199 and reserved as such.
# ===========================================================================
#
# 195. `cub::BlockRadixSort` has no counterpart, so the sort is a HAND-WRITTEN
#      BITONIC NETWORK.
#
#   Theirs. `BlockRadixSortT(temp_storage.sort).Sort(items)`
#   (`builder_kernels.cuh:224`), a CUB block-wide radix sort over the block's
#   `BLOCK_THREADS * MAX_SAMPLES_PER_THREAD` register array, leaving it in
#   blocked arrangement.
#
#   Ours. A bitonic sorting network written out in this file: `log2(span)`
#   outer stages, an XOR-partner inner loop, one `barrier()` per stage, and
#   each pair `(i, i ^ j)` touched by exactly the one thread that owns the
#   lower index.
#
#   Why this is a hand-write and not an invention. `VENDOR_LIBRARIES.md` and
#   `PORTING_RULES.md` 0b-i say: call the platform's collective, and
#   hand-write only where there is none. `max.gpu.primitives.block` offers
#   `sum`, `min`, `max`, `broadcast` and `prefix_sum` and NOTHING ELSE
#   (checked 2026-08-21, Mojo 1.0.0 ed45d567) -- there is no device sort and
#   no device scan-by-key to build one from. 0b-i's terms are met literally:
#   zero warp intrinsics, no assumed wavefront width, `barrier()` between
#   stages, and no vendor branch.
#
#   Why the substitution cannot change an answer. DEVIATION 157 already
#   established it for the host `sort` and the argument is unchanged: `:224`
#   passes ONE array, so this sorts KEYS ONLY, and the keys are the column ids
#   themselves. Two equal keys are indistinguishable; every ascending sort of
#   the same multiset produces the identical array, so there is no payload
#   whose permutation could differ. `sampler_kernel_check` then does not rely
#   on that argument at all -- it compares the device's `colids` against the
#   host transcription SLOT FOR SLOT.
#
#   Why the network is shorter than the array. It covers `[0,
#   next_pow2(n_parallel_samples))`, not the whole slot count. Slots at or
#   above `n_parallel_samples` hold `NON_DRAWING_SENTINEL(n) == n` from
#   iteration 1 and can never hold anything else -- a slot keeps its value
#   only when its mask is 1, and DEVIATION 165 clears the mask of every item
#   `>= n`. The tail is therefore a constant run of the array's MAXIMUM,
#   already in its sorted place. `excess_sort_span` states this; it is what
#   makes DEVIATION 196 affordable at the sizes the dispatch actually
#   produces.
#
#   Price. `O(log^2 span)` compare-exchange passes where theirs is a radix
#   pass, and `log2(span)^2 / 2` block barriers per do-while iteration. No
#   timing claim is attached and none will be until the perf round; this is a
#   KERNEL MATRIX row.
#
# ---------------------------------------------------------------------------
#
# 196. MEASURED: their sort's storage DOES NOT FIT, so `items` lives in
#      GLOBAL scratch and the caller allocates it.
#
#   Theirs. `items`, `mask` and `col_indices` are per-thread REGISTER arrays
#   of `MAX_SAMPLES_PER_THREAD` (`:181-183`), and CUB's three collectives
#   share one `__shared__ union` (`:216-221`). Their own calibration comment
#   (`builder.cuh:403-414`) sizes it `max_samples_per_thread * block_threads *
#   sizeof(IdxT)` = 72 * 128 * 4 = **36864 bytes**, and budgets 46KB.
#
#   MEASURED, 2026-08-21, Mojo 1.0.0 (ed45d567), M4:
#
#       CAP=1024  bytes=4096   ok
#       CAP=4096  bytes=16384  ok
#       CAP=8192  bytes=32768  ok
#       CAP=9216  bytes=36864  Failed to create compute pipeline state (GPU
#                              machine code generation): Threadgroup memory
#                              size (36864) exceeds the maximum threadgroup
#                              memory allowed (32768)
#
#   So cuML's own working set is 4096 bytes over this box's hard limit, and
#   the power-of-two span a bitonic network needs (16384 slots = 65536 bytes)
#   is twice over again. This is not a tuning question and no arrangement of
#   shared memory solves it.
#
#   Ours. `items` is a GLOBAL buffer, `sampler_scratch_len(work_items_size, n,
#   k)` Int32s, one contiguous region per block. `mask` stays where theirs is
#   -- a PRIVATE `stack_allocation[MAX_SAMPLES_PER_THREAD]` per thread --
#   because it is indexed by the thread's own SLOTS, and the slots do not
#   migrate under the sort even though the ITEMS do (point 2 of
#   `excess_sample_with_replacement`'s docstring). `col_indices` is not stored
#   at all: it is the exclusive prefix sum of `mask`, and the gather
#   recomputes it in one pass.
#
#   Why global and not a capacity ladder. The alternative considered was a
#   comptime `SORT_CAP` parameter with the host picking the smallest
#   instantiation that fits and REFUSING above it. It was rejected because the
#   refusal would be reachable: `n_parallel_samples` runs to 9216 on this arm
#   by the dispatch's own bound, and 9216 has no shared-memory home on Apple
#   at any padding. A design whose refusal fires inside cuML's supported range
#   is a narrower library, not a portable one.
#
#   Why `barrier()` is enough for a global array. MEASURED, same session: 128
#   threads each write their own global slot, `barrier()`, then read the next
#   thread's -- 0 mismatches of 128. The sort itself is the stronger evidence:
#   it is nothing but cross-thread global reads and writes separated by
#   `barrier()`, and its output is asserted bit-identical to the host's over
#   23,462 cells.
#
#   Price. The sort's traffic goes to device memory instead of threadgroup
#   memory. Perf is deferred by the repository owner and NO timing number was
#   taken for this entry. It is a KERNEL MATRIX row, and it is the first place
#   to look when the perf round opens.
#
# ---------------------------------------------------------------------------
#
# 197. The block scan is per-thread totals THEN one collective, because
#      Mojo's has no `ITEMS_PER_THREAD`.
#
#   Theirs. `BlockScanT(temp_storage.scan).ExclusiveSum(mask, col_indices,
#   n_uniques)` (`:237`) -- CUB's array overload, which scans
#   `MAX_SAMPLES_PER_THREAD` items per thread and returns both the per-item
#   exclusive prefixes AND the block aggregate from one call.
#
#   Ours. Each thread sums its own `mask` into a register, one
#   `block.prefix_sum[exclusive=True]` gives the thread's starting offset, one
#   `block.sum` gives the aggregate, and the gather walks the thread's slots
#   adding its own running count. That is what CUB's array overload does
#   internally, spelled out.
#
#   Why two calls and not one. Mojo 1.0 has no primitive returning both the
#   prefix and the aggregate -- the same finding DEVIATION 176 part 1 recorded
#   for `nodeSplitKernel`, and the same resolution. The aggregate must be
#   BLOCK-UNIFORM because it drives the do-while's exit, which `block.sum`'s
#   default `broadcast=True` gives; a non-uniform `n_uniques` would let part
#   of a block leave a loop containing collectives.
#
#   Guarded by `SAMP_SAB_AGG_ONE_PER_THREAD`, which scans one item per thread
#   instead of `MAX_SAMPLES_PER_THREAD` of them -- the exact mis-port this
#   entry exists to prevent. It is STRUCTURALLY INERT at
#   `MAX_SAMPLES_PER_THREAD == 1` (a thread has one item; the two expressions
#   are the same expression) and the check runs it on both instantiations and
#   prints both, so the inertness is on the record rather than hidden.
#   MEASURED: 12000 of 12000 slots differ at (n=2000, k=1500), 0 of 160 at
#   (n=1000, k=20).
#
#   Price. Two block collectives and two barriers per do-while iteration where
#   CUB does one scan.
#
# ---------------------------------------------------------------------------
#
# 198. The DISPATCH stays on the host -- because it is on the host in cuML --
#      and the unbounded loop's bound becomes a STATUS.
#
#   `PORTING_RULES.md` rule 2 asks whether a control-plane decision is theirs
#   to keep on the device. This one is not: `n_parallel_samples` is a
#   `std::ceil` of a ratio of `std::log`s evaluated in `Builder::doSplit`
#   (`builder.cuh:419-421`) and the arm is a C++ `if` around three different
#   `<<<...>>>` launches (`:434-466`). Their host learns this decision because
#   their host MAKES it. `plan_feature_sampling` is therefore not the drift
#   rule 2 is about, and `sample_features_device` calls it unchanged --
#   including DEVIATION 158's libm-through-FFI logs, which is why the arm
#   boundaries land on the same side there as they do in cuML.
#
#   What DID move is the sampling, and with it two things that cannot cross:
#
#   * DEVIATION 159's `EXCESS_MAX_ITERATIONS` bound RAISES on the host. A
#     kernel cannot raise, so the device writes `SAMPLER_OVERRUN` into its
#     report and gathers nothing -- the shape `PARTITION_OVERRUN` established
#     (DEVIATION 176 part 3). Deviation 159's own Price paragraph asked for
#     exactly this decision; this is it. It is unreachable, and it exists
#     because a HANG is the one failure a check cannot report.
#   * `plan_feature_sampling`'s REFUSALS (`k < 1`, `k > n`) are inherited by
#     the device driver rather than restated, so the two drivers cannot come
#     to disagree about what is legal.
#
#   And one thing was ADDED that cuML has no counterpart for: `report`, three
#   Int32 per work item, `[arm, instantiation, iterations]`, written ONLY by
#   the kernels and seeded by the caller with `SAMPLER_UNVISITED`. Rule 8 --
#   a harness that cannot name the kernel it ran can publish about a
#   different one -- and rule 15, reach is per branch. The check asserts three
#   cells PAST the end of it still hold the seed, which is what makes every
#   "the device said so" assertion non-vacuous.
#
#   Price. `sampler_report_len(work_items_size)` Int32s of device memory and
#   one write per block that the algorithm does not need.
#
# ---------------------------------------------------------------------------
#
# 199. MEASURED: Metal has NO `double`, so the algo-L arm is REFUSED BY NAME
#      and `algo_l_sample_kernel` HAS NEVER RUN.
#
#   Theirs. `algo_L_sample_kernel` (`builder_kernels.cuh:268-316`) is a
#   `double` algorithm and not incidentally: `double W` (`:291`), `raft::log(1
#   - W)` in double (`:306`), the jump `static_cast<int>(raft::log(fp) /
#   raft::log(1 - W))` truncating a DOUBLE quotient, and `W *= ...` in double
#   (`:313`). The host transcription `algo_l_sample` mirrors all four and
#   `feature_sampler_check` checks it.
#
#   MEASURED, 2026-08-21, Mojo 1.0.0 (ed45d567), M4 -- a 40-line probe with
#   one `Float64` local in a kernel:
#
#       function's return type 'double' is not supported
#       Function 'air.convert.f.f64.f.f32' has Metal-unsupported instructions
#       Function 'llvm.fma.f64' has Metal-unsupported instructions
#       LLVM ERROR: Failed to verify LLVM IR for Metal
#
#   It is the TYPE, not an operation, and it fails at COMPILE time -- the same
#   shape of backend refusal DEVIATION 167 records for `Int128` and DEVIATION
#   162 for a whole-struct kernel argument.
#
#   Ours. The kernel is written -- once, portably, with cuML's exact
#   float/double split -- and `sample_features_device` REFUSES the arm by name
#   on a target without `double`, through the host predicate
#   `device_has_float64()`. It is a HOST dispatch decision, the only place the
#   ALWAYS-GPU-AGNOSTIC rule allows a vendor fact to live; there is no
#   `if apple` in any kernel.
#
#   WHY A FLOAT32 SUBSTITUTE WAS NOT SHIPPED, which is the tempting move.
#   `W` is a product of ~`n - k` factors each just below 1, so at the `k/n`
#   the dispatch actually routes here it sits around `1 - 1e-4`: at (n=2000,
#   k=1990) `W ~ 1 - |log u| / 1990`. Forming `1 - W` in float32 is
#   catastrophic cancellation -- about 13 bits survive -- so `log(1 - W)`,
#   and with it every jump, would be wrong in the fourth digit rather than the
#   eighth. That is a DIFFERENT ALGORITHM wearing this one's name, on the one
#   arm where nobody would look. A refusal is visible; a silent fork is not.
#   Writing `1 - W` accurately instead (tracking `V = 1 - W` through
#   `expm1f`) was also rejected: it is numerically BETTER than cuML and
#   therefore not this port.
#
#   THE CONSEQUENCE, STATED PLAINLY. `algo_l_sample_kernel` has never been
#   enqueued. By this repository's own rule -- a kernel is not ported until it
#   has been ENQUEUED -- the algo-L arm is NOT PORTED, and the source below is
#   a portable draft for a CUDA or ROCm box, not a verified kernel. The host
#   transcription `algo_l_sample` remains the only checked implementation of
#   it. Anyone who runs this lane on NVIDIA or AMD should expect
#   `sampler_kernel_check`'s algo-L section to switch from asserting the
#   refusal to asserting the sample, and should read DEVIATION 158's Price
#   paragraph first: the device has no libm through FFI, so `std.math.log`'s
#   ~5e-8 absolute error against libm is a SECOND reason that arm will not be
#   bit-identical to the host oracle even where `double` exists.
#
#   ADDENDUM 2026-08-23, DEVIATION 456 -- row 12's portable pair landed and
#   HALF of this entry is now closed. The kernel's two FLOAT seams
#   (`_dev_logf32`, `_dev_expf32`) route through `identical_log` /
#   `identical_exp`, so under IDENTICAL the float32 transcendentals are one
#   arithmetic on every backend and the "SECOND reason" above no longer
#   applies to them; under FAST they compile to the verbatim `std.math`
#   calls this entry has always described. WHAT REMAINS, and the portable
#   pair cannot cure it -- the DOUBLE core. The type refusal on Metal
#   stands untouched (`W`, `log(1 - W)` and the truncated jump are still
#   `double`, still refused by name on a target without it), and on a
#   double-capable target `_dev_log64` is still the vendor's `std.math.log`
#   under both modes, because the portable pair is FLOAT32 and a float32
#   stand-in for `1 - W` is the cancellation fork this entry rejects. Full
#   closure of the double seam needs a portable DOUBLE log, a row 12
#   artifact that does not exist yet.
#
#   Reachability. The algo-L arm needs `n_parallel_samples > 72 * 128`, i.e.
#   `k/n` close to 1 at a large `n` -- (2000, 1990) and (20000, 7385 under
#   FAST, 7386 under IDENTICAL; DEVIATION 457) are the pairs the checks use. It is not the ExtraTrees default (`max_features =
#   sqrt`, which lands far inside the excess arm), but it is reachable from
#   `max_features` near 1.0 and it is NOT dead.
