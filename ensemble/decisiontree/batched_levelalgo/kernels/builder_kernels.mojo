"""The work-item and workload structs the builder hands every kernel, plus
their bin search.

MIRRORS `cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh`
at rapidsai/cuml `v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`),
checked out read-only at `~/CascadeProjects/upstream/cuml-v26.08.00`.

Their file is a header of declarations plus four small definitions. The
declarations (`:96-104`, `:106-114`, `:135-147`, `:149-160`) are the four
kernel launchers, whose bodies live in `builder_kernels_impl.cuh` and are
ported in `builder_kernels_impl.mojo`. What is HERE is everything that has
a body in their header and is not a launcher:

  * `InstanceRange`, `NodeWorkItem`, `WorkloadInfo`, `SharedMemoryConfig`
    -- the four plain structs the builder fills on the host and passes down
    (`:31-57`).
  * `lower_bound` (`:118-133`) -- the bin search every histogram thread runs
    once per instance.
  * `alignPointer` (`:60-64`) -- NOT ported; see the deviation block.
  * `sample_features` (`:66-94`) -- the per-node feature sampler. PORTED,
    and held cell for cell to CCCL's own compiled output; see DEVIATION 121.
  * `packHistograms` / `unpackHistograms` (`:162-194`) -- NOT ported; the
    multi-GPU all-reduce path.

WHAT `WorkloadInfo` IS FOR, because the name does not say it. The builder
splits a BATCH of nodes across `gridDim.x`, giving each node as many blocks
as its instance count needs (`builder.cuh:393-407`:
`n_blocks_per_node = max(ceildiv(count, TPB), 1)`). Block `i` of the grid
then looks up `workload_info[blockIdx.x]` to learn which node it serves,
which slice of that node's rows it owns, and how many blocks share the
node. That indirection is why one launch can cover a ragged batch of nodes
of wildly different sizes -- and it is also why the node-partition scan can
be segmented by `workload_info[slot / TPB].nodeid`
(`builder_kernels_impl.cuh:190-192`) rather than by a separate key array.

================= DEVIATION BLOCK (whole file) =================

DEVIATION 120. `alignPointer` (`:60-64`) is not ported. Theirs exists
because their histogram kernel carves several typed arrays out of ONE
`extern __shared__ char[]` block (`builder_kernels_impl.cuh:221`, `:298`)
and must hand-align each cast. Mojo's `stack_allocation` is typed and
returns an aligned pointer per allocation, so the carve-and-align step has
nothing to do. Price of declining it: none in behaviour; the shared-memory
SIZE their `computeSharedMemoryConfig` computes still includes their
alignment padding (`builder.cuh:531`: `sizeof(BinT) + sizeof(DataT)`), and
that padding is transcribed rather than dropped, so the two implementations
still agree on whether a given configuration fits in shared memory. Dropping
the padding would have been an "improvement" that silently changes which
path their dispatch takes.

DEVIATION 121 (OPENED AND NOW CLOSED). `sample_features` (`:66-94`) IS
ported, below. Their body is:

    uint32_t rng_seed = fnv1a32_hash(seed, treeid, nodeid);
    cuda::shuffle_iterator<IdxT> shuffled_features(
        n, cuda::std::minstd_rand(rng_seed), sample_offset);
    column_samples[sample_idx] = shuffled_features[column_index];

`fnv1a32_hash` is `random_utils.mojo`. The other two are CCCL --
`cuda::std::minstd_rand` and `cuda::shuffle_iterator` -- and are ported in
`ensemble/mojo_only/shuffle_iterator.mojo` against CCCL 3.4.3
(`9d65c77f`), the version rapids-cmake v26.08.00 resolves for cuML
v26.08.00. They live in `mojo_only/` rather than in a mirrored path
because CCCL is a general library this tree does not mirror file for file,
the same way `cluster/mojo_only/` holds ported RAFT primitives.

THIS FILE WAS HELD OPEN ON PURPOSE UNTIL THAT WAS EXACT, because the
failure mode is uniquely bad: a plausible-looking sampler that is not
bit-exact compiles, passes every check that does not already know the
answer, and makes the forest wrong in a way that looks like variance. A
one-index error changes every tree and nothing downstream can attribute
the difference.

WHAT CLOSED IT: `ensemble/tools/shuffle_oracle/` compiles CCCL's own
headers -- host-only, no CUDA toolkit, no GPU, because
`cuda::shuffle_iterator` is `_CCCL_HOST_DEVICE` -- and dumps their output
into `ensemble/bench/shuffle_oracle.txt`. `shuffle_check.mojo` compares the
port against it at four separable layers and reports **1085 of 1085 cells
matching**: the raw LCG stream at 8 seeds, the 24 Feistel keys at 4 seeds,
847 permutation indices at 30 adversarial `n` (exact powers of two, either
side of a power of two, everything under the Feistel's `max(8, bit_width)`
floor of 256, and one crossing 2^8), and 78 columns at 12 end-to-end cuML
call sites with their seed chain and round loop included.

Same discipline as `tools/permutation_oracle/`, which compiles CatBoost's
Mersenne twister for the same reason. The sabotage rule relaxes here for
the stated reason: the expected values are the INCUMBENT'S own per-cell
output from a different compiler and language, so the match IS the reach
proof -- 1085 of somebody else's integers do not agree by accident.

DEVIATION 122. `packHistograms` / `unpackHistograms` (`:162-194`) and
`reduction_buffer_size_v` (`:163-164`) are not ported. They exist only to
pack a `BinT` into a homogeneous `double` buffer for
`comm.allreduce` on the multi-GPU path (`builder.cuh:553-568`), which
`Builder::distributed` gates on a RAFT communicator with more than one rank
(`builder.cuh:246`). This is a single-device library, so that branch is
unreachable by construction and the plan above already scoped it out.
Price of declining it: multi-GPU RF is not available. Two further reasons
it could not be transcribed as written even if it were wanted: the buffer
is `double`, and this device has no float64; and their own comment at
`:169-171` justifies the packing by exact integer representation up to
2^53, an argument that does not survive the narrowing.

NOT A DEVIATION, recorded because the number looks arbitrary and is not
ours: `lower_bound`'s clamping. Their comment at `:115-117` states it
directly -- "Values outside the quantile range are clamped to the edge
bins: values below the first quantile return 0, and values above the last
quantile return len - 1." The loop searches `[0, len-1]` rather than
`[0, len]`, which is what produces the upper clamp, and it is transcribed
exactly. This differs from `std::lower_bound`, and a reader who assumes
standard semantics will get the last bin wrong.
=================================================================
"""

# All four structs below are `ImplicitlyCopyable`, not merely `Copyable`.
# Theirs are C++ aggregates of scalars -- trivially copyable PODs passed and
# returned by value throughout `builder.cuh` (`:70-78`, `:91-143`, `:393-407`)
# -- so requiring an explicit `.copy()` at every read would be this port
# imposing a ceremony their code does not have, on types where a copy is a
# register move.

from std.gpu import block_dim, block_idx, grid_dim, thread_idx

from ensemble.decisiontree.batched_levelalgo.random_utils import (
    fnv1a32_hash_seed_tree_node,
)
from ensemble.mojo_only.shuffle_iterator import shuffled_feature


@fieldwise_init
struct InstanceRange(ImplicitlyCopyable, Movable):
    """`builder_kernels.cuh:31-34`. A range in the device array
    `dataset.row_ids`. Their fields are `std::size_t`."""

    var begin: Int
    var count: Int


@fieldwise_init
struct NodeWorkItem(ImplicitlyCopyable, Movable):
    """`builder_kernels.cuh:36-40`.

    `idx` is the node's index IN THE TREE (`tree->sparsetree`), not its
    index in the batch -- `builder.cuh:117` and `:129` set it to
    `tree->sparsetree.size() - 1`. The batch index is separate and is what
    `WorkloadInfo.nodeid` carries. Conflating the two is a real hazard:
    their feature sampler seeds on `work_items[node_idx].idx`, the TREE
    index (`builder_kernels.cuh:87`), so that a node's column sample
    depends on where it sits in the tree and not on how the batch happened
    to be cut.
    """

    var idx: Int
    var depth: Int32
    var instances: InstanceRange


@fieldwise_init
struct WorkloadInfo(ImplicitlyCopyable, Movable):
    """`builder_kernels.cuh:46-52`, with IdxT = Int32.

    Their comment: "This struct has information about workload of a single
    threadblock of computeSplit kernels of classification and regression."
    """

    # Node in the batch on which the threadblock needs to work. This is the
    # BATCH index (`builder.cuh:404` writes `int(i)`, the loop counter over
    # `work_items`), not the tree index.
    var nodeid: Int32
    # Offset threadblock id among all the blocks working on this node.
    var offset_blockid: Int32
    # Total number of blocks working on this node.
    var num_blocks: Int32


@fieldwise_init
struct SharedMemoryConfig(ImplicitlyCopyable, Movable):
    """`builder_kernels.cuh:54-57`. Computed once per batch by
    `Builder::computeSharedMemoryConfig` (`builder.cuh:522-551`) and
    passed down, rather than recomputed per launch."""

    var use_global_memory_histogram: Bool
    var histogram_dynamic_smem_size: Int


def sample_features_kernel(
    column_samples: MutPointer[Int32, MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    n_column_samples: Int32,
    treeid: Int32,
    seed_lo: Int32,
    seed_hi: Int32,
    sample_offset: Int32,
    n: Int32,
    k: Int32,
):
    """`sample_features`, `builder_kernels.cuh:66-94`.

    Theirs is a `thrust::for_each` over a counting iterator of
    `work_items_size * k` indices (`:77-81`), which is a flat grid-stride
    map; this is that map with their lambda body inlined. Nothing is being
    stood in for -- `thrust::for_each` over a counting iterator materializes
    no sequence, and neither does this.

    Their body, transcribed line for line:

        node_idx     = sample_idx / k
        column_index = sample_idx % k
        nodeid       = work_items[node_idx].idx        // the TREE index
        rng_seed     = fnv1a32_hash(seed, treeid, nodeid)
        column_samples[sample_idx] =
            shuffle_iterator(n, minstd_rand(rng_seed), sample_offset)[column_index]

    NOTE WHICH INDEX SEEDS THE RNG. `work_items[node_idx].idx` is the node's
    index in the TREE (`builder.cuh:117`, `:129`), not its index in the
    batch. A node's column sample therefore depends on where it sits in the
    tree and not on how the batch happened to be cut -- which is what makes
    the sample independent of `max_batch_size`, and part of why their
    `n_streams` fan-out changes no output.

    AND NOTE WHAT `sample_offset` DOES, because it is the whole multi-round
    design. It is a pure INDEX OFFSET into the permutation
    (`shuffle_iterator.h:135-141`); it does not perturb the seed. Since
    `rng_seed` does not include the round, every round rebuilds the SAME
    permutation for a node and slices the next disjoint block out of it. So
    a node that found no valid split in round 0 sees, in round 1, features
    it provably has not tried -- and across all
    `ceil(n_cols / n_sampled_cols)` rounds it visits every feature exactly
    once. Their comment at `builder.cuh:456-457` says this deliberately
    matches sklearn's "search beyond max_features" behaviour.

    THE SEED IS SPLIT INTO TWO Int32 HALVES because a Metal kernel argument
    must be Int32 (this repository's standing rule). `fnv1a32_hash` folds a
    uint64 as low word then high word (`random_utils.cuh:37-39`), so the
    halves are recombined here in that order before hashing, and the check
    holds the recombination to their compiled value.

    THE SPLIT COST TWO BUGS, both silent, and the second is a Mojo trap
    worth more than this function.

    FIRST: `Int32(3735928559)` -- packing 0xDEADBEEF into the high half --
    does not raise in Mojo 1.0. It produces a wrong value, and the kernel
    read the high word back as 0xFFFFFFFF. `.cast[DType.int32]()` /
    `.cast[DType.uint32]()` reinterpret the bits and are what both ends
    must use.

    SECOND, AND IT IS THE GENERAL ONE, WORTH MORE THAN THIS FUNCTION:
    **widening an Int32 to 64 bits through an unsigned intermediate
    SIGN-EXTENDS, and binding the intermediate to a `var` does not reliably
    stop it.** MEASURED at Mojo 1.0, on device, with `seed_hi` holding the
    bits 0xDEADBEEF:

        var hi32 = seed_hi.cast[DType.uint32]()   # reads 3735928559
        var hi64 = hi32.cast[DType.uint64]()      # 0xFFFFFFFFDEADBEEF

    The `var` looks like it forces the truncation and sometimes does; in
    the kernel below it did not, and the sign bits survived into the high
    half. The failure then HID, because `hi64 << 32` shifts the bad bits
    straight out and is correct on its own -- only the `| lo64` exposed it,
    turning 0xDEADBEEFCAFEBABE into 0xFFFFFFFFCAFEBABE. A probe that
    checked the shift alone would have reported the cast working.

    THE FIX THAT HOLDS IS AN EXPLICIT MASK, `& 0xFFFFFFFF`, because a mask
    is arithmetic the folder cannot discard. Do not replace it with a
    `var`, and do not replace it with a narrower cast.

    The inline forms fail the same way and are worth recording too:

        i32.cast[DType.uint32]().cast[DType.uint64]()  -> 0xFFFFFFFFDEADBEEF
        UInt64(Int(i32.cast[DType.uint32]()))          -> 0xFFFFFFFFDEADBEEF

    Both were caught only because this check compares against CCCL's own
    compiled output. A check that compared this port against itself would
    have agreed with the bug on both sides -- which is the argument for
    incumbent oracles in one sentence. Same family as this repository's
    recorded `&+`-is-bitwise-AND and `String(Float32)` traps: assume a Mojo
    numeric conversion is a VALUE conversion, and assume a chained one is
    not a conversion at all, until measured.

    Redundant work, transcribed rather than hoisted: every one of a node's
    `k` threads rebuilds the whole bijection -- 48 LCG draws and a 24-round
    cycle walk -- because their lambda constructs a fresh `shuffle_iterator`
    per `sample_idx` (`:90-92`). Hoisting it per node would be a deviation
    with a measurement attached, and no measurement is being taken this
    round.
    """
    # Recombine BY BIT PATTERN, never by value, and note the `var`s: they
    # are load-bearing, not style. See the deviation block.
    var hi64 = seed_hi.cast[DType.uint64]() & 0xFFFFFFFF
    var lo64 = seed_lo.cast[DType.uint64]() & 0xFFFFFFFF
    var seed = (hi64 << 32) | lo64
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while idx < Int(n_column_samples):
        var node_idx = idx // Int(k)
        var column_index = idx % Int(k)
        var nodeid = UInt32(work_items[unsafe_offset=node_idx].idx)
        var rng_seed = fnv1a32_hash_seed_tree_node(seed, treeid, nodeid)
        column_samples[unsafe_offset=idx] = Int32(
            shuffled_feature(
                Int(n), rng_seed, Int(sample_offset), column_index
            )
        )
        idx += stride


@always_inline
def lower_bound[
    dtype: DType, ao: MutOrigin, //
](
    array: MutPointer[Scalar[dtype], ao],
    len: Int32,
    element: Scalar[dtype],
) -> Int32:
    """`builder_kernels.cuh:118-133`.

    "Returns the lowest index in `array` whose value is greater or equal to
    `element`. Values outside the quantile range are clamped to the edge
    bins: values below the first quantile return 0, and values above the
    last quantile return len - 1." (their `:115-117`)

    Note the search interval: `end` starts at `len - 1`, NOT `len`. That
    one character is the upper clamp, and it is why this is not
    `std::lower_bound`.
    """
    var start = Int32(0)
    var end = len - Int32(1)
    while start < end:
        var mid = (start + end) // Int32(2)
        if array[unsafe_offset = Int(mid)] < element:
            start = mid + Int32(1)
        else:
            end = mid
    return start
