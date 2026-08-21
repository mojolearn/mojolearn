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
  * `sample_features` (`:66-94`) -- the per-node feature sampler. NOT
    ported yet; see the deviation block, because it is the single riskiest
    transcription in this directory and it is being read separately.
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

DEVIATION 121, AND IT IS AN OPEN ITEM, NOT A RESOLVED ONE.
`sample_features` (`:66-94`) is NOT PORTED IN THIS FILE and this file must
not be read as though it were. Their body is:

    uint32_t rng_seed = fnv1a32_hash(seed, treeid, nodeid);
    cuda::shuffle_iterator<IdxT> shuffled_features(
        n, cuda::std::minstd_rand(rng_seed), sample_offset);
    column_samples[sample_idx] = shuffled_features[column_index];

`fnv1a32_hash` is ported and checked (`random_utils.mojo`). The other two
are CCCL: `cuda::std::minstd_rand` and `cuda::shuffle_iterator`, the latter
being a random-access permutation whose exact construction -- rounds, round
function, the bit split for a non-power-of-two `n`, the cycle-walking rule,
and how many draws it takes from the generator -- decides which features
every node in every tree ever sees. A one-index error there changes every
tree, and NO downstream check can attribute the difference.

CCCL is open source, so this is a PORT target and not a substitution
target, and it is being read at the pin in a separate pass rather than
guessed at here. Until that lands, this file exposes no feature sampler at
all, deliberately: a plausible-looking sampler that is not bit-exact is
strictly worse than an absent one, because it compiles, it passes every
check that does not know the right answer, and it makes the forest wrong in
a way that looks like variance.

Price of the delay: the builder cannot select columns, so `computeSplit`
cannot run, so nothing in this directory trains yet. That is the correct
order.

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


@fieldwise_init
struct InstanceRange(Copyable, Movable):
    """`builder_kernels.cuh:31-34`. A range in the device array
    `dataset.row_ids`. Their fields are `std::size_t`."""

    var begin: Int
    var count: Int


@fieldwise_init
struct NodeWorkItem(Copyable, Movable):
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
struct WorkloadInfo(Copyable, Movable):
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
struct SharedMemoryConfig(Copyable, Movable):
    """`builder_kernels.cuh:54-57`. Computed once per batch by
    `Builder::computeSharedMemoryConfig` (`builder.cuh:522-551`) and
    passed down, rather than recomputed per launch."""

    var use_global_memory_histogram: Bool
    var histogram_dynamic_smem_size: Int


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
