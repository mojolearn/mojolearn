"""The host control plane: the node queue, the workspace, and the batch shape.

MIRRORS `cpp/src/decisiontree/batched-levelalgo/builder.cuh` at rapidsai/cuml
`v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`), checked out
read-only at `~/CascadeProjects/upstream/cuml-v26.08.00`.

THE CONTROL PLANE IS CODE TOO, and it gets ported like everything else. If
they do something on the GPU, we do it on the GPU; if they keep a decision on
the device so the host never learns it, we keep it there; if they pass a value
as a kernel argument, we pass it as a kernel argument. The host/device split
is part of the algorithm, not an implementation detail this port re-decides.

WHAT IS IN THIS FILE, and what is not
--------------------------------------
Their `builder.cuh` is one class (`NodeQueue`, `:41-145`) plus one struct
(`Builder`, `:147-698`). Everything in it that runs on the HOST is here and
is complete:

  * `NodeQueue` -- the tree, the frontier, the expandability rule, and the
    parent/left/right bookkeeping that turns a batch of chosen splits into
    new nodes. This is where a tree's SHAPE is decided.
  * `maxNodes`, `workspaceSize`, `assignWorkspace` -- the one-shot allocation.
  * `computeSharedMemoryConfig` -- their shared-vs-global histogram dispatch.
  * `updateWorkloadInfo` -- the ragged-batch block map.
  * `sampled_cols_for` and the multi-round feature-resampling schedule.

The four kernel LAUNCHES (`doSplit`, `computeBestSplits`, `computeSplit`,
`SetLeafPredictions`) are NOT wired here yet; see DEVIATION 130's note at the
end of the deviation block. They are the next commit, and the launcher
contract they will call is written down there so the two halves cannot drift.

THE THING MOST LIKELY TO BE MISSED IN THIS FILE, said up front
---------------------------------------------------------------
`doSplit` (`:410-482`) is not one pass over the batch. It is a LOOP over
sampling ROUNDS, added to match sklearn's behaviour of searching beyond
`max_features` when the sampled features yield no valid split -- their own
comment at `:441-442`. Each round:

    sample_offset  = round * original_n_sampled_cols          (`:438`)
    n_sampled_cols = min(original_n_sampled_cols,
                         n_cols - sample_offset)              (`:439-440`)

and the loop runs at most `ceil(n_cols / original_n_sampled_cols)` times
(`:420-421`), shrinking the active set each round to only those nodes that
still have no valid split (`:452-458`). Because `sample_offset` is a pure
INDEX OFFSET into a permutation whose seed does NOT include the round (see
`kernels/builder_kernels.mojo`), every round hands a node features it has
provably not tried, and after the last round it has seen every feature
exactly once.

A port that ran one round would compile, would train, would produce
plausible trees, and would silently make a leaf out of every node whose
first `max_features` columns happened to be uninformative.

================= DEVIATION BLOCK (whole file) =================

DEVIATION 130. `n_streams` and the whole stream/OpenMP fan-out are absent.
That belongs to the estimator above this file and is priced in full under
DEVIATION 117 in `randomforest.mojo`; nothing in `builder.cuh` itself
touches streams except by taking a `cudaStream_t` parameter, which becomes a
`DeviceContext` here. No output bit depends on it.

Recorded here only because `Builder`'s constructor takes `cudaStream_t s`
(`:214`) and a reader diffing the two signatures will notice it missing.

DEVIATION 131. THE FOUR KERNEL LAUNCHES ARE NOT WIRED YET, and this file
does not pretend otherwise. `train()` below RAISES by name rather than
returning an empty tree, because a builder that silently returns a
single-leaf forest is exactly the defect class this repository has been
bitten by: it compiles, it runs, and its output looks like a very
conservative model.

PRICE: `ensemble/` cannot train until the next commit. What IS usable now is
everything that decides tree shape and allocation, and it is checked --
`ensemble/mojo_only/builder_check.mojo` drives `NodeQueue` through a full
synthetic tree build with hand-computed expected node counts, depths and
instance ranges, and drives the workspace and shared-memory dispatch across
their whole parameter range.

THE LAUNCHER CONTRACT the next commit will call, written here so the two
halves cannot drift (mirroring `kernels/builder_kernels.cuh:96-160`):

    launch_node_split_kernel(ctx, dataset, work_items, splits, workload_info,
                             n_blocks_dimx, n_work_items, partition_row_ids)
    launch_leaf_kernel(ctx, objective, dataset, tree, instance_ranges,
                       leaves, batch_size, smem_size)
    launch_build_histograms_kernel(ctx, histograms, max_n_bins, dataset,
                                   quantiles, work_items, col_start,
                                   column_samples, objective, workload_info,
                                   grid_x, grid_y, smem_config)
    launch_find_best_splits_kernel(ctx, histograms, max_n_bins, dataset,
                                   quantiles, col_start, column_samples,
                                   mutex, splits, objective, grid_x, grid_y)

DEVIATION 132. `allReduceHistograms` (`:553-568`), `packedHistogramWorkspaceSize`
(`:269-282`) and the `distributed` flag (`:211`, `:246`) are not ported.
They gate on a RAFT communicator with more than one rank
(`raft::resource::comms_initialized(handle) && get_size() > 1`), which a
single-device library never has. PRICE: multi-GPU RF is unavailable, and the
histogram workspace is smaller by exactly `packedHistogramWorkspaceSize`,
which is zero on their single-rank path too (`:271`: `if (!distributed)
{ return 0; }`). So the workspace this file computes is byte-for-byte the
size THEIRS computes on one device. That equality is checked.

DEVIATION 133. `MLCommon::TimerCPU` and `tree->train_time` (`:377`, `:387`)
are not ported and the field is left at its initial value. Timing is out of
scope this round by instruction, and a field that would report a duration is
better empty than filled with a number nobody measured. PRICE: their tree
summary prints a milliseconds line and ours does not; already recorded from
the other side as DEVIATION 118(c) in `decisiontree.mojo`.

NOT A DEVIATION, and worth stating because the constant looks arbitrary:
`tunable_split_histogram_dynamic_smem_limit_bytes = 16 * 1024` (`:163`) is
transcribed, not chosen. Their comment at `:158-162` explains it -- large
per-block histograms, usually from large `n_classes`, cut occupancy enough
that global memory wins even when shared would fit. What matters for a
PORTABLE build is that 16 KiB is BELOW the per-block shared-memory limit of
every vendor column in `mojo_only/kernel_matrix.column_shared_limit` (Apple
32 KB, NVIDIA 48 KB, AMD 64 KB, and even the 16 KB spec baseline meets it).
So their shared-vs-global dispatch is decided by THEIR tuned constant on
every backend, not by the hardware -- which means this port takes the same
branch they do, on every device, without a device query. Checked.

NOT A DEVIATION: `maxNodes()` (`:283-297`) returns `2^(max_depth+1) - 1`
below depth 13 and a FIXED 8191 at or above it. That is not a bound on the
tree -- it is the initial `reserve()` for a `std::vector` that grows past it
(`:53-54`). Ours reserves the same number for the same reason and likewise
does not treat it as a cap; a port that clamped node count to `maxNodes()`
would silently truncate every tree deeper than 12.
=================================================================
"""

from std.math import ceildiv

from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    SharedMemoryConfig,
    WorkloadInfo,
)
from ensemble.decisiontree.decisiontree import (
    DecisionTreeParams,
    TreeMetaDataNode,
)
from ensemble.flatnode import SparseTreeNode

# `builder.cuh:161` -- "default threads per block for most kernels in here"
comptime TPB_DEFAULT = 128

# `builder.cuh:163`, with their comment transcribed in the deviation block.
comptime TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES = 16 * 1024

# `builder.cuh:203` -- "number of blocks used to parallelize column-wise
# computations". A plain member initialised to 10 and never reassigned.
comptime N_BLKS_FOR_COLS = 10

# `builder.cuh:205` -- "Memory alignment value"
comptime ALIGN_VALUE = 512


@always_inline
def align_to(value: Int, alignment: Int) -> Int:
    """`raft::alignTo`, used by `calculateAlignedBytes` (`builder.cuh:264-267`).
    """
    return ceildiv(value, alignment) * alignment


@always_inline
def calculate_aligned_bytes(actual_size: Int) -> Int:
    """`builder.cuh:264-267`."""
    return align_to(actual_size, ALIGN_VALUE)


def max_nodes(max_depth: Int32) -> Int:
    """`Builder::maxNodes`, `builder.cuh:283-297`.

    Their comments: "Start with allocation for a dense tree for depth < 13"
    and "Start with fixed size allocation for depth >= 13". This is a
    RESERVE, not a cap -- see the deviation block.
    """
    if Int(max_depth) < 13:
        return (1 << (Int(max_depth) + 1)) - 1
    return 8191


def n_sampled_cols_for(max_features: Float32, n_cols: Int) -> Int:
    """`builder.cuh:240`, their sampled-column count.

    Inside their `dataset{...}` initializer:

        max(1, IdxT(params.max_features * n_cols))

    THE CAST IS A TRUNCATION TOWARD ZERO, not a round, and it is a
    knife-edge worth naming. `max_features` arrives as a float32
    (`decisiontree.hpp:33`), so the product is computed in float32; one ULP
    low anywhere turns 4.0 into 3.9999998 and hands the node THREE columns
    instead of four. Their `int` cast is transcribed exactly rather than
    replaced with a round, because rounding would be a different algorithm
    at every ratio that lands near an integer -- which, for `max_features`
    values like `sqrt(n)/n`, is most of them.
    """
    var raw = Int(Float32(n_cols) * max_features)
    return max(1, raw)


def max_sampling_rounds_for(n_cols: Int, original_n_sampled_cols: Int) -> Int:
    """`builder.cuh:420-421`.

        (dataset.n_cols + original_n_sampled_cols - 1) / original_n_sampled_cols

    i.e. `ceil(n_cols / n_sampled_cols)`. See the module docstring for why
    the loop exists at all.
    """
    return ceildiv(n_cols, original_n_sampled_cols)


def sampled_cols_in_round(
    n_cols: Int, original_n_sampled_cols: Int, round: Int
) -> Int:
    """`builder.cuh:438-440`, the per-round column budget.

    The final round is SHORT whenever `n_cols` is not a multiple of the
    sample size, and their `min` is what makes it so. A port that kept the
    full width on the last round would index past `n_cols`.
    """
    var sample_offset = round * original_n_sampled_cols
    return min(original_n_sampled_cols, n_cols - sample_offset)


@fieldwise_init
struct NodeQueue[dtype: DType, sabotage: Int = 0](Copyable, Movable):
    """`ML::DT::NodeQueue<DataT, LabelT>`, `builder.cuh:41-145`.

    `sabotage` is a comptime hook that is 0 in every shipping instantiation
    and is corrupted only by `ensemble/mojo_only/builder_check.mojo`. It is
    IN THE SHIPPED CODE PATH on purpose: sabotaging a copy of an algorithm
    proves nothing about the algorithm, so the corruption has to live where
    the real branch lives. Same pattern as `core/block_scan.mojo`.

    Their `LabelT` is phantom here as it is on `SparseTreeNode` (recorded as
    DEVIATION 116(b) in `flatnode.mojo`), so this carries only `dtype`.

    Their `std::deque<NodeWorkItem> work_items_` is a `List` used as a FIFO
    with a head index. `Pop` takes from the FRONT (`:73-74`) and `Push`
    appends to the BACK (`:117`, `:129`), which is what makes the traversal
    breadth-first and therefore what makes `max_leaves` cut the tree the way
    theirs does. A `List` used as a stack would produce a different tree from
    the same data with no other change.
    """

    var params: DecisionTreeParams
    var tree: TreeMetaDataNode[Self.dtype]
    # `builder.cuh:45` -- one entry per node in `tree.sparsetree`, same index
    var node_instances_: List[InstanceRange]
    # `builder.cuh:46` -- the frontier
    var work_items_: List[NodeWorkItem]
    # NOT in their struct: the head index that makes `List` a deque.
    var head_: Int

    def __init__(
        out self,
        params: DecisionTreeParams,
        max_nodes_hint: Int,
        sampled_rows: Int,
        num_outputs: Int32,
    ) raises:
        """`builder.cuh:50-62`.

        The root starts as a LEAF holding every sampled row, `leaf_counter`
        starts at 1 and `depth_counter` at 0, and the root becomes a work
        item only if `IsExpandable` says so -- so `min_samples_split` and a
        `max_depth` of 0 are honoured before any kernel runs.
        """
        self.params = params
        self.tree = TreeMetaDataNode[Self.dtype](
            treeid=Int32(0),
            depth_counter=Int32(0),
            leaf_counter=Int32(1),
            train_time=Float64(0.0),
            vector_leaf=List[Scalar[Self.dtype]](),
            sparsetree=List[SparseTreeNode[Self.dtype]](),
            num_outputs=num_outputs,
        )
        self.tree.sparsetree.reserve(max_nodes_hint)
        self.tree.sparsetree.append(
            SparseTreeNode[Self.dtype].CreateLeafNode(Int32(sampled_rows))
        )
        self.node_instances_ = List[InstanceRange](capacity=max_nodes_hint)
        self.node_instances_.append(InstanceRange(0, sampled_rows))
        self.work_items_ = List[NodeWorkItem]()
        self.head_ = 0
        if self._is_expandable(self.tree.sparsetree[0], 0):
            self.work_items_.append(
                NodeWorkItem(0, Int32(0), self.node_instances_[0])
            )

    @always_inline
    def has_work(self) -> Bool:
        """`builder.cuh:68`."""
        return len(self.work_items_) - self.head_ > 0

    def pop(mut self) -> List[NodeWorkItem]:
        """`builder.cuh:70-78`.

        Takes up to `max_batch_size` items from the FRONT. Their `while`
        tests `work_items_.size() > 0 && result.size() < max_batch_size`, in
        that order.
        """
        var result = List[NodeWorkItem]()
        var cap = Int(self.params.max_batch_size)
        comptime if Self.sabotage == 3:
            # Take from the BACK: a stack instead of their deque. Predicted
            # movement: a different TREE from the same splits, because the
            # frontier order decides which nodes reach the `max_leaves`
            # budget first.
            while self.head_ < len(self.work_items_) and len(result) < cap:
                result.append(self.work_items_[len(self.work_items_) - 1])
                _ = self.work_items_.pop()
            return result^
        while self.head_ < len(self.work_items_) and len(result) < cap:
            result.append(self.work_items_[self.head_])
            self.head_ += 1
        return result^

    @always_inline
    def _is_expandable(
        self, n: SparseTreeNode[Self.dtype], depth: Int
    ) -> Bool:
        """`builder.cuh:82-88`, their `IsExpandable`, branch for branch.

        Their comment: "This node is allowed to be expanded further (if its
        split gain is high enough)". Note the THIRD test reads
        `tree->leaf_counter`, which is mutated as the batch is pushed, so
        whether a node is expandable depends on how many leaves already
        exist -- and therefore on the ORDER the batch is processed. That is
        theirs.
        """
        if depth >= Int(self.params.max_depth):
            return False
        if Int(n.InstanceCount()) < Int(self.params.min_samples_split):
            return False
        if (
            self.params.max_leaves != Int32(-1)
            and self.tree.leaf_counter >= self.params.max_leaves
        ):
            return False
        return True

    def push(
        mut self,
        work_items: List[NodeWorkItem],
        h_splits: List[SplitSummary[Self.dtype]],
    ) raises:
        """`builder.cuh:91-143`. THE TREE'S SHAPE IS DECIDED HERE.

        Transcribed statement for statement, because the order of the six
        mutations is load-bearing:

        1. `if (!split.IsValid()) continue;` (`:99`) -- the node stays the
           leaf it already is.
        2. `if (max_leaves != -1 && leaf_counter >= max_leaves) break;`
           (`:101`) -- once the leaf budget is reached the rest of the batch
           is abandoned, including nodes with perfectly valid splits.

           AND THEIR `break` IS AN EARLY EXIT, NOT A SEMANTIC. This file
           first claimed the `break` was load-bearing and that a `continue`
           would build a different tree. A sabotage measured NO movement and
           was right: `leaf_counter` is monotonically non-decreasing (it is
           only ever `++`'d, at `:111`), so once
           `leaf_counter >= max_leaves` holds it holds for every remaining
           item, and a `continue` would skip each of them in turn to reach
           the same end state. The two spellings are observationally
           equivalent on every input. Theirs is transcribed anyway -- copy,
           do not improve -- but a reader should know that nothing depends
           on it, and `builder_check` records it rather than pretending the
           `break` is checked.
        3. The parent is OVERWRITTEN in place with a split node whose
           `left_child_id` is `tree->sparsetree.size()` READ BEFORE either
           child is appended (`:105-110`).
        4. `leaf_counter++` (`:111`) -- incremented once per SPLIT, not once
           per leaf created. Splitting one leaf into two is a net +1.
        5. Left child appended, then its instance range, then it is pushed
           as a work item only `if IsExpandable` (`:113-120`).
        6. Right child likewise, with its count computed as
           `parent.InstanceCount() - split.global_nLeft` READ BACK OFF THE
           PARENT NODE (`:123-125`) -- the parent that step 3 just rewrote,
           whose count is the parent RANGE's count.

        And note the two different left counts. The CHILD NODE's instance
        count uses `split.global_nLeft`; the child's instance RANGE uses
        `split.local_nLeft` (`:103`, `:115`, `:117`). On a single device
        these are equal, and their comment at `split.cuh:139-140` says the
        local count is filled just before partitioning. Keeping them
        distinct is what would let a multi-rank build work, and conflating
        them is invisible until it does not.
        """
        for i in range(len(work_items)):
            var split = h_splits[i]
            var item = work_items[i]
            var parent_range = self.node_instances_[item.idx]
            if not split.is_valid:
                continue

            # `:101` -- a BREAK, deliberately, not a continue.
            if (
                self.params.max_leaves != Int32(-1)
                and self.tree.leaf_counter >= self.params.max_leaves
            ):
                comptime if Self.sabotage == 1:
                    # DROP the budget test entirely. Predicted movement: MORE
                    # nodes, because the batch keeps splitting past
                    # `max_leaves`.
                    #
                    # The first version of this sabotage was `continue`
                    # instead of `break`, predicting a different tree. It
                    # measured NO movement, and the measurement was right:
                    # see the note under `push`'s docstring. Their `break`
                    # and a `continue` are observationally equivalent.
                    pass
                else:
                    break

            var local_left_count = Int(split.local_n_left)
            comptime if Self.sabotage == 2:
                # Use the GLOBAL left count for the instance RANGE. On one
                # device these are equal, so predicted movement is NONE here
                # -- which is the finding: this fixture cannot tell the two
                # apart, and only a multi-rank build could.
                local_left_count = Int(split.global_n_left)

            # parent, `:105-110`
            var left_child_id = Int64(len(self.tree.sparsetree))
            self.tree.sparsetree[item.idx] = SparseTreeNode[
                Self.dtype
            ].CreateSplitNode(
                split.colid,
                split.quesval,
                split.best_metric_val,
                left_child_id,
                Int32(parent_range.count),
            )
            self.tree.leaf_counter += 1

            # left, `:113-120`
            self.tree.sparsetree.append(
                SparseTreeNode[Self.dtype].CreateLeafNode(
                    Int32(Int(split.global_n_left))
                )
            )
            self.node_instances_.append(
                InstanceRange(parent_range.begin, local_left_count)
            )
            if self._is_expandable(
                self.tree.sparsetree[len(self.tree.sparsetree) - 1],
                Int(item.depth) + 1,
            ):
                self.work_items_.append(
                    NodeWorkItem(
                        len(self.tree.sparsetree) - 1,
                        item.depth + Int32(1),
                        self.node_instances_[len(self.node_instances_) - 1],
                    )
                )

            # right, `:122-131`. The count is read back off the parent node
            # this iteration just rewrote.
            var parent_count = Int(
                self.tree.sparsetree[item.idx].InstanceCount()
            )
            self.tree.sparsetree.append(
                SparseTreeNode[Self.dtype].CreateLeafNode(
                    Int32(parent_count - Int(split.global_n_left))
                )
            )
            self.node_instances_.append(
                InstanceRange(
                    parent_range.begin + local_left_count,
                    parent_range.count - local_left_count,
                )
            )
            if self._is_expandable(
                self.tree.sparsetree[len(self.tree.sparsetree) - 1],
                Int(item.depth) + 1,
            ):
                self.work_items_.append(
                    NodeWorkItem(
                        len(self.tree.sparsetree) - 1,
                        item.depth + Int32(1),
                        self.node_instances_[len(self.node_instances_) - 1],
                    )
                )

            # `:133-134`
            var d = item.depth + Int32(1)
            if d > self.tree.depth_counter:
                self.tree.depth_counter = d


@fieldwise_init
struct SplitSummary[dtype: DType](TrivialRegisterPassable):
    """The five fields of `Split<DataT, IdxT>` that `NodeQueue::Push` reads.

    NOT A DEVIATION AND NOT A NEW TYPE IN THEIR SENSE: their `Push` takes a
    `SplitT*` and touches exactly `IsValid()`, `colid`, `quesval`,
    `best_metric_val`, `global_nLeft` and `local_nLeft` (`builder.cuh:95-125`).
    This is that read surface, so the host queue does not have to depend on
    the device-side `Split`'s shared-memory and atomic machinery. `Split` in
    `split.mojo` remains the ported type; this is the projection of it that
    crosses back to the host, and `doSplit` fills it from the `h_splits`
    copy their `raft::update_host` produces (`:479`).
    """

    var is_valid: Bool
    var colid: Int32
    var quesval: Scalar[Self.dtype]
    var best_metric_val: Scalar[Self.dtype]
    var global_n_left: Int64
    var local_n_left: Int64


def update_workload_info[
    sabotage: Int = 0
](
    work_items: List[NodeWorkItem],
    mut h_workload_info: List[WorkloadInfo],
) -> Int:
    """`Builder::updateWorkloadInfo`, `builder.cuh:393-407`.

    THE RAGGED-BATCH BLOCK MAP, and the reason one launch can cover a batch
    of nodes with wildly different instance counts. Each node gets

        n_blocks_per_node = max(ceildiv(count, TPB_DEFAULT), 1)

    consecutive entries, each carrying `{batch index i, b, n_blocks_per_node}`,
    and the running total is the grid's x extent. Note the `max(..., 1)`:
    a node with ZERO instances still gets one block, so `n_blocks_dimx` is
    never short of the batch size.

    `nodeid` here is the BATCH index `i` (their `int(i)`), NOT the tree index.
    The tree index is `NodeWorkItem.idx` and is what the feature sampler
    seeds on. Conflating the two is a live hazard and is called out in
    `kernels/builder_kernels.mojo` from the other side.

    Returns their `n_blocks_dimx`. The device copy their `raft::update_device`
    at `:405` performs is the caller's, so this stays a pure host function
    and can be checked without a GPU.
    """
    h_workload_info.clear()
    var n_blocks_dimx = 0
    for i in range(len(work_items)):
        var item = work_items[i]
        var n_blocks_per_node = max(
            ceildiv(item.instances.count, TPB_DEFAULT), 1
        )
        comptime if sabotage == 4:
            # Drop their `max(..., 1)`. Predicted movement: an EMPTY node
            # gets zero blocks, so the grid is short and that node is never
            # visited by any kernel.
            n_blocks_per_node = ceildiv(item.instances.count, TPB_DEFAULT)
        for b in range(n_blocks_per_node):
            h_workload_info.append(
                WorkloadInfo(Int32(i), Int32(b), Int32(n_blocks_per_node))
            )
        n_blocks_dimx += n_blocks_per_node
    return n_blocks_dimx


def max_blocks_dimx_for(max_batch_size: Int32, n_sampled_rows: Int) -> Int:
    """`builder.cuh:250`, the upper bound on the histogram grid's x extent.

    Their constructor's

        max_blocks_dimx = 1 + params.max_batch_size + n_sampled_rows / TPB_DEFAULT

    An UPPER BOUND on what `updateWorkloadInfo` can return, and the
    allocation size of both `workload_info` arrays. The `1 +` and the
    integer division are theirs; the bound holds because the batch's
    instance counts sum to at most `n_sampled_rows`, and each node
    contributes at most `ceil(count / TPB) <= count / TPB + 1` blocks.
    """
    return 1 + Int(max_batch_size) + n_sampled_rows // TPB_DEFAULT


@fieldwise_init
struct WorkspaceLayout(Copyable, Movable):
    """Byte offsets of everything `assignWorkspace` carves out of the two
    buffers (`builder.cuh:334-372`), in their order.

    Their code walks a `char*` and reinterpret-casts; ours computes the same
    offsets and hands them out, because a Mojo buffer is typed. The ORDER and
    the per-item alignment are transcribed, so the total is byte-for-byte
    theirs -- which is what `builder_check` compares.
    """

    var n_nodes: Int
    var histograms: Int
    var mutex: Int
    var splits: Int
    var d_work_items: Int
    var workload_info: Int
    var column_samples: Int
    var partition_row_ids: Int
    var device_total: Int
    var h_workload_info: Int
    var h_splits: Int
    var host_total: Int


def workspace_layout(
    max_batch_size: Int32,
    max_n_bins: Int32,
    num_outputs: Int32,
    n_sampled_rows: Int,
    n_sampled_cols: Int,
    size_of_bin: Int,
    size_of_split: Int,
    size_of_work_item: Int,
    size_of_workload_info: Int,
    size_of_idx: Int,
) -> WorkspaceLayout:
    """`Builder::workspaceSize` + `assignWorkspace`, `builder.cuh:299-372`.

    Their two functions compute the same running sum twice -- once to size
    the buffer and once to place the pointers in it -- and the second must
    agree with the first or the last allocation runs off the end. Ours
    computes it ONCE and returns the offsets, which is the same arithmetic
    with the duplication removed; the total is unchanged and is checked
    against a transcription of their sizing expression.

        max_len_histograms = max_batch * max_n_bins
                             * n_blks_for_cols * num_outputs   (`:304-305`)

    Note `n_blks_for_cols` (10) in that product: the histogram buffer holds
    up to ten COLUMNS' histograms at once, which is what lets
    `computeSplit`'s `gridDim.y` cover ten columns per launch.

    `packedHistogramWorkspaceSize` is omitted; it returns 0 on a
    single-device build (`:271`), so the total is theirs exactly. See
    DEVIATION 132.
    """
    var max_batch = Int(max_batch_size)
    var max_len_histograms = (
        max_batch * Int(max_n_bins) * N_BLKS_FOR_COLS * Int(num_outputs)
    )
    var max_blocks = max_blocks_dimx_for(max_batch_size, n_sampled_rows)

    var d = 0
    var off_n_nodes = d
    d += calculate_aligned_bytes(size_of_idx)
    var off_histograms = d
    d += calculate_aligned_bytes(size_of_bin * max_len_histograms)
    var off_mutex = d
    d += calculate_aligned_bytes(4 * max_batch)  # sizeof(int)
    var off_splits = d
    d += calculate_aligned_bytes(size_of_split * max_batch)
    var off_work_items = d
    d += calculate_aligned_bytes(size_of_work_item * max_batch)
    var off_workload = d
    d += calculate_aligned_bytes(size_of_workload_info * max_blocks)
    var off_column_samples = d
    d += calculate_aligned_bytes(size_of_idx * max_batch * n_sampled_cols)
    var off_partition = d
    d += calculate_aligned_bytes(size_of_idx * n_sampled_rows)

    var h = 0
    var off_h_workload = h
    h += calculate_aligned_bytes(size_of_workload_info * max_blocks)
    var off_h_splits = h
    h += calculate_aligned_bytes(size_of_split * max_batch)

    return WorkspaceLayout(
        off_n_nodes,
        off_histograms,
        off_mutex,
        off_splits,
        off_work_items,
        off_workload,
        off_column_samples,
        off_partition,
        d,
        off_h_workload,
        off_h_splits,
        h,
    )


def compute_shared_memory_config(
    max_n_bins: Int32,
    num_outputs: Int32,
    size_of_bin: Int,
    size_of_data: Int,
    size_of_split: Int,
    available_smem: Int,
    cdf_scan_smem_size: Int,
    warp_size: Int,
) raises -> SharedMemoryConfig:
    """`Builder::computeSharedMemoryConfig`, `builder.cuh:522-551`.

    Transcribed expression for expression:

        shared_histogram_size    = max_n_bins * num_outputs * sizeof(BinT)
        shared_quantiles_size    = max_n_bins * sizeof(DataT)
        histogram_dynamic        = the two, plus sizeof(BinT) + sizeof(DataT)
                                   of ALIGNMENT PADDING (`:531`)
        cdf_scan_smem_size       = sizeof(BlockScan<BinT, TPB>::TempStorage)
        split_scratch_smem_size  = ceildiv(TPB, WarpSize) * sizeof(SplitT)
        split_static             = cdf_scan + split_scratch
        use_global_memory_histogram =
              histogram_dynamic > available_smem
           || split_static      > available_smem
           || histogram_dynamic > 16 KiB

    THE ALIGNMENT PADDING IS KEPT even though this port does not need
    `alignPointer` (DEVIATION 120): dropping it would shrink
    `histogram_dynamic_smem_size` and could silently move a configuration
    from their global arm to our shared arm. That would be an "improvement"
    that changes which of THEIR two kernels runs.

    `available_smem` is `handle.get_device_properties().sharedMemPerBlock`
    in their code. It is a PARAMETER here rather than a query because
    `ctx.get_attribute` costs about 1.26 ms per call on Metal and their own
    `TArchProps` caches it for the same reason -- so the caller queries once
    per fit and threads it through. `warp_size` is threaded for the same
    reason and because it must not be assumed.

    Their `ASSERT` at `:540-541` is kept as a raise: if the split
    bookkeeping alone does not fit, there is no configuration that works and
    failing loudly beats launching something that cannot run.
    """
    var shared_histogram_size = (
        Int(max_n_bins) * Int(num_outputs) * size_of_bin
    )
    var shared_quantiles_size = Int(max_n_bins) * size_of_data
    var histogram_dynamic_smem_size = (
        shared_histogram_size + shared_quantiles_size
    )
    # `:531` -- their alignment allowance, kept. See above.
    var histogram_alignment_smem_size = size_of_bin + size_of_data
    histogram_dynamic_smem_size += histogram_alignment_smem_size

    var split_scratch_smem_size = ceildiv(TPB_DEFAULT, warp_size) * (
        size_of_split
    )
    var split_static_smem_size = cdf_scan_smem_size + split_scratch_smem_size

    if available_smem < split_static_smem_size:
        raise Error(
            "Not enough shared memory for RF split bookkeeping. available="
            + String(available_smem)
            + " required="
            + String(split_static_smem_size)
        )

    var use_global_memory_histogram = (
        histogram_dynamic_smem_size > available_smem
        or split_static_smem_size > available_smem
        or histogram_dynamic_smem_size
        > TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES
    )

    return SharedMemoryConfig(
        use_global_memory_histogram,
        0 if use_global_memory_histogram else histogram_dynamic_smem_size,
    )
