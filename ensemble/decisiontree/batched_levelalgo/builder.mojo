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
`SetLeafPredictions`) are NOT wired here yet; see DEVIATION 300's note at the
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

DEVIATION 300. `n_streams` and the whole stream/OpenMP fan-out are absent.
That belongs to the estimator above this file and is priced in full under
DEVIATION 117 in `randomforest.mojo`; nothing in `builder.cuh` itself
touches streams except by taking a `cudaStream_t` parameter, which becomes a
`DeviceContext` here. No output bit depends on it.

Recorded here only because `Builder`'s constructor takes `cudaStream_t s`
(`:214`) and a reader diffing the two signatures will notice it missing.

DEVIATION 301 (CLOSED). The four kernel launches ARE wired: `train` ->
`do_split` -> `_compute_best_splits` -> `_compute_split`, plus
`set_leaf_predictions`, calling the launchers in
`kernels/builder_kernels_impl.mojo`. `ensemble/mojo_only/train_check.mojo`
grows real trees on the device end to end -- quantiles, feature sampling,
histogram, cdf, gain, split reduction, partition, leaf.

WHAT REMAINS OPEN, and it is not the launches: this `Builder` is
CLASSIFICATION-ONLY. Theirs is `Builder<ObjectiveT>`, generic over the
objective. Ours cannot be yet, because the launchers are OVERLOADED on the
concrete objective type (their DEVIATION 129a) -- `objectives.mojo` declares
no trait and Mojo traits are nominal, so a generic `Builder` has nothing to
dispatch on. The one-line upstream fix, declaring the two objective structs
conformant to an objective trait moved somewhere `objectives.mojo` can
import without a cycle, deletes the two adapters, the six launcher overloads
AND this restriction together. PRICE until then: regression forests do not
train, and say so by not existing rather than by training wrongly.

Classification first was the plan for an independent reason anyway --
`ClassificationBin` is an integer counter under an integer atomic, so its
histogram is order-independent by construction, and `train_check` arm E
holds two fits of a 3-class 4-feature dataset to BIT-IDENTICAL trees and
leaf values.

DEVIATION 302. `allReduceHistograms` (`:553-568`), `packedHistogramWorkspaceSize`
(`:269-282`) and the `distributed` flag (`:211`, `:246`) are not ported.
They gate on a RAFT communicator with more than one rank
(`raft::resource::comms_initialized(handle) && get_size() > 1`), which a
single-device library never has. PRICE: multi-GPU RF is unavailable, and the
histogram workspace is smaller by exactly `packedHistogramWorkspaceSize`,
which is zero on their single-rank path too (`:271`: `if (!distributed)
{ return 0; }`). So the workspace this file computes is byte-for-byte the
size THEIRS computes on one device. That equality is checked.

DEVIATION 303. `MLCommon::TimerCPU` and `tree->train_time` (`:377`, `:387`)
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

from std.gpu import WARP_SIZE
from std.math import ceildiv
from std.sys.info import size_of

from mojo_only.kernel_matrix import TARGET_COLUMN, column_shared_limit

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from ensemble.decisiontree.batched_levelalgo.bins import Bin, BinScales
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ObjectiveLike,
)
from ensemble.decisiontree.batched_levelalgo.quantiles import Quantiles
from ensemble.decisiontree.batched_levelalgo.split import (
    Split,
    init_split_kernel,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    SharedMemoryConfig,
    WorkloadInfo,
    sample_features_kernel,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    DeviceArgs,
    FindBestSplitsArgs,
    HistogramArgs,
    LeafArgs,
    NodeSplitArgs,
    NodeSplitScratch,
    launch_build_histograms_kernel,
    launch_find_best_splits_kernel,
    launch_leaf_kernel,
    launch_node_split_kernel,
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
    DEVIATION 302.
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


# ===========================================================================
# `Builder`, `builder.cuh:147-698`. The device half, wired.
# ===========================================================================


struct Builder[O: ObjectiveLike](Movable):
    """`ML::DT::Builder<ObjectiveT>`, `builder.cuh:147-698`.

    THE BUFFERS ARE FIELDS, AND THAT IS LOAD-BEARING RATHER THAN TIDY.
    Mojo destroys a value at its LAST USE, not at end of scope, so a
    `DeviceBuffer` handed to a kernel as a raw pointer is dead the moment
    `.unsafe_ptr()` is taken and the next `enqueue_create_buffer` is handed
    the same address. That is not theoretical: the kernels lane measured a
    kernel reading `n_bins` as `-8388609`, the bit pattern of
    `Split::Min()`, because the `splits` buffer had been allocated on top of
    a freed `n_bins` buffer -- AND THE ARM HAD PASSED ONCE BEFORE THAT. A
    struct field keeps its buffer alive for the struct's lifetime, so
    holding every buffer here removes the hazard for every launch at once
    instead of requiring a `_ = buf^` after each `synchronize`.

    It is also what their code does, for their own reason: `Builder` owns
    one `rmm::device_uvector<char> d_buff` and one pinned
    `ML::pinned_host_vector<char> h_buff` and carves every pointer out of
    them once in `assignWorkspace` (`:334-372`), because a kernel inside a
    tree builder must not allocate.

    PARAMETERIZED ON CLASSIFICATION ONLY, and that is an OPEN ITEM rather
    than a design. Theirs is `Builder<ObjectiveT>`, generic over the
    objective. Ours cannot be yet: the launchers in
    `kernels/builder_kernels_impl.mojo` are OVERLOADED on the concrete
    objective type (their DEVIATION 129a) because `objectives.mojo` declares
    no trait and Mojo traits are nominal, so a generic `Builder` has nothing
    to dispatch on. The one-line upstream fix -- declare the two objective
    structs conformant to an objective trait, moved out of
    `builder_kernels_impl.mojo` so `objectives.mojo` can import it without a
    cycle -- deletes the adapters, the six launcher overloads AND this
    restriction together. Recorded in `ensemble/PLAN.md`.

    Until then this builder trains CLASSIFICATION forests, which is the
    ship-first path this directory was planned around for an independent
    reason: `ClassificationBin` is an integer counter with an integer
    atomic, so its histogram is order-independent by construction.
    """

    var params: DecisionTreeParams
    var treeid: Int32
    var seed: UInt64
    var n_sampled_rows: Int
    var n_cols: Int
    var original_n_sampled_cols: Int
    var num_outputs: Int32
    # THEIRS constructs the objective FRESH inside `computeSplit`
    # (`builder.cuh:592-596`) from `params`, every level, every column
    # block. It is the same value every time -- `params`, `num_outputs`
    # and the scales do not move inside a fit -- so ours is constructed
    # once by the caller and stored. Same value, one construction; a
    # spelling difference, and the only reason to note it is that a reader
    # diffing `computeSplit` will find their four-argument constructor
    # missing from ours.
    var objective: Self.O

    # --- the workspace, `builder.cuh:176-212` ---------------------------
    var histograms: DeviceBuffer[DType.uint8]
    var mutex: DeviceBuffer[DType.int32]
    var splits: DeviceBuffer[DType.uint8]
    var d_work_items: DeviceBuffer[DType.uint8]
    var workload_info: DeviceBuffer[DType.uint8]
    var column_samples: DeviceBuffer[DType.int32]
    var partition_row_ids: DeviceBuffer[DType.int32]
    var h_work_items: HostBuffer[DType.uint8]
    var h_workload_info: HostBuffer[DType.uint8]
    var h_splits: HostBuffer[DType.uint8]

    # --- DEVIATION 128a's argument blobs, one per launcher --------------
    var hist_args: DeviceArgs[HistogramArgs[Self.O]]
    var find_args: DeviceArgs[FindBestSplitsArgs[Self.O]]
    var leaf_args: DeviceArgs[LeafArgs[Self.O]]
    var node_split_args: DeviceArgs[NodeSplitArgs[Self.O.DataT, Self.O.LabelT]]
    var node_split_scratch: NodeSplitScratch[Self.O.DataT, TPB_DEFAULT]

    def __init__(
        out self,
        ctx: DeviceContext,
        params: DecisionTreeParams,
        treeid: Int32,
        seed: UInt64,
        n_sampled_rows: Int,
        n_cols: Int,
        num_outputs: Int32,
        var objective: Self.O,
    ) raises:
        """`builder.cuh:213-261`, their constructor.

        `n_sampled_cols = max(1, IdxT(max_features * n_cols))` is `:240`;
        `max_blocks_dimx = 1 + max_batch_size + n_sampled_rows / TPB` is
        `:250`; their two `ASSERT`s at `:251-254` become raises.
        """
        self.params = params
        self.treeid = treeid
        self.seed = seed
        self.n_sampled_rows = n_sampled_rows
        self.n_cols = n_cols
        self.num_outputs = num_outputs
        self.objective = objective^
        self.original_n_sampled_cols = n_sampled_cols_for(
            params.max_features, n_cols
        )

        if num_outputs < Int32(1):
            raise Error("n_classes should be at least 1")
        if self.original_n_sampled_cols < 1 or (
            self.original_n_sampled_cols > n_cols
        ):
            raise Error(
                "n_sampled_cols must be in [1, n_cols]; got "
                + String(self.original_n_sampled_cols)
            )

        var max_batch = Int(params.max_batch_size)
        var max_blocks = max_blocks_dimx_for(
            params.max_batch_size, n_sampled_rows
        )
        var max_len_histograms = (
            max_batch
            * Int(params.max_n_bins)
            * N_BLKS_FOR_COLS
            * Int(num_outputs)
        )

        self.histograms = ctx.enqueue_create_buffer[DType.uint8](
            size_of[Self.O.BinT]() * max_len_histograms
        )
        self.mutex = ctx.enqueue_create_buffer[DType.int32](max_batch)
        self.splits = ctx.enqueue_create_buffer[DType.uint8](
            size_of[Split[Self.O.DataT]]() * max_batch
        )
        self.d_work_items = ctx.enqueue_create_buffer[DType.uint8](
            size_of[NodeWorkItem]() * max_batch
        )
        self.workload_info = ctx.enqueue_create_buffer[DType.uint8](
            size_of[WorkloadInfo]() * max_blocks
        )
        self.column_samples = ctx.enqueue_create_buffer[DType.int32](
            max_batch * self.original_n_sampled_cols
        )
        self.partition_row_ids = ctx.enqueue_create_buffer[DType.int32](
            n_sampled_rows if n_sampled_rows > 0 else 1
        )
        self.h_work_items = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[NodeWorkItem]() * max_batch
        )
        self.h_workload_info = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[WorkloadInfo]() * max_blocks
        )
        self.h_splits = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[Split[Self.O.DataT]]() * max_batch
        )

        self.hist_args = DeviceArgs[HistogramArgs[Self.O]](ctx)
        self.find_args = DeviceArgs[FindBestSplitsArgs[Self.O]](ctx)
        self.leaf_args = DeviceArgs[LeafArgs[Self.O]](ctx)
        self.node_split_args = DeviceArgs[NodeSplitArgs[Self.O.DataT, Self.O.LabelT]](
            ctx
        )
        self.node_split_scratch = NodeSplitScratch[Self.O.DataT, TPB_DEFAULT](
            ctx, max_blocks * TPB_DEFAULT, max_batch
        )
        ctx.enqueue_memset(self.mutex, Int32(0))
        ctx.synchronize()

    @always_inline
    def _splits_ptr(mut self) -> MutPointer[Split[Self.O.DataT], MutUntrackedOrigin]:
        return (
            self.splits.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[Split[Self.O.DataT]]()
        )

    @always_inline
    def _work_items_ptr(mut self) -> MutPointer[NodeWorkItem, MutUntrackedOrigin]:
        return (
            self.d_work_items.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[NodeWorkItem]()
        )

    @always_inline
    def _workload_ptr(mut self) -> MutPointer[WorkloadInfo, MutUntrackedOrigin]:
        return (
            self.workload_info.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[WorkloadInfo]()
        )

    @always_inline
    def _hist_ptr(mut self) -> MutPointer[Self.O.BinT, MutUntrackedOrigin]:
        return (
            self.histograms.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[Self.O.BinT]()
        )

    def _objective(self) -> Self.O:
        """`builder.cuh:592-596`. See the field's comment for why this
        returns the stored one rather than constructing a new one."""
        return self.objective.copy()

    def _upload_work_items(
        mut self, ctx: DeviceContext, items: List[NodeWorkItem]
    ) raises:
        """`raft::update_device(d_work_items, ...)`, `:466`, `:492`."""
        var p = self.h_work_items.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
        for i in range(len(items)):
            p[unsafe_offset=i] = items[i]
        ctx.enqueue_copy(
            dst_buf=self.d_work_items,
            src_ptr=self.h_work_items.unsafe_ptr(),
        )

    def _upload_workload(
        mut self, ctx: DeviceContext, wl: List[WorkloadInfo]
    ) raises:
        """`raft::update_device(workload_info, ...)`, `:405`."""
        var p = (
            self.h_workload_info.unsafe_ptr().unsafe_bitcast[WorkloadInfo]()
        )
        for i in range(len(wl)):
            p[unsafe_offset=i] = wl[i]
        ctx.enqueue_copy(
            dst_buf=self.workload_info,
            src_ptr=self.h_workload_info.unsafe_ptr(),
        )

    def _download_splits(
        mut self, ctx: DeviceContext, n: Int
    ) raises -> List[SplitSummary[Self.O.DataT]]:
        """`raft::update_host(h_splits, splits, ...)` + `sync_stream`,
        `:479-480`, `:676-677`. Their host copy is what `NodeQueue::Push`
        reads, projected onto the six fields it touches."""
        ctx.enqueue_copy(dst_buf=self.h_splits, src_buf=self.splits)
        ctx.synchronize()
        var p = self.h_splits.unsafe_ptr().unsafe_bitcast[Split[Self.O.DataT]]()
        var out = List[SplitSummary[Self.O.DataT]]()
        for i in range(n):
            ref s = p[unsafe_offset=i]
            out.append(
                SplitSummary[Self.O.DataT](
                    s.colid != Int32(-1),
                    s.colid,
                    s.quesval,
                    s.best_metric_val,
                    s.global_nLeft,
                    s.local_nLeft,
                )
            )
        return out^

    def _sample_features(
        mut self,
        ctx: DeviceContext,
        n_work_items: Int,
        sample_offset: Int,
        n_sampled_cols: Int,
    ) raises:
        """`Builder::sampleFeatures`, `:505-520`, calling `sample_features`
        (`kernels/builder_kernels.cuh:66-94`).

        The seed is split into two Int32 halves BY BIT PATTERN with every
        intermediate bound to a `var`; chaining the conversions inline folds
        to one sign-extending cast and silently delivers the wrong word.
        """
        var n_column_samples = n_work_items * n_sampled_cols
        if n_column_samples == 0:
            return
        var hi_u = (self.seed >> 32).cast[DType.uint32]()
        var lo_u = (self.seed & 0xFFFFFFFF).cast[DType.uint32]()
        var hi_arg = hi_u.cast[DType.int32]()
        var lo_arg = lo_u.cast[DType.int32]()
        var blocks = ceildiv(n_column_samples, 256)
        ctx.enqueue_function[sample_features_kernel](
            self.column_samples.unsafe_ptr(),
            self._work_items_ptr(),
            Int32(n_column_samples),
            self.treeid,
            lo_arg,
            hi_arg,
            Int32(sample_offset),
            Int32(self.n_cols),
            Int32(n_sampled_cols),
            grid_dim=blocks,
            block_dim=256,
        )

    def _compute_split(
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
        col: Int,
        n_blocks_dimx: Int,
        n_work_items: Int,
        n_sampled_cols: Int,
        smem_config: SharedMemoryConfig,
    ) raises:
        """`Builder::computeSplit`, `:570-626`.

        Their `n_blocks_dimy = min(n_blks_for_cols, n_sampled_cols - col)`
        is what makes the histogram buffer hold up to ten columns at once,
        and `len_histograms` is sized from that same `n_blocks_dimy` rather
        than from `n_blks_for_cols`, so the memset is exactly the region the
        launch will touch.
        """
        if n_blocks_dimx == 0:
            return
        var n_bins = Int(self.params.max_n_bins)
        var n_classes = Int(self.num_outputs)
        var n_blocks_dimy = min(N_BLKS_FOR_COLS, n_sampled_cols - col)
        # `:588-590` -- theirs zeroes exactly `sizeof(BinT) *
        # len_histograms` bytes, where
        # `len_histograms = n_bins * n_classes * n_blocks_dimy *
        # n_work_items`. DEVIATION 304: `enqueue_memset` takes a BUFFER and
        # not a range, so this zeroes the whole histogram workspace --
        # sized for `max_batch_size` nodes and `N_BLKS_FOR_COLS` columns.
        # PRICE: strictly more zeroing than theirs, never less, so every
        # cell their launch reads is zeroed here too and no value can
        # differ. The excess is bounded by the workspace, which is
        # allocated once per tree, and is a candidate for a sub-buffer
        # view if one is ever measured to matter. Nothing is measured this
        # round.
        _ = n_bins * n_classes * n_blocks_dimy * n_work_items
        ctx.enqueue_memset(self.histograms, UInt8(0))
        var objective = self._objective()

        launch_build_histograms_kernel[Self.O](
            ctx,
            self._hist_ptr(),
            n_bins,
            dataset,
            quantiles,
            self._work_items_ptr(),
            col,
            self.column_samples.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            objective,
            self._workload_ptr(),
            n_blocks_dimx,
            n_blocks_dimy,
            smem_config,
            self.hist_args,
        )
        # DEVIATION 302: their `if (distributed) allReduceHistograms(...)`
        # sits exactly here (`:613`) and is unreachable on one device.
        launch_find_best_splits_kernel[Self.O](
            ctx,
            self._hist_ptr(),
            n_bins,
            dataset,
            quantiles,
            col,
            self.column_samples.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            self.mutex.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            self._splits_ptr(),
            objective,
            n_work_items,
            n_blocks_dimy,
            self.find_args,
        )

    def _compute_best_splits(
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
        work_items: List[NodeWorkItem],
        sample_offset: Int,
        n_sampled_cols: Int,
        smem_config: SharedMemoryConfig,
    ) raises -> List[SplitSummary[Self.O.DataT]]:
        """`Builder::computeBestSplits`, `:485-503`, in their order."""
        var n = len(work_items)
        # `:489` -- initSplit
        ctx.enqueue_function[init_split_kernel[Self.O.DataT]](
            self._splits_ptr(), Int32(n), grid_dim=ceildiv(n, 128), block_dim=128
        )
        # `:490` -- the mutex is re-zeroed EVERY batch, over max_batch_size
        # entries and not just the live ones.
        ctx.enqueue_memset(self.mutex, Int32(0))
        self._upload_work_items(ctx, work_items)

        var wl = List[WorkloadInfo]()
        var n_blocks_dimx = update_workload_info(work_items, wl)
        self._upload_workload(ctx, wl)

        self._sample_features(ctx, n, sample_offset, n_sampled_cols)

        # `:497-500` -- ten columns per launch.
        var c = 0
        while c < n_sampled_cols:
            self._compute_split(
                ctx, dataset, quantiles, c, n_blocks_dimx, n,
                n_sampled_cols, smem_config,
            )
            c += N_BLKS_FOR_COLS
        return self._download_splits(ctx, n)

    def do_split[
        sabotage: Int = 0
    ](
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
        work_items: List[NodeWorkItem],
        smem_config: SharedMemoryConfig,
    ) raises -> List[SplitSummary[Self.O.DataT]]:
        """`Builder::doSplit`, `:410-482`. THE MULTI-ROUND LOOP.

        See the module docstring for why the rounds exist. The active set
        shrinks to the nodes that still have no valid split (`:452-458`),
        and their `if (round + 1 >= max_sampling_rounds) break;` at `:456`
        means the LAST round does not bother rebuilding it.
        """
        var n = len(work_items)
        # `:437` -- `const IdxT original_n_sampled_cols = dataset.n_sampled_cols;`
        var ds = dataset.copy()
        var max_rounds = max_sampling_rounds_for(
            self.n_cols, self.original_n_sampled_cols
        )

        var final_splits = List[SplitSummary[Self.O.DataT]]()
        for _ in range(n):
            final_splits.append(
                SplitSummary[Self.O.DataT](
                    False, Int32(-1), Scalar[Self.O.DataT](0),
                    Scalar[Self.O.DataT](0), Int64(0), Int64(0),
                )
            )
        var active_items = work_items.copy()
        var active_to_original = List[Int]()
        for i in range(n):
            active_to_original.append(i)

        var round = 0
        while len(active_items) > 0 and round < max_rounds:
            var sample_offset = round * self.original_n_sampled_cols
            var n_sampled_cols = sampled_cols_in_round(
                self.n_cols, self.original_n_sampled_cols, round
            )
            # `:439-440` -- they NARROW the dataset field for this round:
            #     dataset.n_sampled_cols =
            #         min(original_n_sampled_cols, n_cols - sample_offset);
            # The last round is short whenever n_sampled_cols does not
            # divide n_cols, and the kernels' stride must follow it.
            ds.n_sampled_cols = Int32(n_sampled_cols)
            # CHECK HOOK. 8 restores the pre-fix value -- the DatasetView
            # field left at whatever the caller passed, i.e. `n_cols` -- so
            # a check can watch the writer/reader strides come apart
            # instead of taking the fix on trust. THIS is the load-bearing
            # write: a hook on the `train` assignment alone is inert,
            # because this line overwrites it every round.
            comptime if sabotage == 8:
                ds.n_sampled_cols = Int32(self.n_cols)
            var h = self._compute_best_splits(
                ctx, ds, quantiles, active_items, sample_offset,
                n_sampled_cols, smem_config,
            )
            var retry_items = List[NodeWorkItem]()
            var retry_to_original = List[Int]()
            for i in range(len(active_items)):
                var orig = active_to_original[i]
                final_splits[orig] = h[i]
                if not h[i].is_valid:
                    retry_items.append(active_items[i])
                    retry_to_original.append(orig)
            if round + 1 >= max_rounds:
                break
            active_items = retry_items^
            active_to_original = retry_to_original^
            round += 1

        # `:458` -- `dataset.n_sampled_cols = original_n_sampled_cols;`
        ds.n_sampled_cols = Int32(self.original_n_sampled_cols)

        # `:474-478` -- the chosen splits go back to the device once, and
        # the partition runs ONCE over the whole batch.
        var sp = self.h_splits.unsafe_ptr().unsafe_bitcast[Split[Self.O.DataT]]()
        for i in range(n):
            sp[unsafe_offset=i] = Split[Self.O.DataT](
                final_splits[i].quesval,
                final_splits[i].colid,
                final_splits[i].best_metric_val,
                final_splits[i].global_n_left,
                final_splits[i].local_n_left,
                Int32(-1),
                Int32(-1),
            )
        ctx.enqueue_copy(
            dst_buf=self.splits, src_ptr=self.h_splits.unsafe_ptr()
        )
        self._upload_work_items(ctx, work_items)
        var wl = List[WorkloadInfo]()
        var n_partition_blocks = update_workload_info(work_items, wl)
        self._upload_workload(ctx, wl)

        launch_node_split_kernel(
            ctx,
            ds,
            self._work_items_ptr(),
            self._splits_ptr(),
            self._workload_ptr(),
            n_partition_blocks,
            n,
            self.partition_row_ids.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            self.node_split_scratch,
            self.node_split_args,
        )
        return self._download_splits(ctx, n)

    def shared_memory_config(self) raises -> SharedMemoryConfig:
        """`Builder::computeSharedMemoryConfig`, `:522-551`, with this
        build's sizes filled in.

        Theirs is called once per `computeBestSplits` (`:493`). Ours is
        called once per `train` and threaded down, because in this port it
        is a pure function of `params`, `num_outputs` and the two `size_of`s
        -- none of which move inside a fit -- and because the one input that
        WOULD have been a device query is a kernel-matrix row here rather
        than a `get_attribute` (1.26 ms per call on Metal). Same value,
        computed once.

        `cdf_scan_smem_size` is `sizeof(cub::BlockScan<BinT, TPB>::
        TempStorage)` in their expression; ours is what
        `core/block_scan.block_inclusive_sum` actually allocates,
        `(TPB + WARPS) * size_of[BinT]`, so the two implementations agree on
        whether a configuration fits.
        """
        var warps = ceildiv(TPB_DEFAULT, WARP_SIZE)
        var cdf_scan_smem_size = (TPB_DEFAULT + warps) * size_of[Self.O.BinT]()
        return compute_shared_memory_config(
            self.params.max_n_bins,
            self.num_outputs,
            size_of[Self.O.BinT](),
            size_of[Scalar[Self.O.DataT]](),
            size_of[Split[Self.O.DataT]](),
            column_shared_limit(TARGET_COLUMN),
            cdf_scan_smem_size,
            WARP_SIZE,
        )

    def set_leaf_predictions(
        mut self,
        ctx: DeviceContext,
        mut tree: TreeMetaDataNode[Self.O.DataT],
        instance_ranges: List[InstanceRange],
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
    ) raises:
        """`Builder::SetLeafPredictions`, `:630-687`.

        Their comment at `:636`: "do this in batch to reduce peak memory
        usage in extreme cases", with `max_batch_size = min(100000,
        sparsetree.size())` (`:637`). Transcribed, including the batching,
        because a tree with more nodes than that is exactly the case the
        batching exists for.

        Note their `leafKernel` runs over EVERY node in the batch and
        returns immediately for the ones that are not leaves (`:225`), so
        the launch is sized by node count and not by leaf count. The zero
        fill at `:659-660` is what leaves the internal nodes' slots at 0.
        """
        var n_nodes = len(tree.sparsetree)
        var n_out = Int(dataset.num_outputs)
        if n_nodes != len(instance_ranges):
            raise Error(
                "Expected instance range for each node; "
                + String(n_nodes)
                + " nodes vs "
                + String(len(instance_ranges))
                + " ranges"
            )
        tree.vector_leaf = List[Scalar[Self.O.DataT]]()
        tree.vector_leaf.resize(n_nodes * n_out, Scalar[Self.O.DataT](0))

        var batch = min(100000, n_nodes)
        if batch == 0:
            return
        var d_tree = ctx.enqueue_create_buffer[DType.uint8](
            size_of[SparseTreeNode[Self.O.DataT]]() * batch
        )
        var h_tree = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[SparseTreeNode[Self.O.DataT]]() * batch
        )
        var d_ranges = ctx.enqueue_create_buffer[DType.uint8](
            size_of[InstanceRange]() * batch
        )
        var h_ranges = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[InstanceRange]() * batch
        )
        var d_leaves = ctx.enqueue_create_buffer[Self.O.DataT](batch * n_out)
        var h_leaves = ctx.enqueue_create_host_buffer[Self.O.DataT](batch * n_out)
        var objective = self._objective()
        # `:652` -- their smem is `sizeof(BinT) * num_outputs`.
        var smem_size = size_of[Self.O.BinT]() * n_out

        var begin = 0
        while begin < n_nodes:
            var end = min(begin + batch, n_nodes)
            var size = end - begin
            var tp = h_tree.unsafe_ptr().unsafe_bitcast[
                SparseTreeNode[Self.O.DataT]
            ]()
            var rp = h_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange]()
            for i in range(size):
                tp[unsafe_offset=i] = tree.sparsetree[begin + i]
                rp[unsafe_offset=i] = instance_ranges[begin + i]
            ctx.enqueue_copy(dst_buf=d_tree, src_ptr=h_tree.unsafe_ptr())
            ctx.enqueue_copy(dst_buf=d_ranges, src_ptr=h_ranges.unsafe_ptr())
            ctx.enqueue_memset(d_leaves, Scalar[Self.O.DataT](0))

            launch_leaf_kernel[Self.O](
                ctx,
                objective,
                dataset,
                d_tree.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
                .unsafe_bitcast[SparseTreeNode[Self.O.DataT]](),
                d_ranges.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
                .unsafe_bitcast[InstanceRange](),
                d_leaves.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin](),
                size,
                smem_size,
                self.leaf_args,
            )
            ctx.enqueue_copy(dst_buf=h_leaves, src_buf=d_leaves)
            ctx.synchronize()
            for i in range(size * n_out):
                tree.vector_leaf[begin * n_out + i] = (
                    h_leaves.unsafe_ptr().unsafe_load(i)
                )
            begin = end

        # The buffers above are LOCALS, and Mojo frees a value at its last
        # use rather than at end of scope -- so a device buffer handed to a
        # kernel as a raw pointer can be freed and reallocated under the
        # running launch. These uses keep them alive past the final
        # `synchronize`. Measured hazard, not a precaution.
        _ = d_tree^
        _ = h_tree^
        _ = d_ranges^
        _ = h_ranges^
        _ = d_leaves^
        _ = h_leaves^

    def train[
        sabotage: Int = 0
    ](
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
    ) raises -> TreeMetaDataNode[Self.O.DataT]:
        """`Builder::train`, `:375-389`. Their loop, verbatim:

            NodeQueue queue(params, maxNodes(), n_sampled_rows, num_outputs);
            while (queue.HasWork()) {
              auto work_items = queue.Pop();
              auto [h_splits, n] = doSplit(work_items);
              queue.Push(work_items, h_splits);
            }
            auto tree = queue.GetTree();
            SetLeafPredictions(tree, queue.GetInstanceRanges());

        `train_time` is not set; see DEVIATION 303.
        """
        # `:240` -- THEIR CTOR SETS THIS FIELD:
        #     dataset.n_sampled_cols = max(1, IdxT(params.max_features * n_cols))
        # and both split kernels then index `column_samples` with it
        # (`kernels/builder_kernels_impl.cuh:310`, `:373`) at exactly the
        # stride `sample_features` wrote at, because `:509` passes the same
        # field as `k`. ONE VARIABLE, so writer and reader cannot disagree.
        # Ours took the sampled count onto the Builder instead and left the
        # DatasetView field at whatever the caller passed -- `n_cols` -- so
        # the writer strode by max_features*n_cols and the reader by n_cols.
        # Node 0 coincided; every later node in a batch read another node's
        # columns, and deep enough into a batch, off the end of the
        # allocation. Default classifier path (`max_features='sqrt'`).
        var ds = dataset.copy()
        ds.n_sampled_cols = Int32(self.original_n_sampled_cols)

        var smem_config = self.shared_memory_config()
        var queue = NodeQueue[Self.O.DataT](
            self.params,
            max_nodes(self.params.max_depth),
            self.n_sampled_rows,
            self.num_outputs,
        )
        while queue.has_work():
            var work_items = queue.pop()
            var h_splits = self.do_split[sabotage](
                ctx, ds, quantiles, work_items, smem_config
            )
            queue.push(work_items, h_splits)
        var tree = queue.tree.copy()
        tree.treeid = self.treeid
        self.set_leaf_predictions(
            ctx, tree, queue.node_instances_.copy(), ds
        )
        return tree^
