# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Random Forest decision-tree builder and device training pipeline, aligned with the pinned cuML batched-level algorithm."""

from std.gpu import WARP_SIZE
from std.math import ceildiv
from std.sys.info import size_of

from checks.kernel_matrix import TARGET_COLUMN, column_shared_limit

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.launch_log import log_launch
from ensemble.instruments import FitInstruments
from ensemble.decisiontree.batched_levelalgo.bins import Bin, BinScales
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ObjectiveLike,
)
from ensemble.decisiontree.batched_levelalgo.quantiles import Quantiles
from ensemble.decisiontree.batched_levelalgo.split import (
    Split,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    SharedMemoryConfig,
    WorkloadInfo,
)
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels_impl import (
    DeviceArgs,
    FindBestSplitsArgs,
    HistogramArgs,
    LeafArgs,
    NodeSplitArgs,
    NodeSplitScratch,
    TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES,
    launch_build_histograms_kernel,
    launch_find_best_splits_kernel,
    launch_gather_sampled_order_kernel,
    launch_leaf_kernel,
    launch_node_split_kernel,
    launch_phase_setup_kernel,
)
from ensemble.decisiontree.decisiontree import (
    CRITERION_END,
    DecisionTreeParams,
    GINI,
    MSE,
    TreeMetaDataNode,
)
from ensemble.flatnode import SparseTreeNode

# `builder.cuh:161` -- "default threads per block for most kernels in here"
comptime TPB_DEFAULT = 128

# `builder.cuh:163` is DECLARED ONCE, in
# `kernels/builder_kernels_impl.mojo`, and imported above.
#
# It used to be declared here TOO, as a second independent `16 * 1024`.
# The two agreed, and nothing made them agree: this file's copy decides
# WHICH ARM RUNS (`shared_memory_config` below) while the kernels file's
# copy sizes the BLOB THE SHARED ARM WRITES INTO
# (`default_smem_bin_slots`). Move either one and the dispatch admits the
# shared arm at a size its blob cannot hold, which is a shared-memory
# overflow and not a wrong answer -- exactly the property DEVIATION 103a's
# sizing argument rests on. Theirs cannot drift because theirs is one
# `static constexpr` member and the kernels take the size as a launch
# argument; ours needs it in both places, so it must be one symbol.
#
# The declaration lives in the kernels file rather than here because
# `builder.mojo` imports THAT file, so the other direction is a cycle.

# `builder.cuh:203` -- "number of blocks used to parallelize column-wise
# computations". A plain member initialised to 10 and never reassigned.
comptime N_BLKS_FOR_COLS = 10

# `builder.cuh:205` -- "Memory alignment value"
comptime ALIGN_VALUE = 512

# Histogram work uses the upstream one-item-per-thread mapping. If this is
# tuned above one, partition-phase workload reuse must remain disabled because
# its slot mapping requires TPB-granular rows; histogram integer accumulation
# itself remains order-independent.
comptime HIST_ITEMS_PER_THREAD = 1
comptime HIST_WORKLOAD_GRANULARITY = TPB_DEFAULT * HIST_ITEMS_PER_THREAD


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
    and is corrupted only by `ensemble/checks/builder_check.mojo`. It is
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
    o: MutOrigin, //, sabotage: Int = 0
](
    work_items: List[NodeWorkItem],
    h_workload_info: MutPointer[WorkloadInfo, o],
    granularity: Int = TPB_DEFAULT,
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

    Returns their `n_blocks_dimx`. WRITES STRAIGHT INTO `h_workload_info`
    -- RESTORES upstream: theirs fills the pinned `h_workload_info` array
    in place (`:401`) and this used to rebuild a `List` (growth reallocs
    on every level) that the upload then copied into the pinned buffer
    element by element. The pointer must hold at least
    `max_blocks_dimx_for(...)` entries, the same bound their array is
    allocated at (`:250`, proven in `max_blocks_dimx_for`'s docstring).
    The device copy their `raft::update_device` at `:405` performs is the
    caller's, so this stays a pure host function and can be checked
    without a GPU (hand it any host array's pointer).

    `granularity` is DEVIATION 2011: rows per block. `TPB_DEFAULT` (the
    default, and the only value the partition may ever use -- the scan's
    slot math is TPB-granular) reproduces their arithmetic exactly;
    `enqueue_best_splits` passes `HIST_WORKLOAD_GRANULARITY`, which is
    the same value until the flag is flipped.
    """
    var n_blocks_dimx = 0
    for i in range(len(work_items)):
        var item = work_items[i]
        var n_blocks_per_node = max(
            ceildiv(item.instances.count, granularity), 1
        )
        comptime if sabotage == 4:
            # Drop their `max(..., 1)`. Predicted movement: an EMPTY node
            # gets zero blocks, so the grid is short and that node is never
            # visited by any kernel.
            n_blocks_per_node = ceildiv(item.instances.count, granularity)
        for b in range(n_blocks_per_node):
            h_workload_info[unsafe_offset = n_blocks_dimx + b] = (
                WorkloadInfo(Int32(i), Int32(b), Int32(n_blocks_per_node))
            )
        n_blocks_dimx += n_blocks_per_node
    return n_blocks_dimx


def workload_blocks_for(work_items: List[NodeWorkItem]) -> Int:
    """`updateWorkloadInfo`'s running total WITHOUT its writes -- the same
    `max(ceildiv(count, TPB_DEFAULT), 1)` sum, so a caller that can prove
    the staged map is already on the device (DEVIATION 1919) can size the
    grid without restaging. Any edit to `update_workload_info`'s block
    arithmetic must land here in the same commit; the two must agree or
    the partition grid is short.

    DEVIATION 2011 note: this function is TPB-granular ON PURPOSE and
    grew no granularity argument -- its one caller is the 1919 reuse
    path, which exists only when the staged map is also TPB-granular
    (the reuse is disabled under `HIST_ITEMS_PER_THREAD > 1`)."""
    var n_blocks_dimx = 0
    for i in range(len(work_items)):
        n_blocks_dimx += max(
            ceildiv(work_items[i].instances.count, TPB_DEFAULT), 1
        )
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
# DEVIATION 117's state carriers. cuML's shipped default runs the forest
# loop `#pragma omp parallel for num_threads(n_streams)` with n_streams=4
# (`randomforest.cuh:336-337`, `randomforestclassifier.py:94`): four host
# threads, four CUDA streams, four trees in flight. Metal has one queue and
# this port has one host thread, so the same overlap is expressed as K-WAY
# PIPELINING: each tree's `doSplit` is cut at its two sync points
# (`builder.cuh:479-481`, `:501-502`), K trees enqueue their next phase,
# and ONE synchronize serves all of them. No output bit can move: every
# per-tree and per-node draw is a pure hash of (seed, treeid[, nodeid])
# (`randomforest.cuh:120-122`, `builder_kernels.cuh:88`), each in-flight
# tree owns its whole workspace, and the only shared device objects are
# read-only (the data, the labels, the quantiles).
# ===========================================================================


@fieldwise_init
struct BatchState[O: ObjectiveLike](Movable):
    """One batch of `doSplit` (`builder.cuh:410-482`), suspended at a sync
    point. `phase` 0 means a best-splits download is pending; 1 means the
    node-split download is. `do_split` drives this serially and is
    bit-identical to the pre-pipeline transcription."""

    var work_items: List[NodeWorkItem]
    var active_items: List[NodeWorkItem]
    var active_to_original: List[Int]
    var final_splits: List[SplitSummary[Self.O.DataT]]
    var round: Int
    var max_rounds: Int
    var phase: Int
    var result: List[SplitSummary[Self.O.DataT]]


@fieldwise_init
struct TreeState[O: ObjectiveLike](Movable):
    """One tree's `Builder::train` (`:375-389`), suspended between
    batches. `ds` carries the `n_sampled_cols` correction train() makes
    (their ctor's `:240` -- see `train`'s docstring); `tree` is valid
    only once `done` is set."""

    var queue: NodeQueue[Self.O.DataT]
    var ds: DatasetView[Self.O.DataT, Self.O.LabelT]
    var smem_config: SharedMemoryConfig
    var batch: BatchState[Self.O]
    var tree: TreeMetaDataNode[Self.O.DataT]
    var done: Bool


struct _DevPrefixView(Movable):
    """A cached prefix view of one device workspace buffer.

    RESTORES their carve-once shape: upstream carves every pointer out of
    `d_buff` ONCE (`builder.cuh:334-368`) and then passes COUNTS to
    count-parameterized APIs, so no per-level object is ever created.
    `enqueue_copy`/`enqueue_memset` here are buffer-shaped (the byte
    count IS the buffer), so the live-prefix byte counts -- which are
    THEIRS, see e.g. `_stage_work_items` -- need a view object; this
    cache recreates that view only when the byte count actually moves
    (level to level it is usually pinned at `max_batch_size`'s worth once
    the frontier saturates). Replacing a view whose enqueue is still in
    flight is the already-established contract of this file (`_ = dst^`
    directly after `enqueue_copy` predates this cache).
    """

    var view: DeviceBuffer[DType.uint8]
    var nbytes: Int
    # DEVIATION 1908 -- a fixed base offset into `parent`, so a view can
    # track the live prefix OF A REGION (a packed phase span, a staging
    # slot) and not only of a whole buffer. 0 preserves the original
    # meaning at every pre-1908 site.
    var offset: Int

    def __init__(
        out self,
        parent: DeviceBuffer[DType.uint8],
        nbytes: Int,
        offset: Int = 0,
    ) raises:
        self.view = parent.create_sub_buffer[DType.uint8](offset, nbytes)
        self.nbytes = nbytes
        self.offset = offset

    def ensure(
        mut self, parent: DeviceBuffer[DType.uint8], nbytes: Int
    ) raises:
        """Point `view` at `nbytes` of `parent` starting at the fixed
        `offset`; reuse the existing view when the bounds already match."""
        if nbytes != self.nbytes:
            self.view = parent.create_sub_buffer[DType.uint8](
                self.offset, nbytes
            )
            self.nbytes = nbytes


struct _HostPrefixView(Movable):
    """`_DevPrefixView`'s pinned-host sibling."""

    var view: HostBuffer[DType.uint8]
    var nbytes: Int
    var offset: Int

    def __init__(
        out self,
        parent: HostBuffer[DType.uint8],
        nbytes: Int,
        offset: Int = 0,
    ) raises:
        self.view = parent.create_sub_buffer[DType.uint8](offset, nbytes)
        self.nbytes = nbytes
        self.offset = offset

    def ensure(
        mut self, parent: HostBuffer[DType.uint8], nbytes: Int
    ) raises:
        if nbytes != self.nbytes:
            self.view = parent.create_sub_buffer[DType.uint8](
                self.offset, nbytes
            )
            self.nbytes = nbytes


# ===========================================================================
# DEVIATION 1908 -- the K slots' split results travel as ONE readback per
# pipeline cycle.
# ===========================================================================


struct SplitStaging(Movable):
    """One contiguous device/pinned pair holding EVERY pipeline slot's
    `splits` region, DEVIATION 1908.

    cuML's four streams each run their own cheap async `update_host`
    (`builder.cuh:479`, `:501`); this port's one queue paid a separate
    host-priced transfer PER IN-FLIGHT TREE per cycle instead. With every
    slot's region carved out of this one allocation (512-byte slot
    stride, the arena's own `ALIGN_VALUE`), the forest loop reads all of
    them back in ONE `enqueue_copy` per cycle (`flush_splits_downloads`),
    ~K transfers per cycle down to 1. NO VALUE MOVES: the kernels write
    the same split bytes at a different address, each builder still reads
    exactly its own `[0, n)` prefix, and the copy itself computes
    nothing.

    The prefix copy does carry the dead stretch between a slot's live
    `n` splits and the next slot's base -- the price of one transfer
    instead of K. Bounded by `(K-1) * slot_stride` (~590 KB at K=4,
    max_batch_size 4096, float32) per cycle, against a saved host-priced
    enqueue per slot per cycle; the census, not this comment, prices the
    trade on each vendor.
    """

    var d: DeviceBuffer[DType.uint8]
    var h: HostBuffer[DType.uint8]
    var slot_stride: Int
    var d_view: _DevPrefixView
    var h_view: _HostPrefixView

    def __init__(
        out self, ctx: DeviceContext, n_slots: Int, slot_bytes: Int
    ) raises:
        self.slot_stride = calculate_aligned_bytes(slot_bytes)
        var total = self.slot_stride * n_slots
        self.d = ctx.enqueue_create_buffer[DType.uint8](total)
        self.h = ctx.enqueue_create_host_buffer[DType.uint8](total)
        self.d_view = _DevPrefixView(self.d, total)
        self.h_view = _HostPrefixView(self.h, total)


def flush_splits_downloads[
    O: ObjectiveLike, sampled_labels: Bool = False
](
    ctx: DeviceContext,
    mut staging: SplitStaging,
    mut builders: List[Builder[O, sampled_labels]],
) raises:
    """The cycle's ONE splits readback, DEVIATION 1908.

    Collects every adopted builder's pending byte count (recorded by
    `_enqueue_splits_download` instead of enqueued), clears them, and
    copies the staging prefix that covers the furthest pending slot.
    Builders are indexed BY SLOT: `builders[k]` must have adopted slot
    `k`, which is how `fit_forest` wires them. Call after a cycle's
    enqueues and before its synchronize -- the queue is in-order, so the
    copy lands after every kernel that writes a slot's splits."""
    var extent = 0
    for k in range(len(builders)):
        var pending = builders[k].pending_splits_bytes
        if pending > 0:
            builders[k].pending_splits_bytes = 0
            var end = k * staging.slot_stride + pending
            if end > extent:
                extent = end
    if extent == 0:
        return
    staging.d_view.ensure(staging.d, extent)
    staging.h_view.ensure(staging.h, extent)
    log_launch("xfer_splits_download")
    ctx.enqueue_copy(dst_buf=staging.h_view.view, src_buf=staging.d_view.view)


# ===========================================================================
# `Builder`, `builder.cuh:147-698`. The device half, wired.
# ===========================================================================


struct Builder[O: ObjectiveLike, sampled_labels: Bool = False](Movable):
    """`ML::DT::Builder<ObjectiveT>`, `builder.cuh:147-698`.

    DEVIATION 2001 (`sampled_labels`, default False = the pre-2001
    builder): when True, the builder keeps a sampled-order copy of the
    labels (and sample weights) -- `labels_s[i] == labels[row_ids[i]]`,
    gathered once per tree by `stage_sampled_order` and re-permuted
    alongside `row_ids` at every split -- so the histogram and leaf
    kernels read their stat streams sequentially instead of gathering
    per element. The flag threads from ONE site
    (`randomforest.mojo`'s `LABELS_SAMPLED_ORDER`); every check that
    constructs `Builder[O]` gets the default and the old code.

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

    GENERIC OVER THE OBJECTIVE, as theirs is. `Builder<ObjectiveT>` gets
    that from an unconstrained `typename`; ours gets it from
    `ObjectiveLike` in `objectives.mojo`, which closed DEVIATION 129a and
    took the adapters and the launcher overloads with it. Regression and
    classification are the same code with a different `O`.

    THE OBJECTIVE IS BUILT FROM `params`, not handed in. `builder.cuh:592-596`
    constructs `ObjectiveT` fresh inside `computeSplit` from
    `params.min_samples_leaf`, `params.split_criterion` and
    `params.min_impurity_decrease`, and `:642` builds a three-argument one
    for the leaf pass. That is what makes `params` the single source of
    truth for those three settings, and it is why their two objective
    constructors take the same four arguments even though the regression
    one never stores the first.
"""

    var params: DecisionTreeParams
    var treeid: Int32
    # DEVIATION 401 -- the 0-based index of the batch within the current
    # tree, for identity-trace tags ("treeT.batchB.roundR..."). A POSITION
    # IN THE ALGORITHM (the queue pops in deterministic order), never a
    # machine property. -1 between trees; `begin_batch` pre-increments.
    var trace_batch: Int
    var seed: UInt64
    var n_sampled_rows: Int
    var n_cols: Int
    var original_n_sampled_cols: Int
    var num_outputs: Int32
    # DEVIATION 101b/112c's fixed-point scales -- the ONE input to the
    # objective that is not in `params` and has no cuML counterpart.
    # Everything else the objective needs comes from `params`, as theirs
    # does: `builder.cuh:592-596` constructs `ObjectiveT` fresh inside
    # `computeSplit` from `params.min_samples_leaf`,
    # `params.split_criterion` and `params.min_impurity_decrease`.
    #
    # This used to be a whole pre-built `objective` handed in by the
    # caller, and that was not a spelling difference: it gave those three
    # settings a SECOND source of truth. Setting them on `params` was
    # silently ignored, because nothing read them from there and nothing
    # checked that the two agreed.
    var scales: BinScales

    # --- the workspace, `builder.cuh:176-212` ---------------------------
    # `:207-209` -- `rmm::device_uvector<char> d_buff` and
    # `pinned_host_vector<char> h_buff`. ONE device allocation and one
    # host allocation for the whole builder; everything below is a VIEW
    # carved out of them by `assignWorkspace` (`:334-368`), 512-byte
    # aligned, in their order.
    #
    # This used to be ten separate allocations, with `workspace_layout`
    # computing the arena and nobody calling it -- so `builder_check` was
    # verifying that dead code agreed with their sizing. A Builder is
    # constructed once per TREE, so that was ten allocations per tree
    # instead of two, and it gave up the property their design is for:
    # no allocation inside the tree loop.
    var d_buff: DeviceBuffer[DType.uint8]
    var h_buff: HostBuffer[DType.uint8]

    var histograms: DeviceBuffer[DType.uint8]
    var mutex: DeviceBuffer[DType.int32]
    var splits: DeviceBuffer[DType.uint8]
    var d_work_items: DeviceBuffer[DType.uint8]
    var workload_info: DeviceBuffer[DType.uint8]
    var column_samples: DeviceBuffer[DType.int32]
    var partition_row_ids: DeviceBuffer[DType.int32]
    var h_workload_info: HostBuffer[DType.uint8]
    var h_splits: HostBuffer[DType.uint8]

    # --- DEVIATION 1908's phase-upload staging ---------------------------
    # One pinned span per builder holding a phase's [work items | pad |
    # workload map], sent to the `d_work_items`/`workload_info` arena
    # stretch as ONE H2D copy (`_enqueue_phase_upload`) where the port
    # used to pay two (their `update_device` pair, `:466` + `:405`).
    # `h_work_items` (a separate pinned staging -- theirs copies from a
    # pageable vector, `:467`, so their arena has none) is SUBSUMED by
    # this span; `h_workload_info` above keeps its carve so the host
    # arena's layout stays byte-for-byte theirs (builder_check), but the
    # live workload now stages here, at `cur_wl_rel` -- the next 512-byte
    # boundary past the phase's live work items.
    var h_phase: HostBuffer[DType.uint8]
    var phase_view: _DevPrefixView
    var cur_wl_rel: Int

    # --- DEVIATION 1908's shared splits staging --------------------------
    # False until `adopt_shared_splits` repoints `splits`/`h_splits` at a
    # `SplitStaging` slot; then `_enqueue_splits_download` RECORDS its
    # byte count here and the forest loop's `flush_splits_downloads`
    # copies every slot in one transfer per cycle.
    var splits_shared: Bool
    var pending_splits_bytes: Int

    # --- cached live-prefix views of the buffers above (see
    # `_DevPrefixView`) -----------------------------------------------------
    var splits_d_view: _DevPrefixView
    var splits_h_view: _HostPrefixView
    var hist_view: _DevPrefixView

    # --- DEVIATION 128a's argument blobs, one per launcher --------------
    # DEVIATION 1909: staged PER TREE, not per round/batch. Inside one
    # tree the split blobs are a pure function of the round's
    # `n_sampled_cols` -- everything else in them (dataset pointers and
    # counts, quantiles, the objective built from `params`) is fixed by
    # `begin_tree`/`reset_for_tree` -- and that width takes at most TWO
    # values per tree: the full `original_n_sampled_cols` and the short
    # last round's remainder (`sampled_cols_in_round`). Two cached slots
    # per launcher hold both; `args_cols_a`/`args_cols_b` record which
    # width each slot's device bytes carry, -1 = not staged for this
    # tree. The partition blob (`node_split_args`, width always restored
    # to `original_n_sampled_cols`, `:458`) needs one Bool. `leaf_args`
    # was already once per tree (DEVIATION 1893) and keeps its cadence.
    var hist_args: DeviceArgs[HistogramArgs[Self.O]]
    var find_args: DeviceArgs[FindBestSplitsArgs[Self.O]]
    var hist_args_alt: DeviceArgs[HistogramArgs[Self.O]]
    var find_args_alt: DeviceArgs[FindBestSplitsArgs[Self.O]]
    var args_cols_a: Int
    var args_cols_b: Int
    var leaf_args: DeviceArgs[LeafArgs[Self.O]]
    var node_split_args: DeviceArgs[NodeSplitArgs[Self.O.DataT, Self.O.LabelT]]
    var node_split_args_ready: Bool
    var node_split_scratch: NodeSplitScratch[
        Self.O.DataT, TPB_DEFAULT, Self.O.LabelT, Self.sampled_labels
    ]

    # --- DEVIATION 2001's sampled-order stat streams ---------------------
    # `labels_s[i] = labels[row_ids[i]]`, gathered once per tree
    # (`stage_sampled_order`) and re-permuted with `row_ids` at every
    # split; the partition-side twins live in `node_split_scratch` (that
    # struct's charter). NOT in `WorkspaceLayout`: builder_check
    # byte-compares that arena against cuML's `assignWorkspace`, and
    # these buffers have no counterpart there. `Optional` so the default
    # (flag-off) builder allocates NOTHING; held as fields so the
    # buffers outlive every launch that reads them (this struct's
    # last-use rule, see the docstring).
    var labels_s: Optional[DeviceBuffer[Self.O.LabelT]]
    var sample_weight_s: Optional[DeviceBuffer[DType.float32]]

    # --- the leaf pass's staging, DEVIATION 313's second half ------------
    # Theirs allocates `d_tree` / `d_instance_ranges` / `d_leaves` inside
    # `SetLeafPredictions` per tree (`:638-650`) -- out of RMM's pool, so
    # a pointer bump. Six Metal buffer creates per tree is the un-pooled
    # price; these live with the workspace instead and grow only when a
    # tree's node count exceeds every earlier tree's. `leaf_capacity` is
    # in NODES, batched the way theirs batches (`min(100000, n_nodes)`).
    var leaf_capacity: Int
    # The HOST staging is sized in NODES-OF-THE-TREE, not batch: upstream
    # stages each batch out of the FULL-SIZE host vectors
    # (`tree->sparsetree`, `tree->vector_leaf`, `builder.cuh:648-651`,
    # `:663-666`), which is what lets its batches enqueue with no sync
    # between them. `leaf_host_capacity` tracks that full size; the
    # device trio stays batch-sized, as theirs is.
    var leaf_host_capacity: Int
    var leaf_d_tree: DeviceBuffer[DType.uint8]
    var leaf_h_tree: HostBuffer[DType.uint8]
    var leaf_d_ranges: DeviceBuffer[DType.uint8]
    var leaf_h_ranges: HostBuffer[DType.uint8]
    var leaf_d_leaves: DeviceBuffer[Self.O.DataT]
    var leaf_h_leaves: HostBuffer[Self.O.DataT]

    def __init__(
        out self,
        ctx: DeviceContext,
        params: DecisionTreeParams,
        treeid: Int32,
        seed: UInt64,
        n_sampled_rows: Int,
        n_cols: Int,
        num_outputs: Int32,
        scales: BinScales = BinScales(1.0, 1.0),
    ) raises:
        """`builder.cuh:213-261`, their constructor.

        `n_sampled_cols = max(1, IdxT(max_features * n_cols))` is `:240`;
        `max_blocks_dimx = 1 + max_batch_size + n_sampled_rows / TPB` is
        `:250`; their two `ASSERT`s at `:251-254` become raises.
        """
        self.params = params
        self.treeid = treeid
        self.trace_batch = -1
        self.seed = seed
        self.n_sampled_rows = n_sampled_rows
        self.n_cols = n_cols
        self.num_outputs = num_outputs
        self.scales = scales

        # `decisiontree.cuh:251-256`. CRITERION_END is the "unset" sentinel
        # their header defaults `split_criterion` to (`decisiontree.hpp:89`),
        # and THEY resolve it before the objective is ever built:
        #
        #   if (params.split_criterion == CRITERION::CRITERION_END) {
        #     CRITERION default_criterion =
        #       (std::numeric_limits<LabelT>::is_integer) ? GINI : MSE;
        #     params.split_criterion = default_criterion;
        #   }
        #
        # They do it in `DecisionTree::fit`, which sits between their
        # `RandomForest::fit` and this constructor and also dispatches the
        # objective FAMILY on the same test. Ours has the family already
        # fixed by `O`, so only the value is left to resolve, and this is
        # the first point that both sees `params` and knows `O.LabelT`.
        # Their mutation of the params is kept: it is what makes the
        # resolved value the one every later read gets.
        if self.params.split_criterion == CRITERION_END:
            comptime if Self.O.LabelT.is_integral():
                self.params.split_criterion = GINI
            else:
                self.params.split_criterion = MSE

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

        # `:252-255` -- size the two buffers, then `:334-368` carve them.
        # ONE arithmetic, not two: `workspace_layout` returns the offsets
        # their `assignWorkspace` recomputes.
        var wl = workspace_layout(
            self.params.max_batch_size,
            self.params.max_n_bins,
            num_outputs,
            n_sampled_rows,
            self.original_n_sampled_cols,
            size_of[Self.O.BinT](),
            size_of[Split[Self.O.DataT]](),
            size_of[NodeWorkItem](),
            size_of[WorkloadInfo](),
            4,  # sizeof(IdxT), IdxT = int
        )
        self.d_buff = ctx.enqueue_create_buffer[DType.uint8](wl.device_total)
        self.h_buff = ctx.enqueue_create_host_buffer[DType.uint8](
            wl.host_total
        )

        # `:340-366`, in their order. Every offset is 512-byte aligned
        # (`:202`), so the int32 views below divide exactly by 4.
        self.histograms = self.d_buff.create_sub_buffer[DType.uint8](
            wl.histograms, size_of[Self.O.BinT]() * max_len_histograms
        )
        self.mutex = self.d_buff.create_sub_buffer[DType.int32](
            wl.mutex // 4, max_batch
        )
        self.splits = self.d_buff.create_sub_buffer[DType.uint8](
            wl.splits, size_of[Split[Self.O.DataT]]() * max_batch
        )
        self.d_work_items = self.d_buff.create_sub_buffer[DType.uint8](
            wl.d_work_items, size_of[NodeWorkItem]() * max_batch
        )
        # DEVIATION 1908: the live workload map now rides the packed
        # phase span and lands at `cur_wl_rel` past `wl.d_work_items`
        # (still inside this stretch); the carve is kept so the arena's
        # layout stays byte-for-byte theirs.
        self.workload_info = self.d_buff.create_sub_buffer[DType.uint8](
            wl.workload_info, size_of[WorkloadInfo]() * max_blocks
        )
        self.column_samples = self.d_buff.create_sub_buffer[DType.int32](
            wl.column_samples // 4,
            max_batch * self.original_n_sampled_cols,
        )
        self.partition_row_ids = self.d_buff.create_sub_buffer[DType.int32](
            wl.partition_row_ids // 4,
            n_sampled_rows if n_sampled_rows > 0 else 1,
        )
        self.h_workload_info = self.h_buff.create_sub_buffer[DType.uint8](
            wl.h_workload_info, size_of[WorkloadInfo]() * max_blocks
        )
        self.h_splits = self.h_buff.create_sub_buffer[DType.uint8](
            wl.h_splits, size_of[Split[Self.O.DataT]]() * max_batch
        )
        # NOT in their arena: theirs copies `d_work_items` from a pageable
        # `std::vector` (`:467`), so there is no host staging to carve.
        # `enqueue_copy` here is host<->device only, so ours needs one --
        # DEVIATION 1908 shapes it as the packed phase span: aligned
        # work-item capacity, then workload capacity. The device target is
        # the existing `d_work_items` + `workload_info` arena stretch,
        # which the layout above places adjacently (their order), so a
        # full-capacity span ends exactly at the workload region's live
        # end and never reaches `column_samples`.
        var wi_span = calculate_aligned_bytes(
            size_of[NodeWorkItem]() * max_batch
        )
        self.h_phase = ctx.enqueue_create_host_buffer[DType.uint8](
            wi_span + size_of[WorkloadInfo]() * max_blocks
        )
        self.phase_view = _DevPrefixView(
            self.d_buff,
            wi_span + size_of[WorkloadInfo]() * max_blocks,
            offset=wl.d_work_items,
        )
        self.cur_wl_rel = 0
        self.splits_shared = False
        self.pending_splits_bytes = 0

        # Cached live-prefix views, seeded at full capacity; the first
        # use at a smaller live count re-carves (see `_DevPrefixView`).
        self.splits_d_view = _DevPrefixView(
            self.splits, size_of[Split[Self.O.DataT]]() * max_batch
        )
        self.splits_h_view = _HostPrefixView(
            self.h_splits, size_of[Split[Self.O.DataT]]() * max_batch
        )
        self.hist_view = _DevPrefixView(
            self.histograms, size_of[Self.O.BinT]() * max_len_histograms
        )

        self.hist_args = DeviceArgs[HistogramArgs[Self.O]](ctx)
        self.find_args = DeviceArgs[FindBestSplitsArgs[Self.O]](ctx)
        self.hist_args_alt = DeviceArgs[HistogramArgs[Self.O]](ctx)
        self.find_args_alt = DeviceArgs[FindBestSplitsArgs[Self.O]](ctx)
        self.args_cols_a = -1
        self.args_cols_b = -1
        self.leaf_args = DeviceArgs[LeafArgs[Self.O]](ctx)
        self.node_split_args = DeviceArgs[NodeSplitArgs[Self.O.DataT, Self.O.LabelT]](
            ctx
        )
        self.node_split_args_ready = False
        self.node_split_scratch = NodeSplitScratch[
            Self.O.DataT, TPB_DEFAULT, Self.O.LabelT, Self.sampled_labels
        ](
            ctx,
            max_blocks * TPB_DEFAULT,
            max_batch,
            n_sampled_rows,
        )
        # DEVIATION 2001 -- the per-tree sampled-order streams, sized
        # once at the workspace's row maximum (`reset_for_tree` raises on
        # any tree that would disagree). Flag off: None, no allocation.
        comptime if Self.sampled_labels:
            self.labels_s = Optional(
                ctx.enqueue_create_buffer[Self.O.LabelT](
                    n_sampled_rows if n_sampled_rows > 0 else 1
                )
            )
            self.sample_weight_s = Optional(
                ctx.enqueue_create_buffer[DType.float32](
                    n_sampled_rows if n_sampled_rows > 0 else 1
                )
            )
        else:
            self.labels_s = Optional[DeviceBuffer[Self.O.LabelT]]()
            self.sample_weight_s = Optional[DeviceBuffer[DType.float32]]()

        # Leaf-pass staging, sized by their batching rule up front: a
        # depth < 13 tree can never exceed the dense bound, so for the
        # common case this never regrows; past their `min(100000, ...)`
        # cap the batching loop reuses the same buffers anyway.
        self.leaf_capacity = min(100000, max_nodes(params.max_depth))
        self.leaf_host_capacity = self.leaf_capacity
        var n_out = Int(num_outputs)
        self.leaf_d_tree = ctx.enqueue_create_buffer[DType.uint8](
            size_of[SparseTreeNode[Self.O.DataT]]() * self.leaf_capacity
        )
        self.leaf_h_tree = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[SparseTreeNode[Self.O.DataT]]() * self.leaf_capacity
        )
        self.leaf_d_ranges = ctx.enqueue_create_buffer[DType.uint8](
            size_of[InstanceRange]() * self.leaf_capacity
        )
        self.leaf_h_ranges = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[InstanceRange]() * self.leaf_capacity
        )
        self.leaf_d_leaves = ctx.enqueue_create_buffer[Self.O.DataT](
            self.leaf_capacity * n_out
        )
        self.leaf_h_leaves = ctx.enqueue_create_host_buffer[Self.O.DataT](
            self.leaf_capacity * n_out
        )

        ctx.enqueue_memset(self.mutex, Int32(0))
        ctx.synchronize()

    def reset_for_tree(mut self, treeid: Int32, n_sampled_rows: Int) raises:
        """DEVIATION 313: ONE Builder serves the whole forest; this is the
        per-tree half of its constructor.

        Their Builder IS constructed per tree (`decisiontree.cuh:258-260`),
        but every byte it allocates comes from RMM's POOLED resources --
        `rmm::device_uvector` / `pinned_host_vector` draw from the
        handle's memory resource, so their per-tree construction is
        pointer carving, not a driver allocation. On Metal a fresh
        `enqueue_create_buffer` per tree pays the driver every time, a
        cost their design never has. Pooling the workspace across trees is
        their allocator's semantics, ported; the same ruling as gbdt's
        TTreeWorkspace, pool-of-one.

        `treeid` is the only constructor input that varies across a
        forest's trees -- it feeds the feature-sampler seed chain
        `fnv1a32(seed, treeid, nodeid)`. Every workspace region is
        (re)written before it is read each batch: `mutex` is re-zeroed
        every `computeBestSplits` (`:490`), `histograms` before every
        column block (`:588-591`), `splits` by `initSplitKernel`, the work
        items and workload by their uploads, `partition_row_ids` by
        `nodeSplitKernel` before anything reads it. The fingerprint gate
        (`checks/fingerprint_probe.mojo`) held bit-exact across this
        change on all five configs, and the sabotage that freezes this
        method's `treeid` write moves every multi-tree line of it.

        The guard below cannot fire today -- every `RowSampler` arm holds
        `n_selected` constant across one forest -- and exists so a future
        arm that varies it rebuilds instead of reading a mis-sized
        workspace."""
        if n_sampled_rows != self.n_sampled_rows:
            raise Error(
                "Builder workspace sized for "
                + String(self.n_sampled_rows)
                + " sampled rows; tree asked for "
                + String(n_sampled_rows)
            )
        self.treeid = treeid
        # DEVIATION 401 -- per-tree batch numbering restarts with the tree.
        self.trace_batch = -1
        self._invalidate_args_cache()

    def stage_sampled_order[
        sabotage: Int = 0
    ](
        mut self,
        ctx: DeviceContext,
        mut dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
    ) raises:
        """DEVIATION 2001: gather this tree's sampled-order stat streams
        and REPOINT the view at them.

        Call between `reset_for_tree` and `begin_tree`, after the row
        sampler has written this tree's `row_ids` (the in-order queue
        sequences the gather after the sampler's kernels and before the
        first batch's). On return `dataset.labels` -- and, when
        `has_sample_weight`, `dataset.sample_weight` -- point at this
        builder's `labels_s`/`sample_weight_s`, so every args blob staged
        from this view (histogram, leaf, node split) carries the
        sampled-order pointers with no further plumbing. The ORIGINAL
        arrays are only read from, never written: the host OOB path keeps
        original order untouched.

        `sabotage` forwards to the gather kernel's CHECK HOOK (see the
        DEVIATION 2001 block in `builder_kernels_impl.mojo`); 0 is the
        only value a caller may pass.
        """
        comptime assert Self.sampled_labels, (
            "stage_sampled_order requires Builder[..., sampled_labels=True]"
            " (DEVIATION 2001)"
        )
        if Int(dataset.n_sampled_rows) != self.n_sampled_rows:
            raise Error(
                "stage_sampled_order: labels_s sized for "
                + String(self.n_sampled_rows)
                + " sampled rows; the view carries "
                + String(Int(dataset.n_sampled_rows))
            )
        var labels_s_ptr = (
            self.labels_s.value()
            .unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        var sample_weight_s_ptr = (
            self.sample_weight_s.value()
            .unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        launch_gather_sampled_order_kernel[sabotage=sabotage](
            ctx,
            dataset.labels,
            dataset.sample_weight,
            dataset.row_ids,
            labels_s_ptr,
            sample_weight_s_ptr,
            Int(dataset.n_sampled_rows),
            dataset.has_sample_weight,
        )
        dataset.labels = labels_s_ptr
        if dataset.has_sample_weight:
            dataset.sample_weight = sample_weight_s_ptr

    def _invalidate_args_cache(mut self):
        """DEVIATION 1909: the cached args blobs embed the CURRENT
        tree's dataset (row pointers, sampled-row count), so they are
        dropped wherever the dataset a drive hands down may change --
        `reset_for_tree` and every drive entry (`begin_tree`,
        `do_split`, `_compute_best_splits`). A reused builder can
        therefore never launch against a stale blob; the cost of the
        widest guard is two re-uploads per drive entry, paid only on
        the serial check arms."""
        self.args_cols_a = -1
        self.args_cols_b = -1
        self.node_split_args_ready = False

    def splits_capacity_bytes(self) -> Int:
        """One pipeline slot's worth of split staging -- what
        `SplitStaging` must reserve per adopted builder (DEVIATION
        1908)."""
        return size_of[Split[Self.O.DataT]]() * Int(self.params.max_batch_size)

    def adopt_shared_splits(
        mut self, staging: SplitStaging, slot: Int
    ) raises:
        """DEVIATION 1908: repoint this builder's split traffic at slot
        `slot` of a staging shared by every pipeline slot, so the forest
        loop can read all K slots back in one copy per cycle
        (`flush_splits_downloads`).

        The builder's own `wl.splits` / `wl.h_splits` carves go dormant;
        the arena keeps their layout, which is what `builder_check`
        compares to their sizing. Call after the constructor (whose
        trailing synchronize has drained the arena's enqueues) and before
        any tree work.

        AN ADOPTED BUILDER MUST BE DRIVEN BY A LOOP THAT FLUSHES: its
        `_enqueue_splits_download` only records a pending count, and only
        `flush_splits_downloads` (per cycle) or `_download_splits`' own
        flush actually moves the bytes. `fit_forest` is that loop; the
        serial `train`/`do_split` drives run on never-adopted builders,
        as every check constructs its own."""
        var cap = self.splits_capacity_bytes()
        if cap > staging.slot_stride:
            raise Error(
                "SplitStaging slot stride "
                + String(staging.slot_stride)
                + " smaller than this builder's splits capacity "
                + String(cap)
            )
        var off = slot * staging.slot_stride
        self.splits = staging.d.create_sub_buffer[DType.uint8](off, cap)
        self.h_splits = staging.h.create_sub_buffer[DType.uint8](off, cap)
        self.splits_d_view = _DevPrefixView(self.splits, cap)
        self.splits_h_view = _HostPrefixView(self.h_splits, cap)
        self.splits_shared = True
        self.pending_splits_bytes = 0

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
        """DEVIATION 1908: the CURRENT phase's workload map, which the
        packed upload put at `cur_wl_rel` bytes past the work items --
        inside the `d_work_items`/`workload_info` arena stretch, at a
        512-byte boundary. Valid between `_stage_work_items` and the next
        phase's staging; every launch captures the value by then."""
        return (
            self.d_work_items.unsafe_ptr()
            .unsafe_offset(self.cur_wl_rel)
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
        """`builder.cuh:592-596`, their four-argument construction:

            ObjectiveT objective(dataset.num_outputs,
                                 params.min_samples_leaf,
                                 params.split_criterion,
                                 params.min_impurity_decrease);

        Built FRESH from `params` on every call. Theirs constructs once
        per column block inside `computeSplit`; since DEVIATIONS
        1893/1909 the one caller is `enqueue_best_splits`, at the
        args-blob staging site -- once per tree per sampled-cols width.
        The value cannot move inside a fit, so the cadence is
        immaterial; `params` being the only source is the property worth
        keeping, not the call count."""
        return Self.O(
            self.num_outputs,
            self.params.min_samples_leaf,
            Int32(self.params.split_criterion),
            Scalar[Self.O.DataT](self.params.min_impurity_decrease),
            self.scales,
        )

    def _leaf_objective(self) -> Self.O:
        """`builder.cuh:642` -- THREE arguments, not four:

            ObjectiveT objective(dataset.num_outputs,
                                 params.min_samples_leaf,
                                 params.split_criterion);

        so `min_impurity_decrease` takes its default of `DataT{0}`
        (`objectives.cuh:142`, `:344`). The asymmetry with `_objective`
        above is theirs and is kept. It is inert -- `SetLeafVector` reads
        neither field -- but a reader diffing the two call sites should
        find the same difference here that is in their source."""
        return Self.O(
            self.num_outputs,
            self.params.min_samples_leaf,
            Int32(self.params.split_criterion),
            Scalar[Self.O.DataT](0),
            self.scales,
        )

    def _stage_work_items(mut self, items: List[NodeWorkItem]):
        """The pinned half of `raft::update_device(d_work_items,
        work_items.data(), work_items.size(), stream)` (`:466`, `:492`).
        DEVIATION 1908: the copy itself is deferred to
        `_enqueue_phase_upload`, which sends this phase's work items and
        its workload map as ONE transfer. THEIR COUNT stays
        `work_items.size()` -- the packed span's live bytes -- not the
        buffer's capacity; the workload block begins at the next 512-byte
        boundary (`ALIGN_VALUE`, the arena's own alignment) past the live
        items, so the device pointer the kernels get is as aligned as the
        old carve's."""
        var p = self.h_phase.unsafe_ptr().unsafe_bitcast[NodeWorkItem]()
        for i in range(len(items)):
            p[unsafe_offset=i] = items[i]
        self.cur_wl_rel = calculate_aligned_bytes(
            len(items) * size_of[NodeWorkItem]()
        )

    @always_inline
    def _h_workload_ptr(
        mut self,
    ) -> MutPointer[WorkloadInfo, MutUntrackedOrigin]:
        """The pinned array `update_workload_info` fills in place --
        upstream's `h_workload_info` member (`builder.cuh:198`).
        DEVIATION 1908: it lives in the packed phase span, directly after
        this phase's work items; call only after `_stage_work_items` has
        set `cur_wl_rel`."""
        return (
            self.h_phase.unsafe_ptr()
            .unsafe_offset(self.cur_wl_rel)
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[WorkloadInfo]()
        )

    def _enqueue_phase_upload(
        mut self, ctx: DeviceContext, n_blocks_dimx: Int
    ) raises:
        """DEVIATION 1908: ONE H2D copy for the phase's control plane.
        Their two `update_device`s (`:466` work items, `:405` workload)
        are cheap CUDA async ops; here each enqueue is host-priced, so
        the pair rides one packed span into the SAME
        `d_work_items`/`workload_info` arena stretch. Byte count is
        live-only, exactly the sum of theirs plus the alignment gap.
        Same bytes at the same-or-aligned addresses; no value moves."""
        var nbytes = self.cur_wl_rel + n_blocks_dimx * size_of[WorkloadInfo]()
        if nbytes == 0:
            return
        self.phase_view.ensure(self.d_buff, nbytes)
        log_launch("xfer_phase_upload")
        ctx.enqueue_copy(
            dst_buf=self.phase_view.view,
            src_ptr=self.h_phase.unsafe_ptr(),
        )

    def _copy_splits_now(mut self, ctx: DeviceContext, nbytes: Int) raises:
        """This builder's own splits readback -- the pre-1908 copy,
        shared by the private-mode path and the serial arm's flush."""
        # `:479` -- their count is `work_items.size()`.
        self.splits_h_view.ensure(self.h_splits, nbytes)
        self.splits_d_view.ensure(self.splits, nbytes)
        log_launch("xfer_splits_download")
        ctx.enqueue_copy(
            dst_buf=self.splits_h_view.view,
            src_buf=self.splits_d_view.view,
        )

    def _enqueue_splits_download(mut self, ctx: DeviceContext, n: Int) raises:
        """The COPY half of `raft::update_host(h_splits, splits, ...)`
        (`:479`, `:501`). The sync is the CALLER's, because in the
        pipelined forest loop (DEVIATION 117) one synchronize serves every
        in-flight tree's downloads at once.

        DEVIATION 1908: under a shared staging (`adopt_shared_splits`)
        nothing is enqueued here either -- the byte count is RECORDED and
        the forest loop's `flush_splits_downloads` sends every in-flight
        tree's slot as ONE copy per pipeline cycle."""
        var nbytes = n * size_of[Split[Self.O.DataT]]()
        if nbytes > 0:
            if self.splits_shared:
                self.pending_splits_bytes = nbytes
                return
            self._copy_splits_now(ctx, nbytes)

    def _read_splits(
        mut self, n: Int
    ) raises -> List[SplitSummary[Self.O.DataT]]:
        """The host read that follows the sync -- what `NodeQueue::Push`
        reads, projected onto the six fields it touches. Only valid after
        a `synchronize` that covers `_enqueue_splits_download`."""
        var p = self.h_splits.unsafe_ptr().unsafe_bitcast[Split[Self.O.DataT]]()
        var out = List[SplitSummary[Self.O.DataT]]()
        # The summaries are the return contract (NodeQueue::Push consumes
        # them); reserving up front removes the growth reallocs, which is
        # all the waste there was -- the reads are already in place.
        out.reserve(n)
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

    def _download_splits(
        mut self, ctx: DeviceContext, n: Int
    ) raises -> List[SplitSummary[Self.O.DataT]]:
        """`raft::update_host(h_splits, splits, ...)` + `sync_stream`,
        `:479-480`, `:676-677` -- the serial composition of the two
        halves above, kept so `train()`'s single-tree path reads exactly
        as their code does.

        DEVIATION 1908: on an ADOPTED builder the enqueue only records a
        pending count, so this serial arm flushes its own slot here --
        a check driving `train`/`do_split` on an adopted builder still
        reads real bytes, at the pre-1908 cadence."""
        self._enqueue_splits_download(ctx, n)
        if self.splits_shared and self.pending_splits_bytes > 0:
            var nbytes = self.pending_splits_bytes
            self.pending_splits_bytes = 0
            self._copy_splits_now(ctx, nbytes)
        ctx.synchronize()
        return self._read_splits(n)

    # `Builder::sampleFeatures` (`:505-520`) no longer has a method of its
    # own: DEVIATION 1916 fused it with `:489`'s initSplit and `:490`'s
    # mutex re-zero into `launch_phase_setup_kernel`'s one launch (see
    # `enqueue_best_splits`). The seed-split incident its docstring
    # carried -- two Int32 halves BY BIT PATTERN, `var`-bound, because the
    # chained conversion folds to one sign-extending cast -- lives on in
    # `launch_phase_setup_kernel` and `recombine_seed_halves`, and the
    # kernel `sample_features_kernel` itself stands (shuffle_check drives
    # it directly), sharing the same `sampled_column_at` body.

    def _compute_split(
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        col: Int,
        n_blocks_dimx: Int,
        n_work_items: Int,
        n_sampled_cols: Int,
        smem_config: SharedMemoryConfig,
        mut instr: FitInstruments,
        tag_prefix: String,
        hist_argsp: MutPointer[HistogramArgs[Self.O], MutUntrackedOrigin],
        find_argsp: MutPointer[
            FindBestSplitsArgs[Self.O], MutUntrackedOrigin
        ],
    ) raises:
        """`Builder::computeSplit`, `:570-626`. The quantiles and the
        objective travel in the staged args blobs -- per tree per width
        since DEVIATION 1909, selected by `enqueue_best_splits` and
        handed down here as `hist_argsp`/`find_argsp`; `dataset` remains
        for the `has_bins` dispatch.

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
        # `:588-591` -- theirs zeroes exactly `sizeof(BinT) *
        # len_histograms` bytes, where
        # `len_histograms = n_bins * n_classes * n_blocks_dimy *
        # n_work_items`. DEVIATION 304, REVISED: `enqueue_memset` takes a
        # BUFFER and not a range, so their byte count is expressed as a
        # prefix sub-buffer view. The launch reads and writes only that
        # prefix -- their memset size is the proof, since they never zero
        # past it -- so the bytes zeroed here are THEIR bytes, per launch.
        # (This used to zero the WHOLE workspace, sized for
        # `max_batch_size` nodes, every column block of every batch: at
        # the default `max_batch_size` 4096 that is a constant ~100x the
        # bytes their launch touches whenever the live batch is small.)
        var len_histograms = n_bins * n_classes * n_blocks_dimy * n_work_items
        self.hist_view.ensure(
            self.histograms, size_of[Self.O.BinT]() * len_histograms
        )
        ctx.enqueue_memset(self.hist_view.view, UInt8(0))

        # DEVIATION 1893/1909: the args blobs were staged by
        # `enqueue_best_splits` (once per tree per sampled-cols width);
        # the launchers take the device pointers. The objective their
        # `:592-596` builds per column block is built at the staging
        # site -- same value, `params` still the only source.
        launch_build_histograms_kernel[
            Self.O, sampled_labels = Self.sampled_labels
        ](
            ctx,
            self._hist_ptr(),
            n_bins,
            dataset,
            self._work_items_ptr(),
            col,
            self.column_samples.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            self._workload_ptr(),
            n_blocks_dimx,
            n_blocks_dimy,
            smem_config,
            hist_argsp,
        )
        # DEVIATION 401 -- the column block's REDUCED histograms, hashed
        # between the histogram kernel and the split kernel so the record
        # is the pdf the atomics produced (order-independent by
        # construction: integer atomics for classification, fixed-point
        # Int32 for regression -- DEVIATION 101). The hashed prefix is
        # EXACTLY the region their memset sizes (`:588-591`), so no
        # uninitialized tail is folded in (identity_trace rule 3); its
        # shape is (nodes x colblock x bins x outputs), all algorithm
        # quantities, never an SM count. Draining here is rule 4's price:
        # a traced run is a trace subject, never a timing.
        if instr.trace.enabled:
            instr.trace.record_device(
                ctx,
                tag_prefix + ".cols" + String(col) + ".hist",
                self.histograms,
                size_of[Self.O.BinT]() * len_histograms,
            )
        # DEVIATION 302: their `if (distributed) allReduceHistograms(...)`
        # sits exactly here (`:613`) and is unreachable on one device.
        launch_find_best_splits_kernel[Self.O](
            ctx,
            self._hist_ptr(),
            n_bins,
            col,
            self.column_samples.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            self.mutex.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            self._splits_ptr(),
            n_work_items,
            n_blocks_dimy,
            find_argsp,
        )

    def enqueue_best_splits(
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
        work_items: List[NodeWorkItem],
        sample_offset: Int,
        n_sampled_cols: Int,
        smem_config: SharedMemoryConfig,
        mut instr: FitInstruments,
        tag_prefix: String,
    ) raises:
        """`Builder::computeBestSplits`, `:485-503`, in their order --
        MINUS the trailing sync, which belongs to the caller so the
        pipelined forest loop (DEVIATION 117) can share one synchronize
        across every in-flight tree. `_compute_best_splits` below is the
        serial composition."""
        var n = len(work_items)
        # DEVIATION 1916: `:489`'s initSplit, `:490`'s mutex re-zero and
        # `:505-520`'s sampleFeatures now ride ONE fused launch, enqueued
        # below AFTER the phase upload (the feature sample reads the
        # uploaded work items; the other two writes have no reader before
        # `find_best_splits`, so the later position is equivalent to the
        # old one for every reader on the in-order queue).
        # DEVIATION 1908: stage both, then ONE packed upload.
        self._stage_work_items(work_items)
        # `:393-407` -- straight into the pinned array, as theirs.
        # DEVIATION 2011: the granularity constant folds to TPB_DEFAULT
        # under the default flag, i.e. this line IS theirs until the
        # flag flips; under R > 1 the table is coarser and its only
        # reader on this phase is the histogram launch (see the
        # deviation block by the flag).
        var n_blocks_dimx = update_workload_info(
            work_items, self._h_workload_ptr(), HIST_WORKLOAD_GRANULARITY
        )
        self._enqueue_phase_upload(ctx, n_blocks_dimx)

        # DEVIATION 1893/1909: the two split kernels' argument blobs are
        # a pure function of this tree's dataset and the round's
        # `n_sampled_cols` (`_enqueue_round` narrows it; the objective is
        # a pure function of `params`, their `:592-596`) -- at most two
        # widths per tree, so each width uploads once per tree into its
        # own slot and every later round with that width re-hands the
        # device pointer. Overwriting a slot is safe at this cadence:
        # this builder's previous phase drained at the caller's last
        # synchronize before a new enqueue can reach here.
        var cols = Int(dataset.n_sampled_cols)
        var hist_argsp: MutPointer[HistogramArgs[Self.O], MutUntrackedOrigin]
        var find_argsp: MutPointer[
            FindBestSplitsArgs[Self.O], MutUntrackedOrigin
        ]
        if cols == self.args_cols_a:
            hist_argsp = self.hist_args.device_ptr()
            find_argsp = self.find_args.device_ptr()
        elif cols == self.args_cols_b:
            hist_argsp = self.hist_args_alt.device_ptr()
            find_argsp = self.find_args_alt.device_ptr()
        else:
            var objective = self._objective()
            if self.args_cols_a == -1:
                self.args_cols_a = cols
                hist_argsp = self.hist_args.upload(
                    ctx,
                    HistogramArgs[Self.O](
                        dataset.copy(), quantiles.copy(), objective.copy()
                    ),
                )
                find_argsp = self.find_args.upload(
                    ctx,
                    FindBestSplitsArgs[Self.O](
                        dataset.copy(), quantiles.copy(), objective.copy()
                    ),
                )
            else:
                # A third width cannot occur on the fit paths (see the
                # field block); a sabotage arm that forces one lands
                # here and simply restages the alt slot.
                self.args_cols_b = cols
                hist_argsp = self.hist_args_alt.upload(
                    ctx,
                    HistogramArgs[Self.O](
                        dataset.copy(), quantiles.copy(), objective.copy()
                    ),
                )
                find_argsp = self.find_args_alt.upload(
                    ctx,
                    FindBestSplitsArgs[Self.O](
                        dataset.copy(), quantiles.copy(), objective.copy()
                    ),
                )

        # DEVIATION 1916 -- the fused setup launch: initSplit over the
        # live `n`, the mutex over max_batch_size, the feature sample
        # over `n * n_sampled_cols`. One op where the port paid three
        # (two kernels + one memset) per sampling round.
        launch_phase_setup_kernel[Self.O.DataT](
            ctx,
            self._splits_ptr(),
            n,
            self.mutex.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            Int(self.params.max_batch_size),
            self.column_samples.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin](),
            self._work_items_ptr(),
            n * n_sampled_cols,
            self.treeid,
            self.seed,
            sample_offset,
            self.n_cols,
            n_sampled_cols,
        )
        # DEVIATION 401 (added with the 2026-08-22 identity audit) -- the
        # round's sampled COLUMNS, the per-node feature-sample RNG's
        # (`fnv1a32_hash(seed, treeid, nodeid)` -> minstd_rand ->
        # shuffle_iterator) only output. Without this tag an RNG
        # divergence would first surface in the histograms, one stage
        # late; with it the cross-vendor bisection separates "sampled
        # different columns" from "histogrammed the same columns
        # differently". The hashed prefix is the LIVE region
        # (n work items x this round's sampled columns), a pure algorithm
        # shape; the buffer's tail is capacity (rule 3). Draining here is
        # rule 4's price -- a traced run is never a timing.
        if instr.trace.enabled:
            instr.trace.record_device(
                ctx,
                tag_prefix + ".colsamples",
                self.column_samples,
                n * n_sampled_cols,
            )

        # `:497-500` -- ten columns per launch.
        var c = 0
        while c < n_sampled_cols:
            self._compute_split(
                ctx, dataset, c, n_blocks_dimx, n,
                n_sampled_cols, smem_config, instr, tag_prefix,
                hist_argsp, find_argsp,
            )
            c += N_BLKS_FOR_COLS
        self._enqueue_splits_download(ctx, n)

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
        """The serial composition: enqueue, drain, read -- exactly their
        `computeBestSplits` including `:479-480`'s `update_host` +
        `sync_stream`. Instruments are DISABLED here (DEVIATION 401): the
        callers of this serial arm are checks, and a check must not change
        behavior under an exported trace variable."""
        # DEVIATION 1909 -- serial entry with a caller-supplied dataset:
        # never trust an earlier drive's cached blobs.
        self._invalidate_args_cache()
        var instr = FitInstruments.disabled()
        self.enqueue_best_splits(
            ctx, dataset, quantiles, work_items, sample_offset,
            n_sampled_cols, smem_config, instr, String("untraced"),
        )
        ctx.synchronize()
        return self._read_splits(len(work_items))

    def _enqueue_round[
        sabotage: Int = 0
    ](
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
        smem_config: SharedMemoryConfig,
        mut st: BatchState[Self.O],
        mut instr: FitInstruments,
    ) raises:
        """One sampling round's enqueue, `:437-450`."""
        # DEVIATION 401 -- the tag names a position in the algorithm:
        # which tree, which queue pop, which sampling round. Built only
        # under an enabled trace, so the shipping path pays no String.
        var tag_prefix = String("")
        if instr.trace.enabled:
            tag_prefix = (
                "tree"
                + String(self.treeid)
                + ".batch"
                + String(self.trace_batch)
                + ".round"
                + String(st.round)
            )
        var sample_offset = st.round * self.original_n_sampled_cols
        var n_sampled_cols = sampled_cols_in_round(
            self.n_cols, self.original_n_sampled_cols, st.round
        )
        # `:439-440` -- they NARROW the dataset field for this round:
        #     dataset.n_sampled_cols =
        #         min(original_n_sampled_cols, n_cols - sample_offset);
        # The last round is short whenever n_sampled_cols does not
        # divide n_cols, and the kernels' stride must follow it.
        var ds = dataset.copy()
        ds.n_sampled_cols = Int32(n_sampled_cols)
        # CHECK HOOK. 8 restores the pre-fix value -- the DatasetView
        # field left at whatever the caller passed, i.e. `n_cols` -- so
        # a check can watch the writer/reader strides come apart
        # instead of taking the fix on trust. THIS is the load-bearing
        # write: a hook on the `train` assignment alone is inert,
        # because this line overwrites it every round.
        comptime if sabotage == 8:
            ds.n_sampled_cols = Int32(self.n_cols)
        self.enqueue_best_splits(
            ctx, ds, quantiles, st.active_items, sample_offset,
            n_sampled_cols, smem_config, instr, tag_prefix,
        )

    def _record_splits(
        self,
        mut instr: FitInstruments,
        tag: String,
        splits: List[SplitSummary[Self.O.DataT]],
    ) raises:
        """DEVIATION 401 -- a batch of `SplitSummary` as an identity-trace
        record, FIELD BY FIELD rather than raw struct bytes: a host
        struct's padding bytes are unspecified, and hashing them would
        report divergence where there is none (the same failure rule 3
        exists to prevent, from the host side). Every field travels as bit
        patterns in u32 lanes; 64-bit values are split low-then-high with
        an explicit mask, because Int widening sign-extends
        (`[[mojo-int-widening-sign-extends]]`)."""
        if not instr.trace.enabled:
            return
        var flat = List[UInt32]()
        for i in range(len(splits)):
            ref s = splits[i]
            flat.append(UInt32(1) if s.is_valid else UInt32(0))
            flat.append(UInt32(Int(s.colid) & 0xFFFFFFFF))
            var qb = UInt64(s.quesval.to_bits())
            flat.append(UInt32(qb & 0xFFFFFFFF))
            flat.append(UInt32(qb >> 32))
            var mb = UInt64(s.best_metric_val.to_bits())
            flat.append(UInt32(mb & 0xFFFFFFFF))
            flat.append(UInt32(mb >> 32))
            var g = s.global_n_left.cast[DType.uint64]()
            flat.append(UInt32(g & 0xFFFFFFFF))
            flat.append(UInt32(g >> 32))
            var l = s.local_n_left.cast[DType.uint64]()
            flat.append(UInt32(l & 0xFFFFFFFF))
            flat.append(UInt32(l >> 32))
        instr.trace.record_host(tag, flat.unsafe_ptr(), len(flat))
        # `[[mojo-buffer-freed-at-last-use]]`: `.unsafe_ptr()` above would
        # otherwise be `flat`'s last use, freeing it under the hash.
        _ = flat^

    def begin_batch[
        sabotage: Int = 0
    ](
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
        work_items: List[NodeWorkItem],
        smem_config: SharedMemoryConfig,
        mut instr: FitInstruments,
    ) raises -> BatchState[Self.O]:
        """`doSplit`'s head (`:410-437`) plus round zero's enqueue. The
        returned state is pending a synchronize."""
        # DEVIATION 401 -- the batch takes its 0-based index within the
        # tree. Incremented unconditionally so the numbering is the same
        # whether or not this run is traced.
        self.trace_batch += 1
        var n = len(work_items)
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
        var st = BatchState[Self.O](
            work_items.copy(),
            active_items^,
            active_to_original^,
            final_splits^,
            0,
            max_rounds,
            0,
            List[SplitSummary[Self.O.DataT]](),
        )
        self._enqueue_round[sabotage](
            ctx, dataset, quantiles, smem_config, st, instr
        )
        return st^

    def advance_batch[
        sabotage: Int = 0
    ](
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
        smem_config: SharedMemoryConfig,
        mut st: BatchState[Self.O],
        mut instr: FitInstruments,
    ) raises -> Bool:
        """`doSplit`'s consume side: the retry logic of `:452-458` after a
        best-splits sync, or the final read of `:501` after the node-split
        sync. Call ONLY after a synchronize covering this builder's
        enqueues. Returns True when the batch is finished, with the pushed
        splits in `st.result`."""
        if st.phase == 1:
            st.result = self._read_splits(len(st.work_items))
            # DEVIATION 401 -- the batch's splits as the PARTITION read
            # them back: chosen column, threshold bits, child counts.
            # This is "after split selection" for the whole batch.
            if instr.trace.enabled:
                self._record_splits(
                    instr,
                    "tree"
                    + String(self.treeid)
                    + ".batch"
                    + String(self.trace_batch)
                    + ".splits",
                    st.result,
                )
            return True
        var h = self._read_splits(len(st.active_items))
        # DEVIATION 401 -- the round's CANDIDATE splits, before the retry
        # dispatch, so a divergence is pinned to a sampling round rather
        # than to the batch's final answer.
        if instr.trace.enabled:
            self._record_splits(
                instr,
                "tree"
                + String(self.treeid)
                + ".batch"
                + String(self.trace_batch)
                + ".round"
                + String(st.round)
                + ".cand",
                h,
            )
        var retry_items = List[NodeWorkItem]()
        var retry_to_original = List[Int]()
        for i in range(len(st.active_items)):
            var orig = st.active_to_original[i]
            st.final_splits[orig] = h[i]
            if not h[i].is_valid:
                retry_items.append(st.active_items[i])
                retry_to_original.append(orig)
        # Their `:456`: `if (round + 1 >= max_sampling_rounds) break;` --
        # the LAST round does not bother rebuilding the active set.
        if len(retry_items) > 0 and st.round + 1 < st.max_rounds:
            st.active_items = retry_items^
            st.active_to_original = retry_to_original^
            st.round += 1
            self._enqueue_round[sabotage](
                ctx, dataset, quantiles, smem_config, st, instr
            )
            return False
        # DEVIATION 1919 -- at round 0 the device span already holds this
        # batch's items and block map (round zero staged `active_items`,
        # a copy of `work_items`, and no retry restaged a subset).
        # DEVIATION 2011: under R > 1 the staged map is HISTOGRAM
        # granularity, which the partition scan's TPB-granular slot math
        # must never read -- the reuse is off and every batch restages
        # (priced in the 2011 block). Comptime-folds to `st.round == 0`
        # under the default flag.
        self.enqueue_node_split(
            ctx,
            dataset,
            st.work_items,
            st.final_splits,
            reuse_phase_span=(st.round == 0 and HIST_ITEMS_PER_THREAD == 1),
        )
        st.phase = 1
        return False

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

        See the module docstring for why the rounds exist. Since the
        pipelined forest loop (DEVIATION 117) this is the SERIAL DRIVE of
        `begin_batch`/`advance_batch` -- same operations, same order, same
        sync count (one per round plus one for the partition), verified
        bit-identical by the fingerprint probe when the split was made.
        Instruments are DISABLED (DEVIATION 401): this arm serves checks,
        which must not change behavior under an exported trace variable.
        """
        # DEVIATION 1909 -- serial entry with a caller-supplied dataset:
        # never trust an earlier drive's cached blobs.
        self._invalidate_args_cache()
        var instr = FitInstruments.disabled()
        var st = self.begin_batch[sabotage](
            ctx, dataset, quantiles, work_items, smem_config, instr
        )
        while True:
            ctx.synchronize()
            if self.advance_batch[sabotage](
                ctx, dataset, quantiles, smem_config, st, instr
            ):
                var out = st.result.copy()
                return out^

    def enqueue_node_split(
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        work_items: List[NodeWorkItem],
        final_splits: List[SplitSummary[Self.O.DataT]],
        reuse_phase_span: Bool = False,
    ) raises:
        """The partition half of `doSplit`, `:458-501`, MINUS the trailing
        sync -- the caller's, per the pipelined forest loop (DEVIATION
        117).

        DEVIATION 1919: `reuse_phase_span=True` asserts the device's
        packed [items | workload] span ALREADY holds exactly this
        `work_items` list's staging, so the restage and its
        `xfer_phase_upload` are skipped. The ONE caller that may pass
        True is `advance_batch`, and only when `st.round == 0`: round
        zero staged `st.active_items`, which `begin_batch` made a copy
        of this very `st.work_items`, and no retry round restaged a
        subset in between -- so the span's bytes, `cur_wl_rel`, and the
        block map are byte-for-byte what restaging would produce (and
        nothing else writes the `d_work_items`/`workload_info` stretch:
        the splits upload and 1916's fused setup write other regions).
        Any retry (`st.round > 0`) restages, because the last staging
        was the shrunken active set. The grid size is recomputed by
        `workload_blocks_for`, the same arithmetic `update_workload_info`
        runs, sans writes."""
        var n = len(work_items)
        # `:458` -- `dataset.n_sampled_cols = original_n_sampled_cols;`
        var ds = dataset.copy()
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
        # `:462-466` -- `sizeof(SplitT) * work_items.size()`, not the
        # buffer. NOT foldable into the phase span (DEVIATION 1908): the
        # partition kernels read AND update these splits in place and the
        # batch's final download reads them back, so the upload's
        # destination must be the `splits` region itself -- in shared
        # mode, this builder's `SplitStaging` slot.
        var sp_bytes = n * size_of[Split[Self.O.DataT]]()
        if sp_bytes > 0:
            self.splits_d_view.ensure(self.splits, sp_bytes)
            log_launch("xfer_splits_upload")
            ctx.enqueue_copy(
                dst_buf=self.splits_d_view.view,
                src_ptr=self.h_splits.unsafe_ptr(),
            )
        # DEVIATION 1908: stage both, then ONE packed upload -- unless
        # DEVIATION 1919 proved the span is already there (see the
        # docstring).
        var n_partition_blocks: Int
        if reuse_phase_span:
            n_partition_blocks = workload_blocks_for(work_items)
        else:
            self._stage_work_items(work_items)
            # `:393-407` -- straight into the pinned array, as theirs.
            n_partition_blocks = update_workload_info(
                work_items, self._h_workload_ptr()
            )
            self._enqueue_phase_upload(ctx, n_partition_blocks)

        # DEVIATION 1893/1909: the partition's args blob (dataset only,
        # `n_sampled_cols` restored to `original_n_sampled_cols` above)
        # is the same bytes for every batch of a tree -- staged on the
        # tree's first partition, re-handed until the args cache is
        # invalidated (`reset_for_tree` / the drive entries).
        if not self.node_split_args_ready:
            _ = self.node_split_args.upload(ctx, NodeSplitArgs(ds.copy()))
            self.node_split_args_ready = True
        var argsp = self.node_split_args.device_ptr()
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
            argsp,
        )
        self._enqueue_splits_download(ctx, n)

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
        # DEVIATION 313: the staging is pooled on the builder and grows
        # only when a tree outgrows every earlier one. At entry the
        # previous tree's leaf pass has drained (its own final
        # `synchronize` below), so reassigning the buffers cannot free
        # memory under an in-flight launch. The DEVICE trio is
        # batch-sized, as their `d_tree`/`d_instance_ranges`/`d_leaves`
        # are (`builder.cuh:638-641`); the HOST trio is tree-sized,
        # because upstream stages every batch out of the FULL-SIZE host
        # vectors (`:648-651`, `:663-666`) -- that per-batch-disjoint
        # host staging is what lets the batches below enqueue with no
        # sync between them.
        if batch > self.leaf_capacity:
            self.leaf_capacity = batch
            self.leaf_d_tree = ctx.enqueue_create_buffer[DType.uint8](
                size_of[SparseTreeNode[Self.O.DataT]]() * batch
            )
            self.leaf_d_ranges = ctx.enqueue_create_buffer[DType.uint8](
                size_of[InstanceRange]() * batch
            )
            self.leaf_d_leaves = ctx.enqueue_create_buffer[Self.O.DataT](
                batch * n_out
            )
        if n_nodes > self.leaf_host_capacity:
            self.leaf_host_capacity = n_nodes
            self.leaf_h_tree = ctx.enqueue_create_host_buffer[DType.uint8](
                size_of[SparseTreeNode[Self.O.DataT]]() * n_nodes
            )
            self.leaf_h_ranges = ctx.enqueue_create_host_buffer[
                DType.uint8
            ](size_of[InstanceRange]() * n_nodes)
            self.leaf_h_leaves = ctx.enqueue_create_host_buffer[
                Self.O.DataT
            ](n_nodes * n_out)
        var objective = self._leaf_objective()
        # `:652` -- their smem is `sizeof(BinT) * num_outputs`.
        var smem_size = size_of[Self.O.BinT]() * n_out

        # Stage EVERY node once -- the pinned trio plays their full-size
        # host vectors' part, so each batch's H2D reads a disjoint
        # region and no batch waits for an earlier one's copy to drain.
        var tp = self.leaf_h_tree.unsafe_ptr().unsafe_bitcast[
            SparseTreeNode[Self.O.DataT]
        ]()
        var rp = (
            self.leaf_h_ranges.unsafe_ptr().unsafe_bitcast[InstanceRange]()
        )
        for i in range(n_nodes):
            tp[unsafe_offset=i] = tree.sparsetree[i]
            rp[unsafe_offset=i] = instance_ranges[i]

        # DEVIATION 1893: the leaf args blob (objective + dataset),
        # staged once per tree and shared by every batch's launch.
        var leaf_argsp = self.leaf_args.upload(
            ctx, LeafArgs[Self.O](objective.copy(), dataset.copy())
        )

        # Per-batch views, kept alive past the drain below.
        var d_views = List[DeviceBuffer[DType.uint8]]()
        var l_views = List[DeviceBuffer[Self.O.DataT]]()
        var h_views = List[HostBuffer[Self.O.DataT]]()

        var begin = 0
        while begin < n_nodes:
            var end = min(begin + batch, n_nodes)
            var size = end - begin
            # Prefix views keep the copied and zeroed byte counts at
            # THEIRS -- `update_device(..., batch size)` at `:655-658` and
            # `sizeof(DataT) * d_leaves.size()` at `:652-653` -- now that
            # the pooled buffers can be larger than the live batch. The
            # device trio is REUSED by every batch with no sync: the
            # queue is in-order, so batch N+1's uploads execute after
            # batch N's kernel and download, exactly the stream ordering
            # their reuse of `d_tree` relies on.
            var dt = self.leaf_d_tree.create_sub_buffer[DType.uint8](
                0, size_of[SparseTreeNode[Self.O.DataT]]() * size
            )
            log_launch("xfer_leaf_tree")
            ctx.enqueue_copy(
                dst_buf=dt,
                src_ptr=self.leaf_h_tree.unsafe_ptr().unsafe_offset(
                    begin * size_of[SparseTreeNode[Self.O.DataT]]()
                ),
            )
            var dr = self.leaf_d_ranges.create_sub_buffer[DType.uint8](
                0, size_of[InstanceRange]() * size
            )
            log_launch("xfer_leaf_ranges")
            ctx.enqueue_copy(
                dst_buf=dr,
                src_ptr=self.leaf_h_ranges.unsafe_ptr().unsafe_offset(
                    begin * size_of[InstanceRange]()
                ),
            )
            # DEVIATION 1918 -- their zero fill at `:659-660` (this used
            # to be an `enqueue_memset` over the batch's slots) is fused
            # into the leaf kernel: each block zeroes its OWN node's slot
            # before the IsLeaf early-return, so internal nodes' slots
            # end 0 exactly as the memset left them and a leaf's zeros
            # are overwritten under the block's own barriers. One block
            # owns one slot; no cross-block order exists.
            launch_leaf_kernel[
                Self.O, zero_fill=True, sampled_labels = Self.sampled_labels
            ](
                ctx,
                self.leaf_d_tree.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
                .unsafe_bitcast[SparseTreeNode[Self.O.DataT]](),
                self.leaf_d_ranges.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
                .unsafe_bitcast[InstanceRange](),
                self.leaf_d_leaves.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin](),
                size,
                smem_size,
                leaf_argsp,
            )
            # `:663-666` -- downloaded to THIS batch's slice of the
            # host staging (their `tree->vector_leaf.data() +
            # batch_begin * num_outputs`), so no batch overwrites
            # another's landing zone.
            var hl = self.leaf_h_leaves.create_sub_buffer[Self.O.DataT](
                begin * n_out, size * n_out
            )
            var dls = self.leaf_d_leaves.create_sub_buffer[Self.O.DataT](
                0, size * n_out
            )
            log_launch("xfer_leaf_download")
            ctx.enqueue_copy(dst_buf=hl, src_buf=dls)
            d_views.append(dt^)
            d_views.append(dr^)
            l_views.append(dls^)
            h_views.append(hl^)
            begin = end

        # RESTORES upstream: their `SetLeafPredictions` carries NO sync
        # inside the batch loop (`builder.cuh:643-667`); this used to
        # drain the whole device -- all K pipelined trees -- once per
        # batch. One drain covers every batch, then the host reads.
        ctx.synchronize()
        _ = d_views^
        _ = l_views^
        _ = h_views^
        for i in range(n_nodes * n_out):
            tree.vector_leaf[i] = (
                self.leaf_h_leaves.unsafe_ptr().unsafe_load(i)
            )

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
        # (The fix lives in `begin_tree` now; this docstring keeps the
        # incident where a reader of `train` will find it.)
        #
        # Since the pipelined forest loop (DEVIATION 117) this is the
        # SERIAL DRIVE of `begin_tree`/`advance_tree` -- same operations,
        # same order, same sync count, verified bit-identical by the
        # fingerprint probe when the split was made. Instruments are
        # DISABLED (DEVIATION 401): this arm serves checks, which must not
        # change behavior under an exported trace variable. Traced fits go
        # through `fit_forest`.
        var instr = FitInstruments.disabled()
        var ts = self.begin_tree[sabotage](ctx, dataset, quantiles, instr)
        while not ts.done:
            ctx.synchronize()
            _ = self.advance_tree[sabotage](ctx, quantiles, ts, instr)
        var tree = ts.tree.copy()
        return tree^

    def begin_tree[
        sabotage: Int = 0
    ](
        mut self,
        ctx: DeviceContext,
        dataset: DatasetView[Self.O.DataT, Self.O.LabelT],
        quantiles: Quantiles[Self.O.DataT],
        mut instr: FitInstruments,
    ) raises -> TreeState[Self.O]:
        """`Builder::train`'s head: the `n_sampled_cols` correction (their
        ctor's `:240` -- see `train`'s docstring for the incident it
        fixed), the queue, and the first batch's enqueue. If the root is
        not expandable (max_depth 0, min_samples_split larger than the
        node) the tree finishes HERE, leaf pass included -- check `done`
        before waiting on a sync for it."""
        # DEVIATION 1909 -- this dataset is the one the cached args
        # blobs will embed; drop whatever an earlier drive staged.
        self._invalidate_args_cache()
        var ds = dataset.copy()
        ds.n_sampled_cols = Int32(self.original_n_sampled_cols)
        var smem_config = self.shared_memory_config()
        var queue = NodeQueue[Self.O.DataT](
            self.params,
            max_nodes(self.params.max_depth),
            self.n_sampled_rows,
            self.num_outputs,
        )
        var ts = TreeState[Self.O](
            queue^,
            ds^,
            smem_config,
            BatchState[Self.O](
                List[NodeWorkItem](),
                List[NodeWorkItem](),
                List[Int](),
                List[SplitSummary[Self.O.DataT]](),
                0,
                0,
                0,
                List[SplitSummary[Self.O.DataT]](),
            ),
            TreeMetaDataNode[Self.O.DataT](
                Int32(-1),
                Int32(0),
                Int32(0),
                Float64(0),
                List[Scalar[Self.O.DataT]](),
                List[SparseTreeNode[Self.O.DataT]](),
                Int32(0),
            ),
            False,
        )
        if ts.queue.has_work():
            var work_items = ts.queue.pop()
            ts.batch = self.begin_batch[sabotage](
                ctx, ts.ds, quantiles, work_items, ts.smem_config, instr
            )
        else:
            self._finish_tree(ctx, ts, instr)
        return ts^

    def advance_tree[
        sabotage: Int = 0
    ](
        mut self,
        ctx: DeviceContext,
        quantiles: Quantiles[Self.O.DataT],
        mut ts: TreeState[Self.O],
        mut instr: FitInstruments,
    ) raises -> Bool:
        """One consume step of `Builder::train`'s loop (`:375-389`). Call
        ONLY after a synchronize covering this builder's enqueues. Returns
        True when the TREE is done -- the leaf pass runs inside that final
        step and carries its own syncs, exactly as the serial train did."""
        if ts.done:
            return True
        if not self.advance_batch[sabotage](
            ctx, ts.ds, quantiles, ts.smem_config, ts.batch, instr
        ):
            return False
        ts.queue.push(ts.batch.work_items, ts.batch.result)
        if ts.queue.has_work():
            var work_items = ts.queue.pop()
            ts.batch = self.begin_batch[sabotage](
                ctx, ts.ds, quantiles, work_items, ts.smem_config, instr
            )
            return False
        self._finish_tree(ctx, ts, instr)
        return True

    def _finish_tree(
        mut self,
        ctx: DeviceContext,
        mut ts: TreeState[Self.O],
        mut instr: FitInstruments,
    ) raises:
        """`train`'s tail: `GetTree` + `SetLeafPredictions` (`:386-388`).
        DEVIATION 402 times the leaf pass here, where it runs, because
        `set_leaf_predictions` carries its own per-batch syncs and its
        wall time is real device work."""
        var tree = ts.queue.tree.copy()
        tree.treeid = self.treeid
        var t0 = instr.times.start()
        self.set_leaf_predictions(
            ctx, tree, ts.queue.node_instances_.copy(), ts.ds
        )
        instr.times.stop(ctx, "leaf_values", t0)
        ts.tree = tree^
        ts.done = True
