# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The state a growing tree carries between its two phases.

PORT OF `TPointsSubsets`, `TLeaf` and `TBestSplitProperties` from
`catboost/cuda/methods/greedy_subsets_searcher/split_properties_helper.h:40`
and `catboost/cuda/gpu_data/gpu_structures.h:64` at CatBoost `54a8143a`.
Transliterated. Do not improve.

Every function in their level loop takes `TPointsSubsets*`. It is the reason
`ComputeOptimalSplits` and `SplitLeaves` are two functions and not one long
body, and porting it is what lets ours be two functions as well.

Their comments on the fields are kept verbatim where they had them, because
they say which buffer is already reduced and which is not.
"""

from max.gpu.host import DeviceBuffer, HostBuffer

from gbdt.data.leaf_path import TLeafPath


struct TBestSplitProperties(Copyable, ImplicitlyCopyable, Movable):
    """`gpu_structures.h:64`. Their defaults, exactly.

    `FeatureId` defaults to `(ui32)-1` and `ComputeOptimalSplits` raises on
    it, which is their check that a level found any finite score at all
    (`greedy_search_helper.cpp:534`).
    """

    var feature_id: Int32
    var bin_id: Int32
    var score: Float32
    var gain: Float32
    var defined: Bool

    def __init__(out self):
        self.feature_id = -1
        self.bin_id = 0
        self.score = Float32.MAX
        self.gain = Float32.MAX
        self.defined = False

    def __init__(
        out self,
        feature_id: Int32,
        bin_id: Int32,
        score: Float32,
        gain: Float32,
    ):
        self.feature_id = feature_id
        self.bin_id = bin_id
        self.score = score
        self.gain = gain
        self.defined = True


struct EHistogramsType(Copyable, ImplicitlyCopyable, Movable):
    """Their `EHistogramsType`. Which histogram a leaf already holds.

    The sentence that used to sit here, "not yet consulted: our
    `build_necessary_histograms` rebuilds every level rather than reusing a
    parent's", is FALSE and is deleted. `run_tree_layout` consults it every
    level through `split_properties_helper.build_necessary_histograms`, which
    is what builds the smaller sibling and derives the larger.

    What is still true is that the LIVE path carries these three states on
    `split_properties_helper.LeafRecord` and not on the `TLeaf` below, so
    this enum is the transliterated shape and `LeafRecord.histograms_type`
    is the field actually read.
    """

    var value: Int32

    comptime Zeroes = Self(0)
    comptime PreviousPath = Self(1)
    comptime CurrentPath = Self(2)

    def __init__(out self, value: Int32):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value


struct TLeaf(Copyable, Movable):
    """`split_properties_helper.h`, their `TLeaf`, same five fields."""

    var size: Int
    """Their `Leaf.Size`, the ROW COUNT of the leaf's partition, summed over
    devices by `RebuildLeavesSizes` (`split_properties_helper.cpp:800-812`)
    and by `FastUpdateLeavesSizes` on the single-leaf split path (`:815-828`).

    **Two decisions read it and both are silent when it is stale.** The
    smaller-of-the-pair choice at `:1318` picks which sibling to build and
    which to derive, and the `NonZeroLeaves` / `ZeroLeaves` partition at
    `:1342-1344` decides which leaves get a histogram built at all. A stale
    size makes the first slow and the second WRONG.

    Their maintenance points, all of them: `CreateInitialSubsets` (`:1078`),
    the multi-leaf split (`:950`), the single-leaf split (`:1031`), and the
    lossguide fast path (`:948`)."""

    var histograms_type: EHistogramsType
    var best_split: TBestSplitProperties
    var is_terminal: Bool
    var path: TLeafPath
    """Their `Path`, the whole `TLeafPath`.

    THIS WAS AN `Int depth` UNTIL 2026-08-22, with the note "when the model
    builder lands, this becomes the path". The model builder landed
    (`greedy_subsets_searcher/model_builder.mojo`, the depthwise lane), so
    the note is now the change: `get_depth()` below is their
    `Path.GetDepth()` and nothing carries a second copy of the depth.

    An OBLIVIOUS tree does not need the path -- every leaf of a symmetric
    level shares one split list, which is why the substitution went
    unnoticed for three days. A DEPTHWISE tree does: two leaves at the same
    depth were split on different features, and the model builder rebuilds
    the node tree from these paths and from nothing else."""

    def __init__(out self):
        self.size = 0
        self.histograms_type = EHistogramsType.Zeroes
        self.best_split = TBestSplitProperties()
        self.is_terminal = False
        self.path = TLeafPath()

    def get_depth(self) -> Int:
        """Their `Path.GetDepth()`, which `IsTerminalLeaf` compares against
        `MaxDepth` (`greedy_search_helper.cpp:686`) and `FindMaxDepth` maxes
        over (`:311-317`)."""
        return self.path.get_depth()

    def update_best_split(mut self, best: TBestSplitProperties):
        """Their `UpdateBestSplit`."""
        self.best_split = best


struct TPointsSubsets(Movable):
    """`split_properties_helper.h:40`, their `TPointsSubsets`.

    Their field comments, kept:

        Partitions      how to access this leaves
        PartitionsCpu   this leaf sizes
        PartitionStats  sum of stats in leaves for each devices
        Histograms      stripped between devices final histograms
                        (already reduced)

    ================= DEVIATION BLOCK =================
    `PartitionsCpu` is a plain host buffer here, not their pinned
    `EPtrType::CudaHost`. Measured 2026-08-19: a Mojo kernel handed an
    `enqueue_create_host_buffer` pointer writes nothing, silently, and
    `map_to_host` is 2x slower than the copy it would replace. So this one
    is filled by an explicit copy. See `gbdt/gpu_lib/NOT_PORTED.md`.

    Their `Partitions` is one `TDataPartition` array of `{Offset, Size}`
    pairs; ours is two parallel `UInt32` buffers, which is how
    `gbdt/gpu_util/gpu_data/partitions.mojo` already had it.
    ===================================================
    """

    var partitions_offset: DeviceBuffer[DType.uint32]
    var partitions_size: DeviceBuffer[DType.uint32]
    var partitions_cpu_offset: HostBuffer[DType.uint32]
    var partitions_cpu_size: HostBuffer[DType.uint32]
    var partition_stats: DeviceBuffer[DType.float32]
    var histograms: DeviceBuffer[DType.float32]
    var leaves: List[TLeaf]
    var stat_count: Int

    def __init__(
        out self,
        var partitions_offset: DeviceBuffer[DType.uint32],
        var partitions_size: DeviceBuffer[DType.uint32],
        var partitions_cpu_offset: HostBuffer[DType.uint32],
        var partitions_cpu_size: HostBuffer[DType.uint32],
        var partition_stats: DeviceBuffer[DType.float32],
        var histograms: DeviceBuffer[DType.float32],
        stat_count: Int,
    ):
        self.partitions_offset = partitions_offset^
        self.partitions_size = partitions_size^
        self.partitions_cpu_offset = partitions_cpu_offset^
        self.partitions_cpu_size = partitions_cpu_size^
        self.partition_stats = partition_stats^
        self.histograms = histograms^
        self.leaves = List[TLeaf]()
        self.leaves.append(TLeaf())
        self.stat_count = stat_count

    def get_stat_count(self) -> Int:
        """Their `GetStatCount()`."""
        return self.stat_count

    def leaf_count(self) -> Int:
        """Their `Leaves.size()`."""
        return len(self.leaves)
