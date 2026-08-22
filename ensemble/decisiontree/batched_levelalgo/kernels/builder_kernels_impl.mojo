"""The four device kernels of the cuML random forest: histogram, split
scoring, node partition, leaf.

MIRRORS
`cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels_impl.cuh`
at rapidsai/cuml `v26.08.00` (`265b9da6a0e75dbef071a3168398b993a5ff6f0e`),
checked out read-only at `~/CascadeProjects/upstream/cuml-v26.08.00`.
Their file is 470 lines; every construct in it is either ported below with
a `builder_kernels_impl.cuh:<line>` citation or named in the deviation
block as declined with a price.

WHAT THIS FILE IS. `builder.cuh` is a control plane -- it pops a batch of
nodes, decides a grid, and copies splits back to the host. Everything that
actually reads the dataset is here, in four kernels that run in this order
per level:

  1. `buildHistogramsKernel`   (`:285-352`)  one launch per column round
  2. `findBestSplitsKernel`    (`:353-393`)  one launch per column round
  3. `launchNodeSplitKernel`   (`:144-212`)  once, after the last round
  4. `leafKernel`              (`:213-241`)  once, at the end of the tree

THE ONE STRUCTURAL THING WORTH KNOWING BEFORE READING. Step 3 is a FUSED
segmented scan and not a sequence of passes. Their input is a transform
iterator that gathers `row_ids` and evaluates the split predicate, their
keys are a transform iterator over `workload_info[slot / TPB].nodeid`, and
their output is a `tabulate_output_iterator` wrapping
`NodeSplitPartitionWriter` -- so `partition_row_ids` is SCATTERED during
the scan. Their own comment at `:196-199` says exactly that: "the scan
output is a tabulated writer, so partition_row_ids is populated during the
scan rather than by a second scatter kernel". `core/scan_by_key.mojo`
exists to keep that true here, and its `key` / `load` / `store` are their
three iterators. Nothing between phase 1 and phase 3 materialises a row
list.

AND THE ONE ARITHMETIC THING. Their scan operator (`:40-46`) is

    {lhs.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left}

which sums the left operand's count into the right operand's FLAGS. It is
associative and it is not commutative, and both halves of that sentence
are load-bearing: associativity is what lets the scan be blocked at all,
and the right-hand flags are what make the last slot of a segment carry
the segment's own side rather than its predecessor's.

================= DEVIATION BLOCK (whole file) =================

DEVIATION 103. NO DYNAMIC SHARED MEMORY. Their two shared-memory kernels
take a runtime byte count (`extern __shared__ char smem[]`,
`:220-221` and `:297-298`, sized by `launchBuildHistogramsKernel`'s
`split_smem_config.histogram_dynamic_smem_size` at `:398` and by
`launchLeafKernel`'s `smem_size` at `:252`). Mojo 1.0's
`stack_allocation[..., address_space = AddressSpace.SHARED]` is STATIC:
the slot count is a comptime expression. Resolved in three parts.

--- 103a. THE HISTOGRAM BLOB IS A COMPTIME 16 KiB CARVE-OUT ---------
THEIRS: one `char[]` of exactly
`max_n_bins * num_outputs * sizeof(BinT) + max_n_bins * sizeof(DataT)
 + sizeof(BinT) + sizeof(DataT)` bytes (`builder.cuh:526-533`).
OURS: one comptime `stack_allocation[SMEM_BIN_SLOTS, BinT, SHARED]`,
with `SMEM_BIN_SLOTS` defaulting to
`TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES // size_of[BinT]()`,
and the quantile copy carved out of its tail exactly as theirs is.
WHY THE DEFAULT IS THEIR CONSTANT AND NOT THIS LAPTOP'S 32 KiB:
`builder.cuh:163` sets
`tunable_split_histogram_dynamic_smem_limit_bytes = 16 * 1024` and
`builder.cuh:545-547` sends any configuration ABOVE it to the global
path regardless of how much shared memory the device has. 16 KiB is below
every row of `mojo_only/kernel_matrix.column_shared_limit`, so their
dispatch is already vendor-independent and the constant is transcribed
rather than re-derived from a queried budget.
IS THE BLOB BIG ENOUGH FOR EVERY CONFIGURATION THEY SEND TO SHARED? Yes,
and it is arithmetic rather than hope. They take the shared arm only when
  hist_bytes + quant_bytes + size_of[BinT] + size_of[DataT] <= 16384,
i.e. `hist_bytes + quant_bytes <= 16384 - size_of[BinT] - size_of[DataT]`.
The blob holds `floor(16384 / size_of[BinT]) * size_of[BinT]` bytes,
which is at least `16384 - size_of[BinT] + 1`. For the four bins
(`size_of[BinT]` is 4, 8, 8 and 12) that is 16384, 16384, 16384 and 16380
against required maxima of 16376, 16372, 16372 and 16368. Their
`sizeof(BinT) + sizeof(DataT)` alignment padding is what buys the margin;
DEVIATION 120 in `builder_kernels.mojo` already recorded that the padding
is transcribed and not dropped, and this is where that pays.
THE QUANTILE CARVE-OUT NEEDS NO `alignPointer`. Their `alignPointer`
(`builder_kernels.cuh:60-64`) rounds a `char*` up to `sizeof(OutT)`. Here
the blob's base is `BinT`-aligned by construction and the carve-out is at
`histogram_len` WHOLE BinT slots, so the quantile pointer is a multiple of
`size_of[BinT]()` from a `BinT`-aligned base. `size_of[BinT]()` is 4, 8, 8
or 12 and `align_of[DataT]()` is 4, so every one of those offsets is
already `DataT`-aligned and the round-up is the identity. This is the same
conclusion DEVIATION 120 reached, re-derived at the one site that carves.
PRICE: the shared arm reserves 16 KiB of threadgroup memory whatever the
actual histogram size, instead of the exact `histogram_dynamic_smem_size`.
Their own heuristic already permits exactly that worst case -- a
configuration at 16383 bytes takes the shared arm -- so no configuration
becomes unschedulable that was schedulable for them. What it does cost is
occupancy on the SMALL configurations, where they would have asked for a
few hundred bytes and we ask for 16 KiB. Measuring that is out of scope
this round (no timing may be taken), so it is recorded as an OPEN item:
if it matters, the fix is a second comptime instantiation at a smaller
`SMEM_BIN_SLOTS`, which is a parameter the launcher already exposes.

--- 103b. `use_global_memory_histogram` BECOMES A COMPTIME PARAMETER -
THEIRS: a `bool` KERNEL ARGUMENT (`:294`), branched on at `:322` and
`:346`, so one instantiation serves both arms.
OURS: a comptime `USE_GLOBAL_MEMORY_HISTOGRAM` parameter, so the two arms
are two instantiations and the launcher picks between them on the SAME
runtime `SharedMemoryConfig.use_global_memory_histogram` their launcher
passes down. The dispatch decision is unchanged, byte for byte; only the
moment it is taken moves from inside the kernel to the launch site.
REASON: `stack_allocation` is unconditional at kernel scope, so a single
instantiation would reserve the 16 KiB blob on the GLOBAL arm too, which
is the arm that exists precisely because the histogram does not fit.
Reserving it there would make the fallback worse than the path it is
falling back from.
PRICE: two kernel objects instead of one, i.e. two entries in the
compiled binary per (objective, bin, TPB) tuple. Zero at runtime, zero in
value, and both arms are separately named and separately checked in
`ensemble/mojo_only/builder_kernels_check.mojo` -- an opt-in path is an
unchecked path, so neither of these is opt-in.

--- 103c. THE LEAF HISTOGRAM IS A COMPTIME CAP ---------------------
THEIRS: `smem_size = sizeof(BinT) * dataset.num_outputs`
(`builder.cuh:654`), a runtime product.
OURS: `LEAF_SMEM_BIN_SLOTS`, comptime, defaulting to the same
`TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES // size_of[BinT]()`.
Their `smem_size` argument is still ACCEPTED by `launch_leaf_kernel`, so
the caller's arithmetic is unchanged and the two files still diff, and it
is asserted against the cap at launch rather than silently ignored --
a caller asking for more slots than the instantiation holds gets an error
and not a corrupted leaf.
PRICE: identical in character to 103a; a regression tree
(`num_outputs == 1`) reserves the cap rather than `sizeof(BinT)` bytes.
The same OPEN item and the same one-parameter fix apply.

DEVIATION 127. NO 64-BIT INTEGER ATOMIC. `countLocalLeftKernel` ends
(`:79-81`) with

    atomicAdd(reinterpret_cast<unsigned long long*>(&splits[nid].local_nLeft),
              static_cast<unsigned long long>(block_count));

`Atomic.fetch_add` on a `UInt64` is a hard COMPILE error on Apple GPU
(measured, `ensemble/mojo_only/atomic_width_probe.mojo`), so the 64-bit
atomic cannot be transcribed. Resolved as a 32-bit shadow counter.

WHAT WAS DONE. A device array `local_nleft` of `Int32`, one slot per work
item, is zeroed by `reset_local_left_counts_kernel` alongside their
`splits[idx].local_nLeft = 0`; `count_local_left_kernel` adds into it
through `core/block_reduce.block_flush_count_i32`, which is that flush,
their `block_count > 0` guard included; and a third kernel,
`publish_local_left_counts_kernel`, widens each slot back into
`splits[idx].local_nLeft` before the partition scan reads it at `:113`.
The field keeps its Int64 width, so `builder.cuh:103`'s host-side
`narrow_cast<std::size_t>(split.local_nLeft)` reads exactly what theirs
reads.

WHY 32 BITS IS EXACT AND NOT A NARROWING. `local_nLeft` counts rows of ONE
node's instance range that go left. That range is a slice of
`dataset.row_ids`, whose length is `n_sampled_rows`, declared `IdxT`
(`dataset.h:32`) with `IdxT = int` in every RF instantiation cuML
compiles. A value that cannot exceed 2^31-1 cannot overflow an Int32. The
64-bit width is headroom their own index type forbids them from using --
the same argument `bins.mojo` DEVIATION 101a makes for the bin counter,
and it is the same bound.

THE BLOCK REDUCTION IS ALSO NARROWED, and for a second reason. Theirs is
`cub::BlockReduce<std::int64_t, TPB>` (`:59`). Ours is
`block_reduce_sum[DType.int32, TPB]`. The summand is 0 or 1 (`:76-77`) and
there are `TPB` of them, so the block sum is bounded by TPB -- 128 in
every cuML launch -- and Int32 is exact by construction rather than by the
range argument above. The additional reason to narrow is that a 64-bit
`warp.sum` is a 64-bit shuffle, which this repository has not established
on all three columns; a 32-bit one it has.

PRICE, counted in launches and bytes because no timing may be taken this
round:
  * ONE EXTRA KERNEL LAUNCH per `launch_node_split_kernel`
    (`publish_local_left_counts_kernel`), grid
    `ceil(n_work_items / 128)`, which for a batch of nodes is one or two
    blocks. Their sequence is 3 launches (reset, count, copy-back) plus a
    device scan; ours is 4 plus a three-phase scan.
  * `4 * n_work_items` BYTES of scratch, allocated once by the caller in
    `NodeSplitScratch` and reused every level, never per launch.
  * ONE EXTRA READ AND WRITE of `n_work_items` counters. `n_work_items`
    is a node batch, not a row count.
  * ZERO CHANGE IN VALUE at every reachable input.

DEVIATION 128. A MOJO STRUCT IS REFUSED AS A KERNEL ARGUMENT unless it
conforms to `std.builtin.device_passable.DevicePassable`, which cannot
currently be satisfied from outside the stdlib. `Dataset`, `Quantiles`,
the objective and `NodeSplitPartitionWriter` are all passed BY VALUE into
their kernels (`:57-58`, `:287-293`, `:355-361`, `:216-217`, `:195-199`).
Two consequences, both structural, both priced.

--- 128a. ARGUMENTS TRAVEL IN A ONE-ELEMENT DEVICE BUFFER ----------
The dataset, the quantiles and the objective are packed into one plain
struct (`HistogramArgs`, `FindBestSplitsArgs`, `LeafArgs`,
`NodeSplitArgs`), written once into a `size_of[Args]()`-byte device
buffer, and the kernel loads it in its first line. This is the pattern
`core/scan_by_key.upload_device_functor` already established and priced
(its DEVIATION 115b); `DeviceArgs` below is the same mechanism with a
`Copyable & Deinitable` bound instead of `TrivialRegisterPassable`,
because `DatasetView` and `Quantiles` are `Copyable, Movable` and the
compiler requires the explicit `Deinitable` for a struct field of a
generic parameter type.
PRICE: one device allocation of `size_of[Args]()` bytes and one
host-to-device copy of the same size PER LAUNCHER CALL -- not per launch
inside it -- plus one uniform (therefore broadcast, therefore cached)
global load per thread. The buffers are owned by the CALLER
(`DeviceArgs`, `NodeSplitScratch`) and reused every level, because a
kernel inside a tree builder must not allocate. No value changes. If a
later toolchain documents `DevicePassable` for user structs, every
`argsp[unsafe_offset=0]` becomes a plain parameter and no caller moves.

--- 128b. THE SCAN FUNCTOR DUPLICATES FOUR `Dataset` FIELDS --------
`core/scan_by_key.ScanByKeyOps` requires `TrivialRegisterPassable`,
because its functor is copied into shared memory and into registers.
`DatasetView` is `Copyable, Movable` and cannot be a field of a trivial
struct. `NodeSplitPartitionOps` therefore carries `data`, `row_stride`,
`col_stride` and `row_ids` directly -- the only four fields their
partition path reads -- plus a two-line `value()` that is
`dataset.h:40-44` verbatim.
PRICE: one duplicated accessor, which must not drift from `dataset.mojo`.
Zero in value: same four fields, same two `Int64` casts, same product and
sum. Recorded as a MERGE-TIME item -- if `DatasetView` ever becomes
`TrivialRegisterPassable`, the four fields and the accessor delete and the
struct holds a `DatasetView` again.

DEVIATION 129. TRAIT PLUMBING FOR WHAT C++ GETS FROM DUCK TYPING. Three
places, all of them the same shape: their template parameter is
constrained by nothing and satisfied structurally, and Mojo's traits are
nominal.

--- 129a. `ObjectiveLike` -- CLOSED ---------------------------------
`buildHistogramsKernel`, `findBestSplitsKernel` and `leafKernel` are
templated on `typename ObjectiveT` with no constraint; Mojo traits are
nominal, so that list has to be written down somewhere.

It is written down in `objectives.mojo` as `trait ObjectiveLike`, and
every launcher here is generic on `O: ObjectiveLike`. The merge-time
item this block used to describe -- two forwarding adapter structs and
six overloaded launchers, pending someone moving the trait -- HAPPENED.
The adapters and the overloads are gone; there is nothing left to
delete. The paragraph describing them is deleted rather than annotated,
because it described code that is not in this file.

--- 129b. `ScanBin`, SO `pdf_to_cdf` CAN SEE A BIN ------------------
`core/block_scan.pdf_to_cdf` -- which IS `:259-283`, already ported, and
is used rather than rewritten -- is generic over `BlockScanElement`
(`zero()`, `plus()`). `bins.mojo`'s `Bin` declares `__init__` and
`__add__` and does not declare that trait. `ScanBin[B]` is a
single-field wrapper that does, and the histogram pointer is
`unsafe_bitcast` to it at the one call site. A one-field struct has the
layout of its field, and the check reads the cdf back THROUGH THE BIN
TYPE per cell, so a layout surprise is visible rather than assumed away.
PRICE: one adapter struct. MERGE-TIME item, same as 129a.

--- 129c. `lower_bound` IS DUPLICATED TO CARRY AN ADDRESS SPACE -----
`builder_kernels.mojo`'s `lower_bound` -- their `:118-133`, already ported
-- takes `MutPointer[Scalar[dtype], ao]`, whose address space defaults to
GENERIC. The histogram kernel's shared arm calls it on a THREADGROUP
pointer (`:340`, after `quantiles_for_split = shared_quantiles` at
`:329`), and Metal has no generic address space to cast that into. The
function is duplicated below with an `address_space` parameter added and
NOTHING else changed -- same `end = len - 1`, same clamp, same loop.
PRICE: two copies of a nine-line search that must not drift. MERGE-TIME
item: adding the parameter to the original deletes this copy, and that is
a one-line change to a file this lane does not own.

NOT A DEVIATION, recorded because the brief asked and the answer is
"none": THERE IS NO `double` IN `builder_kernels_impl.cuh`. `grep -n
double` over their whole file returns nothing. Every wide type in it is
`std::int64_t` (`NodeSplitPartitionState::left_count` at `:37`,
`cub::BlockReduce<std::int64_t>` at `:59`, `thread_count` at `:70`,
`global_sample_count` at `:373`) or `std::size_t` (`:50`, `:71`, `:101`,
`:110-113`, `:164`). The float64 that the RF learner does carry is in
`bins.cuh` (DEVIATION 101b, fixed-point), `dataset.h` (DEVIATION 100,
`sample_weight` at Float32) and `builder_kernels.cuh`'s
`packHistograms` (DEVIATION 122, declined with the multi-GPU path). Per
site in THIS file: `left_count` stays Int64 in `NodeSplitPartitionState`,
`global_sample_count` stays Int64, and the only narrowing is DEVIATION
127's block reduction, priced there.

NOT A DEVIATION, A HOST-SIDE HAZARD THE CALLER MUST KNOW ABOUT, and the
most important paragraph in this block for whoever writes `builder.mojo`.
**MOJO DESTROYS A VALUE AT ITS LAST USE, NOT AT THE END OF SCOPE, AND
EVERY LAUNCHER BELOW TAKES RAW `MutPointer`s.** So

    var q_nbins = ctx.enqueue_create_buffer[DType.int32](n_cols)
    var quantiles = Quantiles[dtype](q_arr.unsafe_ptr(), q_nbins.unsafe_ptr())
    ...
    launch_find_best_splits_kernel(ctx, ..., quantiles, ...)

frees `q_nbins` at the `.unsafe_ptr()` line -- its last use -- and the NEXT
`enqueue_create_buffer` is handed the same address. MEASURED, not
theorised: `builder_kernels_check.mojo`'s arm C read `n_bins` as
-8388609, which is the bit pattern of `Split::Min()`, because `q_nbins`
had been freed and the `splits` buffer allocated on top of it. The arm had
gone GREEN once before that, so this is a RACE and a green run is not
evidence it is absent.

THE RULE FOR EVERY CALLER OF THESE FOUR LAUNCHERS: any `DeviceBuffer`
whose contents a kernel will read must have a use AFTER the
`ctx.synchronize()` that waits on the launch. `_ = buf^` on the line after
the synchronize is enough and is what the check does. The same applies to
the `HostBuffer` behind an `enqueue_copy` -- `enqueue_copy` is
asynchronous, so a host staging buffer freed at `.unsafe_ptr()` is read
after free. `builder.mojo` holds these buffers as struct FIELDS, which is
immune by construction; anything holding one as a local is not.

NOT A DEVIATION, A MEASURED PLATFORM TRAP, recorded because it cost this
lane an hour and it will cost the next one the same. **BINDING AN
`ImplicitlyCopyable` STRUCT LOADED FROM A DEVICE POINTER TO A LOCAL `var`
CRASHES THE METAL COMPILER.** Not a Mojo error -- a clean compile followed
by `error: Metal Compiler failed to compile metallib. Please submit a bug
report.` with no line number and no symbol. Bisected to a nine-line
reproducer this session:

    @fieldwise_init
    struct NWI(ImplicitlyCopyable, Movable):
        var idx: Int
        var depth: Int32
        var instances: IR          # IR is two Ints

    def k(p: MutPointer[NWI, MutAnyOrigin], o: ...):
        var wi = p[unsafe_offset=0]        # <- CRASHES metallib
        o[unsafe_offset=0] = Int32(wi.instances.count)

The three spellings that DO compile and run, measured one at a time:

    ref wi = p[unsafe_offset=0]                       # OK
    o[...] = Int32(p[unsafe_offset=0].instances.count)  # OK, inline
    var x = p[unsafe_offset=0].copy()                 # OK *only* when the
                                                     # struct is `Copyable,
                                                     # Movable` and NOT
                                                     # `ImplicitlyCopyable`

`.copy()` on an `ImplicitlyCopyable` struct crashes exactly as the implicit
form does, so "add `.copy()`" is NOT the fix and a reader who tries it will
conclude the trap is elsewhere. THE FIX IS `ref`. Every read of
`NodeWorkItem`, `WorkloadInfo`, `InstanceRange` and `SparseTreeNode` in
this file is therefore a `ref` binding, which is also what their
`const auto work_item = work_items[nid];` costs on their compiler -- a
copy of a POD that the optimiser folds into the loads it already needed.
`DatasetView`, `Quantiles`, the objectives and the `*Args` blobs are
`Copyable, Movable`, so those keep `.copy()`.

NOT A DEVIATION EITHER, recorded because the number looks like a choice:
`raft::WarpSize` at `:365` is a hardcoded 32 and `n_split_warps` is
`ceildiv(TPB, WarpSize)`. This port uses Mojo's queried `WARP_SIZE`, per
this repository's standing rule and per DEVIATION 104 in `split.mojo`,
which already priced the width for the reduction this scratch feeds.
=================================================================
"""

from std.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from core.block_reduce import block_flush_count_i32, block_reduce_sum
from core.launch_log import log_launch
from core.block_scan import BlockScanElement, pdf_to_cdf
from core.scan_by_key import (
    ScanByKeyElement,
    ScanByKeyOps,
    launch_inclusive_scan_by_key,
    scan_by_key_block_agg_bytes,
    scan_by_key_temp_bytes,
)

from ensemble.decisiontree.batched_levelalgo.bins import (
    Bin,
    BinScales,
    RegressionBinLike,
)
from ensemble.decisiontree.batched_levelalgo.dataset import DatasetView
from ensemble.decisiontree.batched_levelalgo.objectives import (
    ObjectiveLike,
    ClassificationObjectiveFunction,
    RegressionObjectiveFunction,
)
from ensemble.decisiontree.batched_levelalgo.quantiles import Quantiles
from ensemble.decisiontree.batched_levelalgo.split import (
    PINNED_SPLIT_REDUCE_LANES,
    Split,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from ensemble.decisiontree.batched_levelalgo.kernels.builder_kernels import (
    InstanceRange,
    NodeWorkItem,
    SharedMemoryConfig,
    WorkloadInfo,
)
from ensemble.flatnode import SparseTreeNode


# `builder_kernels_impl.cuh:34` -- `static constexpr int TPB_DEFAULT = 128;`
# (and `builder.cuh:161`, the same constant declared twice in their tree).
comptime TPB_DEFAULT = 128

# DEVIATION 404 -- the one line the gbdt kernel files also carry
# (`comptime BUILD_MODE = GLOBAL_NUMERIC_MODE`): the numeric mode is read
# from `mojo_only/numerics` ONCE here, and the split reduction's width pin
# defaults from it. Checks may instantiate either arm explicitly through
# `find_best_splits_kernel`'s `pinned_reduce` parameter without flipping
# the global.
comptime BUILD_MODE = GLOBAL_NUMERIC_MODE
comptime SPLIT_REDUCE_PINNED_DEFAULT = BUILD_MODE == NUMERIC_IDENTICAL

# `builder.cuh:163`. Their comment: "Tunable performance heuristic for the
# shared-memory histogram path. Large per-block histograms, usually from
# large n_classes, can reduce occupancy enough that global memory is faster
# even when the histogram fits in shared memory. 16 KiB keeps small/default
# histograms in shared memory while avoiding the large-class shared-memory
# slowdown measured locally."
comptime TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES = 16 * 1024


def default_smem_bin_slots[B: Bin]() -> Int:
    """DEVIATION 103a's default blob size, in whole `BinT` slots."""
    return TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES // size_of[B]()


# ===========================================================================
# DEVIATION 129c -- `lower_bound` with an address space.
# ===========================================================================


@always_inline
def lower_bound_aspace[
    dtype: DType, ao: MutOrigin, aspace: AddressSpace, //
](
    array: MutPointer[Scalar[dtype], ao, address_space=aspace],
    len: Int32,
    element: Scalar[dtype],
) -> Int32:
    """`builder_kernels.cuh:118-133`, duplicated per DEVIATION 129c.

    "Returns the lowest index in `array` whose value is greater or equal
    to `element`. Values outside the quantile range are clamped to the
    edge bins: values below the first quantile return 0, and values above
    the last quantile return len - 1." (their `:115-117`)

    `end` starts at `len - 1`, NOT `len`. That one character is the upper
    clamp and is why this is not `std::lower_bound`.
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


# ===========================================================================
# DEVIATION 129a -- the objective trait their `typename ObjectiveT` implies.
# ===========================================================================


@fieldwise_init
struct ScanBin[B: Bin](BlockScanElement):
    """One `BinT`, wearing `core/block_scan`'s trait.

    A single-field struct has the layout of its field, so a
    `MutPointer[BinT]` bitcast to a `MutPointer[ScanBin[BinT]]` addresses
    the same bins. `pdf_to_cdf` is `:259-283` and is used, not rewritten.
    """

    var b: Self.B

    @staticmethod
    @always_inline
    def zero() -> Self:
        """Their `BinT()`, e.g. `bins.cuh:22`."""
        return Self(Self.B())

    @always_inline
    def plus(self, rhs: Self) -> Self:
        """Their `operator+`, e.g. `bins.cuh:48-52`. Operand order is
        theirs: `a.plus(b)` is `a + b`."""
        return Self(self.b + rhs.b)


# ===========================================================================
# DEVIATION 128a -- one-element device buffers for the by-value arguments.
# ===========================================================================


struct DeviceArgs[F: Copyable & Deinitable](Movable):
    """`core/scan_by_key.upload_device_functor`'s mechanism, widened from
    `TrivialRegisterPassable` to `Copyable & Movable`.

    Owned by the CALLER and reused every level: a kernel inside a tree
    builder must not allocate. `host` must come from
    `enqueue_create_host_buffer`, because `enqueue_copy` is host<->device
    only and a device pointer handed in as `src_ptr` is a silent no-op on
    this target.
    """

    var host: HostBuffer[DType.uint8]
    var dev: DeviceBuffer[DType.uint8]

    def __init__(out self, ctx: DeviceContext) raises:
        self.host = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[Self.F]()
        )
        self.dev = ctx.enqueue_create_buffer[DType.uint8](size_of[Self.F]())

    def upload(
        mut self, ctx: DeviceContext, args: Self.F
    ) raises -> MutPointer[Self.F, MutUntrackedOrigin]:
        self.host.unsafe_ptr().unsafe_bitcast[Self.F]()[
            unsafe_offset=0
        ] = args.copy()
        log_launch("xfer_args_upload")
        ctx.enqueue_copy(dst_buf=self.dev, src_ptr=self.host.unsafe_ptr())
        return (
            self.dev.unsafe_ptr()
            .unsafe_origin_cast[MutUntrackedOrigin]()
            .unsafe_bitcast[Self.F]()
        )


# ===========================================================================
# THE NODE PARTITION -- `:36-46`, `:48-53`, `:55-82`, `:87-115`, `:117-142`,
# `:144-212`.
# ===========================================================================


@fieldwise_init
struct NodeSplitPartitionState(ScanByKeyElement):
    """`struct NodeSplitPartitionState`, `:36-38`."""

    # `:37` -- std::int64_t left_count;
    var left_count: Int64
    # `:38a` -- bool valid_row;
    var valid_row: Bool
    # `:38b` -- bool goes_left;
    var goes_left: Bool

    @staticmethod
    @always_inline
    def zero() -> Self:
        """`NodeSplitPartitionState{std::int64_t{0}, false, false}`, the
        value their own input functor returns for an invalid split
        (`:198`) and for an out-of-range row (`:201`)."""
        return Self(Int64(0), False, False)

    @always_inline
    def combine(self, rhs: Self) -> Self:
        """`NodeSplitPartitionScanOp::operator()`, `:42-44`:

            return {lhs.left_count + rhs.left_count, rhs.valid_row,
                    rhs.goes_left};

        `self` is `lhs`. THE FLAGS COME FROM THE RIGHT OPERAND -- copied,
        not paraphrased, because a reader who assumes a sum will get the
        flags wrong and the counts right, which is the failure a total
        check cannot see.
        """
        return Self(
            self.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left
        )


def reset_local_left_counts_kernel[
    dtype: DType
](
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    local_nleft: MutPointer[Int32, MutAnyOrigin],
    n_splits: Int32,
):
    """`resetLocalLeftCountsKernel`, `:48-53`.

        const auto idx = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx < n_splits) { splits[idx].local_nLeft = 0; }

    `local_nleft` is DEVIATION 127's shadow counter and is zeroed here so
    that the two live in lockstep from the same launch.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n_splits):
        var s = splits[unsafe_offset=idx]
        s.local_nLeft = 0
        splits[unsafe_offset=idx] = s
        local_nleft[unsafe_offset=idx] = Int32(0)


def publish_local_left_counts_kernel[
    dtype: DType
](
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    local_nleft: MutPointer[Int32, MutAnyOrigin],
    n_splits: Int32,
):
    """NOT IN THEIR SOURCE. DEVIATION 127's second half.

    Theirs accumulates straight into `splits[nid].local_nLeft` with a
    64-bit atomic; ours accumulates into a 32-bit shadow and widens it
    back here, before `NodeSplitPartitionWriter` reads the field at
    `:113` and before `builder.cuh:103` reads it on the host.
    """
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < Int(n_splits):
        var s = splits[unsafe_offset=idx]
        s.local_nLeft = Int64(Int(local_nleft[unsafe_offset=idx]))
        splits[unsafe_offset=idx] = s


@fieldwise_init
struct NodeSplitArgs[dtype: DType, label_dtype: DType](Copyable, Movable):
    """DEVIATION 128a's blob for `countLocalLeftKernel` (`:56`) and
    `nodeSplitCopyBackKernel` (`:122`), whose only by-value argument is
    the dataset."""

    var dataset: DatasetView[Self.dtype, Self.label_dtype]


def count_local_left_kernel[
    dtype: DType, label_dtype: DType, TPB: Int, sabotage: Int = 0
](
    argsp: MutPointer[NodeSplitArgs[dtype, label_dtype], MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    workload_info: MutPointer[WorkloadInfo, MutAnyOrigin],
    local_nleft: MutPointer[Int32, MutAnyOrigin],
):
    """`countLocalLeftKernel`, `:55-82`, transcribed line for line.

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may pass;
    1 drops their `split.IsValid()` guard (`:75`), so a node that found no
    split still counts rows and the partition writer -- which keeps the
    guard -- then places against a count that should have been zero.
    """
    var args = argsp[unsafe_offset=0].copy()
    var dataset = args.dataset.copy()

    # `:63-67`
    ref workload_info_cta = workload_info[unsafe_offset = Int(block_idx.x)]
    var nid = workload_info_cta.nodeid
    ref work_item = work_items[unsafe_offset = Int(nid)]
    var split = splits[unsafe_offset = Int(nid)]

    # `:69-78`. Their accumulator is `std::int64_t`; DEVIATION 127 prices
    # the Int32 here, and the summand is 0 or 1 with TPB of them.
    var thread_count = Int32(0)
    var range_pos = (
        Int(workload_info_cta.offset_blockid) * Int(block_dim.x)
        + Int(thread_idx.x)
    )
    comptime valid_guard = sabotage != 1
    var split_ok = split.IsValid()
    comptime if not valid_guard:
        # SABOTAGE: the `split.IsValid()` half of their `:75` guard goes.
        split_ok = True
    if split_ok and range_pos < work_item.instances.count:
        var row = dataset.row_ids[
            unsafe_offset = work_item.instances.begin + range_pos
        ]
        thread_count = Int32(1) if dataset.value(
            row, split.colid
        ) <= split.quesval else Int32(0)

    # `:79`
    var block_count = block_reduce_sum[DType.int32, TPB](thread_count)
    # `:80-82`, through `core/block_reduce`, which IS that flush.
    block_flush_count_i32(local_nleft, Int(nid), Int64(Int(block_count)))


struct NodeSplitPartitionOps[dtype: DType, TPB: Int, sabotage: Int = 0](
    ScanByKeyOps
):
    """Their three iterators, as one functor.

    `key` is `node_key` (`:190-192`), `load` is the `partition_state`
    lambda (`:193-...`) and `store` is
    `NodeSplitPartitionWriter::operator()` (`:97-114`) reached through
    `thrust::make_tabulate_output_iterator` (`:195-199`).

    THE FOUR DATASET FIELDS AND `value()` ARE DUPLICATED HERE, not held as
    a `DatasetView`; see DEVIATION 128b for why and for the price.

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may pass.
    4 uses `left_count` where their `:110` uses `left_count - 1`, which
    shifts every left row one slot and leaves every count right. 5 drops
    the `state.valid_row` early return at `:99`.
    """

    comptime Elem = NodeSplitPartitionState

    # DEVIATION 128b -- `dataset.h:18`, `:28`, `:30`, `:36`.
    var data: MutPointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var row_stride: Int64
    var col_stride: Int64
    var row_ids: MutPointer[Int32, MutUntrackedOrigin]
    # `:90-94` -- their writer's other four members.
    var work_items: MutPointer[NodeWorkItem, MutUntrackedOrigin]
    var splits: MutPointer[Split[Self.dtype], MutUntrackedOrigin]
    var workload_info: MutPointer[WorkloadInfo, MutUntrackedOrigin]
    var partition_row_ids: MutPointer[Int32, MutUntrackedOrigin]

    @always_inline
    def __init__[
        label_dtype: DType, //
    ](
        out self,
        dataset: DatasetView[Self.dtype, label_dtype],
        work_items: MutPointer[NodeWorkItem, MutUntrackedOrigin],
        splits: MutPointer[Split[Self.dtype], MutUntrackedOrigin],
        workload_info: MutPointer[WorkloadInfo, MutUntrackedOrigin],
        partition_row_ids: MutPointer[Int32, MutUntrackedOrigin],
    ):
        self.data = dataset.data
        self.row_stride = dataset.row_stride
        self.col_stride = dataset.col_stride
        self.row_ids = dataset.row_ids
        self.work_items = work_items
        self.splits = splits
        self.workload_info = workload_info
        self.partition_row_ids = partition_row_ids

    @always_inline
    def value(self, row: Int32, col: Int32) -> Scalar[Self.dtype]:
        """`dataset.h:40-44`, duplicated per DEVIATION 128b. The two
        int64 casts are theirs."""
        return self.data[
            unsafe_offset = Int(
                Int64(Int(row)) * self.row_stride
                + Int64(Int(col)) * self.col_stride
            )
        ]

    @always_inline
    def key(self, slot: Int) -> Int32:
        """`:190-192` -- `workload_info[slot / TPB].nodeid`."""
        return self.workload_info[unsafe_offset = slot // Self.TPB].nodeid

    @always_inline
    def load(self, slot: Int) -> Self.Elem:
        """The `partition_state` lambda, `:193-204`, branch for branch."""
        ref workload_info_cta = self.workload_info[
            unsafe_offset = slot // Self.TPB
        ]
        var nid = workload_info_cta.nodeid
        ref work_item = self.work_items[unsafe_offset = Int(nid)]
        var split = self.splits[unsafe_offset = Int(nid)]
        # `:198`
        if not split.IsValid():
            return Self.Elem(Int64(0), False, False)

        # `:200`
        var range_pos = (
            Int(workload_info_cta.offset_blockid) * Self.TPB
            + slot % Self.TPB
        )
        # `:201-203`
        if range_pos >= work_item.instances.count:
            return Self.Elem(Int64(0), False, False)

        # `:203-206`
        var row = self.row_ids[
            unsafe_offset = work_item.instances.begin + range_pos
        ]
        var goes_left = self.value(row, split.colid) <= split.quesval
        return Self.Elem(
            Int64(1) if goes_left else Int64(0), True, goes_left
        )

    @always_inline
    def store(self, slot: Int, state: Self.Elem):
        """`NodeSplitPartitionWriter::operator()`, `:97-114`."""
        # `:99`
        comptime if Self.sabotage != 5:
            if not state.valid_row:
                return

        # `:101-105`
        ref workload_info_cta = self.workload_info[
            unsafe_offset = slot // Self.TPB
        ]
        var nid = workload_info_cta.nodeid
        ref work_item = self.work_items[unsafe_offset = Int(nid)]
        var split = self.splits[unsafe_offset = Int(nid)]

        # `:107-108`
        var range_start = work_item.instances.begin
        var range_pos = (
            Int(workload_info_cta.offset_blockid) * Self.TPB
            + slot % Self.TPB
        )

        # `:110-114`
        var row = self.row_ids[unsafe_offset = range_start + range_pos]
        comptime off = 0 if Self.sabotage == 4 else 1
        var rank = (
            Int(state.left_count) - off
        ) if state.goes_left else range_pos - Int(state.left_count)
        var local_left_count = Int(split.local_nLeft)
        var out_idx = range_start + (
            rank if state.goes_left else local_left_count + rank
        )
        self.partition_row_ids[unsafe_offset=out_idx] = row


def node_split_copy_back_kernel[
    dtype: DType, label_dtype: DType, TPB: Int, sabotage: Int = 0
](
    argsp: MutPointer[NodeSplitArgs[dtype, label_dtype], MutAnyOrigin],
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    splits: MutPointer[Split[dtype], MutAnyOrigin],
    workload_info: MutPointer[WorkloadInfo, MutAnyOrigin],
    partition_row_ids: MutPointer[Int32, MutAnyOrigin],
):
    """`nodeSplitCopyBackKernel`, `:117-142`.

    Their comment at `:116-117`: "Copy back only ranges for nodes that
    actually split. Leaf/invalid nodes keep their existing row-id order
    because the scan writer skips them too."

    `sabotage` is a CHECK HOOK; 6 drops the `split.IsValid()` early return
    at `:127`, so an invalid node's range is overwritten with whatever
    `partition_row_ids` happens to hold -- exactly the corruption their
    guard exists to prevent, and exactly what a check that only inspects
    the nodes that DID split cannot see.
    """
    var args = argsp[unsafe_offset=0].copy()
    var dataset = args.dataset.copy()

    # `:123-127`
    ref workload_info_cta = workload_info[unsafe_offset = Int(block_idx.x)]
    var nid = workload_info_cta.nodeid
    ref work_item = work_items[unsafe_offset = Int(nid)]
    var split = splits[unsafe_offset = Int(nid)]
    comptime if sabotage != 6:
        if not split.IsValid():
            return

    # `:129-141`
    var range_start = work_item.instances.begin
    var range_len = work_item.instances.count
    var range_pos = (
        Int(workload_info_cta.offset_blockid) * Int(block_dim.x)
        + Int(thread_idx.x)
    )
    if range_pos < range_len:
        var idx = range_start + range_pos
        dataset.row_ids[unsafe_offset=idx] = partition_row_ids[
            unsafe_offset=idx
        ]


struct NodeSplitScratch[dtype: DType, TPB: Int](Movable):
    """The buffers `launch_node_split_kernel` needs and must not allocate.

    Theirs come from `rmm::exec_policy(builder_stream)`'s async resource
    at `:167`, i.e. from a pool that is also not a fresh `cudaMalloc`. The
    sizes are DEVIATION 127's shadow counter plus
    `core/scan_by_key`'s four scratch buffers plus DEVIATION 128a's two
    argument blobs; construct once per `Builder`, at the maxima the batch
    can reach, and pass every level.
    """

    comptime Ops = NodeSplitPartitionOps[Self.dtype, Self.TPB]

    var local_nleft: DeviceBuffer[DType.int32]
    var states: DeviceBuffer[DType.uint8]
    var heads: DeviceBuffer[DType.uint8]
    var block_agg: DeviceBuffer[DType.uint8]
    var block_head: DeviceBuffer[DType.uint8]
    var ops_host: HostBuffer[DType.uint8]
    var ops_dev: DeviceBuffer[DType.uint8]

    def __init__(
        out self, ctx: DeviceContext, max_slots: Int, max_work_items: Int
    ) raises:
        var n_blocks = ceildiv(max_slots, Self.TPB)
        self.local_nleft = ctx.enqueue_create_buffer[DType.int32](
            max_work_items if max_work_items > 0 else 1
        )
        self.states = ctx.enqueue_create_buffer[DType.uint8](
            scan_by_key_temp_bytes[Self.Ops, Self.TPB](max_slots) + 1
        )
        self.heads = ctx.enqueue_create_buffer[DType.uint8](max_slots + 1)
        self.block_agg = ctx.enqueue_create_buffer[DType.uint8](
            scan_by_key_block_agg_bytes[Self.Ops, Self.TPB](max_slots) + 1
        )
        self.block_head = ctx.enqueue_create_buffer[DType.uint8](
            n_blocks + 1
        )
        self.ops_host = ctx.enqueue_create_host_buffer[DType.uint8](
            size_of[Self.Ops]()
        )
        self.ops_dev = ctx.enqueue_create_buffer[DType.uint8](
            size_of[Self.Ops]()
        )


def launch_node_split_kernel[
    dtype: DType,
    label_dtype: DType, //,
    TPB: Int = TPB_DEFAULT,
    sabotage: Int = 0,
](
    ctx: DeviceContext,
    dataset: DatasetView[dtype, label_dtype],
    work_items: MutPointer[NodeWorkItem, MutUntrackedOrigin],
    splits: MutPointer[Split[dtype], MutUntrackedOrigin],
    workload_info: MutPointer[WorkloadInfo, MutUntrackedOrigin],
    n_blocks_dimx: Int,
    n_work_items: Int,
    partition_row_ids: MutPointer[Int32, MutUntrackedOrigin],
    mut scratch: NodeSplitScratch[dtype, TPB],
    mut args_blob: DeviceArgs[NodeSplitArgs[dtype, label_dtype]],
) raises:
    """`launchNodeSplitKernel`, `:144-212`, in their order.

    `sabotage` is a CHECK HOOK: 1, 2 and 3 are forwarded to
    `core/scan_by_key` (segment reset, inter-block carry, head flag);
    4 and 5 go to `NodeSplitPartitionOps`; 6 goes to the copy-back.
    """
    # `:153`
    if n_blocks_dimx == 0:
        return

    var argsp = args_blob.upload(ctx, NodeSplitArgs(dataset.copy()))

    # `:155-160` -- `constexpr int reset_tpb = 128;`
    comptime RESET_TPB = 128
    var reset_grid = ceildiv(n_work_items, RESET_TPB)
    comptime k_reset = reset_local_left_counts_kernel[dtype]
    log_launch("nodesplit_reset")
    ctx.enqueue_function[k_reset](
        splits.unsafe_origin_cast[MutAnyOrigin](),
        scratch.local_nleft.unsafe_ptr(),
        Int32(n_work_items),
        grid_dim=reset_grid if reset_grid > 0 else 1,
        block_dim=RESET_TPB,
    )

    # `:161-164`
    comptime count_sab = 1 if sabotage == 7 else 0
    comptime k_count = count_local_left_kernel[
        dtype, label_dtype, TPB, count_sab
    ]
    log_launch("nodesplit_count_left")
    ctx.enqueue_function[k_count](
        argsp.unsafe_origin_cast[MutAnyOrigin](),
        work_items.unsafe_origin_cast[MutAnyOrigin](),
        splits.unsafe_origin_cast[MutAnyOrigin](),
        workload_info.unsafe_origin_cast[MutAnyOrigin](),
        scratch.local_nleft.unsafe_ptr(),
        grid_dim=n_blocks_dimx,
        block_dim=TPB,
    )

    # DEVIATION 127: their 64-bit atomic landed straight in the field;
    # ours widens the shadow into it here, before `:113` reads it.
    comptime k_pub = publish_local_left_counts_kernel[dtype]
    log_launch("nodesplit_publish")
    ctx.enqueue_function[k_pub](
        splits.unsafe_origin_cast[MutAnyOrigin](),
        scratch.local_nleft.unsafe_ptr(),
        Int32(n_work_items),
        grid_dim=reset_grid if reset_grid > 0 else 1,
        block_dim=RESET_TPB,
    )

    # `:166-206` -- the fused segmented scan. "Each slot corresponds to
    # one thread lane in the tiled workload_info layout. workload_info is
    # grouped by node, so scan-by-key resets ranks at node boundaries."
    var n_slots = n_blocks_dimx * TPB
    comptime ops_sab = 4 if sabotage == 4 else (5 if sabotage == 5 else 0)
    comptime OpsT = NodeSplitPartitionOps[dtype, TPB, ops_sab]
    var ops = OpsT(
        dataset, work_items, splits, workload_info, partition_row_ids
    )
    scratch.ops_host.unsafe_ptr().unsafe_bitcast[OpsT]()[
        unsafe_offset=0
    ] = ops
    log_launch("xfer_nodesplit_ops")
    ctx.enqueue_copy(
        dst_buf=scratch.ops_dev, src_ptr=scratch.ops_host.unsafe_ptr()
    )
    var ops_ptr = (
        scratch.ops_dev.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[OpsT]()
    )
    comptime scan_sab = sabotage if sabotage >= 1 and sabotage <= 3 else 0
    launch_inclusive_scan_by_key[OpsT, TPB, scan_sab](
        ctx,
        ops_ptr,
        n_slots,
        scratch.states,
        scratch.heads,
        scratch.block_agg,
        scratch.block_head,
    )

    # `:208-212` -- "The original row_ids buffer remains the source during
    # the scan, so copy back after it finishes."
    comptime copy_sab = 6 if sabotage == 6 else 0
    comptime k_copy = node_split_copy_back_kernel[
        dtype, label_dtype, TPB, copy_sab
    ]
    log_launch("nodesplit_copy_back")
    ctx.enqueue_function[k_copy](
        argsp.unsafe_origin_cast[MutAnyOrigin](),
        work_items.unsafe_origin_cast[MutAnyOrigin](),
        splits.unsafe_origin_cast[MutAnyOrigin](),
        workload_info.unsafe_origin_cast[MutAnyOrigin](),
        partition_row_ids.unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=n_blocks_dimx,
        block_dim=TPB,
    )


# ===========================================================================
# THE LEAF PASS -- `:213-241`, `:243-255`.
# ===========================================================================


@fieldwise_init
struct LeafArgs[O: ObjectiveLike](Copyable, Movable):
    """DEVIATION 128a's blob for `leafKernel`'s two by-value arguments,
    `objective` and `dataset` (`:216-217`)."""

    var objective: Self.O
    var dataset: DatasetView[Self.O.DataT, Self.O.LabelT]


def leaf_kernel[
    O: ObjectiveLike, TPB: Int, LEAF_SMEM_BIN_SLOTS: Int, sabotage: Int = 0
](
    argsp: MutPointer[LeafArgs[O], MutAnyOrigin],
    tree: MutPointer[SparseTreeNode[O.DataT], MutAnyOrigin],
    instance_ranges: MutPointer[InstanceRange, MutAnyOrigin],
    leaves: MutPointer[Scalar[O.DataT], MutAnyOrigin],
):
    """`leafKernel`, `:213-241`, transcribed line for line.

    One block per node of the batch. The shared histogram is
    `num_outputs` bins wide -- `IncrementHistogram(histogram, 1, 0,
    label, ...)` at `:234` passes `n_bins = 1` and `b = 0`, so
    `bins.cuh`'s offset `label * n_bins + b` is just `label` and the
    histogram is indexed by CLASS, not by bin. For regression
    `num_outputs` is 1 and the single bin accumulates the label sum.

    `LEAF_SMEM_BIN_SLOTS` is DEVIATION 103c's comptime cap on their
    runtime `smem_size = sizeof(BinT) * dataset.num_outputs`
    (`builder.cuh:654`).

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may pass;
    1 writes the leaf vector at `leaves + node_id` instead of
    `leaves + num_outputs * node_id` (a mis-indexed write that is
    invisible when `num_outputs == 1` and when every node's vector is the
    same), and 2 drops the `IsLeaf()` early return at `:227` so that
    internal nodes get leaf values written over them.
    """
    var args = argsp[unsafe_offset=0].copy()
    var objective = args.objective.copy()
    var dataset = args.dataset.copy()

    # `:220-221` -- their `extern __shared__ char shared_memory[]` and the
    # `reinterpret_cast<BinT*>`. DEVIATION 103c.
    var histogram = stack_allocation[
        LEAF_SMEM_BIN_SLOTS, O.BinT, address_space = AddressSpace.SHARED
    ]()

    # `:222-224`
    var node_id = Int(block_idx.x)
    ref node = tree[unsafe_offset=node_id]
    ref range = instance_ranges[unsafe_offset=node_id]
    # `:227`
    comptime if sabotage != 2:
        if not node.IsLeaf():
            return

    # `:228-232`
    var tid = Int(thread_idx.x)
    var i = tid
    while i < Int(dataset.num_outputs):
        histogram[unsafe_offset=i] = O.BinT()
        i += Int(block_dim.x)
    barrier()

    # `:233-238`
    var j = range.begin + tid
    while j < range.begin + range.count:
        var row = dataset.row_ids[unsafe_offset=j]
        var label = dataset.labels[unsafe_offset = Int(row)]
        objective.IncrementHistogram(
            histogram, Int32(1), Int32(0), label, dataset, row
        )
        j += Int(block_dim.x)
    barrier()

    # `:239-241`
    if tid == 0:
        comptime scale = 1 if sabotage == 1 else 0
        var out_off = (
            node_id if scale == 1 else Int(dataset.num_outputs) * node_id
        )
        O.SetLeafVector(
            histogram,
            dataset.num_outputs,
            leaves.unsafe_offset(out_off),
            objective.Scales(),
        )


# DEVIATION 129a IS CLOSED. Until `objectives.mojo` declared
# `ObjectiveLike`, this file carried two forwarding adapters
# (`ClassificationObjective`, `RegressionObjective`) and each launcher had
# TWO overloads, one per concrete objective, because Mojo traits are
# nominal and there was nothing to dispatch on. The trait now lives beside
# the objectives, both conform to it, and the adapters and overloads are
# deleted -- which also un-blocks a generic `Builder` and therefore
# regression forests.

def launch_leaf_kernel[
    O: ObjectiveLike,
    TPB: Int = TPB_DEFAULT,
    LEAF_SMEM_BIN_SLOTS: Int = default_smem_bin_slots[O.BinT](),
    sabotage: Int = 0,
](
    ctx: DeviceContext,
    objective: O,
    dataset: DatasetView[O.DataT, O.LabelT],
    tree: MutPointer[SparseTreeNode[O.DataT], MutUntrackedOrigin],
    instance_ranges: MutPointer[InstanceRange, MutUntrackedOrigin],
    leaves: MutPointer[Scalar[O.DataT], MutUntrackedOrigin],
    batch_size: Int,
    smem_size: Int,
    mut args_blob: DeviceArgs[LeafArgs[O]],
) raises:
    """`launchLeafKernel`, `:243-255`. `int num_blocks = batch_size;`"""
    if batch_size <= 0:
        return
    # DEVIATION 103c: their `smem_size` is still the caller's arithmetic
    # and is CHECKED against the instantiation rather than ignored.
    if smem_size > LEAF_SMEM_BIN_SLOTS * size_of[O.BinT]():
        raise Error(
            "launch_leaf_kernel: smem_size "
            + String(smem_size)
            + " exceeds LEAF_SMEM_BIN_SLOTS * size_of[BinT] = "
            + String(LEAF_SMEM_BIN_SLOTS * size_of[O.BinT]())
            + " (DEVIATION 103c)"
        )
    var argsp = args_blob.upload(ctx, LeafArgs[O](objective.copy(), dataset.copy()))
    comptime k = leaf_kernel[O, TPB, LEAF_SMEM_BIN_SLOTS, sabotage]
    log_launch("leaf")
    ctx.enqueue_function[k](
        argsp.unsafe_origin_cast[MutAnyOrigin](),
        tree.unsafe_origin_cast[MutAnyOrigin](),
        instance_ranges.unsafe_origin_cast[MutAnyOrigin](),
        leaves.unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=batch_size,
        block_dim=TPB,
    )


@fieldwise_init
struct HistogramArgs[O: ObjectiveLike](Copyable, Movable):
    """DEVIATION 128a's blob for `buildHistogramsKernel`'s three by-value
    arguments: `dataset` (`:288`), `quantiles` (`:289`) and `objective`
    (`:292`)."""

    var dataset: DatasetView[Self.O.DataT, Self.O.LabelT]
    var quantiles: Quantiles[Self.O.DataT]
    var objective: Self.O


@always_inline
def _histogram_inner_loop[
    O: ObjectiveLike,
    ho: MutOrigin,
    ha: AddressSpace,
    qo: MutOrigin,
    qa: AddressSpace, //,
    sabotage: Int = 0,
](
    objective: O,
    dataset: DatasetView[O.DataT, O.LabelT],
    histogram: MutPointer[O.BinT, ho, address_space=ha],
    quantiles_for_split: MutPointer[Scalar[O.DataT], qo, address_space=qa],
    col: Int32,
    n_bins: Int32,
    range_start: Int,
    end: Int,
    tid: Int,
    stride: Int,
):
    """`:336-342`, their grid-stride accumulation loop:

        for (auto i = range_start + tid; i < end; i += stride) {
          auto row   = dataset.row_ids[i];
          auto data  = dataset.value(row, col);
          auto label = dataset.labels[row];
          IdxT start = lower_bound(quantiles_for_split, n_bins, data);
          objective.IncrementHistogram(histogram, n_bins, start, label,
                                       dataset, row);
        }

    One function rather than two copies because their `histogram` and
    `quantiles_for_split` are ONE pointer pair that points either into
    shared memory or into global memory (`:322-333`); Mojo carries the
    address space in the type, so the two arms are two instantiations of
    this and the source is written once, as theirs is.

    `sabotage` is a CHECK HOOK: 3 replaces the bin search with a constant
    0, so every instance lands in bin 0 of its class.
    """
    var i = range_start + tid
    while i < end:
        var row = dataset.row_ids[unsafe_offset=i]
        var data = dataset.value(row, col)
        var label = dataset.labels[unsafe_offset = Int(row)]
        var start = lower_bound_aspace(quantiles_for_split, n_bins, data)
        comptime if sabotage == 3:
            # SABOTAGE: the bin search never runs.
            start = Int32(0)
        objective.IncrementHistogram(
            histogram, n_bins, start, label, dataset, row
        )
        i += stride


@always_inline
def _histogram_inner_loop_binned[
    O: ObjectiveLike, ho: MutOrigin, ha: AddressSpace, //, sabotage: Int = 0
](
    objective: O,
    dataset: DatasetView[O.DataT, O.LabelT],
    histogram: MutPointer[O.BinT, ho, address_space=ha],
    col: Int32,
    n_bins: Int32,
    range_start: Int,
    end: Int,
    tid: Int,
    stride: Int,
):
    """DEVIATION 314's inner loop: their `:336-342` with the
    `lower_bound` (`:341`) replaced by the PRECOMPUTED index the search
    would return -- `DatasetView.bin_of`, a 1-byte read at the same
    offset `value()` gathers 4 bytes from. Identical bin, identical
    histogram, identical everything after.

    `sabotage=3` (constant bin 0) is honored here exactly as in the
    searching loop, so the check that proves the searching loop's reach
    proves this one's too."""
    var i = range_start + tid
    while i < end:
        var row = dataset.row_ids[unsafe_offset=i]
        var start = dataset.bin_of(row, col)
        comptime if sabotage == 3:
            start = Int32(0)
        var label = dataset.labels[unsafe_offset = Int(row)]
        objective.IncrementHistogram(
            histogram, n_bins, start, label, dataset, row
        )
        i += stride


def build_histograms_kernel[
    O: ObjectiveLike,
    TPB: Int,
    SMEM_BIN_SLOTS: Int,
    USE_GLOBAL_MEMORY_HISTOGRAM: Bool,
    sabotage: Int = 0,
    USE_BINNED: Bool = False,
](
    argsp: MutPointer[HistogramArgs[O], MutAnyOrigin],
    histograms: MutPointer[O.BinT, MutAnyOrigin],
    max_n_bins: Int32,
    work_items: MutPointer[NodeWorkItem, MutAnyOrigin],
    col_start: Int32,
    column_samples: MutPointer[Int32, MutAnyOrigin],
    workload_info: MutPointer[WorkloadInfo, MutAnyOrigin],
):
    """`buildHistogramsKernel`, `:285-352`, transcribed line for line.

    THE COMPUTE CORE. Grid is 2D: `blockIdx.x` indexes the tiled
    `workload_info` (so one node may own many blocks), `blockIdx.y`
    indexes the column within this sampling round.

    `USE_GLOBAL_MEMORY_HISTOGRAM` is their `bool` kernel argument
    (`:294`) hoisted to comptime; DEVIATION 103b says why and what it
    costs. BOTH arms are reachable and both are separately checked.

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may pass.
    1 drops `offset_blockid` from the starting `tid` (`:335`), so every
    block of a multi-block node re-reads the same slice and the node's
    tail is never read -- correct for any node small enough to get one
    block, wrong for every node that got more. 2 collapses the
    shared->global flush (`:346-351`) onto slot 0, which corrupts the
    SHARED arm only and leaves the global arm bit-identical. 3 is
    forwarded to the bin search.
    """
    comptime assert (
        SMEM_BIN_SLOTS * size_of[O.BinT]()
        <= TUNABLE_SPLIT_HISTOGRAM_DYNAMIC_SMEM_LIMIT_BYTES
    ), (
        "shared histogram blob exceeds"
        " tunable_split_histogram_dynamic_smem_limit_bytes (builder.cuh:163)"
    )

    var args = argsp[unsafe_offset=0].copy()
    var dataset = args.dataset.copy()
    var quantiles = args.quantiles.copy()
    var objective = args.objective.copy()

    # `:299-305`
    ref workload_info_cta = workload_info[unsafe_offset = Int(block_idx.x)]
    var nid = workload_info_cta.nodeid
    ref work_item = work_items[unsafe_offset = Int(nid)]
    var range_start = work_item.instances.begin
    var range_len = work_item.instances.count

    var offset_blockid = Int(workload_info_cta.offset_blockid)
    var num_blocks = Int(workload_info_cta.num_blocks)

    # `:307-310`
    var col_index = col_start + Int32(Int(block_idx.y))
    var col = column_samples[
        unsafe_offset = Int(nid) * Int(dataset.n_sampled_cols)
        + Int(col_index)
    ]
    var n_bins = quantiles.n_bins_array[unsafe_offset = Int(col)]

    # `:312-320`
    var n_classes = objective.NumClasses()
    var end = range_start + range_len
    var histogram_len = Int(n_bins) * Int(n_classes)
    var histograms_offset = (
        (Int(nid) * Int(grid_dim.y) + Int(block_idx.y))
        * Int(max_n_bins)
        * Int(n_classes)
    )
    var global_histogram = histograms.unsafe_offset(histograms_offset)
    var stride = Int(block_dim.x) * num_blocks
    comptime tid_scale = 0 if sabotage == 1 else 1
    var tid = Int(thread_idx.x) + (
        tid_scale * offset_blockid
    ) * Int(block_dim.x)

    comptime if USE_GLOBAL_MEMORY_HISTOGRAM:
        # `:294`, `:317-318` -- `histogram = global_histogram` and
        # `quantiles_for_split = quantiles.quantiles_array + max_n_bins * col`.
        comptime if USE_BINNED:
            _histogram_inner_loop_binned[sabotage=sabotage](
                objective,
                dataset,
                global_histogram,
                col,
                n_bins,
                range_start,
                end,
                tid,
                stride,
            )
        else:
            _histogram_inner_loop[sabotage=sabotage](
                objective,
                dataset,
                global_histogram,
                quantiles.quantiles_array.unsafe_offset(
                    Int(max_n_bins) * Int(col)
                ),
                col,
                n_bins,
                range_start,
                end,
                tid,
                stride,
            )
    else:
        # `:322-333` -- carve the shared blob, zero the histogram, copy
        # the quantiles in, then `__syncthreads()`. DEVIATION 103a.
        var histogram = stack_allocation[
            SMEM_BIN_SLOTS, O.BinT, address_space = AddressSpace.SHARED
        ]()
        var shared_quantiles = histogram.unsafe_offset(
            histogram_len
        ).unsafe_bitcast[Scalar[O.DataT]]()

        var i = Int(thread_idx.x)
        while i < histogram_len:
            histogram[unsafe_offset=i] = O.BinT()
            i += Int(block_dim.x)
        comptime if not USE_BINNED:
            # The quantile copy exists only for the bin SEARCH
            # (`:330-332`); the binned loop reads no quantiles.
            var b = Int(thread_idx.x)
            while b < Int(n_bins):
                shared_quantiles[unsafe_offset=b] = (
                    quantiles.quantiles_array[
                        unsafe_offset = Int(max_n_bins) * Int(col) + b
                    ]
                )
                b += Int(block_dim.x)
        barrier()

        # `:336-342`
        comptime if USE_BINNED:
            _histogram_inner_loop_binned[sabotage=sabotage](
                objective,
                dataset,
                histogram,
                col,
                n_bins,
                range_start,
                end,
                tid,
                stride,
            )
        else:
            _histogram_inner_loop[sabotage=sabotage](
                objective,
                dataset,
                histogram,
                shared_quantiles,
                col,
                n_bins,
                range_start,
                end,
                tid,
                stride,
            )

        # `:344-351`
        barrier()
        var k = Int(thread_idx.x)
        while k < histogram_len:
            comptime dst = 0 if sabotage == 2 else 1
            O.BinT.AtomicAdd(
                global_histogram.unsafe_offset(k * dst),
                histogram[unsafe_offset=k],
            )
            k += Int(block_dim.x)


def launch_build_histograms_kernel[
    O: ObjectiveLike,
    TPB: Int = TPB_DEFAULT,
    SMEM_BIN_SLOTS: Int = default_smem_bin_slots[O.BinT](),
    sabotage: Int = 0,
](
    ctx: DeviceContext,
    histograms: MutPointer[O.BinT, MutUntrackedOrigin],
    max_n_bins: Int,
    dataset: DatasetView[O.DataT, O.LabelT],
    quantiles: Quantiles[O.DataT],
    work_items: MutPointer[NodeWorkItem, MutUntrackedOrigin],
    col_start: Int,
    column_samples: MutPointer[Int32, MutUntrackedOrigin],
    objective: O,
    workload_info: MutPointer[WorkloadInfo, MutUntrackedOrigin],
    histogram_grid_x: Int,
    histogram_grid_y: Int,
    smem_config: SharedMemoryConfig,
    mut args_blob: DeviceArgs[HistogramArgs[O]],
) raises:
    """`launchBuildHistogramsKernel`, `:396-421`.

    Their `dim3 histogram_grid` arrives as two Ints and their
    `split_smem_config.histogram_dynamic_smem_size` is not a launch
    argument here (DEVIATION 103a); the SAME
    `smem_config.use_global_memory_histogram` selects the arm
    (DEVIATION 103b).
    """
    if histogram_grid_x <= 0 or histogram_grid_y <= 0:
        return
    var argsp = args_blob.upload(
        ctx, HistogramArgs[O](
            dataset.copy(), quantiles.copy(), objective.copy()
        )
    )
    if smem_config.use_global_memory_histogram:
        if dataset.has_bins:
            # DEVIATION 314: same launch, the binned loop body.
            comptime kgb = build_histograms_kernel[
                O, TPB, 1, True, sabotage, True
            ]
            log_launch("histogram_global_binned")
            ctx.enqueue_function[kgb](
                argsp.unsafe_origin_cast[MutAnyOrigin](),
                histograms.unsafe_origin_cast[MutAnyOrigin](),
                Int32(max_n_bins),
                work_items.unsafe_origin_cast[MutAnyOrigin](),
                Int32(col_start),
                column_samples.unsafe_origin_cast[MutAnyOrigin](),
                workload_info.unsafe_origin_cast[MutAnyOrigin](),
                grid_dim=(histogram_grid_x, histogram_grid_y),
                block_dim=TPB,
            )
            return
        comptime kg = build_histograms_kernel[
            O, TPB, 1, True, sabotage
        ]
        log_launch("histogram_global")
        ctx.enqueue_function[kg](
            argsp.unsafe_origin_cast[MutAnyOrigin](),
            histograms.unsafe_origin_cast[MutAnyOrigin](),
            Int32(max_n_bins),
            work_items.unsafe_origin_cast[MutAnyOrigin](),
            Int32(col_start),
            column_samples.unsafe_origin_cast[MutAnyOrigin](),
            workload_info.unsafe_origin_cast[MutAnyOrigin](),
            grid_dim=(histogram_grid_x, histogram_grid_y),
            block_dim=TPB,
        )
    else:
        if dataset.has_bins:
            # DEVIATION 314, shared arm. Uses the default 103a blob; the
            # tier question below is orthogonal and pending its own A/B.
            comptime ksb = build_histograms_kernel[
                O, TPB, SMEM_BIN_SLOTS, False, sabotage, True
            ]
            log_launch("histogram_binned")
            ctx.enqueue_function[ksb](
                argsp.unsafe_origin_cast[MutAnyOrigin](),
                histograms.unsafe_origin_cast[MutAnyOrigin](),
                Int32(max_n_bins),
                work_items.unsafe_origin_cast[MutAnyOrigin](),
                Int32(col_start),
                column_samples.unsafe_origin_cast[MutAnyOrigin](),
                workload_info.unsafe_origin_cast[MutAnyOrigin](),
                grid_dim=(histogram_grid_x, histogram_grid_y),
                block_dim=TPB,
            )
            return
        # DEVIATION 103a, TIER MACHINERY (2026-08-22, VERDICT PENDING).
        # Their launcher passes EXACTLY `histogram_dynamic_smem_size` as
        # dynamic shared memory (`builder.cuh:412`, computed at
        # `:526-533`, "stays small enough for good occupancy"); 103a
        # reserves the full 16 KiB tunable limit instead because
        # `stack_allocation` is comptime-static. The launch-log
        # attribution measured this kernel at 85.3% of all device time at
        # the 500k benchmark shape while needing ~2.6 KiB of the 16 KiB
        # reserved, so the dispatch below can launch a tier-sized blob --
        # the smallest byte tier holding `histogram_dynamic_smem_size`
        # (which already includes the copied quantiles and their `:531`
        # alignment padding). Same kernel body, same bytes TOUCHED,
        # bit-identical output (fingerprint gate held through the tiered
        # build); only the RESERVATION changes. The tier taken is visible
        # in the launch log by name.
        # THE TIER MACHINERY SHIPS DISABLED, and the disable is a
        # comptime flag rather than a sentinel: the first shipped form
        # ("tier list [1], nothing fits") was BROKEN BY DESIGN --
        # builder_kernels_check passes `SharedMemoryConfig(use_global, 0)`
        # as a don't-care, 0 <= 1 selected the tier, and
        # `TIER_BYTES // size_of[BinT]` allocated a ZERO-SLOT shared
        # blob: the shared arm accumulated into nothing and every
        # non-zero cell read back 0 (caught by that check, 2026-08-22).
        # Rule 11 needs a measured win to enable the tiers, and the only
        # A/B so far was VOIDed at canary spread 10.5x (a peer GBM
        # benchmark held the GPU); the informal reads suggesting the
        # SMALLER blob is SLOWER on M4 are equally uncertifiable.
        # `build/rf_bench_smem4k` / `_smem16k` are the A/B pair; rerun
        # the gated ABAB on a quiet box before flipping this flag.
        comptime HISTOGRAM_SMEM_TIERS_ENABLED = False
        var launched = False
        comptime if HISTOGRAM_SMEM_TIERS_ENABLED:
            var need_bytes = smem_config.histogram_dynamic_smem_size
            comptime for TIER_BYTES in [2048, 4096, 8192]:
                comptime TIER_SLOTS = TIER_BYTES // size_of[O.BinT]()
                comptime assert TIER_SLOTS > 0, "tier below one BinT slot"
                # `need_bytes > 0` guards the check-suite's don't-care
                # config; a zero size means "not computed", never "fits".
                if not launched and need_bytes > 0 and need_bytes <= TIER_BYTES:
                    comptime kt = build_histograms_kernel[
                        O, TPB, TIER_SLOTS, False, sabotage
                    ]
                    log_launch("histogram_shared_" + String(TIER_BYTES))
                    ctx.enqueue_function[kt](
                        argsp.unsafe_origin_cast[MutAnyOrigin](),
                        histograms.unsafe_origin_cast[MutAnyOrigin](),
                        Int32(max_n_bins),
                        work_items.unsafe_origin_cast[MutAnyOrigin](),
                        Int32(col_start),
                        column_samples.unsafe_origin_cast[MutAnyOrigin](),
                        workload_info.unsafe_origin_cast[MutAnyOrigin](),
                        grid_dim=(histogram_grid_x, histogram_grid_y),
                        block_dim=TPB,
                    )
                    launched = True
        if not launched:
            # Above 8 KiB the full 103a blob remains; anything past
            # 16 KiB never reaches this arm because
            # `compute_shared_memory_config` already chose global.
            comptime ks = build_histograms_kernel[
                O, TPB, SMEM_BIN_SLOTS, False, sabotage
            ]
            log_launch("histogram_shared")
            ctx.enqueue_function[ks](
                argsp.unsafe_origin_cast[MutAnyOrigin](),
                histograms.unsafe_origin_cast[MutAnyOrigin](),
                Int32(max_n_bins),
                work_items.unsafe_origin_cast[MutAnyOrigin](),
                Int32(col_start),
                column_samples.unsafe_origin_cast[MutAnyOrigin](),
                workload_info.unsafe_origin_cast[MutAnyOrigin](),
                grid_dim=(histogram_grid_x, histogram_grid_y),
                block_dim=TPB,
            )


@fieldwise_init
struct FindBestSplitsArgs[O: ObjectiveLike](Copyable, Movable):
    """DEVIATION 128a's blob for `findBestSplitsKernel`'s three by-value
    arguments: `dataset` (`:356`), `quantiles` (`:357`) and `objective`
    (`:362`)."""

    var dataset: DatasetView[Self.O.DataT, Self.O.LabelT]
    var quantiles: Quantiles[Self.O.DataT]
    var objective: Self.O


def find_best_splits_kernel[
    O: ObjectiveLike,
    TPB: Int,
    sabotage: Int = 0,
    pinned_reduce: Bool = SPLIT_REDUCE_PINNED_DEFAULT,
](
    argsp: MutPointer[FindBestSplitsArgs[O], MutAnyOrigin],
    histograms: MutPointer[O.BinT, MutAnyOrigin],
    max_n_bins: Int32,
    col_start: Int32,
    column_samples: MutPointer[Int32, MutAnyOrigin],
    mutex: MutPointer[Int32, MutAnyOrigin],
    splits: MutPointer[Split[O.DataT], MutAnyOrigin],
):
    """`findBestSplitsKernel`, `:353-393`, transcribed line for line.

    Grid is 2D: `blockIdx.x` is the node in the batch, `blockIdx.y` the
    column within this sampling round -- so `splits + nid` and
    `mutex + nid` are contended by all `gridDim.y` column blocks of one
    node, which is what `Split::evalBestSplit`'s mutex is for.

    THE HISTOGRAM READ HERE IS ALWAYS GLOBAL. `buildHistogramsKernel`
    flushes its shared copy out at `:346-351`, so both of its arms leave
    the same bytes in `histograms` and this kernel has one arm.

    `sabotage` is a CHECK HOOK and 0 is the only value a caller may pass;
    1 sums `global_sample_count` over class 0 alone (`:374-378`), which
    leaves a one-class problem bit-identical and makes every
    multi-class `nRight = len - nLeft` wrong. 2 is forwarded to
    `core/block_scan.pdf_to_cdf` as ITS sabotage 2, which drops the chunk
    carry so that only the first `TPB` bins of each class come out right
    -- invisible whenever `n_bins <= TPB`, which is why the check runs
    this arm at `TPB = 32` with `n_bins = 100`.
    """
    # `:364-367` -- their `__shared__ __align__(alignof(Split)) unsigned
    # char split_scratch_storage[sizeof(Split) * n_split_warps]` and the
    # `reinterpret_cast`. A non-trivially-constructible type cannot be
    # `__shared__` in CUDA either, which is why theirs is a byte array;
    # `Split` is `TrivialRegisterPassable` here, so the allocation is
    # typed and the cast has nothing to do. `raft::WarpSize` is their
    # hardcoded 32; `WARP_SIZE` is queried (DEVIATION 104 in split.mojo).
    # DEVIATION 404: the pinned arm spells the shuffle through shared
    # memory, so its scratch is one slot per THREAD rather than per warp;
    # its lane count must divide the block.
    comptime assert (not pinned_reduce) or (
        TPB % PINNED_SPLIT_REDUCE_LANES == 0
    ), "pinned split reduce needs TPB % 32 == 0 (DEVIATION 404)"
    comptime N_SPLIT_SCRATCH = TPB if pinned_reduce else ceildiv(
        TPB, WARP_SIZE
    )
    var split_scratch = stack_allocation[
        N_SPLIT_SCRATCH,
        Split[O.DataT],
        address_space = AddressSpace.SHARED,
    ]()

    var args = argsp[unsafe_offset=0].copy()
    var dataset = args.dataset.copy()
    var quantiles = args.quantiles.copy()
    var objective = args.objective.copy()

    # `:369`
    var nid = Int32(Int(block_idx.x))

    # `:371-374`
    var col_index = col_start + Int32(Int(block_idx.y))
    var col = column_samples[
        unsafe_offset = Int(nid) * Int(dataset.n_sampled_cols)
        + Int(col_index)
    ]
    var n_bins = quantiles.n_bins_array[unsafe_offset = Int(col)]

    # `:376-380`
    var n_classes = objective.NumClasses()
    var histograms_offset = (
        (Int(nid) * Int(grid_dim.y) + Int(block_idx.y))
        * Int(max_n_bins)
        * Int(n_classes)
    )
    var histogram = histograms.unsafe_offset(histograms_offset)
    var quantiles_for_split = quantiles.quantiles_array.unsafe_offset(
        Int(max_n_bins) * Int(col)
    )

    # `:382-387`:
    #
    #     std::int64_t global_sample_count = 0;
    #     for (IdxT c = 0; c < n_classes; ++c) {
    #       global_sample_count += static_cast<std::int64_t>(
    #         pdf_to_cdf<BinT, IdxT, TPB>(histogram + n_bins * c, n_bins).Count());
    #     }
    #
    # EVERY thread of the block must reach every `pdf_to_cdf` -- it
    # contains barriers -- so the loop bound is uniform and the call is
    # unconditional. `ScanBin` is DEVIATION 129b.
    var scan_histogram = histogram.unsafe_bitcast[ScanBin[O.BinT]]()
    var global_sample_count = Int64(0)
    comptime class_limit_all = sabotage != 1
    comptime cdf_sab = 2 if sabotage == 2 else 0
    for c in range(Int(n_classes)):
        var total = pdf_to_cdf[
            ScanBin[O.BinT],
            TPB,
            address_space = AddressSpace.GENERIC,
            sabotage=cdf_sab,
        ](scan_histogram.unsafe_offset(Int(n_bins) * c), n_bins)
        comptime if class_limit_all:
            global_sample_count += Int64(Int(total.b.Count()))
        else:
            # SABOTAGE: only class 0 reaches the total.
            if c == 0:
                global_sample_count += Int64(Int(total.b.Count()))

    # `:389`
    barrier()

    # `:391`
    var sp = objective.Gain(
        histogram, quantiles_for_split, col, global_sample_count, n_bins
    )

    # `:393`
    barrier()

    # `:395` -- `sp.evalBestSplit(split_scratch, splits + nid, mutex + nid,
    # quantiles_for_split, n_bins);` DEVIATION 404 picks the arm: the
    # pinned reduction under NUMERIC_IDENTICAL (width 32 on every vendor,
    # no warp primitives), the transcribed warp-shuffle reduction under
    # FAST. At `WARP_SIZE == 32` the two arms are the same function --
    # `split_check.mojo`'s pinned-parity arm holds that per cell.
    comptime if pinned_reduce:
        sp.eval_best_split_pinned(
            split_scratch,
            splits.unsafe_offset(Int(nid)),
            mutex.unsafe_offset(Int(nid)),
            quantiles_for_split,
            n_bins,
        )
    else:
        sp.eval_best_split(
            split_scratch,
            splits.unsafe_offset(Int(nid)),
            mutex.unsafe_offset(Int(nid)),
            quantiles_for_split,
            n_bins,
        )


def launch_find_best_splits_kernel[
    O: ObjectiveLike,
    TPB: Int = TPB_DEFAULT,
    sabotage: Int = 0,
    pinned_reduce: Bool = SPLIT_REDUCE_PINNED_DEFAULT,
](
    ctx: DeviceContext,
    histograms: MutPointer[O.BinT, MutUntrackedOrigin],
    max_n_bins: Int,
    dataset: DatasetView[O.DataT, O.LabelT],
    quantiles: Quantiles[O.DataT],
    col_start: Int,
    column_samples: MutPointer[Int32, MutUntrackedOrigin],
    mutex: MutPointer[Int32, MutUntrackedOrigin],
    splits: MutPointer[Split[O.DataT], MutUntrackedOrigin],
    objective: O,
    split_grid_x: Int,
    split_grid_y: Int,
    mut args_blob: DeviceArgs[FindBestSplitsArgs[O]],
) raises:
    """`launchFindBestSplitsKernel`, `:423-445`."""
    if split_grid_x <= 0 or split_grid_y <= 0:
        return
    var argsp = args_blob.upload(
        ctx, FindBestSplitsArgs[O](
            dataset.copy(), quantiles.copy(), objective.copy()
        )
    )
    comptime k = find_best_splits_kernel[O, TPB, sabotage, pinned_reduce]
    log_launch("find_best_splits")
    ctx.enqueue_function[k](
        argsp.unsafe_origin_cast[MutAnyOrigin](),
        histograms.unsafe_origin_cast[MutAnyOrigin](),
        Int32(max_n_bins),
        Int32(col_start),
        column_samples.unsafe_origin_cast[MutAnyOrigin](),
        mutex.unsafe_origin_cast[MutAnyOrigin](),
        splits.unsafe_origin_cast[MutAnyOrigin](),
        grid_dim=(split_grid_x, split_grid_y),
        block_dim=TPB,
    )



# ===========================================================================
# DEVIATION 314 -- the dataset is binned ONCE per forest.
# ===========================================================================


def bin_dataset_kernel[dtype: DType](
    data: MutPointer[Scalar[dtype], MutAnyOrigin],
    bins: MutPointer[UInt8, MutAnyOrigin],
    quantiles_array: MutPointer[Scalar[dtype], MutAnyOrigin],
    n_bins_array: MutPointer[Int32, MutAnyOrigin],
    max_n_bins: Int32,
    n_rows: Int32,
    n_cols: Int32,
    row_stride: Int64,
    col_stride: Int64,
):
    """DEVIATION 314. One thread per (row, col): store
    `lower_bound(quantiles[col], value)` as a uint8, at the SAME offset
    formula `value()` uses, so `DatasetView.bin_of` reads what the
    histogram kernel's search (`builder_kernels_impl.cuh:341`) would
    have computed. Grid y is the column, x strides the rows -- the
    quantile row for a column is read once per thread, the data column
    is walked coalesced for the column-major default.

    This kernel has no cuML counterpart: their histogram kernel
    re-searches at every level (`:336-343`). The index it stores is the
    IDENTICAL pure function of (row, col), which is the whole
    bit-identity argument.
    """
    var col = Int32(Int(block_idx.y))
    var n_bins = n_bins_array[unsafe_offset = Int(col)]
    var q = quantiles_array.unsafe_offset(Int(max_n_bins) * Int(col))
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < Int(n_rows):
        var off = Int(
            Int64(i) * row_stride + Int64(Int(col)) * col_stride
        )
        var b = lower_bound_aspace(q, n_bins, data[unsafe_offset=off])
        bins[unsafe_offset=off] = UInt8(Int(b))
        i += stride


def launch_bin_dataset[dtype: DType](
    ctx: DeviceContext,
    data: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    mut bins_buf: DeviceBuffer[DType.uint8],
    quantiles_array: MutPointer[Scalar[dtype], MutUntrackedOrigin],
    n_bins_array: MutPointer[Int32, MutUntrackedOrigin],
    max_n_bins: Int,
    n_rows: Int,
    n_cols: Int,
    row_stride: Int,
    col_stride: Int,
) raises:
    """DEVIATION 314's one launch, once per forest, right after
    `compute_quantiles`. The caller guards `max_n_bins <= 256`."""
    if n_rows <= 0 or n_cols <= 0:
        return
    var blocks_x = ceildiv(n_rows, 256)
    log_launch("bin_dataset")
    ctx.enqueue_function[bin_dataset_kernel[dtype]](
        data.unsafe_origin_cast[MutAnyOrigin](),
        bins_buf.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        quantiles_array.unsafe_origin_cast[MutAnyOrigin](),
        n_bins_array.unsafe_origin_cast[MutAnyOrigin](),
        Int32(max_n_bins),
        Int32(n_rows),
        Int32(n_cols),
        Int64(row_stride),
        Int64(col_stride),
        grid_dim=(blocks_x, n_cols),
        block_dim=256,
    )
    _ = bins_buf.unsafe_ptr()
