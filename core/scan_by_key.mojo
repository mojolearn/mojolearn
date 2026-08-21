"""`thrust::inclusive_scan_by_key`, FUSED: the cuML RF node partition.

PORT OF the call at
`cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels_impl.cuh
:165-206` (`launchNodeSplitKernel`), its operator at `:40-46`
(`NodeSplitPartitionScanOp`), its state at `:36-38`
(`NodeSplitPartitionState`) and its output functor at `:87-115`
(`NodeSplitPartitionWriter`), at rapidsai/cuml `v26.08.00`
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`). The scan itself is CUB's
`DeviceScanByKey`, reached through Thrust, at NVIDIA/cccl
`d10a88a945caa4ea63dd2a909cf789c6dbe085a4`.

## WHY THIS FILE IS SHAPED AROUND FUNCTORS AND NOT AROUND ARRAYS

Their call, in full:

    auto node_key = [workload_info] __host__ __device__(std::size_t slot) {
      return workload_info[slot / TPB].nodeid;                      // :190-192
    };
    auto partition_state = [=] __host__ __device__(std::size_t slot) {
      ... dataset.row_ids[...] ... dataset.value(row, split.colid) <= split.quesval
      return NodeSplitPartitionState{goes_left ? 1 : 0, true, goes_left};
    };                                                              // :193-...
    auto node_keys        = thrust::make_transform_iterator(slots_begin, node_key);
    auto partition_states = thrust::make_transform_iterator(slots_begin, partition_state);
    auto partition_writer = thrust::make_tabulate_output_iterator(
      NodeSplitPartitionWriter<...>{dataset, work_items, splits, workload_info,
                                    partition_row_ids});
    thrust::inclusive_scan_by_key(exec_policy, node_keys, node_keys + n_slots,
                                  partition_states, partition_writer,
                                  thrust::equal_to<IdxT>{},
                                  NodeSplitPartitionScanOp{});      // :200-206

**NOTHING IS MATERIALISED.** The keys are computed on the fly from
`workload_info`; the inputs are computed on the fly by gathering
`row_ids` and comparing against the split; and the output is SCATTERED
into `partition_row_ids` by the writer as the scan produces each element.
Their own comment says so at `:196-199`: "the scan output is a tabulated
writer, so partition_row_ids is populated during the scan rather than by
a second scatter kernel."

A segmented-scan primitive that takes an input ARRAY and writes an output
ARRAY does not port this call. It ports a slower algorithm and freezes the
extra passes into the design, which is the mistake this repository has
already paid for once (`traffic-model-ignores-blocks`; and the k-NN case
where standing a device-wide vendor GEMM plus a vendor top-k in for a
fused kernel forced a distance matrix to exist). So the primitive here
takes a caller-supplied FUNCTOR with `key`, `load` and `store`, and the
transform and the scatter happen inside the scan's own passes exactly as
theirs do.

## Their operator is NOT a sum

    struct NodeSplitPartitionScanOp {                              // :40
      NodeSplitPartitionState operator()(const NodeSplitPartitionState& lhs,
                                         const NodeSplitPartitionState& rhs) const
      { return {lhs.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left}; }
    };                                                             // :43-45

Associative, and it TAKES THE RIGHT OPERAND'S FLAGS. `ScanByKeyElement`
below requires exactly `combine`, not `+`, because a `+` would invite a
reader to assume the flags add.

## The segmented rule, and where it comes from

`inclusive_scan_by_key` restarts the accumulation at every key change.
CUB spells that as a head flag and an operator that discards the left
operand when the right one has already reached a segment start. This
repository already carries that exact shape in
`gbdt/gpu_util/kernel/segmented_scan.mojo`, which mirrors CatBoost's
`TSegmentedSum` (`cuda_util/kernel/segmented_scan_helpers.cuh:11-34`):

    newValue = rightFlag ? right : op(left, right)
    newFlag  = leftFlag | rightFlag

THE BLOCK-LEVEL HILLIS-STEELE LOOP AND THE THREE-PHASE CARRY IN THIS FILE
ARE LIFTED FROM `gbdt/gpu_util/kernel/segmented_scan.mojo` (its
`seg_block_scan_kernel`, `seg_scan_block_sums_kernel` and
`seg_add_block_carry_kernel`), retyped from Float32-with-a-sign-bit-flag
to a generic element with a KEY-DERIVED flag, and with the load and the
store replaced by functor calls. It is duplicated rather than imported
because `ensemble/` must not depend on `gbdt/`, and because the two differ
in exactly the place a shared function would have to be parameterised
anyway -- which is how CUB spells it too, as separate template
instantiations.

## The head flag is derived from the KEYS, not supplied

Theirs compares adjacent keys (`thrust::equal_to<IdxT>{}` at `:205`); the
CatBoost family this repository already ports reads a flag bit out of the
value or the index. This port keeps THEIRS: slot `i` is a segment head iff
`i == 0` or `key(i) != key(i - 1)`. That is two key evaluations per slot,
and their key functor is a single `workload_info[slot / TPB].nodeid`
load, so it is cheap on their side too.

# =========================================================================
# DEVIATION BLOCK
#
# DEVIATION 115. TWO PARTS, BOTH STRUCTURAL, BOTH PRICED.
#
# 115a. THREE-PHASE DEVICE DECOUPLING INSTEAD OF CUB'S SINGLE-PASS
#       DECOUPLED LOOKBACK. The lookback is DECLINED, and a decline is a
#       deviation, so here is its price.
#
# THEIRS: `cub::DeviceScanByKey` is ONE pass. Each tile scans locally and
# then obtains its exclusive prefix by spinning on its predecessors'
# published status words (decoupled lookback), so the output is written
# during that same pass.
#
# OURS: three kernels -- block scan, a serial scan of the per-block
# aggregates, then carry-and-emit -- which is the shape this repository
# already runs in `gbdt/gpu_util/kernel/scan.mojo`,
# `segmented_scan.mojo` and `reorder_one_bit.mojo`, for the reason
# recorded there: MAX ships no device-wide scan, so the decoupling has to
# be written out either way, and a lookback needs a forward-progress
# guarantee across blocks that Mojo 1.0 does not state for any of the
# three GPU targets. Writing one anyway would be inventing, on the one
# construct in a GPU program where being wrong hangs the device instead of
# returning a wrong number.
#
# PRICE, COUNTED IN PASSES, NEVER IN TIME (timing is out of scope for this
# lane):
#   * PASSES OVER `row_ids`: UNCHANGED AT TWO. Their input transform
#     gathers `dataset.row_ids[range_start + range_pos]` (`:203`) and
#     their writer gathers it AGAIN (`:106`) -- the fused single pass
#     already reads it twice. Ours reads it once in phase 1 (`ops.load`)
#     and once in phase 3 (`ops.store`). Nothing was added. This is the
#     number that matters and it is a tie.
#   * PASSES OVER THE SCAN STATE: +2. Phase 1 WRITES one
#     `ScanByKeyElement` and one head byte per slot, and phase 3 READS
#     them. CUB keeps that per-thread state in registers and never spills
#     it. At `n_slots` slots the cost is
#     `n_slots * (size_of[Elem]() + 1)` bytes written and read again;
#     for cuML's `NodeSplitPartitionState` that element is 16 bytes
#     (an Int64 and two flags), so 17 bytes per slot each way.
#   * KERNEL LAUNCHES: 3 instead of 1, plus a serial phase whose grid is
#     ONE THREAD over `ceil(n_slots / TPB)` block aggregates.
#   * WHAT IS NOT PAID: no additional pass over `partition_row_ids`, no
#     separate scatter kernel, and no materialised input array. The
#     fusion their comment at `:196-199` exists to advertise is intact.
#
# 115b. THE FUNCTOR TRAVELS IN A ONE-ELEMENT DEVICE BUFFER, NOT IN THE
#       KERNEL PARAMETER BLOCK.
#
# THEIRS: the lambda's captured state and the `NodeSplitPartitionWriter`
# struct are passed by value into the kernel's parameter block, because
# that is what a C++ functor argument compiles to.
#
# OURS: the functor is written into a `size_of[F]()`-byte device buffer
# once and the kernels take a `MutPointer[F, ...]` to it, loading it into
# a register copy in their first line. MEASURED, not assumed: a Mojo
# struct is not accepted as a kernel argument unless it conforms to
# `std.builtin.device_passable.DevicePassable`, whose `_to_device_type`
# has an internal signature this lane could not satisfy from outside the
# stdlib (the compiler's required form renders as
# `def(self, mut encoder: *?, target: Pointer[NoneType, origin_of(origin)])`
# and nine spellings were rejected); `Pointer` itself IS DevicePassable,
# so the indirection is what is available.
#
# PRICE: one extra device allocation of `size_of[F]()` bytes per scan, one
# host-to-device copy of the same size, and one uniform (therefore
# broadcast, therefore cached) global load per thread in each of the two
# functor-touching kernels. NO CHANGE TO ANY VALUE, and no change to the
# fusion -- the functor's pointers are device pointers either way. If a
# later toolchain documents `DevicePassable` for user structs, this
# indirection is deletable without touching any caller, because
# `upload_device_functor` is the only place that knows about it.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import size_of
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import TARGET_COLUMN, column_shared_limit


trait ScanByKeyElement(TrivialRegisterPassable):
    """`NodeSplitPartitionState` plus its operator,
    `builder_kernels_impl.cuh:36-46`."""

    @staticmethod
    def zero() -> Self:
        """The value a slot outside the scan contributes.

        Theirs is `NodeSplitPartitionState{0, false, false}`, which their
        own input functor returns for an invalid split or an out-of-range
        row (`:198`, `:201`). It must be a LEFT identity of `combine` for
        the padding argument in phase 1 to hold.
        """
        ...

    def combine(self, rhs: Self) -> Self:
        """`NodeSplitPartitionScanOp::operator()(lhs, rhs)`, `:42-44`.

        `self` is `lhs` (the earlier slot), `rhs` the later one. Their
        body:

            {lhs.left_count + rhs.left_count, rhs.valid_row, rhs.goes_left}

        Associative; NOT commutative -- it keeps the RIGHT operand's
        flags -- so the operand order below is load-bearing.
        """
        ...


trait ScanByKeyOps(TrivialRegisterPassable):
    """Their three iterators, as one functor.

    `key` is `node_key` (`:190-192`), `load` is `partition_state`
    (`:193-...`), `store` is `NodeSplitPartitionWriter::operator()`
    (`:97-114`) reached through `make_tabulate_output_iterator`. Keeping
    them in ONE struct rather than three mirrors the fact that all three
    of theirs capture the same five pointers.
    """

    comptime Elem: ScanByKeyElement

    def key(self, slot: Int) -> Int32:
        """`workload_info[slot / TPB].nodeid` (`:191`)."""
        ...

    def load(self, slot: Int) -> Self.Elem:
        """The `partition_state` lambda. Gathers, compares, returns the
        per-slot state. NOTHING IS READ FROM A MATERIALISED INPUT ARRAY --
        that is the point of this trait."""
        ...

    def store(self, slot: Int, state: Self.Elem):
        """`NodeSplitPartitionWriter::operator()(index, state)` (`:97`).

        Theirs returns immediately when `!state.valid_row` (`:99`),
        computes a rank from `state.left_count` and `state.goes_left`, and
        scatters `partition_row_ids[out_idx] = row`. All of that stays in
        the caller: the scan owns the state, the writer owns the placement.
        """
        ...


def scan_by_key_temp_bytes[F: ScanByKeyOps, TPB: Int](n_slots: Int) -> Int:
    """Bytes for the `states` buffer (`n_slots` elements)."""
    return n_slots * size_of[F.Elem]()


def scan_by_key_block_agg_bytes[F: ScanByKeyOps, TPB: Int](
    n_slots: Int,
) -> Int:
    """Bytes for the per-block aggregates (`ceil(n_slots / TPB)`)."""
    return ceildiv(n_slots, TPB) * size_of[F.Elem]()


def upload_device_functor[
    F: TrivialRegisterPassable
](
    ctx: DeviceContext,
    ops: F,
    mut host_blob: HostBuffer[DType.uint8],
    mut device_blob: DeviceBuffer[DType.uint8],
) raises -> MutPointer[F, MutUntrackedOrigin]:
    """DEVIATION 115b's one implementation site.

    Both buffers must hold at least `size_of[F]()` bytes; `host_blob` must
    come from `enqueue_create_host_buffer`, because `enqueue_copy` is
    host<->device only and a device pointer handed in as `src_ptr` is a
    silent no-op on this target.
    """
    host_blob.unsafe_ptr().unsafe_bitcast[F]()[unsafe_offset=0] = ops
    ctx.enqueue_copy(dst_buf=device_blob, src_ptr=host_blob.unsafe_ptr())
    return (
        device_blob.unsafe_ptr()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_bitcast[F]()
    )


# ---------------------------------------------------------------------------
# `sabotage` is a CHECK HOOK, not a tunable, and 0 is the only value any
# caller may pass. This repository's rule is that a digest cannot tell a
# working change from a no-op and that reach is proved PER BRANCH, so each
# mechanism below has a switch the check flips to watch the comparison
# move. Values: 1 drops the segment reset inside a block, 2 drops the
# inter-block carry, 3 applies the carry even to slots that already saw a
# segment head. Every one of them leaves the TOTAL right and the PLACEMENT
# wrong, which is the failure mode a count check cannot see.
# ---------------------------------------------------------------------------


def scan_by_key_block_kernel[
    F: ScanByKeyOps, TPB: Int, sabotage: Int = 0
](
    opsp: MutPointer[F, MutAnyOrigin],
    n_slots_in: Int32,
    states: MutPointer[F.Elem, MutAnyOrigin],
    heads: MutPointer[UInt8, MutAnyOrigin],
    block_agg: MutPointer[F.Elem, MutAnyOrigin],
    block_head: MutPointer[UInt8, MutAnyOrigin],
):
    """Phase 1: the segmented inclusive scan inside one block.

    Writes, per slot, the running combine since the last segment head AT
    OR AFTER the block's first slot, plus whether such a head was seen.
    The slot that saw one takes no carry from earlier blocks; that is the
    reset, and phase 3 is where it lands.
    """
    comptime SCAN_BYTES = TPB * (size_of[F.Elem]() + size_of[Int32]())
    comptime assert (
        SCAN_BYTES <= column_shared_limit(TARGET_COLUMN)
    ), "scan_by_key block state exceeds the vendor threadgroup budget"

    var ops = opsp[unsafe_offset=0]

    var s_val = stack_allocation[
        TPB, F.Elem, address_space = AddressSpace.SHARED
    ]()
    var s_flg = stack_allocation[
        TPB, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    var n = Int(n_slots_in)
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * TPB + tid

    # A slot past `n_slots` contributes the identity and cannot start a
    # segment, so the last thread's accumulator is the block's answer
    # whether or not the block is full. Same padding argument as
    # `seg_block_scan_kernel` in `gbdt/gpu_util/kernel/segmented_scan.mojo`.
    var v = F.Elem.zero()
    var flag = Int32(0)
    if i < n:
        v = ops.load(i)
        # `thrust::equal_to<IdxT>{}` on adjacent keys (`:205`).
        if i == 0 or ops.key(i) != ops.key(i - 1):
            flag = Int32(1)

    comptime if sabotage == 1:
        # SABOTAGE: no slot is a segment head, so the scan runs straight
        # through every key change. Totals per block are unchanged.
        flag = Int32(0)

    s_val[unsafe_offset=tid] = v
    s_flg[unsafe_offset=tid] = flag
    barrier()

    # The operator, as a Hillis-Steele inclusive scan:
    #
    #     resultValue = rightFlag ? right : combine(left, right)
    #     newFlag     = leftFlag | rightFlag
    #
    # `right` is this thread, `left` the neighbour `d` behind it. A thread
    # whose accumulated flag is already set has reached its segment's head
    # and stops absorbing. `d` is uniform across the block, so every
    # barrier is reached by every thread.
    var d = 1
    while d < TPB:
        var lv = F.Elem.zero()
        var lf = Int32(0)
        if tid >= d:
            lv = s_val[unsafe_offset = tid - d]
            lf = s_flg[unsafe_offset = tid - d]
        barrier()
        if tid >= d and s_flg[unsafe_offset=tid] == Int32(0):
            s_val[unsafe_offset=tid] = lv.combine(s_val[unsafe_offset=tid])
            s_flg[unsafe_offset=tid] = lf
        barrier()
        d *= 2

    if i < n:
        states[unsafe_offset=i] = s_val[unsafe_offset=tid]
        var seen = s_flg[unsafe_offset=tid]
        heads[unsafe_offset=i] = UInt8(1) if seen != Int32(0) else UInt8(0)

    if tid == TPB - 1:
        var b = Int(block_idx.x)
        block_agg[unsafe_offset=b] = s_val[unsafe_offset=tid]
        var any = s_flg[unsafe_offset=tid]
        block_head[unsafe_offset=b] = UInt8(1) if any != Int32(0) else UInt8(
            0
        )


def scan_by_key_carry_kernel[
    T: ScanByKeyElement, sabotage: Int = 0
](
    block_agg: MutPointer[T, MutAnyOrigin],
    block_head: MutPointer[UInt8, MutAnyOrigin],
    n_blocks_in: Int32,
):
    """Phase 2: the same operator again, over the per-block aggregates.

    One thread, serial -- the shape `seg_scan_block_sums_kernel`
    (`gbdt/gpu_util/kernel/segmented_scan.mojo`) already runs, and the
    reason it is not a recursion is DEVIATION 115a. `block_agg[b]` leaves
    holding the carry INTO block `b`, and the carry dies at a block that
    contains a segment head -- past that head nothing earlier is in the
    segment any more.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var n = Int(n_blocks_in)
    var running = T.zero()
    for b in range(n):
        var v = block_agg[unsafe_offset=b]
        var f = block_head[unsafe_offset=b]
        block_agg[unsafe_offset=b] = running
        comptime if sabotage == 2:
            # SABOTAGE: the carry never leaves block 0, so every block
            # after the first starts its open segment from the identity.
            running = T.zero()
        else:
            if f != UInt8(0):
                running = v
            else:
                running = running.combine(v)


def scan_by_key_emit_kernel[
    F: ScanByKeyOps, TPB: Int, sabotage: Int = 0
](
    opsp: MutPointer[F, MutAnyOrigin],
    n_slots_in: Int32,
    states: MutPointer[F.Elem, MutAnyOrigin],
    heads: MutPointer[UInt8, MutAnyOrigin],
    block_agg: MutPointer[F.Elem, MutAnyOrigin],
):
    """Phase 3: add the block's carry, then WRITE THROUGH THE FUNCTOR.

    This is `make_tabulate_output_iterator(NodeSplitPartitionWriter{...})`
    (`:195-199`). There is no scatter kernel after it and no output array
    between it and the caller's destination, which is the whole reason
    this file takes a functor.

    A slot that already saw a segment head inside its own block owns a
    complete accumulation and must not take anything from before the
    block. Launched with the SAME `TPB` blocking as phase 1, because
    `block_agg` is indexed by `block_idx.x`.
    """
    var ops = opsp[unsafe_offset=0]
    var n = Int(n_slots_in)
    var i = Int(block_idx.x) * TPB + Int(thread_idx.x)
    if i >= n:
        return

    var state = states[unsafe_offset=i]
    var take_carry = heads[unsafe_offset=i] == UInt8(0)
    comptime if sabotage == 3:
        # SABOTAGE: the head flag is ignored, so a slot that starts a new
        # segment still absorbs the previous segment's carry.
        take_carry = True
    if take_carry:
        state = block_agg[unsafe_offset = Int(block_idx.x)].combine(state)

    ops.store(i, state)


def launch_inclusive_scan_by_key[
    F: ScanByKeyOps, TPB: Int, sabotage: Int = 0
](
    ctx: DeviceContext,
    ops_ptr: MutPointer[F, MutUntrackedOrigin],
    n_slots: Int,
    mut states: DeviceBuffer[DType.uint8],
    mut heads: DeviceBuffer[DType.uint8],
    mut block_agg: DeviceBuffer[DType.uint8],
    mut block_head: DeviceBuffer[DType.uint8],
) raises:
    """`thrust::inclusive_scan_by_key(exec_policy, node_keys,
    node_keys + n_slots, partition_states, partition_writer,
    thrust::equal_to<IdxT>{}, NodeSplitPartitionScanOp{})`,
    `builder_kernels_impl.cuh:200-206`.

    `ops_ptr` comes from `upload_device_functor`. The four buffers are
    scratch, sized by `scan_by_key_temp_bytes` /
    `scan_by_key_block_agg_bytes` / `n_slots` / `ceil(n_slots / TPB)`;
    they are passed in rather than allocated here for the same reason
    CatBoost's `TScanKernelContext.PartResults` is
    (`gbdt/gpu_util/kernel/scan.mojo`): a scan inside a tree builder is
    called once per level and must not allocate.
    """
    if n_slots <= 0:
        return
    var n_blocks = ceildiv(n_slots, TPB)

    var states_p = states.unsafe_ptr().unsafe_bitcast[F.Elem]()
    var agg_p = block_agg.unsafe_ptr().unsafe_bitcast[F.Elem]()

    comptime k1 = scan_by_key_block_kernel[F, TPB, sabotage]
    ctx.enqueue_function[k1](
        ops_ptr,
        Int32(n_slots),
        states_p,
        heads.unsafe_ptr(),
        agg_p,
        block_head.unsafe_ptr(),
        grid_dim=n_blocks,
        block_dim=TPB,
    )

    comptime k2 = scan_by_key_carry_kernel[F.Elem, sabotage]
    ctx.enqueue_function[k2](
        agg_p,
        block_head.unsafe_ptr(),
        Int32(n_blocks),
        grid_dim=1,
        block_dim=1,
    )

    comptime k3 = scan_by_key_emit_kernel[F, TPB, sabotage]
    ctx.enqueue_function[k3](
        ops_ptr,
        Int32(n_slots),
        states_p,
        heads.unsafe_ptr(),
        agg_p,
        grid_dim=n_blocks,
        block_dim=TPB,
    )
