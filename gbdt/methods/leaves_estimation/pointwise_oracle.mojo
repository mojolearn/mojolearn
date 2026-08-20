"""`TBinOptimizedOracle`: the walker's device-side eyes, per-bin.

PORT OF `catboost/cuda/methods/leaves_estimation/pointwise_oracle.{h,cpp}`
at CatBoost `54a8143a`, the rowSize==1 arm -- every single-dim pointwise
loss. Transliterated. Do not improve.

WHAT THE ORACLE HOLDS, in their layout: the target, weights and CURSOR
COPY gathered into BIN ORDER (docs of leaf 0, then leaf 1, ...), with
per-bin offsets/sizes. Their doc-parallel factory sorts by the model's
bins at construction; ours receives the order for free, because the
searcher's `split_points` gathers left `row_index` exactly bin-sorted --
the caller gathers target/weights/cursor by it and hands them here.

THE CALL CYCLE, theirs (`pointwise_oracle.cpp`):

  MoveTo(point)       shift = point - CurrentPoint, added per bin to the
                      cursor copy (`AddBinModelValues`, `:35-57`); caches
                      cleared
  WriteValueAndFirst  ONE fused kernel -- `ApproximateAt(cursor, &value,
                      &der, &der2)`, the rowSize==1 "fast path" (`:73-78`)
                      -- then `ComputePartitionStats` per bin for der AND
                      der2 (`:82-91`), gradient = reduced der, cached
                      Hessian = reduced der2 PLUS LAMBDA (`:86-89`), value
                      read back
  WriteSecondDer      returns the cache (`:114-117`); this port RAISES if
                      the cache is empty, because for rowSize==1 their
                      `ApproximateAt` always fills it and an empty cache
                      here means the call order broke
  Regularize          `RegularizeImpl` (`oracle_interface.h:43-52`): zero
                      every bin whose weight sum is under MinLeafWeight
                      (their hardcoded 1e-20)

================= DEVIATION BLOCK =================
* THE REDUCES RUN IN FLOAT32 where theirs land in `TStripeBuffer<double>`
  (`:81`, `:85`). This is `compute_partition_stats`'s existing port-wide
  width, not a choice made here; the walker's own arithmetic is Float64
  from the readback on, like theirs from `ReadReduce` on.
* UNWEIGHTED WeightsCpu COMES FROM THE LEAF SIZES, exactly: their ctor
  reduces the literal-1.0 weights column in double (`:236-243`); a float32
  reduce of ones loses exactness past 2^24 rows per leaf, so the
  unweighted arm takes the INTEGER count the partition already knows,
  which is the number their double reduce produces. The weighted arm
  reduces the real weights like theirs.
* `AddBinModelValues` is `add_model_value_kernel` over an IDENTITY row
  index at learning_rate 1 -- same kernel the boosting loop applies models
  with, pointed at the bin-sorted cursor copy.
* `function_value` arrives as per-block partials folded in one fixed host
  order, the file-standard substitution for their block-reduce-plus-
  `atomicAdd` scalar.
* `AddRigdeRegulaizationIfNecessary` (`:109-111`) is a no-op unless
  `AddRidgeToTargetFunction`, which no configuration this repository runs
  sets; omitted, like the Langevin hooks (`oracle_interface.mojo` records
  the terms).
===================================================
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute

from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.gpu_util.partitions_reduce import (
    compute_partition_stats,
    partition_stats_chunks,
)
from gbdt.methods.kernel_add_model_value import add_model_value_kernel
from gbdt.methods.leaves_estimation.oracle_interface import (
    LeavesEstimationOracle,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    cross_entropy_kernel,
)


@fieldwise_init
struct BinOptimizedOracle(LeavesEstimationOracle, Movable):
    """One tree's estimation state. Build with `make_bin_optimized_oracle`."""

    var ctx: DeviceContext
    var n_rows: Int
    var bin_count: Int
    var has_weights: Bool
    var has_border: Bool
    var border: Float32
    var lambda_reg: Float64
    var min_leaf_weight: Float64
    var wide: Int

    var d_target: DeviceBuffer[DType.float32]
    var d_weights: DeviceBuffer[DType.float32]
    var d_cursor: DeviceBuffer[DType.float32]
    var d_identity: DeviceBuffer[DType.uint32]
    var d_leaves: DeviceBuffer[DType.uint32]
    var d_p_off: DeviceBuffer[DType.uint32]
    var d_p_sz: DeviceBuffer[DType.uint32]
    var d_shift: DeviceBuffer[DType.float32]
    var h_shift: HostBuffer[DType.float32]
    var d_eval_stats: DeviceBuffer[DType.float32]
    var d_fv: DeviceBuffer[DType.float32]
    var h_fv: HostBuffer[DType.float32]
    var d_mag_dummy: DeviceBuffer[DType.float32]
    var d_partials: DeviceBuffer[DType.float32]
    var d_part_stats: DeviceBuffer[DType.float32]
    var h_part_stats: HostBuffer[DType.float32]
    var sm_count: Int

    var current_point: List[Float32]
    var weights_cpu: List[Float64]
    var cached_der2: List[Float64]

    def point_dim(self) -> Int:
        return self.bin_count

    def move_to(mut self, point: List[Float32]) raises:
        """`TBinOptimizedOracle::MoveTo` (`pointwise_oracle.cpp:35-57`)."""
        # the last eval's synchronize is what makes h_shift reusable here;
        # the walker is strictly move -> eval -> move.
        for i in range(self.bin_count):
            self.h_shift.unsafe_ptr().unsafe_store(
                i, point[i] - self.current_point[i]
            )
        self.ctx.enqueue_copy(
            dst_buf=self.d_shift, src_ptr=self.h_shift.unsafe_ptr()
        )
        self.ctx.enqueue_function[add_model_value_kernel](
            self.d_p_off.unsafe_ptr(),
            self.d_p_sz.unsafe_ptr(),
            self.d_identity.unsafe_ptr(),
            self.d_shift.unsafe_ptr(),
            Float32(1.0),
            self.d_cursor.unsafe_ptr(),
            grid_dim=(self.wide, self.bin_count, 1),
            block_dim=(256, 1, 1),
        )
        # `DerAtPoint.Clear(); Der2AtPoint.Clear();` (`:54-55`)
        self.cached_der2.clear()
        self.current_point.clear()
        for i in range(self.bin_count):
            self.current_point.append(point[i])

    def write_value_and_first_derivatives(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ) raises:
        """`WriteValueAndFirstDerivatives`, rowSize==1 (`:59-112`)."""
        var blocks = (
            self.n_rows + MSE_BLOCK_SIZE - 1
        ) // MSE_BLOCK_SIZE
        if self.has_border:
            self.ctx.enqueue_function[cross_entropy_kernel[True, True]](
                self.d_target.unsafe_ptr(),
                self.d_weights.unsafe_ptr(),
                Int32(self.n_rows),
                self.d_cursor.unsafe_ptr(),
                Int32(1) if self.has_weights else Int32(0),
                self.border,
                self.d_eval_stats.unsafe_ptr(),
                self.d_fv.unsafe_ptr(),
                Int32(1),
                self.d_mag_dummy.unsafe_ptr(),
                Int32(0),
                grid_dim=(blocks, 1, 1),
                block_dim=(MSE_BLOCK_SIZE, 1, 1),
            )
        else:
            self.ctx.enqueue_function[cross_entropy_kernel[False, True]](
                self.d_target.unsafe_ptr(),
                self.d_weights.unsafe_ptr(),
                Int32(self.n_rows),
                self.d_cursor.unsafe_ptr(),
                Int32(1) if self.has_weights else Int32(0),
                self.border,
                self.d_eval_stats.unsafe_ptr(),
                self.d_fv.unsafe_ptr(),
                Int32(1),
                self.d_mag_dummy.unsafe_ptr(),
                Int32(0),
                grid_dim=(blocks, 1, 1),
                block_dim=(MSE_BLOCK_SIZE, 1, 1),
            )
        compute_partition_stats(
            self.ctx, self.bin_count, 0, 2, self.n_rows,
            self.d_leaves, self.d_p_off, self.d_p_sz,
            self.d_eval_stats, self.d_partials, self.d_part_stats,
            sm_count=self.sm_count,
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.h_part_stats.unsafe_ptr(),
            src_buf=self.d_part_stats,
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.h_fv.unsafe_ptr(), src_buf=self.d_fv
        )
        self.ctx.synchronize()

        gradient.clear()
        self.cached_der2.clear()
        for leaf in range(self.bin_count):
            gradient.append(
                Float64(
                    self.h_part_stats.unsafe_ptr().unsafe_load(2 * leaf)
                )
            )
            # `(*Der2AtPoint)[i] += lambda` (`:86-89`)
            self.cached_der2.append(
                Float64(
                    self.h_part_stats.unsafe_ptr().unsafe_load(
                        2 * leaf + 1
                    )
                )
                + self.lambda_reg
            )
        value = 0.0
        for b in range(
            (self.n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
        ):
            value += Float64(self.h_fv.unsafe_ptr().unsafe_load(b))

    def write_second_derivatives(mut self, mut second_der: List[Float64]) raises:
        """`WriteSecondDerivatives`, the Newton cache arm (`:114-117`)."""
        if len(self.cached_der2) != self.bin_count:
            raise Error(
                "WriteSecondDerivatives before WriteValueAndFirst"
                "Derivatives: the der2 cache is empty, the call order broke"
            )
        second_der.clear()
        for leaf in range(self.bin_count):
            second_der.append(self.cached_der2[leaf])

    def regularize(self, mut point: List[Float32]):
        """`RegularizeImpl` (`oracle_interface.h:43-52`)."""
        for bin in range(self.bin_count):
            if self.weights_cpu[bin] < self.min_leaf_weight:
                point[bin] = Float32(0.0)


def make_bin_optimized_oracle(
    ctx: DeviceContext,
    n_rows: Int,
    bin_count: Int,
    leaf_sizes: List[Int],
    var d_target: DeviceBuffer[DType.float32],
    var d_weights: DeviceBuffer[DType.float32],
    var d_cursor: DeviceBuffer[DType.float32],
    var d_p_off: DeviceBuffer[DType.uint32],
    var d_p_sz: DeviceBuffer[DType.uint32],
    has_weights: Bool,
    has_border: Bool,
    border: Float32,
    lambda_reg: Float64,
    sm_count: Int,
) raises -> BinOptimizedOracle:
    """Their ctor (`pointwise_oracle.cpp:218-246`): allocate the eval
    buffers, seed `CurrentPoint` at zero, and settle `WeightsCpu` once --
    the weights never move during estimation."""
    var d_identity = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    launch_make_sequence(ctx, UInt32(0), d_identity, n_rows)

    var d_leaves = ctx.enqueue_create_buffer[DType.uint32](bin_count)
    var h_leaves = ctx.enqueue_create_host_buffer[DType.uint32](bin_count)
    for i in range(bin_count):
        h_leaves.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=d_leaves, src_ptr=h_leaves.unsafe_ptr())

    var d_shift = ctx.enqueue_create_buffer[DType.float32](bin_count)
    var h_shift = ctx.enqueue_create_host_buffer[DType.float32](bin_count)
    var d_eval_stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var d_fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var h_fv = ctx.enqueue_create_host_buffer[DType.float32](blocks)
    var d_mag_dummy = ctx.enqueue_create_buffer[DType.float32](2)

    var sm = sm_count
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var chunks2 = partition_stats_chunks(sm, 2)
    var chunks1 = partition_stats_chunks(sm, 1)
    var partials_len = bin_count * 2 * chunks2
    if bin_count * 1 * chunks1 > partials_len:
        partials_len = bin_count * 1 * chunks1
    var d_partials = ctx.enqueue_create_buffer[DType.float32](partials_len)
    var d_part_stats = ctx.enqueue_create_buffer[DType.float32](
        2 * bin_count
    )
    var h_part_stats = ctx.enqueue_create_host_buffer[DType.float32](
        2 * bin_count
    )

    var current_point = List[Float32]()
    for _ in range(bin_count):
        current_point.append(Float32(0.0))

    # WeightsCpu (`:236-243`): the weighted arm reduces the real weights;
    # the unweighted arm takes the exact integer counts (deviation block).
    var weights_cpu = List[Float64]()
    if has_weights:
        compute_partition_stats(
            ctx, bin_count, 0, 1, n_rows,
            d_leaves, d_p_off, d_p_sz,
            d_weights, d_partials, d_part_stats,
            sm_count=sm,
        )
        ctx.enqueue_copy(
            dst_ptr=h_part_stats.unsafe_ptr(), src_buf=d_part_stats
        )
        ctx.synchronize()
        for leaf in range(bin_count):
            weights_cpu.append(
                Float64(h_part_stats.unsafe_ptr().unsafe_load(leaf))
            )
    else:
        for leaf in range(bin_count):
            weights_cpu.append(Float64(leaf_sizes[leaf]))

    # the widest leaf sizes the shift-add grid; correctness never depends
    # on it because the kernel strides.
    var max_leaf = 1
    for i in range(bin_count):
        if leaf_sizes[i] > max_leaf:
            max_leaf = leaf_sizes[i]
    var wide = (max_leaf + 255) // 256
    if wide < 1:
        wide = 1

    return BinOptimizedOracle(
        ctx,
        n_rows,
        bin_count,
        has_weights,
        has_border,
        border,
        lambda_reg,
        1e-20,  # MinLeafWeight, their hardcoded default
        wide,
        d_target^,
        d_weights^,
        d_cursor^,
        d_identity^,
        d_leaves^,
        d_p_off^,
        d_p_sz^,
        d_shift^,
        h_shift^,
        d_eval_stats^,
        d_fv^,
        h_fv^,
        d_mag_dummy^,
        d_partials^,
        d_part_stats^,
        h_part_stats^,
        sm,
        current_point^,
        weights_cpu^,
        List[Float64](),
    )
