# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
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
* `AddBinModelValues` is `add_bin_model_value_kernel` -- THEIR MoveTo
  kernel (`add_model_value.cu:14-53`), flat over rows against the per-row
  `bins` array built once per tree from the partition. It used to be
  `add_model_value_kernel` over an identity row index, whose
  widest-leaf-sized grid cost 13.6 ms/call on a skewed higgs tree
  against 1.0 ms for the whole Logloss evaluation (DEVIATION 210b in
  `kernel_add_model_value.mojo`; PREP_BILL 2026-08-22 step 32).
* `function_value` arrives as per-block partials folded in one fixed host
  order, the file-standard substitution for their block-reduce-plus-
  `atomicAdd` scalar. The fold is FLOAT32, their accumulator's width, and
  the total passes through their `static_cast<float>` truncation
  (`pointwise_oracle.cpp:106`) before it becomes the walker's double --
  it was Float64 until 2026-08-22, which gave our AnyImprovement test
  sub-float32 resolution their walker does not have (PORTING.md 140's
  two extra accepted rounds were exactly that).
* `AddRigdeRegulaizationIfNecessary` (`:109-111`) is a no-op unless
  `AddRidgeToTargetFunction`, which no configuration this repository runs
  sets; omitted, like the Langevin hooks (`oracle_interface.mojo` records
  the terms).
===================================================
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute

from gbdt.methods.greedy_subsets_searcher.depthwise_stage_times import (
    StageTimes,
)

from gbdt.gpu_util.kernel.fill import launch_make_sequence
from gbdt.gpu_util.partitions_reduce import (
    compute_partition_stats,
    partition_stats_chunks,
)
from gbdt.methods.kernel_add_model_value import (
    add_bin_model_value_kernel,
    add_model_value_kernel,
    fill_bins_from_partition_kernel,
    ABMV_BLOCK,
    ABMV_ELEMENTS,
)
from gbdt.methods.leaves_estimation.oracle_interface import (
    LeavesEstimationOracle,
)
from gbdt.options.catboost_options import (
    LEAF_ESTIMATION_EXACT,
    LEAF_ESTIMATION_GRADIENT,
    LEAF_ESTIMATION_NEWTON,
)
from gbdt.methods.kernel.exact_estimation import (
    compute_exact_value_kernel,
)
from gbdt.methods.leaves_estimation.leaves_estimation_helper import (
    ExactQuantileScratch,
    compute_exact_approx,
    make_exact_quantile_scratch,
)
from gbdt.targets.kernel.multilogit import (
    launch_multilogit_second_der,
    launch_multilogit_value_and_der,
    launch_one_vs_all_second_der,
    launch_one_vs_all_value_and_der,
    multilogit_blocks,
)
from gbdt.targets.kernel.pointwise_targets import (
    OBJECTIVE_MAPE,
    OBJECTIVE_MULTICLASS,
    OBJECTIVE_MULTICLASS_OVA,
)
from gbdt.targets.kernel.pointwise_targets import (
    MSE_BLOCK_SIZE,
    launch_approximate,
)


def merge_stage_times(mut dst: StageTimes, src: StageTimes):
    """Fold one instrument's rows into another's, tag by tag.

    The LEBILL pattern made permanent (PREP_BILL 2026-08-22 step 32): the
    oracle self-times its per-eval phases into its OWN `StageTimes` --
    it is constructed and destroyed per tree, so its rows would die with
    it -- and the walker folds them into the FIT-level table the caller
    reports once. Lives here because the oracle owns the field; the
    struct itself is the depthwise lane's
    (`depthwise_stage_times.mojo`) and is not this file's to grow a
    method on."""
    if not src.enabled:
        return
    for i in range(len(src.tags)):
        var found = False
        for j in range(len(dst.tags)):
            if dst.tags[j] == src.tags[i]:
                dst.ns[j] += src.ns[i]
                found = True
                break
        if not found:
            dst.tags.append(src.tags[i].copy())
            dst.ns.append(src.ns[i])


@fieldwise_init
struct BinOptimizedOracle(LeavesEstimationOracle, Movable):
    """One tree's estimation state. Build with `make_bin_optimized_oracle`."""

    var ctx: DeviceContext
    var n_rows: Int
    var bin_count: Int
    var has_weights: Bool
    #: their `SingleBinDim()` (`pointwise_oracle.h:57-64`): the number of
    #: approxes per LEAF. `cursorDim` for every pointwise loss, and
    #: `cursorDim + 1` for MultiClass -- the leaf carries ONE MORE
    #: dimension than the cursor, because the walker solves in the full
    #: `numClasses`-dimensional space and `make_estimation_result`
    #: projects back by subtracting the pinned component.
    var single_bin_dim: Int
    #: `Cursor.GetColumnCount()`: 1 for every pointwise loss,
    #: `numClasses - 1` for MultiClass.
    var cursor_dim: Int
    #: `numClasses`, needed by the multilogit kernels. 0 when not
    #: MultiClass.
    var num_classes: Int
    #: the der / der2 plane buffer for the multi-dimensional path
    var d_multi_der: DeviceBuffer[DType.float32]
    var h_multi_stats: HostBuffer[DType.float32]
    var d_multi_stats: DeviceBuffer[DType.float32]
    var d_multi_partials: DeviceBuffer[DType.float32]
    #: `DerAtPoint`, kept because MultiClass's gradient reconstruction
    #: reads it back (`pointwise_oracle.cpp:93-101`) and their
    #: `WriteSecondDerivatives` CB_ENSUREs it is defined (`:119`)
    var der_at_point: List[Float64]
    var objective: Int
    #: THE KERNEL'S alpha -- their `GetAlpha()`, the ONE float
    #: `PointwiseTargetKernel` receives (`pointwise_targets.cu:451`).
    var alpha: Float32
    #: THE ESTIMATOR'S alpha, WHICH IS A DIFFERENT NUMBER, and conflating
    #: them was a real defect for a day.
    #:
    #: `ComputeWeightedQuantile` does NOT take the kernel's float. It reads
    #: the quantile level out of the loss params map with its own default:
    #:
    #:     auto it = params.find("alpha");
    #:     float alpha = it == params.end() ? 0.5 : FromString<float>(...);
    #:                                    (`leaves_estimation_helper.h:72-74`)
    #:
    #: For MAE and Quantile the two coincide and nothing shows. For MAPE
    #: they do not: `Init`'s MAPE case is a bare `break`
    #: (`pointwise_target_impl.h:261-266`), so the KERNEL gets the member's
    #: declared `0` -- harmless, MAPE's kernel never reads it -- while the
    #: ESTIMATOR gets 0.5. Feeding the kernel's 0 to the quantile search
    #: makes `needWeights` zero and collapses it to the segment start, so
    #: every MAPE leaf became its leaf's MINIMUM residual instead of the
    #: MAPE-weighted median.
    var estimator_alpha: Float32
    var border: Float32
    var estimation_method: Int
    var lambda_reg: Float64
    var min_leaf_weight: Float64
    var wide: Int

    var d_target: DeviceBuffer[DType.float32]
    var d_weights: DeviceBuffer[DType.float32]
    var d_cursor: DeviceBuffer[DType.float32]
    var d_identity: DeviceBuffer[DType.uint32]
    #: their oracle's per-row `Bins` (`pointwise_oracle.h:70`), read off
    #: the partition once per tree; `MoveTo`'s kernel indexes it flat
    var d_bins: DeviceBuffer[DType.uint32]
    var d_leaves: DeviceBuffer[DType.uint32]
    #: `d_leaves`' host staging, HELD AS A FIELD so its enqueued copy can
    #: never outlive it (DEVIATION 1891): as a local in
    #: `make_bin_optimized_oracle` it forced a full per-tree drain whose
    #: only purpose was making that copy a run before the local died.
    #: Riding the oracle, it dies with the oracle -- which every caller
    #: holds past at least one eval drain (the walker syncs every
    #: evaluation; the boosting loop pins the oracle past its tail drain).
    var h_leaves: HostBuffer[DType.uint32]
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

    #: allocated only when the resolved method is Exact; an
    #: `Optional` because the Newton and Gradient paths must not pay
    #: nineteen buffers of `n_rows` they never read.
    var exact: Optional[ExactQuantileScratch]
    var max_leaf_size: Int
    #: the per-eval phase clock (LEBILL rows: est.move / est.approx /
    #: est.pstats / est.readback). Constructed FORCE-DISABLED -- the env
    #: is read once per fit, not per tree -- and adopted from the
    #: fit-level instrument by the walker's timed overload
    #: (`descent_helpers.newton_like_walker_estimate`), which also folds
    #: the rows back out via `merge_stage_times`. When disabled every
    #: call is one Bool test.
    var times: StageTimes

    def point_dim(self) -> Int:
        return self.bin_count * self.single_bin_dim

    def hessian_block_size(self) -> Int:
        """`HessianBlockSize()` (`pointwise_oracle.h:30-39`), their test
        in their order:

            if (method != Newton)                 return 1;
            if (GetHessianType() == Diagonal)     return 1;
            else                                  return SingleBinDim();

        SO A GRADIENT-METHOD MULTICLASS FIT IS DIAGONAL, not blocked --
        the blocked Cholesky is a property of the METHOD as much as of the
        loss, and reading it as "MultiClass is blocked" would send a
        Gradient fit down the wrong arm. Every pointwise loss reports
        `EHessianType::Symmetric` but has `SingleBinDim() == 1`, so the
        second test is what keeps them diagonal, not the first.
        """
        if self.estimation_method != LEAF_ESTIMATION_NEWTON:
            return 1
        if self.objective != OBJECTIVE_MULTICLASS:
            # `GetHessianType()` is Diagonal for MultiClassOneVsAll
            # (`multiclass_targets.h:118-123`) -- its classes are
            # INDEPENDENT logistic regressions, so the Hessian has no
            # off-diagonal to solve -- and every pointwise loss has
            # `SingleBinDim() == 1`, where the blocked arm would be a 1x1
            # Cholesky anyway.
            return 1
        return self.single_bin_dim

    def move_to(mut self, point: List[Float32]) raises:
        """`TBinOptimizedOracle::MoveTo` (`pointwise_oracle.cpp:35-57`).

        THE POINT ARRIVES IN THE WALKER'S GAUGE AND THE CURSOR LIVES IN
        ANOTHER ONE. `point` has `SingleBinDim()` components per bin;
        `CurrentPoint` and the cursor have `cursorDim`. Their very first
        line of work is `MakeEstimationResult(point)` (`:43`), which
        projects one into the other, and the shift is taken between two
        vectors that are BOTH in the cursor's gauge. For every pointwise
        loss the projection is the identity and the distinction is
        invisible; for MultiClass it is the whole reason the approxes do
        not drift.

        THE TWO LAYOUTS: the shift is BIN-MAJOR
        (`newPoint[bin * cursorDim + dim]`) and the cursor is
        PLANE-MAJOR. `add_model_value_kernel`'s z axis is where they meet.
        """
        if len(point) != self.bin_count * self.single_bin_dim:
            raise Error(
                "MoveTo: point holds " + String(len(point))
                + " for " + String(self.bin_count) + " bins x "
                + String(self.single_bin_dim) + " dims"
            )
        self.times.begin(self.ctx)
        # `TVector<float> newPoint = MakeEstimationResult(point);` (`:43`)
        var new_point = self.make_estimation_result(point)

        # the last eval's synchronize is what makes h_shift reusable here;
        # the walker is strictly move -> eval -> move.
        var shift_len = self.bin_count * self.cursor_dim
        for i in range(shift_len):
            self.h_shift.unsafe_ptr().unsafe_store(
                i, new_point[i] - self.current_point[i]
            )
        self.ctx.enqueue_copy(
            dst_buf=self.d_shift, src_ptr=self.h_shift.unsafe_ptr()
        )
        # their `AddBinModelValue(shift, bins, ...)` (`pointwise_oracle
        # .cpp:50-52`): the FLAT kernel over rows, `CeilDivide(size,
        # blockSize * elementsPerThreads)` blocks (`add_model_value.cu
        # :60-62`). DEVIATION 210b in `kernel_add_model_value.mojo` has
        # the 13.6-ms-per-call grid this replaces.
        var abmv_blocks = (
            self.n_rows + ABMV_BLOCK * ABMV_ELEMENTS - 1
        ) // (ABMV_BLOCK * ABMV_ELEMENTS)
        self.ctx.enqueue_function[add_bin_model_value_kernel](
            self.d_shift.unsafe_ptr(),
            self.d_bins.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.cursor_dim),
            Int32(self.n_rows),
            self.d_cursor.unsafe_ptr(),
            grid_dim=(abmv_blocks, 1, 1),
            block_dim=(ABMV_BLOCK, 1, 1),
        )
        # `DerAtPoint.Clear(); Der2AtPoint.Clear();` (`:54-55`)
        self.cached_der2.clear()
        self.der_at_point.clear()
        self.current_point.clear()
        for i in range(shift_len):
            self.current_point.append(new_point[i])
        self.times.end(self.ctx, "est.move")

    def write_value_and_first_derivatives(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ) raises:
        """`WriteValueAndFirstDerivatives` (`pointwise_oracle.cpp:59-112`).

        TWO ARMS, theirs, on `rowSize == 1` (`:70-77`):

          rowSize == 1  the FAST PATH for pools with few features and many
                        docs: `ApproximateAt(cursor, &value, &der, &der2)`
                        computes value, der AND der2 in ONE kernel, and
                        `Der2AtPoint` is cached with lambda added.
          rowSize >  1  `ComputeValueAndDerivative(cursor, &value, &der)`
                        only. der2 is NOT computed here; it costs a launch
                        per Hessian row and is paid for in
                        `write_second_derivatives`.

        THE MULTICLASS GRADIENT HAS ONE MORE COMPONENT THAN THE KERNEL
        WRITES (`:93-101`). The cursor carries `cursorDim = numClasses - 1`
        free classes; the leaf carries `SingleBinDim() = cursorDim + 1`.
        The missing component is not computed, it is RECONSTRUCTED:

            total = sum over dim of DerAtPoint[bin*cursorDim + dim]
            gradient[bin*rowSize + cursorDim] = -total

        which is exact because the multinomial gradient sums to zero over
        ALL numClasses. `original/multilogit_check.mojo` gates that
        identity on the kernel directly, which is what makes this line
        safe to write rather than merely plausible.
        """
        var blocks = (
            self.n_rows + MSE_BLOCK_SIZE - 1
        ) // MSE_BLOCK_SIZE

        if self.single_bin_dim == 1:
            self.times.begin(self.ctx)
            launch_approximate[True](
                self.ctx, self.objective,
                self.d_target, self.d_weights, Int32(self.n_rows),
                self.d_cursor,
                Int32(1) if self.has_weights else Int32(0),
                self.alpha, self.border,
                self.d_eval_stats, self.d_fv, Int32(1),
                self.d_mag_dummy, Int32(0),
                blocks,
            )
            self.times.end(self.ctx, "est.approx")
            self.times.begin(self.ctx)
            compute_partition_stats(
                self.ctx, self.bin_count, 0, 2, self.n_rows,
                self.d_leaves, self.d_p_off, self.d_p_sz,
                self.d_eval_stats, self.d_partials, self.d_part_stats,
                sm_count=self.sm_count,
            )
            self.times.end(self.ctx, "est.pstats")
            self.times.begin(self.ctx)
            self.ctx.enqueue_copy(
                dst_ptr=self.h_part_stats.unsafe_ptr(),
                src_buf=self.d_part_stats,
            )
            self.ctx.enqueue_copy(
                dst_ptr=self.h_fv.unsafe_ptr(), src_buf=self.d_fv
            )
            # KEPT (DEVIATION 1891 audit): the ONE drain per walker
            # evaluation, and it is required -- the host reads
            # `h_part_stats` and `h_fv` immediately below to decide the
            # line search. Both readbacks are already batched ahead of
            # this single drain, and no per-evaluation allocation exists
            # on this path, so this is the floor CatBoost's pinned-memory
            # `ReadReduce` also pays (theirs is cheaper per drain, not
            # fewer drains).
            self.ctx.synchronize()
            self.times.end(self.ctx, "est.readback")

            gradient.clear()
            self.cached_der2.clear()
            for leaf in range(self.bin_count):
                gradient.append(
                    Float64(
                        self.h_part_stats.unsafe_ptr().unsafe_load(
                            2 * leaf
                        )
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
            # `(*value) = static_cast<float>(ReadReduce(valueGpu)[0]);`
            # (`pointwise_oracle.cpp:106`). THE VALUE IS A FLOAT32 NUMBER
            # in their walker: the target kernel block-reduces and
            # atomicAdds per-block FLOAT partials into one float scalar,
            # and the cast keeps it float on the way into the double. The
            # fold below is the file-standard deterministic substitution
            # for their atomic's order; its WIDTH is now theirs too. It
            # was Float64, which let AnyImprovement see improvements
            # BELOW float32 resolution and accept steps their walker
            # cannot see -- PORTING.md 140 measured ours accepting 8
            # rounds where 6-7 sit at the float32 noise floor. Found
            # 2026-08-22 in the Newton-walk audit; the walk-divergence
            # entry carries the measurement.
            var fv32 = Float32(0.0)
            for b in range(blocks):
                fv32 += self.h_fv.unsafe_ptr().unsafe_load(b)
            value = Float64(fv32)
            return

        # ---- the rowSize > 1 arm: the multiclass family --------------
        var is_ova = self.objective == OBJECTIVE_MULTICLASS_OVA
        if self.objective != OBJECTIVE_MULTICLASS and not is_ova:
            raise Error(
                "the multi-dimensional oracle arm is the multiclass"
                " family only; objective " + String(self.objective)
                + " has SingleBinDim > 1 without a der calcer"
            )
        var ml_blocks = multilogit_blocks(self.n_rows)
        self.times.begin(self.ctx)
        if is_ova:
            launch_one_vs_all_value_and_der[False](
                self.ctx, self.num_classes, self.n_rows,
                self.d_target, self.d_weights, self.has_weights,
                self.d_cursor, self.n_rows,
                self.d_identity, False,
                self.d_fv, True,
                self.d_multi_der, self.n_rows,
                self.d_mag_dummy, False,
            )
        else:
            launch_multilogit_value_and_der(
                self.ctx, self.num_classes, self.n_rows,
                self.d_target, self.d_weights, self.has_weights,
                self.d_cursor, self.n_rows,
                self.d_identity, False,
                self.d_fv, True,
                self.d_multi_der, self.n_rows,
                self.d_mag_dummy, False,
            )
        self.times.end(self.ctx, "est.approx")
        # `ComputePartitionStats(der, Offsets, &reducedDer)` (`:83`), with
        # `cursorDim` columns instead of one
        self.times.begin(self.ctx)
        compute_partition_stats(
            self.ctx, self.bin_count, 0, self.cursor_dim, self.n_rows,
            self.d_leaves, self.d_p_off, self.d_p_sz,
            self.d_multi_der, self.d_multi_partials, self.d_multi_stats,
            sm_count=self.sm_count,
        )
        self.times.end(self.ctx, "est.pstats")
        self.times.begin(self.ctx)
        self.ctx.enqueue_copy(
            dst_ptr=self.h_multi_stats.unsafe_ptr(),
            src_buf=self.d_multi_stats,
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.h_fv.unsafe_ptr(), src_buf=self.d_fv
        )
        self.ctx.synchronize()
        self.times.end(self.ctx, "est.readback")

        # `DerAtPoint = ReadReduce(reducedDer)` (`:84`)
        self.der_at_point.clear()
        for i in range(self.bin_count * self.cursor_dim):
            self.der_at_point.append(
                Float64(self.h_multi_stats.unsafe_ptr().unsafe_load(i))
            )

        # their MultiClass reconstruction (`:93-101`), gated on the LOSS
        # exactly as theirs is: `if (DerCalcer->GetType() ==
        # ELossFunction::MultiClass)`. MultiClassOneVsAll has no pinned
        # class, so `SingleBinDim() == cursorDim` and the gradient is
        # their `(*gradient) = *DerAtPoint` (`:102-104`) unchanged.
        gradient.clear()
        for _ in range(self.bin_count * self.single_bin_dim):
            gradient.append(Float64(0.0))
        if is_ova:
            for i in range(self.bin_count * self.cursor_dim):
                gradient[i] = self.der_at_point[i]
        else:
            for bin in range(self.bin_count):
                var total = Float64(0.0)
                for dim in range(self.cursor_dim):
                    var val = self.der_at_point[
                        bin * self.cursor_dim + dim
                    ]
                    gradient[bin * self.single_bin_dim + dim] = val
                    total += val
                # `sum of der is equal to zero` (`:100`)
                gradient[
                    bin * self.single_bin_dim + self.cursor_dim
                ] = -total

        # der2 is NOT cached on this arm; `write_second_derivatives`
        # recomputes it row by row, exactly as theirs does
        self.cached_der2.clear()

        # the same `static_cast<float>` (`pointwise_oracle.cpp:106`) as the
        # single-dim arm: fold the block partials in FLOAT32, their width.
        var mfv32 = Float32(0.0)
        for b in range(ml_blocks):
            mfv32 += self.h_fv.unsafe_ptr().unsafe_load(b)
        value = Float64(mfv32)

    def write_second_derivatives(mut self, mut second_der: List[Float64]) raises:
        """`WriteSecondDerivatives` (`pointwise_oracle.cpp:114-195`).

        TWO ARMS, theirs, keyed on `LeavesEstimationConfig
        .LeavesEstimationMethod`:

        * NEWTON (`:122-124`) returns the `Der2AtPoint` cache the eval
          pass filled -- the reduced `weight * Der2` per bin, plus lambda.
        * GRADIENT (`:185-193`) IGNORES the second derivative entirely and
          returns `WeightsCpu[bin] + lambda`. The walker's
          `MoveDirection = Gradient / (Hessian + 1e-20)` then becomes a
          weight-normalized gradient step, which is what "gradient
          descent on the leaves" means in their code.

        That second arm is why four objectives whose `Der2` is identically
        zero -- Quantile, MAE, MAPE, LogLinQuantile -- can be trained at
        all: under Newton their Hessian would be `lambda` alone, and the
        leaf value would be the summed gradient over the L2 term rather
        than anything scaled to the leaf.

        Their `CB_ENSURE(method != Exact)` (`:121`) is here too: Exact
        never reaches the walker, it replaces it (`EstimateExact`,
        `:204-213`).
        """
        if self.estimation_method == LEAF_ESTIMATION_GRADIENT:
            # `:185-193`, and note the INNER LOOP over `rowSize`: the
            # weight is repeated across every approx dimension, so a
            # multi-dimensional Gradient fit gets `rowSize` copies of its
            # leaf weight rather than one.
            second_der.clear()
            for leaf in range(self.bin_count):
                var w = self.weights_cpu[leaf] + self.lambda_reg
                for _ in range(self.single_bin_dim):
                    second_der.append(w)
            return
        if self.estimation_method != LEAF_ESTIMATION_NEWTON:
            raise Error(
                "WriteSecondDerivatives: only Newton and Gradient reach"
                " the walker; Exact replaces it"
            )

        # ---- the BLOCKED arm (`pointwise_oracle.cpp:125-184`) --------
        if self.hessian_block_size() > 1:
            self._write_blocked_second_derivatives(second_der)
            return

        # ---- DIAGONAL but MULTI-COLUMN: MultiClassOneVsAll -----------
        # `hessianBlockSize == 1` with `rowSize > 1` is their generic
        # blocked path at `blockCount == rowSize` (`:129-133`), which
        # degenerates to one launch of `ComputeSecondDerRowLowerTriangle`
        # at row 0 producing `rowSize` columns. That IS the OneVsAll
        # second-der kernel, which writes every class plane in one go
        # because its Hessian has no off-diagonal.
        if self.objective == OBJECTIVE_MULTICLASS_OVA:
            launch_one_vs_all_second_der(
                self.ctx, self.num_classes, self.n_rows,
                self.d_weights, self.has_weights,
                self.d_cursor, self.n_rows,
                self.d_multi_der, self.n_rows,
            )
            compute_partition_stats(
                self.ctx, self.bin_count, 0, self.cursor_dim,
                self.n_rows,
                self.d_leaves, self.d_p_off, self.d_p_sz,
                self.d_multi_der, self.d_multi_partials,
                self.d_multi_stats,
                sm_count=self.sm_count,
            )
            self.ctx.enqueue_copy(
                dst_ptr=self.h_multi_stats.unsafe_ptr(),
                src_buf=self.d_multi_stats,
            )
            self.ctx.synchronize()
            second_der.clear()
            for i in range(self.bin_count * self.cursor_dim):
                second_der.append(
                    Float64(
                        self.h_multi_stats.unsafe_ptr().unsafe_load(i)
                    )
                    + self.lambda_reg
                )
            return

        if len(self.cached_der2) != self.bin_count:
            raise Error(
                "WriteSecondDerivatives before WriteValueAndFirst"
                "Derivatives: the der2 cache is empty, the call order broke"
            )
        second_der.clear()
        for leaf in range(self.bin_count):
            second_der.append(self.cached_der2[leaf])

    def estimate_exact(mut self) raises -> List[Float32]:
        """`TBinOptimizedOracle::EstimateExact`
        (`pointwise_oracle.cpp:204-213`).

            auto values  = TStripeBuffer<float>::CopyMapping(Bins);
            auto weights = TStripeBuffer<float>::CopyMapping(Bins);
            DerCalcer->ComputeExactValue(Cursor.AsConstBuf(), &values,
                                         &weights);
            TVector<float> point(BinCount * SingleBinDim());
            ComputeExactApprox(Bins, values, weights, BinCount, point,
                               LossDescription);
            MoveTo(point);
            return MakeEstimationResult(point);

        `MoveTo` at the end is not decoration: the walker never runs for
        this method, so this is the only place the oracle's cursor copy
        learns the value it just produced, and `WeightsCpu`-based
        regularization is applied by the caller exactly as
        `MakeEstimationResult` -> `RegularizeImpl` does for the others.
        """
        if self.estimation_method != LEAF_ESTIMATION_EXACT:
            raise Error(
                "estimate_exact called with a non-Exact estimation method"
            )
        if not self.exact:
            raise Error(
                "estimate_exact: the Exact scratch was never allocated;"
                " make_bin_optimized_oracle was told a different method"
            )
        var blocks = (self.n_rows + 255) // 256
        if blocks < 1:
            blocks = 1
        # `DerCalcer->ComputeExactValue(Cursor, &values, &weights)`
        self.ctx.enqueue_function[compute_exact_value_kernel](
            self.d_target.unsafe_ptr(),
            self.d_cursor.unsafe_ptr(),
            self.d_weights.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(1) if self.has_weights else Int32(0),
            self.exact.value().residuals.unsafe_ptr(),
            self.exact.value().residual_weights.unsafe_ptr(),
            grid_dim=blocks, block_dim=256,
        )
        var point = List[Float32]()
        compute_exact_approx(
            self.ctx,
            self.objective,
            self.objective == OBJECTIVE_MAPE,
            self.n_rows,
            self.bin_count,
            self.max_leaf_size,
            self.estimator_alpha,
            self.d_p_off,
            self.d_p_sz,
            self.exact.value(),
            point,
        )
        # `MoveTo(point)` (`:211`)
        self.move_to(point)
        # `MakeEstimationResult(point)` -> `RegularizeImpl`
        self.regularize(point)
        return point^

    def _write_blocked_second_derivatives(
        mut self, mut second_der: List[Float64]
    ) raises:
        """The blocked lower-triangular Hessian (`:125-184`).

        Their shape, and every constant in it:

            hessianBlockSize          = SingleBinDim()  = numClasses
            matrixSize                = hbs * hbs
            blockCount                = rowSize / hbs   = 1 here
            lowTriangleMatrixSize     = hbs * (hbs + 1) / 2

        then one launch per Hessian ROW (`:140-151`), each producing
        `hessianBlockRow + 1` columns, each reduced per bin; then the
        lower triangle is MIRRORED into the upper one and `lambda` is
        added to the diagonal (`:159-181`).

        NOTE WHICH ROWS EXIST. The loop runs `hessianBlockRow` over
        `[0, numClasses)`, so its LAST iteration asks
        `MultiLogitSecondDerRowImpl` for row `numClasses - 1`, which is
        `der2Row == effectiveClassCount` -- the arm that reads
        `exp(-maxApprox)` instead of a prediction plane
        (`multilogit.cu:157-158`). That branch is not defensive: it is the
        PINNED class's row, and the Hessian is `numClasses x numClasses`
        even though the cursor has `numClasses - 1` planes. A port that
        stopped the loop at `cursorDim` would build a matrix one row
        short and the Cholesky would solve a different system.

        ================= DEVIATION BLOCK =================
        DEVIATION 75: ONE REDUCE PER ROW INTO ITS OWN BUFFER, where theirs
        reduces every row into disjoint SLICES of one
        `reducedHessianGpu` and reads the whole thing back once
        (`:135-157`). Their slice arithmetic exists because `ReadReduce`
        is one call over one buffer; ours reads each row's reduce as it is
        produced, which is `numClasses` copies of `binCount * (row + 1)`
        floats instead of one copy of `binCount * lowTriangleMatrixSize`.
        SAME NUMBERS, same order within a row. It costs `numClasses - 1`
        extra device-to-host copies per estimation iteration and buys not
        having to reproduce their `offset` bookkeeping, which is the part
        of their function most likely to be transcribed wrong.
        ===================================================
        """
        var hbs = self.single_bin_dim
        var matrix_size = hbs * hbs

        # `secondDer->resize(singleBinBlockedMatrixSize * BinCount)`
        second_der.clear()
        for _ in range(matrix_size * self.bin_count):
            second_der.append(Float64(0.0))

        for row in range(hbs):
            var column_count = row + 1
            launch_multilogit_second_der(
                self.ctx, self.num_classes, self.n_rows,
                self.d_weights, self.has_weights,
                self.d_cursor, self.n_rows,
                self.d_multi_der, row, self.n_rows,
            )
            compute_partition_stats(
                self.ctx, self.bin_count, 0, column_count, self.n_rows,
                self.d_leaves, self.d_p_off, self.d_p_sz,
                self.d_multi_der, self.d_multi_partials,
                self.d_multi_stats,
                sm_count=self.sm_count,
            )
            self.ctx.enqueue_copy(
                dst_ptr=self.h_multi_stats.unsafe_ptr(),
                src_buf=self.d_multi_stats,
            )
            self.ctx.synchronize()

            # mirror this row into both triangles (`:166-180`)
            for bin in range(self.bin_count):
                var base = bin * matrix_size
                for col in range(column_count):
                    var val = Float64(
                        self.h_multi_stats.unsafe_ptr().unsafe_load(
                            bin * column_count + col
                        )
                    )
                    if col == row:
                        # `sigma[row*hbs + row] = ... + lambda` (`:178`)
                        second_der[base + row * hbs + row] = (
                            val + self.lambda_reg
                        )
                    else:
                        second_der[base + row * hbs + col] = val
                        second_der[base + col * hbs + row] = val

    def make_estimation_result(
        self, point: List[Float32]
    ) -> List[Float32]:
        """`MakeEstimationResult` (`pointwise_oracle.cpp:18-33`).

        Identity for every single-dimensional loss. For MultiClass it is
        the GAUGE FIXING that makes the whole scheme work:

            newPoint[bin*cursorDim + dim] =
                point[bin*SingleBinDim() + dim]
              - point[bin*SingleBinDim() + cursorDim]

        The walker solves in the full `numClasses`-dimensional space,
        where the softmax's shift invariance leaves the Hessian singular
        and only `+ lambda` makes it solvable. Subtracting the pinned
        component re-pins the last class at zero, which is the gauge the
        CURSOR is stored in. Without it the approxes drift by a common
        constant every iteration -- invisibly, since the predictions do
        not change -- until the exponentials leave float32 range.
        """
        if self.objective != OBJECTIVE_MULTICLASS:
            return point.copy()
        var out = List[Float32]()
        for bin in range(self.bin_count):
            for dim in range(self.cursor_dim):
                out.append(
                    point[bin * self.single_bin_dim + dim]
                    - point[bin * self.single_bin_dim + self.cursor_dim]
                )
        return out^

    def regularize(self, mut point: List[Float32]):
        """`RegularizeImpl` (`oracle_interface.h:43-52`), with their
        `approxDim` argument (`pointwise_oracle.cpp:13-16`).

        A leaf under `MinLeafWeight` is zeroed in EVERY approx dimension,
        not only the first -- their inner `for dim` loop (`:48-50`).
        """
        var dim_count = self.single_bin_dim
        for bin in range(self.bin_count):
            if self.weights_cpu[bin] < self.min_leaf_weight:
                for dim in range(dim_count):
                    point[bin * dim_count + dim] = Float32(0.0)


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
    objective: Int,
    alpha: Float32,
    estimator_alpha: Float32,
    border: Float32,
    lambda_reg: Float64,
    sm_count: Int,
    estimation_method: Int = LEAF_ESTIMATION_NEWTON,
    num_classes: Int = 0,
) raises -> BinOptimizedOracle:
    """Their ctor (`pointwise_oracle.cpp:218-246`): allocate the eval
    buffers, seed `CurrentPoint` at zero, and settle `WeightsCpu` once --
    the weights never move during estimation."""
    # `SingleBinDim()` and `Cursor.GetColumnCount()` (`pointwise_oracle.h
    # :57-64`): MultiClass carries `numClasses - 1` cursor planes and
    # `numClasses` leaf dimensions; everything else is 1 and 1.
    var cursor_dim = 1
    var single_bin_dim = 1
    if objective == OBJECTIVE_MULTICLASS:
        if num_classes < 2:
            raise Error(
                "MultiClass oracle needs num_classes >= 2, got "
                + String(num_classes)
            )
        cursor_dim = num_classes - 1
        single_bin_dim = num_classes
    elif objective == OBJECTIVE_MULTICLASS_OVA:
        # `GetDim()` is `NumClasses` here, not `NumClasses - 1`
        # (`multiclass_targets.h:129-134`): no pinned class, so
        # `SingleBinDim() == cursorDim` and there is no gauge to fix.
        if num_classes < 2:
            raise Error(
                "MultiClassOneVsAll oracle needs num_classes >= 2, got "
                + String(num_classes)
            )
        cursor_dim = num_classes
        single_bin_dim = num_classes
    # one buffer wide enough for both the value pass (`cursor_dim`
    # planes) and the widest Hessian row (`single_bin_dim` columns)
    var multi_planes = single_bin_dim

    var d_identity = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    launch_make_sequence(ctx, UInt32(0), d_identity, n_rows)

    var d_bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)

    var d_leaves = ctx.enqueue_create_buffer[DType.uint32](bin_count)
    var h_leaves = ctx.enqueue_create_host_buffer[DType.uint32](bin_count)
    for i in range(bin_count):
        h_leaves.unsafe_ptr().unsafe_store(i, UInt32(i))
    ctx.enqueue_copy(dst_buf=d_leaves, src_ptr=h_leaves.unsafe_ptr())

    var d_shift = ctx.enqueue_create_buffer[DType.float32](
        bin_count * cursor_dim
    )
    var h_shift = ctx.enqueue_create_host_buffer[DType.float32](
        bin_count * cursor_dim
    )
    var d_eval_stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var d_fv = ctx.enqueue_create_buffer[DType.float32](blocks)
    var h_fv = ctx.enqueue_create_host_buffer[DType.float32](blocks)
    var d_mag_dummy = ctx.enqueue_create_buffer[DType.float32](2)

    var sm = sm_count
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    # their oracle's per-row `Bins`, read off the partition ONCE per tree
    # (their ctor receives it ready-made from the searcher). Machine-sized
    # x, strided; `MoveTo` reads this every evaluation.
    var bins_gx = 2 * sm
    if bins_gx < 1:
        bins_gx = 1
    ctx.enqueue_function[fill_bins_from_partition_kernel](
        d_p_off.unsafe_ptr(), d_p_sz.unsafe_ptr(), d_bins.unsafe_ptr(),
        grid_dim=(bins_gx, bin_count, 1),
        block_dim=(256, 1, 1),
    )

    var chunks2 = partition_stats_chunks(sm, 2)
    var chunks1 = partition_stats_chunks(sm, 1)
    var partials_len = bin_count * 2 * chunks2
    if bin_count * 1 * chunks1 > partials_len:
        partials_len = bin_count * 1 * chunks1
    var d_partials = ctx.enqueue_create_buffer[DType.float32](partials_len)
    # the multi-dimensional reduce needs its own partials, sized for the
    # WIDEST stat count it will ever be asked for
    var multi_partials_len = 1
    for sc in range(1, multi_planes + 1):
        var need = bin_count * sc * partition_stats_chunks(sm, sc)
        if need > multi_partials_len:
            multi_partials_len = need
    var d_multi_partials = ctx.enqueue_create_buffer[DType.float32](
        multi_partials_len
    )
    var d_part_stats = ctx.enqueue_create_buffer[DType.float32](
        2 * bin_count
    )
    var h_part_stats = ctx.enqueue_create_host_buffer[DType.float32](
        2 * bin_count
    )

    # allocated ONCE per tree rather than per Hessian row
    var d_multi_der = ctx.enqueue_create_buffer[DType.float32](
        multi_planes * n_rows
    )
    var d_multi_stats = ctx.enqueue_create_buffer[DType.float32](
        multi_planes * bin_count
    )
    var h_multi_stats = ctx.enqueue_create_host_buffer[DType.float32](
        multi_planes * bin_count
    )

    # `CurrentPoint` lives in the CURSOR's gauge -- `cursorDim` per bin,
    # not `SingleBinDim()` -- because `MoveTo` projects before it
    # subtracts (`pointwise_oracle.cpp:43-47`).
    var current_point = List[Float32]()
    for _ in range(bin_count * cursor_dim):
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

    # the Exact scratch, and ONLY when Exact is the resolved method
    var exact = Optional[ExactQuantileScratch](None)
    if estimation_method == LEAF_ESTIMATION_EXACT:
        exact = Optional(
            make_exact_quantile_scratch(ctx, n_rows, bin_count, max_leaf)
        )

    # THE HOST STAGING BUFFER MUST OUTLIVE ITS ENQUEUED COPY. `h_leaves`'
    # last use used to be its `enqueue_copy`, so Mojo freed it there while
    # the copy sat in the queue -- a use-after-free whose window OPENS
    # UNDER CPU CONTENTION (the freed pages get reused before the queue
    # drains), which is precisely the signature of the divergent
    # full-higgs fit of 2026-08-22 (PREP_BILL step 33). The first fix was
    # a `ctx.synchronize()` here plus `_ = h_leaves^`: the drain made the
    # copy a RUN, only then could the buffer die (the 730cc20 rule: an
    # enqueue is not a run).
    #
    # DEVIATION 1891 keeps the rule and drops the drain: `h_leaves` is now
    # a FIELD of the oracle (declared beside `d_leaves`), so it lives as
    # long as every other buffer the estimation reads and its copy is a
    # run long before the oracle can die -- the walker drains at every
    # evaluation, `estimate_exact`'s readback drains, and the boosting
    # loop additionally pins the oracle past the task's tail drain. One
    # full device drain per estimation task removed; enqueue order is
    # untouched, so the change is bit-inert.

    # FORCE-DISABLED regardless of the environment: the fit reads the env
    # ONCE (its own `StageTimes()`), and the walker's timed overload
    # copies that enabled flag in here per tree. An env read per oracle
    # would be 500 per fit and, worse, would let an UN-merged caller pay
    # per-eval drains for rows nobody reports.
    var est_times = StageTimes()
    est_times.enabled = False

    return BinOptimizedOracle(
        ctx,
        n_rows,
        bin_count,
        has_weights,
        single_bin_dim,
        cursor_dim,
        num_classes,
        d_multi_der^,
        h_multi_stats^,
        d_multi_stats^,
        d_multi_partials^,
        List[Float64](),
        objective,
        alpha,
        estimator_alpha,
        border,
        estimation_method,
        lambda_reg,
        1e-20,  # MinLeafWeight, their hardcoded default
        wide,
        d_target^,
        d_weights^,
        d_cursor^,
        d_identity^,
        d_bins^,
        d_leaves^,
        h_leaves^,
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
        exact^,
        max_leaf,
        est_times^,
    )
