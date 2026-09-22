# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`TBinOptimizedOracle`: the walker's device-side eyes, per-bin.

Reference: `catboost/cuda/methods/leaves_estimation/pointwise_oracle.{h,cpp}`
(CatBoost `54a8143a`), the rowSize==1 arm -- every single-dim pointwise
loss.

WHAT THE ORACLE HOLDS, in the reference layout: the target, weights and CURSOR
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
  WriteSecondDer      returns the cache (`:114-117`); this implementation RAISES if
                      the cache is empty, because for rowSize==1 their
                      `ApproximateAt` always fills it and an empty cache
                      here means the call order broke
  Regularize          `RegularizeImpl` (`oracle_interface.h:43-52`): zero
                      every bin whose weight sum is under MinLeafWeight
                      (their hardcoded 1e-20)

================= DEVIATION BLOCK =================
* THE REDUCES RUN IN FLOAT32 where theirs land in `TStripeBuffer<double>`
  (`:81`, `:85`). This is `compute_partition_stats`'s existing implementation-wide
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
  sub-float32 resolution their walker does not have (the
  two extra accepted rounds were exactly that).
* `AddRigdeRegulaizationIfNecessary` (`:109-111`) is a no-op unless
  `AddRidgeToTargetFunction`, which no configuration this repository runs
  sets; omitted, like the Langevin hooks (`oracle_interface.mojo` records
  the terms).
===================================================
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.host.device_attribute import DeviceAttribute
from gbdt.gpu_util.arena import BufferArena

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
    launch_approximate_move_eval,
)
from gbdt.targets.kernel.query_rmse import (
    QuerywiseTargetBuffers,
    launch_query_rmse_with,
)
from gbdt.targets.kernel.pair_logit import (
    PairwiseTargetBuffers,
    launch_pair_logit_with,
)
from gbdt.targets.kernel.yeti_rank import (
    YetiRankTargetBuffers,
    launch_yeti_rank_with,
)
from gbdt.data.permutation import TRandom
from std.sys.compile import is_defined
from std.gpu import block_idx, thread_idx

from checks.kernel_matrix import COLUMN_NVIDIA, TARGET_COLUMN

# ================= DEVIATION BLOCK 2030 =================
# FUSED MoveTo + evaluation for the Newton walker (single-dim losses).
#
# The walker calls `move_to` and `write_value_and_first_derivatives`
# strictly paired (`descent_helpers.cpp:128-204`: the initial evaluation,
# then every backtracking try). The split schedule pays a full-row
# `add_bin_model_value_kernel` pass per pair whose only purpose is to put
# cursor bytes where the evaluation kernel reads them one launch later.
# Under this flag `move_to` records the shift (host arithmetic + the
# existing h2d copy of `d_shift`) and DEFERS the cursor update; the next
# evaluation launches the fused kernel
# (`pointwise_targets.launch_approximate_move_eval`), which performs the
# identical per-row add, stores the identical cursor bytes, and runs the
# UNCHANGED evaluation body on the just-stored value. Bit-identical in
# every tier -- same operations, same operands, same order; the kernel
# block in pointwise_targets.mojo carries the argument.
#
# SCOPE: `cursor_dim == 1 and single_bin_dim == 1` only (every pointwise
# loss). The multiclass family keeps the split schedule; `pending_shift`
# can never be set on its arms. Any cursor reader outside the pair
# (`estimate_exact`) flushes the pending shift through the ORIGINAL
# `add_bin_model_value_kernel` launch first, so the cursor is never
# observed stale.
#
# OFF BY DEFAULT. Arm B builds with `-D MOJOLEARN_2030_FUSED_EST_MOVE=1`.
# `-D MOJOLEARN_2030_NO_FUSED_EST_MOVE=1` takes precedence for baseline A/B.
# Kernel equivalence is checked by checks/gbdt_fused_move_check.mojo.
# Native fit A/B: checks/gbdt_fused_fit_check.mojo (baseline versus define).
# NVIDIA small-fixture fingerprints passed on L40S and H100. Per-device
# end-to-end timing still governs
# default selection. Additional A/B commands are in
# archive/research/gbdt/UPSTREAM_SURVEY_2026-09.md and the PLAN appendix; the gates that
# must hold are `check-fit-pointwise`, `check-logloss-train` and
# `check-ordered-boosting` built with the define (the changed arm is the
# Logloss estimation path, and A GATE MUST EXERCISE THE CHANGED ARM --
# DEVIATION 2009's lesson).
# ====================================================
comptime FUSED_EST_MOVE_2030 = (
    is_defined["MOJOLEARN_2030_FUSED_EST_MOVE"]()
    and not is_defined["MOJOLEARN_2030_NO_FUSED_EST_MOVE"]()
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
    #: DEVIATION 2030: True between a deferred `move_to` and the fused
    #: evaluation that applies it. Always False unless the build defines
    #: `MOJOLEARN_2030_FUSED_EST_MOVE` AND the loss is single-dim; every
    #: cursor reader outside the move/eval pair flushes it first.
    var pending_shift: Bool
    #: the QUERYWISE target's grouping and scratch, present only for
    #: QueryRMSE: their `TPermutationDerCalcer<TTarget, Querywise>`
    #: (`targets/permutation_der_calcer.h:171-247`) reads the point through
    #: the inverse of this oracle's bin order and keeps the targets in row
    #: order, where the pointwise calcer gathers them (`:57-72`).
    var query: Optional[QuerywiseTargetBuffers]
    #: the PairLogit pairs and scratch, present only for PairLogit: the same
    #: querywise der calcer over per-pair derivatives
    #: (`gbdt/targets/kernel/pair_logit.mojo`)
    var pairs: Optional[PairwiseTargetBuffers]
    #: how many value partials an evaluation writes and the host folds: one
    #: per 256 rows, or one per 256 PAIRS for PairLogit
    var fv_blocks: Int
    #: the YetiRank task table and scratch, present only for YetiRank: the
    #: querywise der calcer over sampled permutations
    #: (`gbdt/targets/kernel/yeti_rank.mojo`)
    var yeti: Optional[YetiRankTargetBuffers]
    #: this tree's YetiRank draw stream: one `NextUniformL` per evaluation,
    #: the seed of that call's task streams (`querywise_targets_impl.h:214`;
    #: the stream is the fit's per-tree draw, see `doc_parallel_boosting.mojo`)
    var yeti_rng: TRandom
    #: the DEFERRED weight sums (`make_bin_optimized_oracle(...,
    #: defer_weights=True)`): the per-leaf weight fold's host copy, read into
    #: `weights_cpu` by `settle_weights` after the caller's next drain. None
    #: on every oracle built the ordinary way, whose constructor drains and
    #: fills `weights_cpu` itself.
    var h_weight_stats: Optional[HostBuffer[DType.float32]]

    def settle_weights(mut self) raises:
        """`WeightsCpu` from the deferred weight fold (see
        `h_weight_stats`): the SAME per-leaf float32 sums the constructor's
        drained readback reads, widened the same way. The caller must have
        drained the queue since construction. A no-op on an ordinary
        oracle."""
        if not self.h_weight_stats.__bool__():
            return
        var h = self.h_weight_stats.take()
        self.weights_cpu.clear()
        for leaf in range(self.bin_count):
            self.weights_cpu.append(Float64(h.unsafe_ptr().unsafe_load(leaf)))
        _ = h^

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

        # DEVIATION 2030, the defensive corner: two `move_to` calls with
        # no evaluation between them (the walker never does this -- its
        # calls are strictly move -> eval -> move). The earlier shift is
        # still whole in `d_shift` (its copy was enqueued before anything
        # below), so apply it through the ORIGINAL kernel, then drain so
        # the `h_shift` rewrite below cannot race the in-flight DMA of
        # the deferred copy.
        @parameter
        if FUSED_EST_MOVE_2030:
            if self.pending_shift:
                self._launch_shift_abmv()
                self.pending_shift = False
                self.ctx.synchronize()

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
        # DEVIATION 2030: under the flag, on the single-dim arm, the
        # cursor update is DEFERRED into the next evaluation's fused
        # kernel -- the launch below is skipped and the shift rides
        # `d_shift` until `write_value_and_first_derivatives` (or a
        # flush) applies it. Every other configuration launches their
        # kernel exactly as before.
        var defer_shift = False

        @parameter
        if FUSED_EST_MOVE_2030:
            defer_shift = (
                self.cursor_dim == 1 and self.single_bin_dim == 1
                and not self.query.__bool__()
                and not self.pairs.__bool__()
                and not self.yeti.__bool__()
            )
        if defer_shift:
            self.pending_shift = True
        else:
            self._launch_shift_abmv()
        # `DerAtPoint.Clear(); Der2AtPoint.Clear();` (`:54-55`)
        self.cached_der2.clear()
        self.der_at_point.clear()
        self.current_point.clear()
        for i in range(shift_len):
            self.current_point.append(new_point[i])
        self.times.end(self.ctx, "est.move")

    def _launch_shift_abmv(mut self) raises:
        """The original MoveTo cursor update, factored so both the split
        schedule and DEVIATION 2030's flush launch the SAME kernel.

        their `AddBinModelValue(shift, bins, ...)` (`pointwise_oracle
        .cpp:50-52`): the FLAT kernel over rows, `CeilDivide(size,
        blockSize * elementsPerThreads)` blocks (`add_model_value.cu
        :60-62`). DEVIATION 210b in `kernel_add_model_value.mojo` has
        the 13.6-ms-per-call grid this replaces.
        """
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
        ALL numClasses. `checks/multilogit_check.mojo` gates that
        identity on the kernel directly, which is what makes this line
        safe to write rather than merely plausible.
        """
        if self.single_bin_dim == 1:
            self.enqueue_single_dim_evaluation()
            self.times.begin(self.ctx)
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
            self.finish_single_dim_evaluation(value, gradient)
            return
        self._write_multi_dim_value_and_first_derivatives(value, gradient)

    def enqueue_single_dim_evaluation(mut self) raises:
        """The single-dim arm of `write_value_and_first_derivatives` up to
        its drain: the evaluation launch, the per-leaf fold, and the two
        readback copies, ENQUEUED. `finish_single_dim_evaluation` reads them
        after the caller's drain. The ordinary method is exactly
        enqueue -> drain -> finish; the Ordered fit's batched estimation
        (`ordered_boosting.mojo`) enqueues every task's evaluation behind
        ONE drain instead of one each."""
        var blocks = (
            self.n_rows + MSE_BLOCK_SIZE - 1
        ) // MSE_BLOCK_SIZE
        if True:
            self.times.begin(self.ctx)
            # the QUERYWISE target (QueryRMSE): `ApproximateAt` through the
            # querywise der calcer (`permutation_der_calcer.h:192-205`),
            # the point read back to row order through `query.inverse`
            # and the der/der2 planes written at each row's bin position.
            if self.yeti.__bool__():
                # YetiRank: one `NextUniformL` per evaluation seeds the
                # call's task streams (`querywise_targets_impl.h:213-229`)
                launch_yeti_rank_with[True](
                    self.ctx, self.yeti.value(), self.d_cursor, True,
                    self.yeti_rng.next_uniform_l(),
                    self.d_eval_stats, self.d_fv, True,
                    self.d_mag_dummy, False,
                )
            elif self.pairs.__bool__():
                launch_pair_logit_with[True, False](
                    self.ctx, self.pairs.value(), self.d_cursor, True,
                    self.d_eval_stats, self.d_fv, True,
                    self.d_mag_dummy, False,
                )
            elif self.query.__bool__():
                launch_query_rmse_with[True](
                    self.ctx, self.query.value(), self.d_cursor, True,
                    self.d_eval_stats, self.d_fv, True,
                    self.d_mag_dummy, False,
                )
            else:
                # DEVIATION 2030: a deferred MoveTo is applied INSIDE this
                # evaluation -- the fused kernel performs the identical
                # per-row cursor add, stores the identical bytes, and runs
                # the unchanged evaluation body on the just-stored value.
                # `pending_shift` clears here because the cursor is current
                # from this launch on (queue order covers every later
                # reader). `comptime if`, the house elision: with the flag
                # off (every shipped build) the fused arm is not compiled
                # and the else arm is byte-for-byte the old call.
                comptime if FUSED_EST_MOVE_2030:
                    if self.pending_shift:
                        launch_approximate_move_eval[True](
                            self.ctx, self.objective,
                            self.d_shift, self.d_bins,
                            self.d_target, self.d_weights, Int32(self.n_rows),
                            self.d_cursor,
                            Int32(1) if self.has_weights else Int32(0),
                            self.alpha, self.border,
                            self.d_eval_stats, self.d_fv, Int32(1),
                            self.d_mag_dummy, Int32(0),
                            blocks,
                        )
                        self.pending_shift = False
                    else:
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
                else:
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
            # the widest leaf bounds every partition exactly
            # (`compute_partition_stats`' `row_bound`)
            compute_partition_stats(
                self.ctx, self.bin_count, 0, 2, self.n_rows,
                self.d_leaves, self.d_p_off, self.d_p_sz,
                self.d_eval_stats, self.d_partials, self.d_part_stats,
                sm_count=self.sm_count,
                row_bound=self.max_leaf_size,
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

    def finish_single_dim_evaluation(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ) raises:
        """The host half of the single-dim arm, after the drain that made
        `enqueue_single_dim_evaluation`'s copies runs."""
        if True:
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
            # cannot see -- a measurement caught ours accepting 8
            # rounds where 6-7 sit at the float32 noise floor. Found
            # 2026-08-22 in the Newton-walk audit; the walk-divergence
            # entry carries the measurement.
            var fv32 = Float32(0.0)
            for b in range(self.fv_blocks):
                fv32 += self.h_fv.unsafe_ptr().unsafe_load(b)
            value = Float64(fv32)

    def _write_multi_dim_value_and_first_derivatives(
        mut self, mut value: Float64, mut gradient: List[Float64]
    ) raises:
        """The rowSize > 1 arm of `write_value_and_first_derivatives`."""
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
        # DEVIATION 2030: no evaluation follows this move (Exact REPLACES
        # the walker), so a deferred shift would never be applied. Flush
        # it through the original kernel so the cursor holds exactly what
        # the split schedule leaves.
        @parameter
        if FUSED_EST_MOVE_2030:
            if self.pending_shift:
                self._launch_shift_abmv()
                self.pending_shift = False
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
        even though the cursor has `numClasses - 1` planes. An implementation that
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
        having to reproduce the reference `offset` bookkeeping, which is the part
        of that function most likely to be reproduced wrong.
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


# ================= DEVIATION 3041: the oracle's device buffers belong to the FIT =================
# MEASURED FIRST (RTX 4090, taxi 1,000,000 x 18, IDENTICAL, 101 symmetric
# trees under nsys, 2026-09-17):
# 1,476 `cuMemAlloc` and 1,476 `cuMemFree`, 58.9 + 57.9 ms, against 126.3 ms of
# GPU kernel time for the whole run. About 16 device allocations a tree, 12 of
# them the ones `make_bin_optimized_oracle` makes for every estimation task
# (three of `n_rows` or more), and every `cuMemFree` is a device drain.
# DEVIATION 1890 moved the estimator's gathers to a fit-owned pool of one and
# named the reference's reason (`TCudaManager` hands these out of a per-device
# memory pool, `cuda_lib/memory_pool.h`); this is the same repair for the
# oracle's own buffers (`OracleScratchPool`).
#
# WHY NO BIT MOVES. No kernel, grid, launch order, drain or host loop changes:
# the oracle reads and writes the same cells through handle copies of buffers
# the fit keeps, instead of through buffers it allocates. The key is EXACT (row
# count, bin count, cursor and leaf dimensions, value blocks, the resolved
# machine count), because several of these buffers are the `dst_buf` or
# `src_buf` of a whole-buffer copy; another bin count gets its own small
# buffers beside the shared `n_rows`-sized ones, another shape rebuilds. Every cell the
# oracle reads it has written first in the same task (`d_identity` is refilled
# by `launch_make_sequence`, `d_bins` by `fill_bins_from_partition_kernel`,
# `d_leaves` by its copy, the rest by the evaluation that reads them), which the
# identity lanes check: a read of a never-written cell would now see the
# previous tree's value where a fresh allocation showed whatever the driver
# left.
#
# The row: on by default on the NVIDIA column, where it was measured;
# `-D MOJOLEARN_3041_ORACLE_POOL=1` opts another column in and
# `-D MOJOLEARN_3041_ORACLE_ALLOC_PER_TREE=1` is the kill switch and the BEFORE
# arm. The caller decides (`oracle_scratch_pooled_for`); with no scratch passed
# this function allocates exactly as before.
# `-D MOJOLEARN_GBDT_ORACLE_POOL_SABOTAGE=1` is the negative control.
# ================================================================================================


def oracle_scratch_pooled_for[column: Int]() -> Bool:
    """SCHEDULING row (DEVIATION 3041)."""
    comptime if is_defined["MOJOLEARN_3041_ORACLE_ALLOC_PER_TREE"]():
        return False
    comptime if is_defined["MOJOLEARN_3041_ORACLE_POOL"]():
        return True
    return column == COLUMN_NVIDIA


comptime ORACLE_SCRATCH_POOLED = oracle_scratch_pooled_for[TARGET_COLUMN]()
#: negative control for DEVIATION 3041 (default off): a task that REUSES the
#: fit's buffers gets ONE cell of `d_bins` moved to another leaf after the
#: fill, in range. It stands in for the stale read this DEVIATION must never
#: cause (a reusing task reading the previous tree's bins). The first form of
#: this control skipped the fill outright; a previous tree's bins can index
#: past a smaller leaf count, and the arm aborted a column with
#: CUDA_ERROR_ILLEGAL_ADDRESS in gbdt-ordered-rmse (2026-09-17). A fit with
#: one leaf has no other leaf to move to and the control is inert there.
comptime ORACLE_POOL_SABOTAGE = is_defined["MOJOLEARN_GBDT_ORACLE_POOL_SABOTAGE"]()


def oracle_pool_sabotage_kernel(
    bins: MutPointer[UInt32, MutAnyOrigin], bin_count: Int32
):
    """The DEVIATION 3041 negative control: row 0 goes to the next leaf."""
    if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
        bins.unsafe_store(0, (bins.unsafe_load(0) + 1) % UInt32(bin_count))


struct OracleDeviceScratch(Movable):
    """The twelve device buffers `make_bin_optimized_oracle` needs, under
    their exact key (DEVIATION 3041). `handles()` gives views onto the same
    memory."""

    var n_rows: Int
    var bin_count: Int
    var cursor_dim: Int
    var multi_planes: Int
    var fv_blocks: Int
    var sm: Int
    var d_identity: DeviceBuffer[DType.uint32]
    var d_bins: DeviceBuffer[DType.uint32]
    var d_leaves: DeviceBuffer[DType.uint32]
    var d_shift: DeviceBuffer[DType.float32]
    var d_eval_stats: DeviceBuffer[DType.float32]
    var d_fv: DeviceBuffer[DType.float32]
    var d_mag_dummy: DeviceBuffer[DType.float32]
    var d_partials: DeviceBuffer[DType.float32]
    var d_multi_partials: DeviceBuffer[DType.float32]
    var d_part_stats: DeviceBuffer[DType.float32]
    var d_multi_der: DeviceBuffer[DType.float32]
    var d_multi_stats: DeviceBuffer[DType.float32]

    def __init__(
        out self,
        n_rows: Int,
        bin_count: Int,
        cursor_dim: Int,
        multi_planes: Int,
        fv_blocks: Int,
        sm: Int,
        var d_identity: DeviceBuffer[DType.uint32],
        var d_bins: DeviceBuffer[DType.uint32],
        var d_leaves: DeviceBuffer[DType.uint32],
        var d_shift: DeviceBuffer[DType.float32],
        var d_eval_stats: DeviceBuffer[DType.float32],
        var d_fv: DeviceBuffer[DType.float32],
        var d_mag_dummy: DeviceBuffer[DType.float32],
        var d_partials: DeviceBuffer[DType.float32],
        var d_multi_partials: DeviceBuffer[DType.float32],
        var d_part_stats: DeviceBuffer[DType.float32],
        var d_multi_der: DeviceBuffer[DType.float32],
        var d_multi_stats: DeviceBuffer[DType.float32],
    ):
        self.n_rows = n_rows
        self.bin_count = bin_count
        self.cursor_dim = cursor_dim
        self.multi_planes = multi_planes
        self.fv_blocks = fv_blocks
        self.sm = sm
        self.d_identity = d_identity^
        self.d_bins = d_bins^
        self.d_leaves = d_leaves^
        self.d_shift = d_shift^
        self.d_eval_stats = d_eval_stats^
        self.d_fv = d_fv^
        self.d_mag_dummy = d_mag_dummy^
        self.d_partials = d_partials^
        self.d_multi_partials = d_multi_partials^
        self.d_part_stats = d_part_stats^
        self.d_multi_der = d_multi_der^
        self.d_multi_stats = d_multi_stats^

    def matches(
        self, n_rows: Int, bin_count: Int, cursor_dim: Int, multi_planes: Int,
        fv_blocks: Int, sm: Int,
    ) -> Bool:
        return (
            self.n_rows == n_rows
            and self.bin_count == bin_count
            and self.cursor_dim == cursor_dim
            and self.multi_planes == multi_planes
            and self.fv_blocks == fv_blocks
            and self.sm == sm
        )

    def row_matches(self, n_rows: Int, multi_planes: Int, fv_blocks: Int) -> Bool:
        """The key of the `n_rows`-sized half."""
        return (
            self.n_rows == n_rows
            and self.multi_planes == multi_planes
            and self.fv_blocks == fv_blocks
        )

    def handles(self) -> OracleDeviceScratch:
        """Handle copies onto the same device memory."""
        return OracleDeviceScratch(
            self.n_rows, self.bin_count, self.cursor_dim, self.multi_planes,
            self.fv_blocks, self.sm,
            self.d_identity.copy(), self.d_bins.copy(), self.d_leaves.copy(),
            self.d_shift.copy(), self.d_eval_stats.copy(), self.d_fv.copy(),
            self.d_mag_dummy.copy(), self.d_partials.copy(),
            self.d_multi_partials.copy(), self.d_part_stats.copy(),
            self.d_multi_der.copy(), self.d_multi_stats.copy(),
        )


def _oracle_dims(objective: Int, num_classes: Int) -> Tuple[Int, Int]:
    """`(cursor_dim, single_bin_dim)`, the factory's own rule, for a caller
    that keys a scratch before the factory has validated `num_classes`."""
    if objective == OBJECTIVE_MULTICLASS:
        return (num_classes - 1, num_classes)
    if objective == OBJECTIVE_MULTICLASS_OVA:
        return (num_classes, num_classes)
    return (1, 1)


def _oracle_partials_len(bin_count: Int, sm: Int) -> Int:
    var chunks2 = partition_stats_chunks(sm, 2)
    var chunks1 = partition_stats_chunks(sm, 1)
    var partials_len = bin_count * 2 * chunks2
    if bin_count * 1 * chunks1 > partials_len:
        partials_len = bin_count * 1 * chunks1
    return partials_len


def _oracle_multi_partials_len(bin_count: Int, sm: Int, multi_planes: Int) -> Int:
    # the multi-dimensional reduce needs its own partials, sized for the
    # WIDEST stat count it will ever be asked for
    var multi_partials_len = 1
    for sc in range(1, multi_planes + 1):
        var need = bin_count * sc * partition_stats_chunks(sm, sc)
        if need > multi_partials_len:
            multi_partials_len = need
    return multi_partials_len


def make_oracle_device_scratch(
    ctx: DeviceContext,
    n_rows: Int,
    bin_count: Int,
    cursor_dim: Int,
    multi_planes: Int,
    fv_blocks: Int,
    sm: Int,
) raises -> OracleDeviceScratch:
    """The allocations `make_bin_optimized_oracle` made inline, in its order
    and at its sizes."""
    var d_identity = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var d_bins = ctx.enqueue_create_buffer[DType.uint32](n_rows)
    var d_leaves = ctx.enqueue_create_buffer[DType.uint32](bin_count)
    var d_shift = ctx.enqueue_create_buffer[DType.float32](
        bin_count * cursor_dim
    )
    var d_eval_stats = ctx.enqueue_create_buffer[DType.float32](2 * n_rows)
    var d_fv = ctx.enqueue_create_buffer[DType.float32](fv_blocks)
    var d_mag_dummy = ctx.enqueue_create_buffer[DType.float32](2)
    var d_partials = ctx.enqueue_create_buffer[DType.float32](
        _oracle_partials_len(bin_count, sm)
    )
    var d_multi_partials = ctx.enqueue_create_buffer[DType.float32](
        _oracle_multi_partials_len(bin_count, sm, multi_planes)
    )
    var d_part_stats = ctx.enqueue_create_buffer[DType.float32](
        2 * bin_count
    )
    # allocated ONCE per tree rather than per Hessian row (and, under
    # DEVIATION 3041, once per fit)
    var d_multi_der = ctx.enqueue_create_buffer[DType.float32](
        multi_planes * n_rows
    )
    var d_multi_stats = ctx.enqueue_create_buffer[DType.float32](
        multi_planes * bin_count
    )
    return OracleDeviceScratch(
        n_rows, bin_count, cursor_dim, multi_planes, fv_blocks, sm,
        d_identity^, d_bins^, d_leaves^, d_shift^, d_eval_stats^, d_fv^,
        d_mag_dummy^, d_partials^, d_multi_partials^, d_part_stats^,
        d_multi_der^, d_multi_stats^,
    )


def make_oracle_device_scratch_in(
    ctx: DeviceContext,
    mut arena: BufferArena,
    n_rows: Int,
    bin_count: Int,
    cursor_dim: Int,
    multi_planes: Int,
    fv_blocks: Int,
    sm: Int,
) raises -> OracleDeviceScratch:
    """`make_oracle_device_scratch` at the same sizes, carved from `arena`
    (`gbdt/gpu_util/arena.mojo`) instead of allocated one by one."""
    var d_identity = arena.device[DType.uint32](ctx, n_rows)
    var d_bins = arena.device[DType.uint32](ctx, n_rows)
    var d_leaves = arena.device[DType.uint32](ctx, bin_count)
    var d_shift = arena.device[DType.float32](ctx, bin_count * cursor_dim)
    var d_eval_stats = arena.device[DType.float32](ctx, 2 * n_rows)
    var d_fv = arena.device[DType.float32](ctx, fv_blocks)
    var d_mag_dummy = arena.device[DType.float32](ctx, 2)
    var d_partials = arena.device[DType.float32](
        ctx, _oracle_partials_len(bin_count, sm)
    )
    var d_multi_partials = arena.device[DType.float32](
        ctx, _oracle_multi_partials_len(bin_count, sm, multi_planes)
    )
    var d_part_stats = arena.device[DType.float32](ctx, 2 * bin_count)
    var d_multi_der = arena.device[DType.float32](ctx, multi_planes * n_rows)
    var d_multi_stats = arena.device[DType.float32](
        ctx, multi_planes * bin_count
    )
    return OracleDeviceScratch(
        n_rows, bin_count, cursor_dim, multi_planes, fv_blocks, sm,
        d_identity^, d_bins^, d_leaves^, d_shift^, d_eval_stats^, d_fv^,
        d_mag_dummy^, d_partials^, d_multi_partials^, d_part_stats^,
        d_multi_der^, d_multi_stats^,
    )


struct OracleHostScratch(Movable):
    """The host staging `make_bin_optimized_oracle` allocates per oracle
    (`h_leaves`, `h_shift`, `h_fv`, `h_part_stats`, `h_multi_stats`, and
    the deferred weight fold's `h_weight_stats`), supplied by a caller that
    keeps them for the fit. Every cell is written (by the host, or by a
    copy) before it is read, in each oracle's life."""

    var bin_count: Int
    var cursor_dim: Int
    var multi_planes: Int
    var fv_blocks: Int
    var h_leaves: HostBuffer[DType.uint32]
    var h_shift: HostBuffer[DType.float32]
    var h_fv: HostBuffer[DType.float32]
    var h_part_stats: HostBuffer[DType.float32]
    var h_multi_stats: HostBuffer[DType.float32]
    var h_weight_stats: HostBuffer[DType.float32]

    def __init__(
        out self,
        ctx: DeviceContext,
        mut arena: BufferArena,
        bin_count: Int,
        cursor_dim: Int,
        multi_planes: Int,
        fv_blocks: Int,
    ) raises:
        self.bin_count = bin_count
        self.cursor_dim = cursor_dim
        self.multi_planes = multi_planes
        self.fv_blocks = fv_blocks
        self.h_leaves = arena.host_buffer[DType.uint32](ctx, bin_count)
        self.h_shift = arena.host_buffer[DType.float32](
            ctx, bin_count * cursor_dim
        )
        self.h_fv = arena.host_buffer[DType.float32](ctx, fv_blocks)
        self.h_part_stats = arena.host_buffer[DType.float32](
            ctx, 2 * bin_count
        )
        self.h_multi_stats = arena.host_buffer[DType.float32](
            ctx, multi_planes * bin_count
        )
        # the size of `d_part_stats`, the WHOLE-BUFFER copy's source (the
        # weight fold fills its first `bin_count` cells): a staging buffer
        # of `bin_count` took a copy of twice its length, and the overrun
        # landed in whatever host memory came next -- a neighbour's
        # `h_leaves` whose upload had not run yet, which is how it was
        # found (a batched fit whose partitions all read leaf 0)
        self.h_weight_stats = arena.host_buffer[DType.float32](
            ctx, 2 * bin_count
        )

    def matches(
        self, bin_count: Int, cursor_dim: Int, multi_planes: Int,
        fv_blocks: Int,
    ) -> Bool:
        return (
            self.bin_count == bin_count
            and self.cursor_dim == cursor_dim
            and self.multi_planes == multi_planes
            and self.fv_blocks == fv_blocks
        )

    def handles(self) -> OracleHostScratch:
        """Handle copies onto the same host memory."""
        return OracleHostScratch(
            self.bin_count, self.cursor_dim, self.multi_planes,
            self.fv_blocks, self.h_leaves.copy(), self.h_shift.copy(),
            self.h_fv.copy(), self.h_part_stats.copy(),
            self.h_multi_stats.copy(), self.h_weight_stats.copy(),
        )

    def __init__(
        out self,
        bin_count: Int,
        cursor_dim: Int,
        multi_planes: Int,
        fv_blocks: Int,
        var h_leaves: HostBuffer[DType.uint32],
        var h_shift: HostBuffer[DType.float32],
        var h_fv: HostBuffer[DType.float32],
        var h_part_stats: HostBuffer[DType.float32],
        var h_multi_stats: HostBuffer[DType.float32],
        var h_weight_stats: HostBuffer[DType.float32],
    ):
        self.bin_count = bin_count
        self.cursor_dim = cursor_dim
        self.multi_planes = multi_planes
        self.fv_blocks = fv_blocks
        self.h_leaves = h_leaves^
        self.h_shift = h_shift^
        self.h_fv = h_fv^
        self.h_part_stats = h_part_stats^
        self.h_multi_stats = h_multi_stats^
        self.h_weight_stats = h_weight_stats^


def make_oracle_device_scratch_sharing_rows(
    ctx: DeviceContext,
    rows: OracleDeviceScratch,
    bin_count: Int,
    cursor_dim: Int,
    sm: Int,
) raises -> OracleDeviceScratch:
    """A scratch for another `bin_count`: fresh bin-sized buffers, and handle
    copies of `rows`' `n_rows`-sized ones."""
    var multi_planes = rows.multi_planes
    var d_leaves = ctx.enqueue_create_buffer[DType.uint32](bin_count)
    var d_shift = ctx.enqueue_create_buffer[DType.float32](
        bin_count * cursor_dim
    )
    var d_partials = ctx.enqueue_create_buffer[DType.float32](
        _oracle_partials_len(bin_count, sm)
    )
    var d_multi_partials = ctx.enqueue_create_buffer[DType.float32](
        _oracle_multi_partials_len(bin_count, sm, multi_planes)
    )
    var d_part_stats = ctx.enqueue_create_buffer[DType.float32](
        2 * bin_count
    )
    var d_multi_stats = ctx.enqueue_create_buffer[DType.float32](
        multi_planes * bin_count
    )
    return OracleDeviceScratch(
        rows.n_rows, bin_count, cursor_dim, multi_planes, rows.fv_blocks, sm,
        rows.d_identity.copy(), rows.d_bins.copy(), d_leaves^, d_shift^,
        rows.d_eval_stats.copy(), rows.d_fv.copy(), rows.d_mag_dummy.copy(),
        d_partials^, d_multi_partials^, d_part_stats^,
        rows.d_multi_der.copy(), d_multi_stats^,
    )


#: the bin-keyed half of the pool holds at most this many keys (a
#: non-symmetric fit sees a handful of leaf counts; each entry is a few KiB)
comptime ORACLE_POOL_MAX_BIN_KEYS = 128


struct OracleScratchPool(Movable):
    """The fit's pool (DEVIATION 3041), in two halves because the two halves
    have different keys. The ROW half (`d_identity`, `d_bins`, `d_eval_stats`,
    `d_fv`, `d_mag_dummy`, `d_multi_der`: everything of `n_rows` size) is a
    pool of ONE. The BIN half (`d_leaves`, `d_shift`, the partials, the part
    stats: a few KiB) is one entry per exact `bin_count`, because a
    non-symmetric fit's trees do not all have the same leaf count and these
    buffers are whole-buffer copy endpoints. An entry is a full
    `OracleDeviceScratch` whose row half is a handle onto the shared one."""

    var entries: List[OracleDeviceScratch]

    def __init__(out self):
        self.entries = List[OracleDeviceScratch]()

    def take(
        mut self,
        ctx: DeviceContext,
        n_rows: Int,
        bin_count: Int,
        objective: Int,
        num_classes: Int,
        sm_count: Int,
        pair_blocks: Int,
    ) raises -> Optional[OracleDeviceScratch]:
        """Handle views for this task's exact key, allocating what the pool
        does not hold. `pair_blocks` is the PairLogit value block count, 0
        otherwise."""
        var dims = _oracle_dims(objective, num_classes)
        if dims[0] < 1:
            # the factory refuses this `num_classes` in its own words
            return Optional[OracleDeviceScratch]()
        var fv_blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
        if pair_blocks > 0:
            fv_blocks = pair_blocks
        var sm = sm_count
        if sm < 0:
            sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        # a row-half key change (another dataset shape) empties the pool
        if len(self.entries) > 0 and not self.entries[0].row_matches(
            n_rows, dims[1], fv_blocks
        ):
            self.entries.clear()
        for i in range(len(self.entries)):
            if self.entries[i].matches(
                n_rows, bin_count, dims[0], dims[1], fv_blocks, sm
            ):
                return Optional(self.entries[i].handles())
        if len(self.entries) >= ORACLE_POOL_MAX_BIN_KEYS:
            self.entries.clear()
        if len(self.entries) == 0:
            self.entries.append(
                make_oracle_device_scratch(
                    ctx, n_rows, bin_count, dims[0], dims[1], fv_blocks, sm
                )
            )
        else:
            self.entries.append(
                make_oracle_device_scratch_sharing_rows(
                    ctx, self.entries[0], bin_count, dims[0], sm
                )
            )
        return Optional(self.entries[len(self.entries) - 1].handles())


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
    var query: Optional[QuerywiseTargetBuffers] = None,
    var pairs: Optional[PairwiseTargetBuffers] = None,
    var yeti: Optional[YetiRankTargetBuffers] = None,
    yeti_seed: UInt64 = UInt64(0),
    # DEVIATION 3041: handle views onto the fit's pool of one; None (every
    # check, and every column the row is off on) allocates as before
    var scratch: Optional[OracleDeviceScratch] = None,
    defer_weights: Bool = False,
    var host_scratch: Optional[OracleHostScratch] = None,
    leaves_ready: Bool = False,
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
    if (query.__bool__() or pairs.__bool__() or yeti.__bool__()) and estimation_method == LEAF_ESTIMATION_EXACT:
        # `ComputeExactValue`'s querywise arm
        # (`targets/permutation_der_calcer.h:206-216`), their message
        raise Error(
            "Exact leaves estimation method on GPU is not supported for"
            " non-pointwise target"
        )
    # one buffer wide enough for both the value pass (`cursor_dim`
    # planes) and the widest Hessian row (`single_bin_dim` columns)
    var multi_planes = single_bin_dim

    var blocks = (n_rows + MSE_BLOCK_SIZE - 1) // MSE_BLOCK_SIZE
    var fv_blocks = blocks
    if pairs.__bool__():
        fv_blocks = pairs.value().blocks()
    var sm = sm_count
    if sm < 0:
        sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    # DEVIATION 3041: the fit's buffers when the key is this task's, fresh
    # ones at the same sizes otherwise
    var have_scratch = False
    if scratch.__bool__():
        have_scratch = scratch.value().matches(
            n_rows, bin_count, cursor_dim, multi_planes, fv_blocks, sm
        )
    if not have_scratch:
        scratch = Optional(
            make_oracle_device_scratch(
                ctx, n_rows, bin_count, cursor_dim, multi_planes, fv_blocks, sm
            )
        )
    var ws = scratch.take()
    var d_identity = ws.d_identity.copy()
    var d_bins = ws.d_bins.copy()
    var d_leaves = ws.d_leaves.copy()
    var d_shift = ws.d_shift.copy()
    var d_eval_stats = ws.d_eval_stats.copy()
    var d_fv = ws.d_fv.copy()
    var d_mag_dummy = ws.d_mag_dummy.copy()
    var d_partials = ws.d_partials.copy()
    var d_multi_partials = ws.d_multi_partials.copy()
    var d_part_stats = ws.d_part_stats.copy()
    var d_multi_der = ws.d_multi_der.copy()
    var d_multi_stats = ws.d_multi_stats.copy()

    # `leaves_ready`: the caller's persistent scratch (`have_scratch`) whose
    # `d_leaves` already holds `[0, bin_count)` from its first use, on a
    # single-dimensional non-Exact walk, which never reads `d_identity`
    # (only the multiclass arms and the Exact scratch do). Neither buffer
    # is written by anything else, so both refills are skipped.
    var skip_static = leaves_ready and have_scratch
    if skip_static and (
        single_bin_dim != 1 or estimation_method == LEAF_ESTIMATION_EXACT
    ):
        raise Error("leaves_ready is for single-dim, non-Exact oracles only")
    if not skip_static:
        launch_make_sequence(ctx, UInt32(0), d_identity, n_rows)

    # the host staging: the caller's (`host_scratch`, kept for the fit) when
    # it matches this oracle's shape, else allocated here as always
    var have_host = False
    if host_scratch.__bool__():
        have_host = host_scratch.value().matches(
            bin_count, cursor_dim, multi_planes, fv_blocks
        )
    var h_leaves: HostBuffer[DType.uint32]
    if have_host:
        h_leaves = host_scratch.value().h_leaves.copy()
    else:
        h_leaves = ctx.enqueue_create_host_buffer[DType.uint32](bin_count)
    if not skip_static:
        for i in range(bin_count):
            h_leaves.unsafe_ptr().unsafe_store(i, UInt32(i))
        ctx.enqueue_copy(dst_buf=d_leaves, src_ptr=h_leaves.unsafe_ptr())

    var h_shift: HostBuffer[DType.float32]
    var h_fv: HostBuffer[DType.float32]
    if have_host:
        h_shift = host_scratch.value().h_shift.copy()
        h_fv = host_scratch.value().h_fv.copy()
    else:
        h_shift = ctx.enqueue_create_host_buffer[DType.float32](
            bin_count * cursor_dim
        )
        h_fv = ctx.enqueue_create_host_buffer[DType.float32](fv_blocks)

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
    comptime if ORACLE_POOL_SABOTAGE:
        # the negative control: a reusing task's row 0 moves to the next
        # leaf, in range (see the comptime above)
        if have_scratch:
            ctx.enqueue_function[oracle_pool_sabotage_kernel](
                d_bins.unsafe_ptr(), Int32(bin_count),
                grid_dim=(1, 1, 1),
                block_dim=(1, 1, 1),
            )

    # `d_partials`, `d_multi_partials`, `d_part_stats`, `d_multi_der` and
    # `d_multi_stats` are sized by `make_oracle_device_scratch` (DEVIATION 3041)
    var h_part_stats: HostBuffer[DType.float32]
    var h_multi_stats: HostBuffer[DType.float32]
    if have_host:
        h_part_stats = host_scratch.value().h_part_stats.copy()
        h_multi_stats = host_scratch.value().h_multi_stats.copy()
    else:
        h_part_stats = ctx.enqueue_create_host_buffer[DType.float32](
            2 * bin_count
        )
        h_multi_stats = ctx.enqueue_create_host_buffer[DType.float32](
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
    # the widest leaf, an exact bound on every partition the weight fold
    # reads (`compute_partition_stats`' `row_bound`)
    var widest_leaf = 0
    for i in range(bin_count):
        if leaf_sizes[i] > widest_leaf:
            widest_leaf = leaf_sizes[i]
    # `defer_weights`: the same fold, copied to its OWN host buffer and read
    # by `settle_weights` after the caller's next drain instead of draining
    # here (the evaluation's readback reuses `h_part_stats`, so the two
    # copies may not share it while both are in flight)
    var h_weight_stats = Optional[HostBuffer[DType.float32]]()
    if has_weights and defer_weights:
        compute_partition_stats(
            ctx, bin_count, 0, 1, n_rows,
            d_leaves, d_p_off, d_p_sz,
            d_weights, d_partials, d_part_stats,
            sm_count=sm,
            row_bound=widest_leaf,
        )
        var h_w: HostBuffer[DType.float32]
        if have_host:
            h_w = host_scratch.value().h_weight_stats.copy()
        else:
            # `d_part_stats`' length: the copy below is whole-buffer
            h_w = ctx.enqueue_create_host_buffer[DType.float32](2 * bin_count)
        ctx.enqueue_copy(dst_ptr=h_w.unsafe_ptr(), src_buf=d_part_stats)
        h_weight_stats = Optional(h_w^)
    elif has_weights:
        compute_partition_stats(
            ctx, bin_count, 0, 1, n_rows,
            d_leaves, d_p_off, d_p_sz,
            d_weights, d_partials, d_part_stats,
            sm_count=sm,
            row_bound=widest_leaf,
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
        False,  # pending_shift (DEVIATION 2030): no deferred move yet
        query^,
        pairs^,
        fv_blocks,
        yeti^,
        TRandom(yeti_seed),
        h_weight_stats^,
    )
